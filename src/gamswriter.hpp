// Copyright (C) 2022 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__GAMSWRITER_HPP
#define MC__GAMSWRITER_HPP

#include <stdexcept>
#include <cassert>
#include <fstream>

#include "base_opt.hpp"
#include "polimage.hpp"

#undef MC__GAMSWRITER_DEBUG
#undef MC__GAMSWRITER_TRACE

namespace mc
{
//! @brief C++ class for exporting a CANON reformulated model into GAMS
////////////////////////////////////////////////////////////////////////
//! mc::GAMSWRITER is a C++ class for exporting a reformulated CANON
//! model into GAMS language.
////////////////////////////////////////////////////////////////////////
template< typename T>
class GAMSWRITER
: public virtual BASE_OPT
{  
  // Typedef for variable set
  typedef std::set< PolVar<T> const*, lt_PolVar<T> > t_Var;

public:

  //! @brief Model type
  enum MODELTYPE{
     LIN=0,  //!< Linear program
     QUAD,   //!< Quadratically constrained program
     NLIN,   //!< Nonlinear program
  };

protected:

  //! @brief Model type
  MODELTYPE _type;

  //! @brief Objective direction
  t_OBJ _DirObj;

  //! @brief Objective variable
  std::string _VarObj;

  //! @brief Continuous variable declaration
  std::ostringstream _CVarDec;

  //! @brief Binary variable declaration
  std::ostringstream _BVarDec;

  //! @brief Integer variable declaration
  std::ostringstream _IVarDec;

  //! @brief Variable bounds
  std::ostringstream _VarBnd;

  //! @brief Variable Fixings
  std::ostringstream _VarFix;

  //! @brief Variable levels
  std::ostringstream _VarIni;

  //! @brief Equation declaration
  std::ostringstream _EqnDec;

  //! @brief Equation definition
  std::ostringstream _EqnDef;

  //! @brief Equation counter
  unsigned long _EqnCnt;

  //! @brief set of variables in GAMS model
  t_Var _GAMSvar;

  //! @brief Polyhedral image environment
  PolImg<T>* _POLenv;

public:

  //! @brief Constructor
  GAMSWRITER
    ()
    : _VarObj( "objvar" ), _EqnCnt( 0 ), _POLenv( nullptr )
    {}

  //! @brief Destructor
  virtual ~GAMSWRITER()
    {}

  //! @brief Reset GAMS model
  void reset
    ();

  //! @brief Write GAMS model to file
  bool write
    ( std::string const filename );

  //! @brief Set variables and cuts in GAMS model
  template <typename T>
  void set_cuts
    ( PolImg<T>* env, bool const reset_=true );

  //! @brief Set objective variable and direction in GAMS model
  template <typename T>
  void set_objective
    ( PolVar<T> const& pObj, t_OBJ const& tObj, bool const appvar_=true );

  //! @brief Set starting point and branch priority of PolImg variable <a>X</a>
  template <typename T>
  bool set_variable
    ( PolVar<T> const& X, double const* pval ); //, unsigned const priority );

protected:

  //! @brief Append variable to MIP model
  template <typename T>
  std::pair<typename t_Var::iterator,bool> _add_var
    ( PolVar<T> const* pVar );

  //! @brief Append constraint to MIP model
  template <typename T>
  void _add_cut
    ( PolCut<T> const* pCut );

  //! @brief Append constraint to MIP model
  template <typename T>
  void _add_cut
    ( PolCut<T> const* pCut, char GRB_Type );

private:

  //! @brief Vector of Gurobi variables in linear terms
  std::vector<GRBVar> _cutvar;

  //! @brief Vector of Gurobi variables in quadratic terms
  std::vector<GRBVar> _cutqvar1;

  //! @brief Vector of Gurobi variables in quadratic terms
  std::vector<GRBVar> _cutqvar2;
};

inline void
GAMSWRITER::reset
()
{
  _type = LIN;
  _EqnCnt = 0;
  _CVarDec.clear(); _CVarDec.set("");
  _BVarDec.clear(); _BVarDec.set("");
  _IVarDec.clear(); _IVarDec.set("");
  _EqnDec.clear();  _EqnDec.set("");
  _EqnDef.clear();  _EqnDef.set("");
  _VarBnd.clear();  _VarBnd.set("");
  _VarFix.clear();  _VarFix.set("");
  _VarIni.clear();  _VarIni.set("");
}

inline bool
GAMSWRITER::write
( std::string const filename )
{
  // Create GAMS file
  std::ofstream GAMSmodel;
  GAMSmodel.open( filename, std::ios::trunc );
  if( !GAMSmodel.is_open() ){
    std::cerr << "Could not create GAMS model. Do you have write permissions in execution directory?"
              << std::endl;
    return false;
  }

  // Write GAMS model to file
  GAMSmode << "VARIABLE " << _VarObj << ";" << std::endl;
  if( _CVarDec.tellp() > 0 )
    GAMSmodel << "VARIABLES " << _CVarDec.str() << ";" << std::endl << std::endl;
  if( _BVarDec.tellp() > 0 )
    GAMSmodel << "BINARY VARIABLES " << _BVarDec.str() << ";" << std::endl << std::endl;
  if( _IVarDec.tellp() > 0 )
    GAMSmodel << "INTEGER VARIABLES " << _IVarDec.str() << ";" << std::endl << std::endl;
  GAMSmodel << "EQUATIONS " << _EqnDec.str() << ";" << std::endl << std::endl;
  GAMSmodel << _EqnDef.str() << std::endl << std::endl;
  GAMSmodel << _VarBnd.str() << std::endl << std::endl;
  GAMSmodel << _VarIni.str() << std::endl << std::endl;
  GAMSmodel << _VarFix.str() << std::endl << std::endl;
  GAMSmodel << "MODEL canon / ALL /;" << std::endl;
  GAMSmodel << "SOLVE canon USING ";
  switch( _type ){
    case LIN  : GAMSmodel << (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MIP"  : "LP" ; break;
    case QUAD : GAMSmodel << (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MIQCP": "QCP"; break;
    case NLIN : GAMSmodel << (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MINLP": "NLP"; break;
  };
  GAMSmodel << " / ALL /;" << std::endl;

  // Close GAMS file
  GAMSmodel.close();
  return true;
}

template <typename T>
inline void
GAMSWRITER::set_cuts
( PolImg<T>* env, bool const reset_ )
{
  if( reset_ ) reset();
  _POLenv = env;

  // Add participating variables into Gurobi model
  for( auto && pvar : _POLenv->Vars() ){
    if( !reset_ && _GAMSvar.find( pvar.second ) != _GAMSvar.end() ) continue;
    _add_var( pvar.second );
  }
  for( auto && paux : _POLenv->Aux() ){
    if( !reset_ && _GAMSvar.find( paux ) != _GAMSvar.end() ) continue;
    _add_var( paux );
  }

  // Add cuts into Gurobi model
  for( auto && pcut : _POLenv->Cuts() ){
#ifdef MC__GAMSWRITER_DEBUG
    std::cout << *pcut << std::endl;
#endif
    _add_cut( pcut );
  }
}

template <typename T>
inline void
GAMSWRITER::_add_var
( PolVar<T> const* pVar )
{
  switch( pVar->id().first ){
    case PolVar<T>::AUXCST:
      _CVarDec << (_CVarDec.tellp()>0?", ":" ") << pVar->name();
      _VarFix  << pVar->name() << ".FX = " << Op<T>::mid(pVar->range()) << ";" << std::endl;
      break;

    case PolVar<T>::VARCONT:
    case PolVar<T>::AUXCONT:
      _CVarDec << (_CVarDec.tellp()>0?", ":" ") << pVar->name();
      if( Op<T>::l(pVar->range()) > -0.999*BASE_OPT::INF )
        _VarBnd << pVar->name() << ".LO = " << Op<T>::l(pVar->range()) << ";" << std::endl;
      if( Op<T>::u(pVar->range()) <  0.999*BASE_OPT::INF )
        _VarBnd << pVar->name() << ".UP = " << Op<T>::u(pVar->range()) << ";" << std::endl;
      break;
      
    case PolVar<T>::VARINT:
    case PolVar<T>::AUXINT:
      if( isequal( Op<T>::l(pVar->range()), 0. ) && isequal( Op<T>::u(pVar->range()), 1. ) )
        _BVarDec << (_BVarDec.tellp()>0?", ":" ") << pVar->name();
      else{
        _IVarDec << (_IVarDec.tellp()>0?", ":" ") << pVar->name();
        if( !isequal( Op<T>::l(pVar->range()), 0. ) )
          _VarBnd << pVar->name() << ".LO = " << Op<T>::l(pVar->range()) << ";" << std::endl;
        if( !isequal( Op<T>::u(pVar->range()), 100. ) )
          _VarBnd << pVar->name() << ".UP = " << Op<T>::u(pVar->range()) << ";" << std::endl;
      }
      break;

    default:
      throw std::runtime_error("GAMSWRITER - Error: Unsupported variable type");
  }
}

template <typename T>
inline bool
GAMSWRITER::set_variable
( PolVar<T> const& pVar, double const* pVal )//, unsigned const priority )
{
  auto itv = _GAMSvar.find( const_cast<PolVar<T>*>(&pVar) );
  if( itv == _GAMSvar.end() ) return false;
  if( pVal ) _VarIni << pVar.name() << ".L = " << *pVal << ";" << std::endl;
  //itv->second.set( GRB_IntAttr_BranchPriority, (int)priority );
  return true;
}

template <typename T>
inline void
GAMSWRITER::set_objective
( PolVar<T> const& pObj, t_OBJ const& tObj )
{
  _DirObj = tObj;
  if( _GAMSvar.find( &pObj ) == _GAMSvar.end() ) _add_var( &pObj );
  _EndDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++EqnCnt;
  _EqnDef << "E" << EqnCnt << " .. " << _VarObj << " =E= " << pObj.name() << ";" << std::endl;
}

//template <typename T>
//inline typename MIPSLV_GUROBI<T>::t_MIPCtr
//MIPSLV_GUROBI<T>::add_constraint
//( PolVar<T> const& pCtr, t_CTR const& tCtr, double const rhs,
//  bool const appvar_ )
//{
//  // Set constraint
//  auto jtctr = _MIPvar.find( &pCtr );
//  if( jtctr == _MIPvar.end() && appvar_ )
//      jtctr = _add_var( &pCtr ).first;
//  else if( jtctr == _MIPvar.end() )
//    throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in constraint");
//  GRBLinExpr lhs( jtctr->second );
//  GRBConstr ctr;
//  switch( tCtr ){
//    case EQ: ctr = _GRBmodel->addConstr( lhs, GRB_EQUAL,         rhs ); break;
//    case LE: ctr = _GRBmodel->addConstr( lhs, GRB_LESS_EQUAL,    rhs ); break;
//    case GE: ctr = _GRBmodel->addConstr( lhs, GRB_GREATER_EQUAL, rhs ); break;
//  }
//  return ctr;
//}

//template <typename T>
//inline typename MIPSLV_GUROBI<T>::t_MIPCtr
//MIPSLV_GUROBI<T>::add_constraint
//( unsigned const nCtr, PolVar<T> const* pCtr, double const* cCtr,
//  t_CTR const& tCtr, double const rhs, bool const appvar_ )
//{
//  // Set constraint
//  GRBLinExpr lhs;
//  _cutvar.resize( nCtr );
//  for( unsigned k=0; k<nCtr; k++ ){
//    auto jtctr = _MIPvar.find( &pCtr[k] );
//    if( jtctr == _MIPvar.end() && appvar_)
//        jtctr = _add_var( &pCtr[k] ).first;
//    else if( jtctr == _MIPvar.end() )
//      throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in constraint");
//    _cutvar[k] = jtctr->second;
//  }
//  lhs.addTerms( cCtr, _cutvar.data(), nCtr );
//  GRBConstr ctr;
//  switch( tCtr ){
//    case EQ: ctr = _GRBmodel->addConstr( lhs, GRB_EQUAL,         rhs ); break;
//    case LE: ctr = _GRBmodel->addConstr( lhs, GRB_LESS_EQUAL,    rhs ); break;
//    case GE: ctr = _GRBmodel->addConstr( lhs, GRB_GREATER_EQUAL, rhs ); break;
//  }
//  return ctr;
//}

//template <typename T>
//inline typename MIPSLV_GUROBI<T>::t_MIPCtr
//MIPSLV_GUROBI<T>::dummy_constraint
//()
//{
//  GRBConstr ctr;
//  return ctr;
//}

template <typename T>
inline void
MIPSLV_GUROBI<T>::_add_cut
( PolCut<T> const* pCut, char GRB_Type )
{
  if( pCut->nqvar() ){
    GRBQuadExpr qlhs;
    qlhs.addTerms( pCut->qcoef(), _cutqvar1.data(), _cutqvar2.data(), pCut->nqvar() );
    if( pCut->nvar() )
      qlhs.addTerms( pCut->coef(), _cutvar.data(), pCut->nvar() );
    _GRBmodel->addQConstr( qlhs, GRB_Type, pCut->rhs() );
  }
  else{
    GRBLinExpr lhs;
    lhs.addTerms( pCut->coef(), _cutvar.data(), pCut->nvar() );
    _GRBmodel->addConstr( lhs, GRB_Type, pCut->rhs() );        
  }
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::_add_cut
( PolCut<T> const* pCut )
{
  // Check valid cut
  if( !pCut->nvar() && !pCut->nqvar() ){
    std::cout << *pCut << std::endl;
    throw std::runtime_error("MIPSLV_GUROBI - Error: Invalid cut with no variable participating");
  }

  // Form vector of variables in linear terms or SOS
  _cutvar.resize( pCut->nvar() );
  for( unsigned k=0; k<pCut->nvar(); k++ ){
    auto itvar = _MIPvar.find( pCut->var()+k );
    //if( itvar->first->var().cst() )
    //  std::cout << "CONSTANT VARIABLE: " << itvar->first->var() << std::endl;
    if( itvar == _MIPvar.end() ){//|| itvar->first->var().cst() )
      std::cerr << *pCut << std::endl;
      throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in cut");
    }
    _cutvar[k] = itvar->second;
  }

  // Form vector of variables in quadratic terms
  _cutqvar1.resize( pCut->nqvar() );
  _cutqvar2.resize( pCut->nqvar() );
  for( unsigned k=0; k<pCut->nqvar(); k++ ){
    auto itvar = _MIPvar.find( pCut->qvar1()+k );
    auto jtvar = _MIPvar.find( pCut->qvar2()+k );
    if( itvar == _MIPvar.end() || jtvar == _MIPvar.end() ){
      std::cerr << *pCut << std::endl;
      throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in cut");
    }
    _cutqvar1[k] = itvar->second;
    _cutqvar2[k] = jtvar->second;
  }

  // Add contraint to MIP model
  //GRBConstr ctr;
  try{
    switch( pCut->type() ){
      case PolCut<T>::EQ:
        _add_cut( pCut, GRB_EQUAL );
        break;

      case PolCut<T>::LE:
        _add_cut( pCut, GRB_LESS_EQUAL );
        break;

      case PolCut<T>::GE:
        _add_cut( pCut, GRB_GREATER_EQUAL );
        break;

      case PolCut<T>::SOS1:
        _GRBmodel->addSOS( _cutvar.data(), pCut->coef(), pCut->nvar(), GRB_SOS_TYPE1 );
        if( pCut->nqvar() )
          throw std::runtime_error("MIPSLV_GUROBI - Error: Invalid SOS1 cut with quadratic terms");
        break;

      case PolCut<T>::SOS2:
        _GRBmodel->addSOS( _cutvar.data(), pCut->coef(), pCut->nvar(), GRB_SOS_TYPE2 );
        if( pCut->nqvar() )
          throw std::runtime_error("MIPSLV_GUROBI - Error: Invalid SOS2 cut with quadratic terms");
        break;
        
      case PolCut<T>::NLIN:
        if( _cutvar.size() < 2 || _cutvar.size() > 3 )
          throw std::runtime_error("MIPSLV_GUROBI - Error: Incorrect number of variables in nonlinear cut");
        switch( pCut->op()->type ){
          case FFOp::IPOW:
          case FFOp::DPOW:{
            double const& dExp = pCut->op()->pops[1]->num().val();
            if( dExp < 0 ) throw std::runtime_error("MIPSLV_GUROBI - Error: Nonlinear cut not yet implemented");
            _GRBmodel->addGenConstrPow( _cutvar[1], _cutvar[0], dExp, "", options.pwl() );
            break;}
          //case FFOp::DPOW:{
            //unsigned const ncoef = pCut->op()->pops[1]->num().n+1;
            //std::vector<double> coef( ncoef, 0. ); coef[0] = 1.;
            //_GRBmodel->addGenConstrPoly( _cutvar[1], _cutvar[0], ncoef, coef.data(), "", options.pwl() );
            //break;}
          case FFOp::CHEB:{
            unsigned const ncoef = pCut->op()->pops[1]->num().n+1;
            std::vector<double>&& coef = chebcoef( ncoef-1 );
            _GRBmodel->addGenConstrPoly( _cutvar[1], _cutvar[0], ncoef, coef.data(), "", options.pwl() );
            break;}
          case FFOp::SQR:{
            _GRBmodel->addGenConstrPow( _cutvar[1], _cutvar[0], 2, "", options.pwl() );
            //unsigned const ncoef = 3;
            //std::vector<double> coef( ncoef, 0. ); coef[0] = 1.;
            //_GRBmodel->addGenConstrPoly( _cutvar[1], _cutvar[0], ncoef, coef.data(), "", options.pwl() );
            break;}
          case FFOp::SQRT:{
            _GRBmodel->addGenConstrPow( _cutvar[1], _cutvar[0], 0.5, "", options.pwl() );
            //const unsigned ncoef = 3;
            //std::vector<double> coef( ncoef, 0. ); coef[0] = 1.;
            //_GRBmodel->addGenConstrPoly( _cutvar[0], _cutvar[1], ncoef, coef.data(), "", options.pwl() );
            break;}
          case FFOp::EXP:
            _GRBmodel->addGenConstrExp( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;
          case FFOp::LOG:
            _GRBmodel->addGenConstrLog( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;
          case FFOp::COS:
            _GRBmodel->addGenConstrCos( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;
          case FFOp::SIN:
            _GRBmodel->addGenConstrSin( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;
          case FFOp::TAN:
            _GRBmodel->addGenConstrTan( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;
          case FFOp::ACOS:
            _GRBmodel->addGenConstrCos( _cutvar[0], _cutvar[1], "", options.pwl() );
            break;
          case FFOp::ASIN:
            _GRBmodel->addGenConstrSin( _cutvar[0], _cutvar[1], "", options.pwl() );
            break;
          case FFOp::ATAN:
            _GRBmodel->addGenConstrTan( _cutvar[0], _cutvar[1], "", options.pwl() );
            break;
          case FFOp::FABS:
            _GRBmodel->addGenConstrAbs( _cutvar[0], _cutvar[1] );
            break;
          case FFOp::MINF:
            if( pCut->nvar() > 2 )
              _GRBmodel->addGenConstrMin( _cutvar[0], _cutvar.data()+1, pCut->nvar()-1 );
            else
              _GRBmodel->addGenConstrMin( _cutvar[0], _cutvar.data()+1, pCut->nvar()-1, pCut->rhs() );
            break;
          case FFOp::MAXF:
            if( pCut->nvar() > 2 )
              _GRBmodel->addGenConstrMax( _cutvar[0], _cutvar.data()+1, pCut->nvar()-1 );
            else
              _GRBmodel->addGenConstrMax( _cutvar[0], _cutvar.data()+1, pCut->nvar()-1, pCut->rhs() );
            break;
          default:
            throw std::runtime_error("MIPSLV_GUROBI - Error: Nonlinear cut not yet implemented");
        }
    }
    //_MIPcut.insert( std::make_pair( pCut, ctr ) );
  }

  catch( GRBException& e ){
    //if( options.DISPLEVEL )
      std::cout << "GRBException - Error code: " << e.getErrorCode() << std::endl
                << e.getMessage() << std::endl;
    throw;
  }
}

} // end namespace mc

#endif

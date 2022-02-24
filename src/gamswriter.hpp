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

//#define MC__GAMSWRITER_DEBUG

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
  void set_cuts
    ( PolImg<T>* env, bool const reset_=true );

  //! @brief Set objective variable and direction in GAMS model
  void set_objective
    ( PolVar<T> const& pObj, t_OBJ const& tObj );

  //! @brief Set starting point and branch priority of PolImg variable <a>X</a>
  bool set_variable
    ( PolVar<T> const& X, double const* pval ); //, unsigned const priority );

protected:

  //! @brief Append variable to MIP model
  void _add_var
    ( PolVar<T> const* pVar );

  //! @brief Append constraint to MIP model
  void _add_cut
    ( PolCut<T> const* pCut );

  //! @brief Append constraint to MIP model
  std::string _lhs_cut
    ( PolCut<T> const* pCut );
};

template <typename T>
inline void
GAMSWRITER<T>::reset
()
{
  _type = LIN;
  _EqnCnt = 0;
  _CVarDec.clear(); _CVarDec.str("");
  _BVarDec.clear(); _BVarDec.str("");
  _IVarDec.clear(); _IVarDec.str("");
  _EqnDec.clear();  _EqnDec.str("");
  _EqnDef.clear();  _EqnDef.str("");
  _VarBnd.clear();  _VarBnd.str(""); _VarBnd << std::setprecision(16);
  _VarFix.clear();  _VarFix.str(""); _VarFix << std::setprecision(16);
  _VarIni.clear();  _VarIni.str(""); _VarIni << std::setprecision(16);
}

template <typename T>
inline bool
GAMSWRITER<T>::write
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
  GAMSmodel << "VARIABLE " << _VarObj << ";" << std::endl;
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
    case LIN  : GAMSmodel << (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MIP"  : "LP") ; break;
    case QUAD : GAMSmodel << (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MIQCP": "QCP"); break;
    case NLIN : GAMSmodel << (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MINLP": "NLP"); break;
  };
  switch( _DirObj ){
    case MIN  : GAMSmodel << " MINIMIZING "; break;
    case MAX  : GAMSmodel << " MAXIMIZING "; break;
  }
  GAMSmodel << _VarObj << ";" << std::endl;

  // Close GAMS file
  GAMSmodel.close();
  return true;
}

template <typename T>
inline void
GAMSWRITER<T>::set_cuts
( PolImg<T>* env, bool const reset_ )
{
  if( reset_ ) reset();

  // Add cuts into GAMS model
  _POLenv = env;
  for( auto && pcut : _POLenv->Cuts() ){
#ifdef MC__GAMSWRITER_DEBUG
    std::cout << *pcut << std::endl;
#endif
    _add_cut( pcut );
#ifdef MC__GAMSWRITER_DEBUG
    std::cout << "VARIABLE " << _VarObj << ";" << std::endl;
    std::cout << "VARIABLES " << _CVarDec.str() << ";" << std::endl << std::endl;
    std::cout << "BINARY VARIABLES " << _BVarDec.str() << ";" << std::endl << std::endl;
    std::cout << "INTEGER VARIABLES " << _IVarDec.str() << ";" << std::endl << std::endl;
    std::cout << "EQUATIONS " << _EqnDec.str() << ";" << std::endl << std::endl;
    std::cout << _EqnDef.str() << std::endl << std::endl;
#endif
  }
}

template <typename T>
inline void
GAMSWRITER<T>::_add_var
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
      if( Op<T>::l(pVar->range()) > -1. && Op<T>::u(pVar->range()) < 2. )
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

  _GAMSvar.insert( pVar );
}

template <typename T>
inline std::string
GAMSWRITER<T>::_lhs_cut
( PolCut<T> const* pCut )//, std::ostringstream& oss )
{
  std::ostringstream lhs;

  // Add linear terms to lhs
  for( unsigned k=0; k<pCut->nvar(); k++ ){
    if( pCut->coef()[k] == 0. ) continue;
    // Append term to lhs
    if( pCut->coef()[k] < 0. )
      lhs << " - ";
    else if( lhs.tellp() > 0 && pCut->coef()[k] > 0. )
      lhs << " + ";
    if( std::fabs(pCut->coef()[k]) != 1. )
      lhs << std::fabs(pCut->coef()[k]) << "*";
    lhs << pCut->var()[k].name();
  }

  // Add quadratic terms to lhs
  for( unsigned k=0; k<pCut->nqvar(); k++ ){
    if( pCut->qcoef()[k] == 0. ) continue;
    // Append term to lhs
    if( pCut->qcoef()[k] < 0. )
      lhs << " - ";
    else if( lhs.tellp() > 0 && pCut->qcoef()[k] > 0. )
      lhs << " + ";
    if( std::fabs(pCut->qcoef()[k]) != 1. )
      lhs << std::fabs(pCut->qcoef()[k]) << "*";
    if( pCut->qvar1()[k].name() == pCut->qvar2()[k].name() )
      lhs << "SQR(" << pCut->qvar1()[k].name() << ")";
    else
      lhs << pCut->qvar1()[k].name() << "*" << pCut->qvar2()[k].name();
    if( _type < QUAD ) _type = QUAD;
  }
  return lhs.str();
}

template <typename T>
inline void
GAMSWRITER<T>::_add_cut
( PolCut<T> const* pCut )
{
  // Check valid cut
  if( !pCut->nvar() && !pCut->nqvar() ){
    std::cout << *pCut << std::endl;
    throw std::runtime_error("GAMSWRITER - Error: Invalid cut without participating variables");
  }

  // Declare equation and initiate definition
  _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
  _EqnDef << "E" << _EqnCnt << " .. ";// << _VarObj << " =E= " << pObj.name() << ";" << std::endl;
  
  // Add participating variables to GAMS model
  for( unsigned k=0; k<pCut->nvar(); k++ ){
    if( _GAMSvar.find( pCut->var()+k ) == _GAMSvar.end() )
      _add_var( pCut->var()+k );
  }
  for( unsigned k=0; k<pCut->nqvar(); k++ ){
    if( _GAMSvar.find( pCut->qvar1()+k ) == _GAMSvar.end() )
      _add_var( pCut->qvar1()+k );
    if( _GAMSvar.find( pCut->qvar2()+k ) == _GAMSvar.end() )
      _add_var( pCut->qvar2()+k );
  }

  // Add contraint to MIP model
  switch( pCut->type() ){

    case PolCut<T>::EQ:
      _EqnDef << _lhs_cut( pCut ) << " =E= " << pCut->rhs() << ";" << std::endl;
      break;

    case PolCut<T>::LE:
      _EqnDef << _lhs_cut( pCut ) << " =L= " << pCut->rhs() << ";" << std::endl;
      break;

    case PolCut<T>::GE:
      _EqnDef << _lhs_cut( pCut ) << " =G= " << pCut->rhs() << ";" << std::endl;
      break;

    case PolCut<T>::SOS1:
    case PolCut<T>::SOS2:
      throw std::runtime_error("GAMSWRITER - Error: SOS variable not supported");
        
    case PolCut<T>::NLIN:
      if( pCut->nvar() < 2 || pCut->nvar() > 3 )
        throw std::runtime_error("GAMSWRITER - Error: Incorrect number of variables in nonlinear cut");
      switch( pCut->op()->type ){

        case FFOp::IPOW:
          if( pCut->op()->pops[1]->num().n < 0 )
            _EqnDef << pCut->var()[0].name() << " * POWER(" << pCut->var()[1].name()
                    << "," << -pCut->op()->pops[1]->num().n << ") =E= 1;" << std::endl;
          else if( pCut->op()->pops[1]->num().n == 0 )
            _EqnDef << pCut->var()[0].name() << " =E= 1;" << std::endl;
          else
            _EqnDef << pCut->var()[0].name() << " - POWER(" << pCut->var()[1].name()
                    << "," << pCut->op()->pops[1]->num().n << ") =E= 0;" << std::endl;
          break;

        case FFOp::DPOW:{
          _EqnDef << pCut->var()[0].name() << " - " << pCut->var()[1].name()
                  << "**" << pCut->op()->pops[1]->num().x << " =E= 0;" << std::endl;
          break;

        case FFOp::CHEB:{
          unsigned const ncoef = pCut->op()->pops[1]->num().n+1;
          std::vector<double>&& coef = chebcoef( ncoef-1 );
          _EqnDef << pCut->var()[0].name() << " - POLY(" << pCut->var()[1].name();
          for( int k=ncoef; k>0; ) _EqnDef << "," << coef[--k];
          for( int k=ncoef; k<3; ++k ) _EqnDef << ",0"; // at least quadratic
          _EqnDef << ") =E= 0;" << std::endl;
          break;}

        case FFOp::SQR:
          _EqnDef << pCut->var()[0].name() << " - SQR(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::SQRT:
          _EqnDef << pCut->var()[0].name() << " - SQRT(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::EXP:
          _EqnDef << pCut->var()[0].name() << " - EXP(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::LOG:
          _EqnDef << pCut->var()[0].name() << " - LOG(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::XLOG:
          _EqnDef << pCut->var()[0].name() << " + ENTROPY(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::COS:
          _EqnDef << pCut->var()[0].name() << " - COS(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::SIN:
          _EqnDef << pCut->var()[0].name() << " - SIN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::TAN:
          _EqnDef << pCut->var()[0].name() << " - TAN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ACOS:
          _EqnDef << pCut->var()[0].name() << " - ARCCOS(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ASIN:
          _EqnDef << pCut->var()[0].name() << " - ARCSIN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ATAN:
          _EqnDef << pCut->var()[0].name() << " - ARCTAN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::COSH:
          _EqnDef << pCut->var()[0].name() << " - COSH(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::SINH:
          _EqnDef << pCut->var()[0].name() << " - SINH(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::TANH:
          _EqnDef << pCut->var()[0].name() << " - TANH(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ERF:
          _EqnDef << pCut->var()[0].name() << " - ERRORF(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::FABS:
          _EqnDef << pCut->var()[0].name() << " - ABS(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::MINF:
          _EqnDef << pCut->var()[0].name() << " - MIN(" << pCut->var()[1].name();
          if( pCut->nvar() == 2 )
            _EqnDef << "," << pCut->rhs();
          else
            for( unsigned k=2; k<pCut->nvar(); ++k )
              _EqnDef << "," << pCut->var()[k].name();
          _EqnDef << ") =E= 0;" << std::endl;
          break;

        case FFOp::MAXF:
          _EqnDef << pCut->var()[0].name() << " - MAX(" << pCut->var()[1].name();
          if( pCut->nvar() == 2 )
            _EqnDef << "," << pCut->rhs();
          else
            for( unsigned k=2; k<pCut->nvar(); ++k )
              _EqnDef << "," << pCut->var()[k].name();
          _EqnDef << ") =E= 0;" << std::endl;
          break;

        default:
          throw std::runtime_error("GAMSWRITER - Error: Nonlinear cut not supported");
      }
    }
    if( _type < NLIN ) _type = NLIN;
  }
}

template <typename T>
inline bool
GAMSWRITER<T>::set_variable
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
GAMSWRITER<T>::set_objective
( PolVar<T> const& pObj, t_OBJ const& tObj )
{
  _DirObj = tObj;
  if( _GAMSvar.find( &pObj ) == _GAMSvar.end() ) _add_var( &pObj );
  _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
  _EqnDef << "E" << _EqnCnt << " .. " << _VarObj << " =E= " << pObj.name() << ";" << std::endl;
}

//template <typename T>
//inline void
//GAMSWRITER<T>::set_objective
//( unsigned const nObj, PolVar<T> const* pObj, double const* cObj,
//  t_OBJ const& tObj )
//{
//  _DirObj = tObj;

//  // Set objective
//  GRBLinExpr linobj;
//  _cutvar.resize( nObj );
//  for( unsigned k=0; k<nObj; k++ ){
//    auto itvar = _MIPvar.find( &pObj[k] );
//    if( itvar == _MIPvar.end() && appvar_)
//      itvar = _add_var( &pObj[k] ).first;
//    else if( itvar == _MIPvar.end() )
//      throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in objective");
//    _cutvar[k] = itvar->second;
//  }
//  linobj.addTerms( cObj, _cutvar.data(), nObj );
//  _GRBmodel->setObjective( linobj );
//  switch( tObj ){
//    case MIN: _GRBmodel->set( GRB_IntAttr_ModelSense,  1 ); break;
//    case MAX: _GRBmodel->set( GRB_IntAttr_ModelSense, -1 ); break;
//  }
//}

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

} // end namespace mc

#endif

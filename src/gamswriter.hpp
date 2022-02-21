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
  // Typedef's
  typedef std::map< const PolVar<T>*, GRBVar, lt_PolVar<T> > t_MIPVar;
  typedef GRBConstr t_MIPCtr;
  
protected:

  //! @brief GAMS output file
  std::ofstream _GAMSmodel;

  //! @brief map of variables in MIP model vs DAG
  t_MIPVar _MIPvar;

  //! @brief Polyhedral image environment
  PolImg<T>* _POLenv;

public:
  /** @defgroup MIPSLV_GUROBI MIP optimization using GUROBI
   *  @{
   */
  //! @brief Constructor
  GAMSWRITER
    ( std::string const GAMSfile )
    : _GAMSfile( GAMSfile ), _POLenv( nullptr )
    {
      _GAMSmodel.open( GAMSfile, std::ios::trunc );
      if( !_GAMSmodel.is_open() )
        std::cerr << "Could not create GAMS model. Do you have write permissions in execution directory?"
                  << std::endl;
    }

  //! @brief Destructor
  virtual ~MIPSLV_GUROBI()
    { if( _GAMSmodel.is_open() ) _GAMSmodel.close(); }

  //! @brief Set variables and cuts in GAMS model
  void set_cuts
    ( PolImg<T>* env );

  //! @brief Set objective in 
  void set_objective
    ( PolVar<T> const& pObj, t_OBJ const& tObj, bool const appvar_=true );

  //! @brief Set objective in MIP
  void set_objective
    ( unsigned const nObj, PolVar<T> const* pObj, double const* cObj,
      t_OBJ const& tObj, bool const appvar_=true );

  //! @brief Add extra constraint in GAMS model
  t_MIPCtr add_constraint
    ( PolVar<T> const& pCtr, t_CTR const& tCtr, const double rhs=0.,
      bool const appvar_=true );

  //! @brief Add extra constraint in GAMS model
  t_MIPCtr add_constraint
    ( unsigned const nCtr, PolVar<T> const* pCtr, double const* cCtr, 
      t_CTR const& tCtr, double const rhs=0., bool const appvar_=true );

  //! @brief Set starting point and branch priority of PolImg variable <a>X</a>
  bool set_variable
    ( PolVar<T> const& X, double const* pval, unsigned const priority )//=GRB_UNDEFINED )
    {
      auto itv = _MIPvar.find( const_cast<PolVar<T>*>(&X) );
      if( itv == _MIPvar.end() ) return false;
      itv->second.set( GRB_DoubleAttr_Start, pval? *pval: GRB_UNDEFINED );
      itv->second.set( GRB_IntAttr_BranchPriority, (int)priority );
      return true;
    }

  //! @brief Close GAMS model
  void close
    ()
    { if( _GAMSmodel.is_open() ) _GAMSmodel.close(); }
  /** @} */

protected:

  //! @brief Set options of MIP solver
  void _set_options
    ();

  //! @brief Append variable to MIP model
  std::pair<typename t_MIPVar::iterator,bool> _add_var
    ( PolVar<T> const* pVar );

  //! @brief Append constraint to MIP model
  void _add_cut
    ( PolCut<T> const* pCut );

  //! @brief Append constraint to MIP model
  void _add_cut
    ( PolCut<T> const* pCut, char GRB_Type );

private:

  //! @brief Vector of Gurobi variables in linear terms
  std::vector<GRBVar> _cutvar;

  //! @brief Vector of Gurobi variables in quadratic terms
  std::vector<GRBVar> _cutqvar1;

  //! @brief Vector of Gurobi variables in quadratic terms
  std::vector<GRBVar> _cutqvar2;

  //! @brief Private methods to block default compiler methods
  MIPSLV_GUROBI
    (const MIPSLV_GUROBI&);
  MIPSLV_GUROBI& operator=
    (const MIPSLV_GUROBI&);
};

template <typename T>
inline void
MIPSLV_GUROBI<T>::solve
()
{
  //_GRBexcpt = false;

  try{
    _set_options();
    _GRBmodel->update();
    if( options.OUTPUTFILE != "" )
      _GRBmodel->write( options.OUTPUTFILE );
    fedisableexcept(FE_ALL_EXCEPT);
    _GRBmodel->optimize();
  }

  catch( GRBException& e ){
    //if( options.DISPLEVEL )
      std::cout << "GRBException - Error code: " << e.getErrorCode() << std::endl
                << e.getMessage() << std::endl;
    //_GRBexcpt = true;
    throw;
  }

  //if( !_GRBexcpt && options.DISPLEVEL > 1
  if( options.DISPLEVEL > 1
   && (get_status() == OPTIMAL || get_status() == TIMELIMIT) ){
    std::cout << "MIP solution complete\n";
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  Optimal solution: " << get_objective() << std::endl;
    for( auto && pvar : _POLenv->Vars() )
      std::cout << pvar.second->var().name() << " = "
                << get_variable( *pvar.second ) << std::endl;
  }
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::_set_options
()
{
  // Gurobi options
  _GRBmodel->getEnv().set( GRB_IntParam_LogToConsole,      options.LOGFILE!=""? 0:1 );
  _GRBmodel->getEnv().set( GRB_StringParam_LogFile,        options.LOGFILE );
  _GRBmodel->getEnv().set( GRB_IntParam_OutputFlag,        options.DISPLEVEL>0?1:0 );
  _GRBmodel->getEnv().set( GRB_DoubleParam_TimeLimit,      options.TIMELIMIT );
  _GRBmodel->getEnv().set( GRB_IntParam_Method,            options.ALGO );
  _GRBmodel->getEnv().set( GRB_DoubleParam_OptimalityTol,  options.OPTIMTOL );
  _GRBmodel->getEnv().set( GRB_DoubleParam_FeasibilityTol, options.FEASTOL );
  _GRBmodel->getEnv().set( GRB_DoubleParam_MIPGap,         options.MIPRELGAP );
  _GRBmodel->getEnv().set( GRB_DoubleParam_MIPGapAbs,      options.MIPABSGAP );
  _GRBmodel->getEnv().set( GRB_DoubleParam_Heuristics,     options.HEURISTICS );
  _GRBmodel->getEnv().set( GRB_IntParam_NumericFocus,      options.NUMERICFOCUS );
  _GRBmodel->getEnv().set( GRB_IntParam_ScaleFlag,         options.SCALEFLAG );
  _GRBmodel->getEnv().set( GRB_IntParam_Presolve,          options.PRESOLVE );
  _GRBmodel->getEnv().set( GRB_DoubleParam_PreSOS1BigM,    options.PRESOS1BIGM );
  _GRBmodel->getEnv().set( GRB_DoubleParam_PreSOS2BigM,    options.PRESOS2BIGM );
  _GRBmodel->getEnv().set( GRB_IntParam_DualReductions,    options.DUALRED );
  _GRBmodel->getEnv().set( GRB_IntParam_NonConvex,         options.NONCONVEX );
  _GRBmodel->getEnv().set( GRB_IntParam_Threads,           options.THREADS );
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::reset
()
{
  delete _GRBmodel;
  _GRBmodel = new GRBModel( *_GRBenv );
  _MIPvar.clear();
  //_MIPcut.clear();
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::set_cuts
( PolImg<T>* env, bool const reset_ )
{
  if( reset_ ) reset();
  _POLenv = env;
    
  // Add participating variables into Gurobi model
  for( auto && pvar : _POLenv->Vars() ){
    if( !reset_ && _MIPvar.find( pvar.second ) != _MIPvar.end() ) continue;
    _add_var( pvar.second );
  }
  for( auto && paux : _POLenv->Aux() ){
    if( !reset_ && _MIPvar.find( paux ) != _MIPvar.end() ) continue;
    _add_var( paux );
  }

  // Update Gurobi model before adding cuts
  _GRBmodel->update();

  // Add cuts into Gurobi model
  for( auto && pcut : _POLenv->Cuts() ){
#ifdef MC__MIPSLV_DEBUG
    std::cout << *pcut << std::endl;
#endif
    _add_cut( pcut );
  }
}

//template <typename T>
//inline void
//MIPSLV_GUROBI<T>::set_objective
//( FFVar const& pObj, t_OBJ const& tObj )
//{
//  auto itp = _POLenv->Vars().find( const_cast<FFVar*>(&pObj) );
//  set_objective( *itp->second, tObj );
//}

template <typename T>
inline void
MIPSLV_GUROBI<T>::set_objective
( PolVar<T> const& pObj, t_OBJ const& tObj, bool const appvar_ )
{
  // Set objective
  auto jtobj = _MIPvar.find( &pObj );
  if( jtobj == _MIPvar.end() && appvar_)
    jtobj = _add_var( &pObj ).first;
  else if( jtobj == _MIPvar.end() )
    throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in objective");
  _GRBmodel->setObjective( GRBLinExpr( jtobj->second ) );
  switch( tObj ){
    case MIN: _GRBmodel->set( GRB_IntAttr_ModelSense,  1 ); break;
    case MAX: _GRBmodel->set( GRB_IntAttr_ModelSense, -1 ); break;
  }
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::set_objective
( unsigned const nObj, PolVar<T> const* pObj, double const* cObj,
  t_OBJ const& tObj, bool const appvar_ )
{
  // Set objective
  GRBLinExpr linobj;
  _cutvar.resize( nObj );
  for( unsigned k=0; k<nObj; k++ ){
    auto itvar = _MIPvar.find( &pObj[k] );
    if( itvar == _MIPvar.end() && appvar_)
      itvar = _add_var( &pObj[k] ).first;
    else if( itvar == _MIPvar.end() )
      throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in objective");
    _cutvar[k] = itvar->second;
  }
  linobj.addTerms( cObj, _cutvar.data(), nObj );
  _GRBmodel->setObjective( linobj );
  switch( tObj ){
    case MIN: _GRBmodel->set( GRB_IntAttr_ModelSense,  1 ); break;
    case MAX: _GRBmodel->set( GRB_IntAttr_ModelSense, -1 ); break;
  }
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::update_objective
( PolVar<T> const& pObj, double const& cObj, bool const appvar_ )
{
  // Update objective
  auto jtobj = _MIPvar.find( &pObj );
  if( jtobj == _MIPvar.end() && appvar_)
    jtobj = _add_var( &pObj ).first;
  else if( jtobj == _MIPvar.end() )
    throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in objective");
  jtobj->second.set( GRB_DoubleAttr_Obj, cObj );
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::update_objective
( unsigned const nObj, PolVar<T> const* pObj, double const* cObj,
  bool const appvar_ )
{
  // Set objective
  for( unsigned k=0; k<nObj; k++ ){
    auto jtobj = _MIPvar.find( &pObj[k] );
    if( jtobj == _MIPvar.end() && appvar_)
      jtobj = _add_var( &pObj[k] ).first;
    else if( jtobj == _MIPvar.end() )
      throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in objective");
    jtobj->second.set( GRB_DoubleAttr_Obj, cObj[k] );
  }
}

template <typename T>
inline void
MIPSLV_GUROBI<T>::remove_constraint
( typename MIPSLV_GUROBI<T>::t_MIPCtr & ctr )
{
  _GRBmodel->remove( ctr );
}

template <typename T>
inline typename MIPSLV_GUROBI<T>::t_MIPCtr &
MIPSLV_GUROBI<T>::update_constraint
( typename MIPSLV_GUROBI<T>::t_MIPCtr & ctr, double const rhs )
{
  ctr.set( GRB_DoubleAttr_RHS, rhs );
  return ctr;
}

//template <typename T>
//inline typename MIPSLV_GUROBI<T>::t_MIPCtr
//MIPSLV_GUROBI<T>::add_constraint
//( FFVar const& pCtr, t_CTR const& tCtr, double const rhs )
//{
//  auto itp = _POLenv->Vars().find( const_cast<FFVar*>(&pCtr) );
//  return add_constraint( *itp->second, tCtr, rhs );
//}

template <typename T>
inline typename MIPSLV_GUROBI<T>::t_MIPCtr
MIPSLV_GUROBI<T>::add_constraint
( PolVar<T> const& pCtr, t_CTR const& tCtr, double const rhs,
  bool const appvar_ )
{
  // Set constraint
  auto jtctr = _MIPvar.find( &pCtr );
  if( jtctr == _MIPvar.end() && appvar_ )
      jtctr = _add_var( &pCtr ).first;
  else if( jtctr == _MIPvar.end() )
    throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in constraint");
  GRBLinExpr lhs( jtctr->second );
  GRBConstr ctr;
  switch( tCtr ){
    case EQ: ctr = _GRBmodel->addConstr( lhs, GRB_EQUAL,         rhs ); break;
    case LE: ctr = _GRBmodel->addConstr( lhs, GRB_LESS_EQUAL,    rhs ); break;
    case GE: ctr = _GRBmodel->addConstr( lhs, GRB_GREATER_EQUAL, rhs ); break;
  }
  return ctr;
}

template <typename T>
inline typename MIPSLV_GUROBI<T>::t_MIPCtr
MIPSLV_GUROBI<T>::add_constraint
( unsigned const nCtr, PolVar<T> const* pCtr, double const* cCtr,
  t_CTR const& tCtr, double const rhs, bool const appvar_ )
{
  // Set constraint
  GRBLinExpr lhs;
  _cutvar.resize( nCtr );
  for( unsigned k=0; k<nCtr; k++ ){
    auto jtctr = _MIPvar.find( &pCtr[k] );
    if( jtctr == _MIPvar.end() && appvar_)
        jtctr = _add_var( &pCtr[k] ).first;
    else if( jtctr == _MIPvar.end() )
      throw std::runtime_error("MIPSLV_GUROBI - Error: Unknown variable in constraint");
    _cutvar[k] = jtctr->second;
  }
  lhs.addTerms( cCtr, _cutvar.data(), nCtr );
  GRBConstr ctr;
  switch( tCtr ){
    case EQ: ctr = _GRBmodel->addConstr( lhs, GRB_EQUAL,         rhs ); break;
    case LE: ctr = _GRBmodel->addConstr( lhs, GRB_LESS_EQUAL,    rhs ); break;
    case GE: ctr = _GRBmodel->addConstr( lhs, GRB_GREATER_EQUAL, rhs ); break;
  }
  return ctr;
}

template <typename T>
inline typename MIPSLV_GUROBI<T>::t_MIPCtr
MIPSLV_GUROBI<T>::dummy_constraint
()
{
  GRBConstr ctr;
  return ctr;
}

template <typename T>
inline std::pair<typename MIPSLV_GUROBI<T>::t_MIPVar::iterator,bool>
MIPSLV_GUROBI<T>::_add_var
( PolVar<T> const* pVar )
{
  GRBVar var;
  try{
    switch( pVar->id().first ){
      case PolVar<T>::VARCONT:
      case PolVar<T>::AUXCONT:
      case PolVar<T>::AUXCST:
        var = _GRBmodel->addVar( Op<T>::l(pVar->range()), Op<T>::u(pVar->range()),
          0., GRB_CONTINUOUS, pVar->name() );
        break;
      
      case PolVar<T>::VARINT:
      case PolVar<T>::AUXINT:
        if( options.CONTRELAX )
        var = _GRBmodel->addVar( Op<T>::l(pVar->range()), Op<T>::u(pVar->range()),
          0., GRB_CONTINUOUS, pVar->name() );
        else if( isequal( Op<T>::l(pVar->range()), 0. ) && isequal( Op<T>::u(pVar->range()), 1. ) )
          var = _GRBmodel->addVar( 0., 1., 0., GRB_BINARY, pVar->name() );
        else
          var = _GRBmodel->addVar( Op<T>::l(pVar->range()), Op<T>::u(pVar->range()),
            0., GRB_INTEGER, pVar->name() );
        break;

      default:
        throw std::runtime_error("MIPSLV_GUROBI - Error: Invalid auxiliary variable type");
    }
  }

  catch( GRBException& e ){
    //if( options.DISPLEVEL )
      std::cout << "GRBException - Error code: " << e.getErrorCode() << std::endl
                << e.getMessage() << std::endl;
      std::cout << "name: " << pVar->name() << "  range: " << pVar->range() << std::endl;
    throw;
  }

  return _MIPvar.insert( std::make_pair( pVar, var ) );
}

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

template <typename T>
inline void
MIPSLV_GUROBI<T>::Options::display
( std::ostream&out ) const
{
  // Display MIP options
  out << std::left << std::scientific << std::setprecision(1)
      << std::setw(15) << "  ALGO"        << ALGO        << std::endl
      << std::setw(15) << "  PRESOLVE"    << PRESOLVE    << std::endl
      << std::setw(15) << "  DUALRED"     << DUALRED     << std::endl
      << std::setw(15) << "  NONCONVEX"   << NONCONVEX   << std::endl
      << std::setw(15) << "  FEASTOL"     << FEASTOL     << std::endl
      << std::setw(15) << "  OPTIMTOL"    << OPTIMTOL    << std::endl
      << std::setw(15) << "  MIPRELGAP"   << MIPRELGAP   << std::endl
      << std::setw(15) << "  MIPABSGAP"   << MIPABSGAP   << std::endl
      << std::setw(15) << "  MIPABSGAP"   << MIPABSGAP   << std::endl
      << std::setw(15) << "  HEURISTICS"  << HEURISTICS  << std::endl
      << std::setw(15) << "  PRESOS1BIGM" << PRESOS1BIGM << std::endl
      << std::setw(15) << "  PRESOS2BIGM" << PRESOS2BIGM << std::endl
      << std::setw(15) << "  TIMELIMIT"   << TIMELIMIT   << std::endl
      << std::setw(15) << "  DISPLEVEL"   << DISPLEVEL   << std::endl
      << std::setw(15) << "  OUTPUTFILE"  << OUTPUTFILE  << std::endl;
}

template <typename T>
inline std::string
MIPSLV_GUROBI<T>::Options::pwl
() const
{
  std::ostringstream oline;
  oline << "FuncPieces=-2, FuncPieceError=" << PWLRELGAP;
  return oline.str();
}

} // end namespace mc

#endif

// Copyright (C) 2020 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__MIPSLV_GUROBI_HPP
#define MC__MIPSLV_GUROBI_HPP

#include <stdexcept>
#include <cassert>

#include "base_opt.hpp"
#include "polimage.hpp"
#include "gurobi_c++.h"

#undef MC__MIPSLV_DEBUG
#undef MC__MIPSLV_TRACE

extern "C"{
  #include <fenv.h>
  int fedisableexcept( int );
}

namespace mc
{
//! @brief C++ class for solution of mixed-integer programs using GUROBI
////////////////////////////////////////////////////////////////////////
//! mc::MIPSLV_GUROBI is a C++ class to enable the construction and 
//! solution of mixed-integer programs (MILP, MIQP, MIQCQP)  using the
//! C++ API of GUROBI (v9.0.1 or later).
////////////////////////////////////////////////////////////////////////
template< typename T>
class MIPSLV_GUROBI
: public virtual BASE_OPT
{
  // Typedef's
  typedef std::map< const PolVar<T>*, GRBVar, lt_PolVar<T> > t_MIPVar;
  typedef GRBConstr t_MIPCtr;
  
protected:

  //! @brief GUROBI environment
  GRBEnv* _GRBenv;

  //! @brief GUROBI model
  GRBModel* _GRBmodel;

  //! @brief map of variables in MIP model vs DAG
  t_MIPVar _MIPvar;

  //! @brief whether GUROBI has sent an exception
  bool _GRBexcpt;

  //! @brief Polyhedral image environment
  PolImg<T>* _POLenv;

public:

  //! @brief Constructor
  MIPSLV_GUROBI
    ()
    : _GRBenv( new GRBEnv( true ) ), _GRBmodel( nullptr ), _POLenv( nullptr )
    { _GRBenv->set( GRB_IntParam_LogToConsole, 1 );//0 );
      _GRBenv->start(); }

  //! @brief Destructor
  virtual ~MIPSLV_GUROBI()
    {
      delete _GRBmodel;
      delete _GRBenv;
    }

  //! @brief MIP solution status
  enum STATUS{
     OPTIMAL=0,   //!< Optimal solution found within tolerances
     SUBOPTIMAL,  //!< Unable to satisfy optimality tolerances, but sub-optimal solution available
     INFEASIBLE,  //!< Infeasible, but not unbounded
     INFORUNBND,  //!< Infeasible or unbounded
     UNBOUNDED,   //!< Unbounded
     TIMELIMIT,   //!< Time limit reached
     OTHER        //!< Undefined
  };
  
  //! @brief MIP options
  struct Options
  {
    //! @brief Constructor
    Options():
      ALGO( -1 ), PRESOLVE( -1 ), LPWARMSTART( 1 ), 
      CONTRELAX( false ), DUALRED( 1 ), NONCONVEX( 2 ), 
      FEASTOL( 1e-7 ), OPTIMTOL( 1e-7 ), MIPRELGAP( 1e-5 ), MIPABSGAP( 1e-5 ),
      NUMERICFOCUS( 0 ), SCALEFLAG( -1 ), HEURISTICS( 0.05 ),
      PRESOS1BIGM( -1. ), PRESOS2BIGM( -1. ), PWLRELGAP( 1e-5 ), FUNCMAXVAL( 1e6 ),
      TIMELIMIT( 6e2 ), THREADS( 0 ), DISPLEVEL( 1 ),
      LOGFILE(), OUTPUTFILE()
      {}
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        ALGO         = options.ALGO;
        PRESOLVE     = options.PRESOLVE;
        LPWARMSTART  = options.LPWARMSTART;        
        CONTRELAX    = options.CONTRELAX;
        DUALRED      = options.DUALRED;
        NONCONVEX    = options.NONCONVEX;
        FEASTOL      = options.FEASTOL;
        OPTIMTOL     = options.OPTIMTOL;
        MIPRELGAP    = options.MIPRELGAP;
        MIPABSGAP    = options.MIPABSGAP;
        NUMERICFOCUS = options.NUMERICFOCUS;
        SCALEFLAG    = options.SCALEFLAG;
        HEURISTICS   = options.HEURISTICS;
        PRESOS1BIGM  = options.PRESOS1BIGM;
        PRESOS2BIGM  = options.PRESOS2BIGM;
        PWLRELGAP    = options.PWLRELGAP;
        FUNCMAXVAL   = options.FUNCMAXVAL;
        TIMELIMIT    = options.TIMELIMIT;
        THREADS      = options.THREADS;
        DISPLEVEL    = options.DISPLEVEL;
        LOGFILE      = options.LOGFILE;
        OUTPUTFILE   = options.OUTPUTFILE;
        return *this ;
      }
    //! @brief Algorithm used to solve continuous models or the root node of a MIP model. The default options is: -1=automatic. Other options are: 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex. 
    int ALGO;
    //! @brief Controls the presolve level. A value of -1 corresponds to an automatic setting. Other options are off (0), conservative (1), or aggressive (2). More aggressive application of presolve takes more time, but can sometimes lead to a significantly tighter model.
    int PRESOLVE;
    //! @brief Controls whether and how Gurobi uses warm start information for an LP optimization. A value of 0 ignores any start information and solves the model from scratch. Setting it to 1 (the default) uses the provided warm start information to solve the original, unpresolved problem, regardless of whether presolve is enabled. Setting it to 2 uses the start information to solve the presolved problem, assuming that presolve is enabled.
    int LPWARMSTART;
    //! @brief Determines whether to relax binary/integer variables as continuous variables during solve.
    bool CONTRELAX;
    //! @brief Determines whether dual reductions are performed in presolve. The default value of 1 enables these redutions. You can set the parameter to 0 to disable these reductions if you received an optimization status of INF_OR_UNBD and would like a more definitive conclusion. 
    int DUALRED;
    //! @brief Sets the strategy for handling non-convex quadratic objectives or non-convex quadratic constraints. With setting 0, an error is reported if the original user model contains non-convex quadratic constructs. With setting 1, an error is reported if non-convex quadratic constructs could not be discarded or linearized during presolve. With the default setting 2, non-convex quadratic problems are solved by means of translating them into bilinear form and applying spatial branching. The default -1 setting is currently equivalent to 1, and may change in future GUROBI releases to be equivalent to 2. 
    int NONCONVEX;
    //! @brief All constraints must be satisfied to this tolerance. Tightening this tolerance can produce smaller constraint violations, but for numerically challenging models it can sometimes lead to much larger iteration counts. 
    double FEASTOL;
     //! @brief Reduced costs must all be smaller than this tolerance in the improving direction in order for a model to be declared optimal.
    double OPTIMTOL;
    //! @brief The MIP solver will terminate (with an optimal result) when the gap between the lower and upper objective bound is less than this tolerance times the absolute value of the upper bound. 
    double MIPRELGAP;
    //! @brief The MIP solver will terminate (with an optimal result) when the gap between the lower and upper objective bound is less than this tolerance. 
    double MIPABSGAP;
    //! @brief Controls the degree to which the code attempts to detect and manage numerical issues. The default setting (0) makes an automatic choice, with a slight preference for speed. Settings 1-3 increasingly shift the focus towards being more careful in numerical computations. With higher values, the code will spend more time checking the numerical accuracy of intermediate results, and it will employ more expensive techniques in order to avoid potential numerical issues. 
    int NUMERICFOCUS;
    //! @brief Controls model scaling. By default, the rows and columns of the model are scaled in order to improve the numerical properties of the constraint matrix. The scaling is removed before the final solution is returned. Scaling typically reduces solution times, but it may lead to larger constraint violations in the original, unscaled model. Turning off scaling (ScaleFlag=0) can sometimes produce smaller constraint violations. Choosing a different scaling setting 1-3 can sometimes improve performance for particularly numerically difficult models.  
    int SCALEFLAG;
    //! @brief Determines the amount of time spent in MIP heuristics. You can think of the value as the desired fraction of total MIP runtime devoted to heuristics. The default value of 0.05 aims to spend 5% of runtime on heuristics. Larger values produce more and better feasible solutions, at a cost of slower progress in the best bound. 
    double HEURISTICS;
    //! @brief Controls the automatic reformulation of SOS1 constraints into binary form. SOS1 constraints are often handled more efficiently using a binary representation. The reformulation often requires big-M values to be introduced as coefficients. This parameter specifies the largest big-M that can be introduced by presolve when performing this reformulation. Larger values increase the chances that an SOS1 constraint will be reformulated, but very large values (e.g., 1e8) can lead to numerical issues. The default value of -1 chooses a threshold automatically. You should set the parameter to 0 to shut off SOS1 reformulation entirely, or a large value to force reformulation. 
    double PRESOS1BIGM;
    //! @brief Controls the automatic reformulation of SOS2 constraints into binary form. SOS2 constraints are often handled more efficiently using a binary representation. The reformulation often requires big-M values to be introduced as coefficients. This parameter specifies the largest big-M that can be introduced by presolve when performing this reformulation. Larger values increase the chances that an SOS2 constraint will be reformulated, but very large values (e.g., 1e8) can lead to numerical issues. The default value of 0 disables the reformulation. You can set the parameter to -1 to choose an automatic approach, or a large value to force reformulation. 
    double PRESOS2BIGM;
    //! @brief Sets the maximum relative error in the piecewise-linear approximations of the nonlinear functions. Gurobi will choose pieces, typically of different sizes, to achieve that error bound. Note that the number of pieces required may be quite large when setting a tight error tolerance. The default value of 0.05 specifies a 5% maximal relative error gap.  
    double PWLRELGAP;
    //! @brief Sets the maximum allowed value for x and y variables in function constraints. Very large values in piecewise-linear approximations can cause numerical errors. The default value is 1e+6. This parameter limits the bounds on the variables that participate in function constraints - any bound larger than this limit will be truncated.
    double FUNCMAXVAL;
    //! @brief Limits the total time expended (in seconds). Optimization returns with a TIMELIMIT status if the limit is exceeded.
    double TIMELIMIT;
    //! @brief Limits the number of threads used by the MIP solver. The default value of 0 allows to use all available threads.
    unsigned THREADS;
    //! @brief Enables or disables solver output. The default value of 1 shows the solver iterations and results.
    int DISPLEVEL;
    //! @brief The name of the Gurobi log file. Modifying this parameter closes the current log file and opens the specified file. Use an empty string for no log file.
    std::string LOGFILE;
    //! @brief The name of the file to be written before solving the model. Valid suffixes are .mps, .rew, .lp, or .rlp for writing the model.
    std::string OUTPUTFILE;
    //! @brief Display
    void display
      ( std::ostream&out = std::cout )
      const;
    //! @brief Piecewise-linear option string
    std::string pwl
      ()
      const;
    //! @brief Piecewise-linear approximation range check
    void check_pwl
      ( GRBVar const& x, GRBVar const& y )
      const; 
  };
  //! @brief MIP options
  Options options;

  //! @brief Reset MIP model
  void reset
    ();

  //! @brief Set variables and cuts in MIP
  void set_cuts
    ( PolImg<T>* env, bool const reset_=true );

//  //! @brief Set objective in MIP
//  void set_objective
//    ( FFVar const& pObj, t_OBJ const& tObj );

  //! @brief Set objective in MIP
  void set_objective
    ( PolVar<T> const& pObj, t_OBJ const& tObj, bool const appvar_=true );

  //! @brief Set objective in MIP
  void set_objective
    ( unsigned const nObj, PolVar<T> const* pObj, double const* cObj,
      t_OBJ const& tObj, bool const appvar_=true );

  //! @brief Update objective term in MIP
  void update_objective
    ( PolVar<T> const& pObj, double const& cObj, bool const appvar_=true );

  //! @brief Update objective terms in MIP
  void update_objective
    ( unsigned const nObj, PolVar<T> const* pObj, double const* cObj,
      bool const appvar_=true );

//  //! @brief Add constraint in MIP
//  t_MIPCtr add_constraint
//    ( FFVar const& pCtr, t_CTR const& tCtr, const double rhs=0. );

  //! @brief Add constraint in MIP
  t_MIPCtr add_constraint
    ( PolVar<T> const& pCtr, t_CTR const& tCtr, const double rhs=0.,
      bool const appvar_=true );

  //! @brief Add constraint in MIP
  t_MIPCtr add_constraint
    ( unsigned const nCtr, PolVar<T> const* pCtr, double const* cCtr, 
      t_CTR const& tCtr, double const rhs=0., bool const appvar_=true );

  //! @brief Modify constraint RHS in MIP
  t_MIPCtr & update_constraint
    ( t_MIPCtr & ctr, double const rhs );

  //! @brief Dummy constraint in MIP
  t_MIPCtr dummy_constraint
    ();
    
  //! @brief Remove constraint from MIP
  void remove_constraint
    ( t_MIPCtr & ctr );

  //! @brief Solve MIP
  void solve
    ();

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

  //! @brief Value of PolImg variable <a>X</a> after last MIP call
  double get_variable
    ( PolVar<T> const& X )
    const
    {
      auto itv = _MIPvar.find( const_cast<PolVar<T>*>(&X) );
      return itv->second.get( GRB_DoubleAttr_X );
    }

  //! @brief Value of DAG variable <a>X</a> after last MIP call
  double get_variable
    ( FFVar const& X )
    const
    {
      auto itp = _POLenv->Vars().find( const_cast<FFVar*>(&X) );
      auto itv = _MIPvar.find( itp->second );
      return itv->second.get( GRB_DoubleAttr_X );
    }

  //! @brief Optimal cost value after last MIP call
  double get_objective
    ()
    const
    {
      return _GRBmodel->get( GRB_DoubleAttr_ObjVal );
    }

  //! @brief Optimal cost bound after last MIP call
  double get_objective_bound
    ()
    const
    {
      return _GRBmodel->get( GRB_IntAttr_IsMIP )?
             _GRBmodel->get( GRB_DoubleAttr_ObjBound ):
             _GRBmodel->get( GRB_DoubleAttr_ObjVal );
    }

  //! @brief Status after last MIP call
  STATUS get_status
    ()
    const
    {
      if( _GRBexcpt )         return OTHER;
      switch( _GRBmodel->get( GRB_IntAttr_Status ) ){
       case GRB_OPTIMAL:      return OPTIMAL;
       case GRB_SUBOPTIMAL:   return SUBOPTIMAL;
       case GRB_INFEASIBLE:   return INFEASIBLE;
       case GRB_INF_OR_UNBD:  return INFORUNBND;
       case GRB_UNBOUNDED:    return UNBOUNDED;
       case GRB_TIME_LIMIT:   return TIMELIMIT;
       default:               return OTHER;
      }
    }

  //! @brief Pointer to MIP model
  GRBModel const* get_model
    ()
    const
    { return _GRBmodel; }

  //! @brief Terminate solve process
  void terminate
    ()
    { if( _GRBmodel ) _GRBmodel->terminate(); }

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
  _GRBexcpt = false;

  try{
    _set_options();
    _GRBmodel->update();
    if( options.OUTPUTFILE != "" )
      _GRBmodel->write( options.OUTPUTFILE );
    fedisableexcept(FE_ALL_EXCEPT);
    _GRBmodel->optimize();
  }

  catch( GRBException& e ){
    std::cout << "GRBException - Error code: " << e.getErrorCode() << std::endl
              << e.getMessage() << std::endl;
    _GRBexcpt = true;
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
  _GRBmodel->getEnv().set( GRB_IntParam_LPWarmStart,       options.LPWARMSTART );
  _GRBmodel->getEnv().set( GRB_DoubleParam_PreSOS1BigM,    options.PRESOS1BIGM );
  _GRBmodel->getEnv().set( GRB_DoubleParam_PreSOS2BigM,    options.PRESOS2BIGM );
  _GRBmodel->getEnv().set( GRB_DoubleParam_FuncMaxVal,     options.FUNCMAXVAL );
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
          case FFOp::IPOW:{
            unsigned const ncoef = pCut->op()->pops[1]->num().n+1;
            std::vector<double> coef( ncoef, 0. ); coef[0] = 1.;
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrPoly( _cutvar[1], _cutvar[0], ncoef, coef.data(), "", options.pwl() );
            break;}

          case FFOp::DPOW:{
            double const& dExp = pCut->op()->pops[1]->num().val();
            if( dExp < 0 ) throw std::runtime_error("MIPSLV_GUROBI - Error: Nonlinear cut not yet implemented");
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrPow( _cutvar[1], _cutvar[0], dExp, "", options.pwl() );
            break;}

          case FFOp::CHEB:{
            unsigned const ncoef = pCut->op()->pops[1]->num().n+1;
            std::vector<double>&& coef = chebcoef( ncoef-1 );
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrPoly( _cutvar[1], _cutvar[0], ncoef, coef.data(), "", options.pwl() );
            break;}

          case FFOp::SQR:{
            //_GRBmodel->addGenConstrPow( _cutvar[1], _cutvar[0], 2, "", options.pwl() );
            unsigned const ncoef = 3;
            std::vector<double> coef( ncoef, 0. ); coef[0] = 1.;
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrPoly( _cutvar[1], _cutvar[0], ncoef, coef.data(), "", options.pwl() );
            break;}

          case FFOp::SQRT:{
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrPow( _cutvar[1], _cutvar[0], 0.5, "", options.pwl() );
            //const unsigned ncoef = 3;
            //std::vector<double> coef( ncoef, 0. ); coef[0] = 1.;
            //_GRBmodel->addGenConstrPoly( _cutvar[0], _cutvar[1], ncoef, coef.data(), "", options.pwl() );
            break;}

          case FFOp::EXP:
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrExp( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;

          case FFOp::LOG:
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrLog( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;

          case FFOp::COS:
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrCos( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;

          case FFOp::SIN:
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrSin( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;

          case FFOp::TAN:
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrTan( _cutvar[1], _cutvar[0], "", options.pwl() );
            break;

          case FFOp::ACOS:
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrCos( _cutvar[0], _cutvar[1], "", options.pwl() );
            break;

          case FFOp::ASIN:
            options.check_pwl( _cutvar[0], _cutvar[1] );
            _GRBmodel->addGenConstrSin( _cutvar[0], _cutvar[1], "", options.pwl() );
            break;

          case FFOp::ATAN:
            options.check_pwl( _cutvar[0], _cutvar[1] );
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

template <typename T>
inline void
MIPSLV_GUROBI<T>::Options::check_pwl
( GRBVar const& x, GRBVar const& y ) const
{
  if( x.get(GRB_DoubleAttr_LB) < -FUNCMAXVAL || x.get(GRB_DoubleAttr_UB) > FUNCMAXVAL
   || y.get(GRB_DoubleAttr_LB) < -FUNCMAXVAL || y.get(GRB_DoubleAttr_UB) > FUNCMAXVAL )
    throw std::runtime_error("MIPSLV_GUROBI - Error: Parameter FuncMaxVal too small for piecewise-linear approximation of function constraints");
}

} // end namespace mc

#endif

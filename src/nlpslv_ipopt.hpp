// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_NLPSLV_IPOPT Local (Continuous) Nonlinear Optimization interfacing IPOPT with MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 2.0
\date 2020
\bug No known bugs.

Consider a nonlinear optimization problem in the form:
\f{align*}
\mathcal{P}:\quad & \min_{x_1,\ldots,x_n}\ f(x_1,\ldots,x_n)\\
& {\rm s.t.}\ \ g_j(x_1,\ldots,x_n)\ \leq,=,\geq\ 0,\ \ j=1,\ldots,m\\
& \qquad x_i^L\leq x_i\leq x_i^U,\ \ i=1,\ldots,n\,,
\f}
where \f$f, g_1, \ldots, g_m\f$ are factorable, potentially nonlinear, real-valued functions; and \f$x_1, \ldots, x_n\f$ are continuous decision variables. The class mc::NLPSLV_IPOPT solves such NLP problems using the software package <A href="https://projects.coin-or.org/Ipopt">IPOPT</A>, which implements a local solution method (interior point). IPOPT requires the first and second derivatives as well as the sparsity pattern of the objective and constraint functions in the NLP model. This information is generated using direct acyclic graphs (DAG) in <A href="https://projects.coin-or.org/MCpp">MC++</A>.

\section sec_NLPSLV_IPOPT_solve How to Solve an NLP Model using mc::NLPSLV_IPOPT?

Consider the following NLP:
\f{align*}
  \max_{\bf p}\ & p_1+p_2 \\
  \text{s.t.} \ & p_1\,p_2 \leq 4 \\
  & 0 \leq p_1 \leq 6\\
  & 0 \leq p_2 \leq 4\,.
\f}

We start by defining an mc::NLPSLV_IPOPT class as below, whereby the class <A href="http://www.coin-or.org/Doxygen/Ipopt/class_ipopt_1_1_smart_ptr.html">Ipopt::SmartPtr</A> is used here to comply with the IPOPT C++ interface:

\code
  #include "nlpslv_ipopt.hpp"
  
  Ipopt::SmartPtr<mc::NLPSLV_IPOPT> NLP = new mc::NLPSLV_IPOPT;
\endcode

Next, we set the variables and objective/constraint functions by creating a DAG of the problem: 

\code
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );
  NLP->set_dag( &DAG );                       // DAG
  NLP->set_var( NP, P );                      // decision variables
  NLP->set_obj( BASE_NLP::MAX, P[0]+P[1] );   // objective
  NLP->add_ctr( BASE_NLP::LE, P[0]*P[1]-4. ); // constraints
  NLP->setup();
\endcode

Given initial bounds \f$P\f$ and initial guesses \f$p_0\f$ on the decision variables, the NLP model is solved as:

\code
  #include "interval.hpp"

  typedef mc::interval I;
  I P[NP] = { I(0.,6.), I(0.,4.) };
  double p0[NP] = { 5., 1. };

  Ipopt::ApplicationReturnStatus status = NLP->solve( p0, P );
\endcode

The return value is of the enumeration type <A href="http://www.coin-or.org/Doxygen/Ipopt/namespace_ipopt.html#efa0497854479cde8b0994cdf132c982">Ipopt::ApplicationReturnStatus</A>. Moreover, the optimization results can be retrieved as an instance of mc::SOLUTION_OPT using the method <a>solution</a> or simply displayed as follows:

\code
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP->solution();
\endcode

The following result is displayed here:

\verbatim
  STATUS: 1
  X[0]:  LEVEL =  6.000000e+00  MARGINAL =  8.888889e-01
  X[1]:  LEVEL =  6.666667e-01  MARGINAL =  0.000000e+00
  F[0]:  LEVEL =  6.666667e+00  MARGINAL =  0.000000e+00
  F[1]:  LEVEL =  8.881784e-16  MARGINAL =  1.666667e-01
\endverbatim

Regarding options, the output level, maximum number of iterations, tolerance, maximum CPU time, etc can all be modified through the public member mc::NLPSLV_IPOPT::options. 
*/

#ifndef MC__NLPSLV_IPOPT_HPP
#define MC__NLPSLV_IPOPT_HPP

#include <stdexcept>
#include <cassert>
#include <thread>
#include "coin-or/IpTNLP.hpp"
#include "coin-or/IpIpoptApplication.hpp"

#include "mctime.hpp"
#include "ffdep.hpp"
#include "base_nlp.hpp"
#include "gamsio.hpp"

#if defined( MC__USE_SOBOL )
  #include <boost/random/sobol.hpp>
  #include <boost/random/uniform_01.hpp>
  #include <boost/random/variate_generator.hpp>
#endif

//#define MC__NLPSLV_IPOPT_DEBUG
//#define MC__NLPSLV_IPOPT_TRACE

namespace mc
{

class NLPSLV_IPOPT;

//! @brief C++ class for calling IPOPT on local threads
////////////////////////////////////////////////////////////////////////
//! mc::WORKER_IPOPT is a C++ class for calling IPOPT on local threads
////////////////////////////////////////////////////////////////////////
struct WORKER_IPOPT:
  public Ipopt::TNLP
{
  //! @brief local copy of DAG
  FFGraph  dag;

  //! @brief size of parameters in DAG
  size_t nP;
  //! @brief vector of parameters in DAG
  std::vector<FFVar>  Pvar;
  //! @brief vector of parameter values
  std::vector<double> Pval;
  //!@brief vector of parameter values in fadbad::F<double> arithmetic
  std::vector<fadbad::F<double>> FPval;
  //!@brief vector of parameter values in fadbad::B<double> arithmetic
  std::vector<fadbad::B<double>> BPval;
  //!@brief vector of parameter values in fadbad::B<fadbad::F<double>> arithmetic
  std::vector<fadbad::B<fadbad::F<double>>> BFPval;

  //! @brief vector of decision variables in DAG
  std::vector<FFVar>  Xvar;
  //! @brief vector of decision variable values
  std::vector<double> Xval;
  //! @brief vector of decision variable lower bounds
  std::vector<double> Xlow;
  //! @brief vector of decision variable upper bounds
  std::vector<double> Xupp;
  //!@brief vector of variable values in fadbad::F<double> arithmetic
  std::vector<fadbad::F<double>> FXval;
  //!@brief vector of variable values in fadbad::B<double> arithmetic
  std::vector<fadbad::B<double>> BXval;
  //!@brief vector of variable values in fadbad::B<fadbad::F<double>> arithmetic
  std::vector<fadbad::B<fadbad::F<double>>> BFXval;

  //! @brief vector of functions in DAG
  std::vector<FFVar>  Fvar;
  //! @brief vector of function multipliers in DAG
  std::vector<FFVar>  Fmul;
  //! @brief vector of function values
  std::vector<double> Fval;
  //! @brief vector of function lower bounds
  std::vector<double> Flow;
  //! @brief vector of function upper bounds
  std::vector<double> Fupp;
  //!@brief vector of function values in fadbad::F<double> arithmetic
  std::vector<fadbad::F<double>> FFval;
  //!@brief vector of function values in fadbad::B<double> arithmetic
  std::vector<fadbad::B<double>> BFval;
  //!@brief vector of multiplier values in fadbad::F<double> arithmetic
  std::vector<fadbad::F<double>> FFmul;
  //!@brief vector of multiplier values in fadbad::B<fadbad::F<double>> arithmetic
  std::vector<fadbad::B<fadbad::F<double>>> BFFmul;

  //! @brief vector of cost gradient in DAG
  std::vector<FFVar>  Cvar;
  //! @brief vector of cost gradient values
  std::vector<double> Cval;
  //! @brief cost multiplier in fadbad::F<double> arithmetic
  fadbad::F<double>   FCmul;
  //! @brief cost multiplier in fadbad::B<fadbad::F<double>> arithmetic
  fadbad::B<fadbad::F<double>> BFCmul;
  //! @brief cost gradient in fadbad::F<double> arithmetic
  fadbad::F<double>   FCval;
  //! @brief cost gradient in fadbad::B<double> arithmetic
  fadbad::B<double>   BCval;

  //! @brief vector of contraint gradients in DAG
  std::vector<FFVar>  Gvar;
  //! @brief vector of constraint gradient values
  std::vector<double> Gval;
  //!@brief row coordinates of nonzero elements in constraint gradients
  std::vector<int>    iGfun;
  //!@brief column coordinates of nonzero elements in constraint gradients
  std::vector<int>    jGvar;

  //! @brief vector of Lagrangian Hessian in DAG
  std::vector<FFVar>  Lvar;
  //! @brief Lagrangian value in fadbad::B<fadbad::F<double>> arithmetic
  fadbad::B<fadbad::F<double>> BFLval;
  //!@brief row coordinates of nonzero elements in Lagrangian Hessian
  std::vector<int>    iLvar;
  //!@brief column coordinates of nonzero elements in Lagrangian Hessian
  std::vector<int>    jLvar;

  //! @brief list of operations for objective evaluation
  FFSubgraph          op_f;
  //! @brief list of operations for objective gradient evaluation
  FFSubgraph          op_df;
  //! @brief list of operations for constraint evaluation
  FFSubgraph          op_g;
  //! @brief list of operations for constraint gradient evaluation
  FFSubgraph          op_dg;
  //! @brief list of operations for Lagrangian Hessian evaluation
  FFSubgraph          op_L;
  //! @brief Storage vector for DAG evaluation in double arithmetic
  std::vector<double> dwk;
  //! @brief Storage vector for DAG evaluation in fadbad::F<double> arithmetic
  std::vector<fadbad::F<double>> Fwk;
  //! @brief Storage vector for DAG evaluation in fadbad::B<double> arithmetic
  std::vector<fadbad::B<double>> Bwk;
  //! @brief Storage vector for DAG evaluation in fadbad::B<fadbad::F<double>> arithmetic
  std::vector<fadbad::B<fadbad::F<double>>> BFwk;

  //! @brief Gradient option
  int                 Gmeth;

  //! @brief Time limit
  double              tMAX;
  //! @brief Structure holding solution information
  SOLUTION_OPT        solution;

  //! @brief Function updating NLP bounds and parameters
  void update
    ( int const nX, double const* Xl, double const* Xu, int const nP,
      double const* P0 );

  //! @brief Function initializing NLP solution
  void initialize
    ( int const nX, double const* Xini );

  //! @brief Method to return some info about the NLP
  virtual bool get_nlp_info
    ( Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
      Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style );

  //! @brief Method to return the variable and constraint bounds
  virtual bool get_bounds_info
    ( Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
      Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u );

  //! @brief Method to return the initial point for the NLP solver
  virtual bool get_starting_point
    ( Ipopt::Index n, bool init_x, Ipopt::Number* x, bool init_z,
      Ipopt::Number* z_L, Ipopt::Number* z_U, Ipopt::Index m,
      bool init_lambda, Ipopt::Number* lambda );

  //! @brief Method to return the objective function value
  virtual bool eval_f
    ( Ipopt::Index n, const Ipopt::Number* x, bool new_x,
      Ipopt::Number& obj_value );

  //! @brief Method to return the objective function gradient
  virtual bool eval_grad_f
    ( Ipopt::Index n, const Ipopt::Number* x, bool new_x,
      Ipopt::Number* grad_f );

  //! @brief Method to return the constraint residuals
  virtual bool eval_g
    ( Ipopt::Index n, const Ipopt::Number* x, bool new_x,
      Ipopt::Index m, Ipopt::Number* g );

  //! @brief Method to return the structure of the jacobian (if "values" is NULL) and the values of the jacobian (otherwise)
  virtual bool eval_jac_g
    ( Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m,
      Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
      Ipopt::Number* values );

  //! @brief Method to return the structure of the hessian of the lagrangian (if "values" is NULL) and the values of the hessian of the lagrangian (otherwise)
  virtual bool eval_h
    ( Ipopt::Index n, const Ipopt::Number* x, bool new_x,
      Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
      bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
      Ipopt::Index* jCol, Ipopt::Number* values );

  //! @brief Method called when the algorithm is complete
  virtual void finalize_solution
    ( Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x,
      const Ipopt::Number* z_L, const Ipopt::Number* z_U, Ipopt::Index m,
      const Ipopt::Number* g, const Ipopt::Number* lambda,
      Ipopt::Number obj_value, const Ipopt::IpoptData* ip_data,
      Ipopt::IpoptCalculatedQuantities* ip_cq );

  //! @brief Method called at the end of each iteration
  virtual bool intermediate_callback
    ( Ipopt::AlgorithmMode mode, Ipopt::Index iter, Ipopt::Number obj_value,
      Ipopt::Number inf_pr, Ipopt::Number inf_du, Ipopt::Number mu,
      Ipopt::Number d_norm, Ipopt::Number regularization_size,
      Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
      Ipopt::Index ls_trials, const Ipopt::IpoptData* ip_data,
      Ipopt::IpoptCalculatedQuantities* ip_cq );

  //! @brief Function testing NLP solution feasibility
  bool feasible
    ( double const CTRTOL, int const nP, int const nX, int const nF );

  //! @brief Function testing NLP solution stationarity
  bool stationary
    ( double const GRADTOL, int const nP, int const nX, int const nF, int const nG );

  //! @brief Function computing NLP cost correction to compensate for infeasibility
  double correction
    ( int const nP, int const nX, int const nF );
};


//! @brief C++ class for NLP solution using IPOPT and MC++
////////////////////////////////////////////////////////////////////////
//! mc::NLPSLV_IPOPT is a C++ class for solving NLP problems
//! using IPOPT and MC++
////////////////////////////////////////////////////////////////////////
class NLPSLV_IPOPT
#if defined( MC__WITH_GAMS )
: protected virtual GAMSIO,
  public virtual BASE_NLP
#else
: public virtual BASE_NLP
#endif
{
public:

  using BASE_NLP::dag;
  using BASE_NLP::set_dag;

  using BASE_NLP::par;
  using BASE_NLP::set_par;
  using BASE_NLP::add_par;
  using BASE_NLP::reset_par;

  using BASE_NLP::var;
  using BASE_NLP::set_var;
  using BASE_NLP::add_var;
  using BASE_NLP::reset_var;
  using BASE_NLP::update_vartyp;

  using BASE_NLP::set;
  using BASE_NLP::set_obj;
  using BASE_NLP::add_ctr;

#if defined( MC__WITH_GAMS )
  using GAMSIO::read;
#endif

protected:

  using BASE_NLP::_dag;
  using BASE_NLP::_var;
  using BASE_NLP::_vartyp;
  using BASE_NLP::_varlb;
  using BASE_NLP::_varlm;
  using BASE_NLP::_varub;
  using BASE_NLP::_varum;
  using BASE_NLP::_par;
  using BASE_NLP::_obj;
  using BASE_NLP::_ctr;

#if defined( MC__WITH_GAMS )
  using GAMSIO::_varini;
#endif

public:

  //! @brief NLP solution status
  enum STATUS{
     SUCCESSFUL=0,      //!< Optimal solution found (possibly not within required accuracy)
     INFEASIBLE,        //!< The problem appears to be infeasible
     UNBOUNDED,	        //!< The problem appears to be unbounded
     INTERRUPTED,       //!< Resource limit reached
     FAILURE,           //!< Terminated after numerical difficulties
     ABORTED            //!< Critical error encountered
  };

private:

  //! @brief vector of IPOPT workers
  std::vector<Ipopt::SmartPtr<WORKER_IPOPT>> _worker;

  //! @brief number of parameters in problem
  int                 _nP;
  //! @brief vector of parameters in DAG
  std::vector<FFVar>  _Pvar;
  //! @brief vector of parameter dependencies
  std::vector<FFDep>  _Pdep;

  //! @brief number of decision variables (independent and dependent) in problem
  int                 _nX;
  //! @brief vector of decision variables in DAG
  std::vector<FFVar>  _Xvar;
  //! @brief vector of decision variable dependencies
  std::vector<FFDep>  _Xdep;
  //! @brief vector of decision variable lower bounds
  std::vector<double> _Xlow;
  //! @brief vector of decision variable upper bounds
  std::vector<double> _Xupp;
  //! @brief vector of decision variable levels
  std::vector<double> _Xini;

  //! @brief number of functions (objective and constraints) in problem
  int                 _nF;
  //! @brief vector of functions in DAG
  std::vector<FFVar>  _Fvar;
  //! @brief vector of functions dependencies
  std::vector<FFDep>  _Fdep;
  //! @brief vector of function lower bounds
  std::vector<double> _Flow;
  //! @brief vector of function upper bounds
  std::vector<double> _Fupp;
  //! @brief vector of function multipliers in DAG
  std::vector<FFVar>  _Fmul;
  //! @brief vector of multiplier dependencies
  std::vector<FFDep>  _Mdep;

  //!@brief number of nonzero elements in constraint gradients
  int                 _nG;
  //!@brief row coordinates of nonzero elements in constraint gradients
  std::vector<int>    _iGfun;
  //!@brief column coordinates of nonzero elements in constraint gradients
  std::vector<int>    _jGvar;
  //! @brief vector of constraint gradients
  std::vector<FFVar>  _Gvar;

  //!@brief number of nonzero elements in Lagrangian Hessian
  int                 _nL;
  //!@brief row coordinates of nonzero elements in Lagrangian Hessian
  std::vector<int>    _iLvar;
  //!@brief column coordinates of nonzero elements in Lagrangian Hessian
  std::vector<int>    _jLvar;
  //! @brief vector of Lagrangian Hessian
  std::vector<FFVar>  _Lvar;
  //! @brief Lagrangian dependencies
  FFDep               _Ldep;

  //! @brief Lagrangian functions
  FFVar               _lagr;
  //! @brief objective gradient
  FFVar*              _obj_grad;
  //! @brief constraint gradient (sparse format)
  std::tuple< unsigned, const unsigned*, const unsigned*, const FFVar* > _ctr_grad;
  //! @brief Lagrangian Hessian (sparse format)
  std::tuple< unsigned, const unsigned*, const unsigned*, const FFVar* > _lagr_hess;

  //! @brief direction of objective function (-1:minmize; 0:feasibility; 1:maximize)
  int                 _obj_dir;

  //! @brief whether a record of the original model is available
  bool                _rec_model;
  //! @brief direction of objective function (-1:minmize; 0:feasibility; 1:maximize)
  int                 _rec_obj_dir;
  //! @brief objective gradient
  std::vector<FFVar>  _rec_obj_grad;

  //! @brief number of functions (objective and constraints) in problem
  int                 _rec_nF;
  //! @brief vector of functions in DAG
  std::vector<FFVar>  _rec_Fvar;
  //! @brief vector of function lower bounds
  std::vector<double> _rec_Flow;
  //! @brief vector of function upper bounds
  std::vector<double> _rec_Fupp;
  //! @brief vector of function multipliers in DAG
  std::vector<FFVar>  _rec_Fmul;
  
  //!@brief number of nonzero elements in constraint gradients
  int                 _rec_nG;
  //!@brief row coordinates of nonzero elements in constraint gradients
  std::vector<int>    _rec_iGfun;
  //!@brief column coordinates of nonzero elements in constraint gradients
  std::vector<int>    _rec_jGvar;
  //! @brief vector of constraint gradients
  std::vector<FFVar>  _rec_Gvar;

  //!@brief number of nonzero elements in Lagrangian Hessian
  int                 _rec_nL;
  //!@brief row coordinates of nonzero elements in Lagrangian Hessian
  std::vector<int>    _rec_iLvar;
  //!@brief column coordinates of nonzero elements in Lagrangian Hessian
  std::vector<int>    _rec_jLvar;
  //! @brief vector of Lagrangian Hessian
  std::vector<FFVar>  _rec_Lvar;

  //! @brief Structure holding solution information
  SOLUTION_OPT        _solution;

public:
  /** @defgroup NLPSLV_IPOPT Local (Continuous) Optimization using IPOPT and MC++
   *  @{
   */
  //! @brief Constructor
  NLPSLV_IPOPT()
    : _nP(0), _nX(0), _nF(0), _nG(0), _nL(0),
      _obj_grad( nullptr ), _ctr_grad( 0, nullptr, nullptr, nullptr ),
      _lagr_hess( 0, nullptr, nullptr, nullptr ), _obj_dir(0),
      _rec_model(false), _solution(FAILURE)
    {}

  //! @brief Destructor
  virtual ~NLPSLV_IPOPT()
    {
      _cleanup_grad();
      _worker.clear();
      //for( auto && th : _worker ) delete th;
    }

  //! @brief NLP solver options
  struct Options
  {
    //! @brief Constructor
    Options
      ()
      {
        reset();
      }
    //! @brief Reset to default options
    void reset
      ()
      {
        FEASTOL     = 1e-7;
        OPTIMTOL    = 1e-5;
        MAXITER     = 200;
        GRADMETH    = FSYM;
        HESSMETH    = LBFGS;
        LINMETH     = MA57;
        GRADCHECK   = false;
        DISPLEVEL   = 1;
        TIMELIMIT   = 72e2;
        MAXTHREAD   = 0;
      }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        FEASTOL     = options.FEASTOL;
        OPTIMTOL    = options.OPTIMTOL;
        MAXITER     = options.MAXITER;
        GRADMETH    = options.GRADMETH;
        HESSMETH    = options.HESSMETH;
        LINMETH     = options.LINMETH;
        GRADCHECK   = options.GRADCHECK;
        DISPLEVEL   = options.DISPLEVEL;
        TIMELIMIT   = options.TIMELIMIT;
        MAXTHREAD   = options.MAXTHREAD;
        return *this;
      }
    //! @brief Enumeration type for Hessian strategy
    enum HESSIAN_STRATEGY{
      EXACT=0, 	//!< Exact second derivatives from AD
      LBFGS,	//!< Limited-memory quasi-Newton approximation
    };
    //! @brief Enumeration type for gradient strategy
    enum GRADIENT_STRATEGY{
      FSYM=0,	//!< Forward symbolic AD
      BSYM,	//!< Backward symbolic AD
      FAD,	//!< Forward numeric AD
      BAD,	//!< Backward numeric AD
      FD        //!< Finite differences
    };
    //! @brief Enumeration type for gradient strategy
    enum LINEAR_SOLVER{
      MA27=0,       //!< Harwell routine MA27
      MA57,         //!< Harwell routine MA57
      MA77,         //!< Harwell routine HSL_MA77
      MA86,         //!< Harwell routine HSL_MA86
      MA97,         //!< Harwell routine HSL_MA97
      PARDISO,      //!< Pardiso package
      WSMP,         //!< WSMP package
      MUMPS         //!< MUMPS package
    };
    //! @brief Corresponds to "constr_viol_tol" in Ipopt, which specifies the final accuracy on the constraints
    double FEASTOL;
    //! @brief Corresponds to "tol", "dual_inf_tol" and "compl_inf_tol" in Ipopt, which specific the final accuracy on the dual and complementarity slackness conditions
    double OPTIMTOL;
   //! @brief Corresponds to "max_iter" in Ipopt, which is the maximum number of iterations.
    int MAXITER;
    //! @brief Specifies the method for computing derivatives, either analytically (FSYM, BSYM), computed using automatic differentiation (FAD, BAD), or estimated using finite differences (FD).
    GRADIENT_STRATEGY GRADMETH;
    //! @brief Corresponds to "hessian_approximation" in  Ipopt, which specifies the method for computing the Hessian, either exactly via symbolic or automatic differentiation (EXACT) or using a limited-memory BFGS update (LBFGS).
    HESSIAN_STRATEGY HESSMETH;
    //! @brief Corresponds to "linear_solver" in Ipopt, which specifies the linear solver.
    LINEAR_SOLVER LINMETH;
    //! @brief Corresponds to "derivative-test" in Ipopt, which enables finite-difference checks on the derivatives computed by the user-provided routines, both first- and second-order derivatives.
    bool GRADCHECK;
    //! @brief Corresponds to  "print_level" in Ipopt, which specifies the verbosity level of the solver between 0 (no output) and 12 (maximum verbosity)
    int DISPLEVEL;
    //! @brief Maximum run-time (in seconds) - this is checked both internally (option "max_cpu_time" in Ipopt) and externally based on the wall clock.
    double TIMELIMIT;
    //! @brief Maximum number of threads for multistart solve.
    unsigned MAXTHREAD;
  } options;

  //! @brief NLPSLV exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for NLPSLV exception handling
    enum TYPE{
      PARAM=-1,	        //!< Undefined parameter values
      INTERN=-33	//!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr(){ return _ierr; }
    //! @brief Inline function returning the error description
    std::string what(){
      switch( _ierr ){
        case PARAM:
          return "NLPSLV_IPOPT::Exceptions  Undefined parameter values";
        case INTERN:
        default:
          return "NLPSLV_IPOPT::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief Setup NLP model before solution
  bool setup();

  //! @brief Change objective function of NLP model
  bool set_obj_lazy
    ( BASE_OPT::t_OBJ const& type, FFVar const& obj );

  //! @brief Append general constraint to NLP model
  bool add_ctr_lazy
    ( BASE_OPT::t_CTR const type, FFVar const& ctr );

  //! @brief Restore original NLP model
  bool restore_model
    ();

  //! @brief Solve NLP model -- return value is IPOPT status
  template <typename T>
  int solve
    ( double const* Xini, T const* Xbnd, double const* Pval=nullptr );

  //! @brief Solve NLP model -- return value is IPOPT status
  int solve
    ( double const* Xini=nullptr, double const* Xlow=nullptr, double const* Xupp=nullptr,
      double const* Pval=nullptr );

#ifdef MC__USE_SOBOL
  //! @brief Solve NLP model using multistart search -- return value is IPOPT status
  template <typename T>
  int solve
    ( unsigned const NSAM, T const* Xbnd, double const* Pval=nullptr,
      bool const* logscal=nullptr, bool const disp=false );

  //! @brief Solve NLP model using multistart search -- return value is IPOPT status
  int solve
    ( unsigned const NSAM, double const* Xlow=nullptr, double const* Xupp=nullptr,
      double const* Pval=nullptr, bool const* logscal=nullptr, bool const disp=false );
#endif

  //! @brief Test primal feasibility
  bool is_feasible
    ( double const* p, double const CTRTOL );

  //! @brief Test primal feasibility of current solution point
  bool is_feasible
    ( const double CTRTOL );

  //! @brief Test dual feasibility
  bool is_stationary
    ( double const* x, double const* ux, double const* uf, double const GRADTOL );

  //! @brief Test dual feasibility of current solution point
  bool is_stationary
    ( double const GRADTOL );//, const double NUMTOL=1e-8 );

  //! @brief Compute cost correction to compensate for infeasibility of current solution
  double cost_correction
    ();

  //! @brief Compute cost correction to compensate for infeasibility
  double cost_correction
    ( double const* x, double const* ux, double const* uf );

  //! @brief Get solution info
  SOLUTION_OPT const& solution() const
    {
      return _solution;
    }

  //! @brief Status after last NLP call
  STATUS get_status
    ()
    const
    {
      return get_status( _solution.stat );
    }

  //! @brief Status
  static STATUS get_status
    ( int const stat )
    {
      if( stat == Ipopt::Solve_Succeeded
       || stat == Ipopt::Solved_To_Acceptable_Level
       || stat == Ipopt::Feasible_Point_Found )
        return SUCCESSFUL;
      if( stat == Ipopt::Infeasible_Problem_Detected )
        return INFEASIBLE;
      if( stat == Ipopt::Diverging_Iterates )
        return UNBOUNDED;
      if( stat == Ipopt::Search_Direction_Becomes_Too_Small
       || stat == Ipopt::Restoration_Failed
       || stat == Ipopt::Error_In_Step_Computation )
        return FAILURE;
      if( stat == Ipopt::User_Requested_Stop
       || stat == Ipopt::Maximum_Iterations_Exceeded
       || stat == Ipopt::Maximum_CpuTime_Exceeded )
        return INTERRUPTED;
      if( stat == Ipopt::Invalid_Option
       || stat == Ipopt::Not_Enough_Degrees_Of_Freedom
       || stat == Ipopt::Invalid_Problem_Definition
       || stat == Ipopt::Unrecoverable_Exception
       || stat == Ipopt::NonIpopt_Exception_Thrown
       || stat == Ipopt::Insufficient_Memory
       || stat == Ipopt::Internal_Error )
        return ABORTED;
      return ABORTED;
    }
  /** @} */

protected:

  //! @brief Cleanup gradient/hessian storage
  void _cleanup_grad
    ( bool const objGrad=true, bool const ctrGrad=true )
    {
      if( objGrad ){
        delete[] _obj_grad;               _obj_grad = nullptr;
      }
      if( ctrGrad ){
        delete[] std::get<1>(_ctr_grad);  std::get<1>(_ctr_grad) = nullptr;
        delete[] std::get<2>(_ctr_grad);  std::get<2>(_ctr_grad) = nullptr;
        delete[] std::get<3>(_ctr_grad);  std::get<3>(_ctr_grad) = nullptr;
      }
      delete[] std::get<1>(_lagr_hess); std::get<1>(_lagr_hess) = nullptr;
      delete[] std::get<2>(_lagr_hess); std::get<2>(_lagr_hess) = nullptr;
      delete[] std::get<3>(_lagr_hess); std::get<3>(_lagr_hess) = nullptr;
    }

  //! @brief Make a local copy of the original NLP model
  bool _record_model
    ();

  //! @brief Set function gradient storage
  void _set_gradient
    ();

  //! @brief Set Lagrangian Hessian storage
  void _set_hessian
    ();

  //! @brief Add function gradient to storage
  bool _add_gradient
    ( unsigned const ndx );

  //! @brief Set IPOPT options
  void _set_options
    ( Ipopt::SmartPtr<Ipopt::IpoptApplication> & IpoptApp );

  //! @brief set the worker internal variables
  void _set_worker
    ( Ipopt::SmartPtr<WORKER_IPOPT> & th );

  //! @brief resize the number of workers
  void _resize_workers
    ( int const noth );
    
#ifdef MC__USE_SOBOL
  //! @brief call multistart NLP solver on worker th
  void _mssolve
    ( int const th, unsigned const NOTHREADS, unsigned const NSAM,
      bool const* logscal, bool const DISP, int& feasible, SOLUTION_OPT& stat );
#endif

private:

  //! @brief Private methods to block default compiler methods
  NLPSLV_IPOPT(const NLPSLV_IPOPT&);
  NLPSLV_IPOPT& operator=(const NLPSLV_IPOPT&);
};

inline
void
WORKER_IPOPT::update
( int const nX, double const* Xl, double const* Xu, int const nP, double const* P0 )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
    std::cout << "  WORKER_IPOPT::update  " << warm << std::endl;
#endif

  // variable bounds
  if( Xl ) Xlow.assign( Xl, Xl+nX );
  if( Xu ) Xupp.assign( Xu, Xu+nX );
#ifdef MC__NLPSLV_IPOPT_DEBUG
  for( int i=0; i<nX; i++ )
    std::cout << "  Xlow[" << i << "] = " << Xlow[i]
              << "  Xupp[" << i << "] = " << Xupp[i] << std::endl;
#endif

  // parameter values
  this->nP = nP;
  if( !Pvar.empty() && !P0 ) throw NLPSLV_IPOPT::Exceptions( NLPSLV_IPOPT::Exceptions::PARAM );
  if( P0 ) Pval.assign( P0, P0+nP );
#ifdef MC__NLPSLV_IPOPT_DEBUG
  for( int i=0; !Pvar.empty() && P0 && i<nP; i++ )
    std::cout << "  Pval[" << i << "] = " << Pval[i] << std::endl;
#endif
}

inline
void
WORKER_IPOPT::initialize
( int const nX, double const* Xini )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
    std::cout << "  WORKER_IPOPT::initialize  " << std::endl;
#endif

  // initial starting point
  if( Xini ) Xval.assign( Xini, Xini+nX );
  else       Xval.assign( nX, 0. );
#ifdef MC__NLPSLV_IPOPT_DEBUG
  for( int i=0; i<nX; i++ )
    std::cout << "  Xval[" << i << "] = " << Xval[i] << std::endl;
#endif
}

inline
bool
WORKER_IPOPT::get_nlp_info
( Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
  Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
    std::cout << "  WORKER_IPOPT::get_nlp_info\n";
#endif

  // set size
  n = Xvar.size();
  m = Fvar.size()-1;
  nnz_jac_g = iGfun.size(); //Gvar.size();
  nnz_h_lag = iLvar.size(); //Lvar.size();

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

#ifdef MC__NLPSLV_IPOPT_DEBUG
  std::cout << "n:" << n << std::endl;
  std::cout << "m:" << m << std::endl;
  std::cout << "nnz_jac_g:" << nnz_jac_g << std::endl;
  std::cout << "nnz_h_lag:" << nnz_h_lag << std::endl;
#endif

  return true;
}

inline
bool
WORKER_IPOPT::get_bounds_info
( Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
  Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
    std::cout << "  WORKER_IPOPT::get_bounds_info\n";
#endif

  // set variable bounds
  for( Ipopt::Index i=0; i<n; i++ ){   
    x_l[i] = Xlow[i];
    x_u[i] = Xupp[i];
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "  x_l[" << i << "] = " << x_l[i]
              << "  x_u[" << i << "] = " << x_u[i] << std::endl;
#endif
  }

  // set constraint bounds
  for( Ipopt::Index j=0; j<m; j++ ){   
    g_l[j] = Flow[1+j]; // index 0 is objective
    g_u[j] = Fupp[1+j];
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "  g_l[" << j << "] = " << g_l[j]
              << "  g_u[" << j << "] = " << g_u[j] << std::endl;
#endif
  }
  return true;
}

inline
bool
WORKER_IPOPT::get_starting_point
( Ipopt::Index n, bool init_x, Ipopt::Number* x, bool init_z,
  Ipopt::Number* z_L, Ipopt::Number* z_U, Ipopt::Index m,
  bool init_lambda, Ipopt::Number* lambda )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
    std::cout << "  WORKER_IPOPT::get_starting_point  "
              << init_x << init_z << init_lambda << std::endl;
#endif

  // Here, we assume we only have starting values for x, if you code
  // your own NLP, you can provide starting values for the dual variables
  // if you wish
  if( !init_x || init_z || init_lambda ) return false;

  // initialize to the given starting point
  for( Ipopt::Index i=0; i<n; i++ ){   
    x[i] = Xval[i];
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "  x_0[" << i << "] = " << x[i] << std::endl;
#endif
  }
  return true;
}

inline
bool
WORKER_IPOPT::eval_f
( Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& f )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
  std::cout << "  WORKER_IPOPT::eval_f  " << new_x << std::endl;
  for( Ipopt::Index i=0; i<n; i++ )
    std::cout << "  x[" << i << "] = " << x[i] << std::endl;
#endif

  // evaluate objective
  try{
    if( nP ) dag.eval( op_f, dwk, 1, Fvar.data(), &f, n, Xvar.data(), x, nP, Pvar.data(), Pval.data() );
    else     dag.eval( op_f, dwk, 1, Fvar.data(), &f, n, Xvar.data(), x );
  }
  catch(...){
    return false;
  }
#ifdef MC__NLPSLV_IPOPT_DEBUG
  std::cout << "  f = " << f << std::endl;
#endif

  return true;
}

inline
bool
WORKER_IPOPT::eval_grad_f
( Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* df )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
  std::cout << "  WORKER_IPOPT::eval_grad_f  " << new_x << std::endl;
  for( Ipopt::Index i=0; i<n; i++ )
    std::cout << "  x[" << i << "] = " << x[i] << std::endl;
#endif

  // evaluate objective gradient
  try{
    switch( Gmeth ){
      // Compute symbolic derivative
      case NLPSLV_IPOPT::Options::FSYM:
      case NLPSLV_IPOPT::Options::BSYM:
        if( nP ) dag.eval( op_df, dwk, n, Cvar.data(), df, n, Xvar.data(), x, nP, Pvar.data(), Pval.data() );
        else     dag.eval( op_df, dwk, n, Cvar.data(), df, n, Xvar.data(), x );
        break;

      // Compute forward numeric derivative
      case NLPSLV_IPOPT::Options::FAD:
        FXval.resize( n );
        // Initialize participating variables in fadbad::F<double>
        for( Ipopt::Index i=0; i<n; i++ ){
          FXval[i] = x[i];
          FXval[i].diff( i, n );
        }
        if( nP ){
          FPval.resize( nP );
          // Initialize parameters in fadbad::F<double>
          for( size_t iP=0; iP<nP; ++iP ) FPval[iP] = Pval[iP];
          dag.eval( op_f, Fwk, 1, Fvar.data(), &FCval, n, Xvar.data(), FXval.data(), nP, Pvar.data(), FPval.data() );
        }
        else{
          dag.eval( op_f, Fwk, 1, Fvar.data(), &FCval, n, Xvar.data(), FXval.data() );
        }
        // Gather derivatives
        for( Ipopt::Index i=0; i<n; i++ )
          df[i] = FCval.d(i);
        break;

      // Compute backward numeric derivative
      case NLPSLV_IPOPT::Options::BAD:
        BXval.resize( n );
        // Initialize participating variables in fadbad::B<double>
        for( Ipopt::Index i=0; i<n; i++ )
          BXval[i] = x[i];
        if( nP ){
          BPval.resize( nP );
          // Initialize parameters in fadbad::B<double>
          for( size_t iP=0; iP<nP; ++iP ) BPval[iP] = Pval[iP];
          dag.eval( op_f, Bwk, 1, Fvar.data(), &BCval, n, Xvar.data(), BXval.data(), nP, Pvar.data(), BPval.data() );
        }
        else{
          dag.eval( op_f, Bwk, 1, Fvar.data(), &BCval, n, Xvar.data(), BXval.data() );
        }
        Bwk.clear();
        BCval.diff( 0, 1 );
        // Gather derivatives
        for( Ipopt::Index i=0; i<n; i++ )
          df[i] = BXval[i].d(0);
        break;

      // Other derivative method - error
      default:
        throw typename NLPSLV_IPOPT::Exceptions( NLPSLV_IPOPT::Exceptions::INTERN );
    }

#ifdef MC__NLPSLV_IPOPT_DEBUG
    for( Ipopt::Index i=0; i<n; i++ )
      std::cout << "  df[" << i << "] = " << df[i] << std::endl;
    { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
  }

  catch(...){
    return false;
  }
  return true;
}

inline
bool
WORKER_IPOPT::eval_g
( Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m,
  Ipopt::Number* g )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
  std::cout << "  WORKER_IPOPT::eval_g  " << new_x << std::endl;
  for( Ipopt::Index i=0; i<n; i++ )
    std::cout << "  x[" << i << "] = " << x[i] << std::endl;
#endif

  // evaluate constraints
  try{
    if( nP ) dag.eval( op_g, dwk, m, Fvar.data()+1, g, n, Xvar.data(), x, nP, Pvar.data(), Pval.data() );
    else     dag.eval( op_g, dwk, m, Fvar.data()+1, g, n, Xvar.data(), x );
#ifdef MC__NLPSLV_IPOPT_DEBUG
    for( Ipopt::Index j=0; j<m; j++ )
      std::cout << "  g[" << j << "] = " << g[j] << std::endl;
#endif
  }
  catch(...){
    return false;
  }
  return true;
}

inline
bool
WORKER_IPOPT::eval_jac_g
( Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m,
  Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
  Ipopt::Number* dg )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
  std::cout << "  WORKER_IPOPT::eval_jac_g  " << new_x << std::endl;
#endif
 
  // return the constraint Jacobian structure
  if( !dg ){
    for( Ipopt::Index ie=0; ie<nele_jac; ++ie ){
      iRow[ie] = iGfun[ie];
      jCol[ie] = jGvar[ie];
#ifdef MC__NLPSLV_IPOPT_DEBUG
      std::cout << "  dg[" << iRow[ie] << ", " << jCol[ie] << "]" << std::endl;
#endif
    }
    return true;
  }

  // evaluate constraint gradient
  try{
    switch( Gmeth ){
      // Compute symbolic derivative
      case NLPSLV_IPOPT::Options::FSYM:
      case NLPSLV_IPOPT::Options::BSYM:
        if( nP ) dag.eval( op_dg, dwk, nele_jac, Gvar.data(), dg, n, Xvar.data(), x, nP, Pvar.data(), Pval.data() );
        else     dag.eval( op_dg, dwk, nele_jac, Gvar.data(), dg, n, Xvar.data(), x );
        break;

      // Compute forward numeric derivative
      case NLPSLV_IPOPT::Options::FAD:
        FXval.resize( n );
        // Initialize participating variables in fadbad::F<double>
        for( Ipopt::Index i=0; i<n; i++ ){
          FXval[i] = x[i];
          FXval[i].diff( i, n );
        }
        FFval.resize( m );
        if( nP ){
          FPval.resize( nP );
          // Initialize parameters in fadbad::F<double>
          for( size_t iP=0; iP<nP; ++iP ) FPval[iP] = Pval[iP];
          dag.eval( op_g, Fwk, m, Fvar.data()+1, FFval.data(), n, Xvar.data(), FXval.data(), nP, Pvar.data(), FPval.data() );
        }
        else{
          dag.eval( op_g, Fwk, m, Fvar.data()+1, FFval.data(), n, Xvar.data(), FXval.data() );
        }

        // Gather derivatives
        for( Ipopt::Index ie=0; ie<nele_jac; ++ie )
          dg[ie] = FFval[ iGfun[ie] ].d( jGvar[ie] );
        break;

      // Compute backward numeric derivative
      case NLPSLV_IPOPT::Options::BAD:
        BXval.resize( n );
        // Initialize participating variables in fadbad::B<double>
        for( Ipopt::Index i=0; i<n; i++ )
          BXval[i] = x[i];
        BFval.resize( m );
        if( nP ){
          BPval.resize( nP );
          // Initialize parameters in fadbad::B<double>
          for( size_t iP=0; iP<nP; ++iP ) BPval[iP] = Pval[iP];
          dag.eval( op_g, Bwk, m, Fvar.data()+1, BFval.data(), n, Xvar.data(), BXval.data(), nP, Pvar.data(), BPval.data() );
        }
        else{
          dag.eval( op_g, Bwk, m, Fvar.data()+1, BFval.data(), n, Xvar.data(), BXval.data() );
        }

        Bwk.clear();
        for( Ipopt::Index j=0; j<m; j++ )
          BFval[j].diff( j, m );
        // Gather derivatives
        for( Ipopt::Index ie=0; ie<nele_jac; ++ie )
          dg[ie] = BXval[ jGvar[ie] ].d( iGfun[ie] );
        break;

      // Other derivative method - error
      default:
        throw typename NLPSLV_IPOPT::Exceptions( NLPSLV_IPOPT::Exceptions::INTERN );
    }

#ifdef MC__NLPSLV_IPOPT_DEBUG
    for( Ipopt::Index ie=0; ie<nele_jac; ++ie )
       std::cout << "  dg[" << iGfun[ie] << ", " << jGvar[ie] << "] = " << dg[ie] << std::endl;
#endif
  }
  
  catch(...){
    return false;
  }
  return true;
}

inline
bool
WORKER_IPOPT::eval_h
( Ipopt::Index n, const Ipopt::Number* x, bool new_x,
  Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
  bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
  Ipopt::Index* jCol, Ipopt::Number* d2L )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
  std::cout << "  WORKER_IPOPT::eval_h  " << new_x  << new_lambda << std::endl;
#endif

  // return the Lagrangian Hessian structure
  if( !d2L ){
    for( Ipopt::Index ie=0; ie<nele_hess; ++ie ){
      iRow[ie] = iLvar[ie];
      jCol[ie] = jLvar[ie];
#ifdef MC__NLPSLV_IPOPT_DEBUG
      std::cout << "  d2L[" << iRow[ie] << ", " << jCol[ie] << "]" << std::endl;
#endif
    }
    return true;
  }

  // evaluate Lagrangian Hessian
  try{
    switch( Gmeth ){
      // Compute symbolic derivative
      case NLPSLV_IPOPT::Options::FSYM:
      case NLPSLV_IPOPT::Options::BSYM:
        if( nP ) dag.eval( op_L, dwk, nele_hess, Lvar.data(), d2L, n, Xvar.data(), x, 1, Fmul.data(), &obj_factor, m, Fmul.data()+1, lambda, nP, Pvar.data(), Pval.data() );
        else     dag.eval( op_L, dwk, nele_hess, Lvar.data(), d2L, n, Xvar.data(), x, 1, Fmul.data(), &obj_factor, m, Fmul.data()+1, lambda );
        break;

      // Compute backward/forward numeric derivative
      case NLPSLV_IPOPT::Options::FAD:
      case NLPSLV_IPOPT::Options::BAD:
        if( Lvar.empty() ) break;
        // Initialize participating variables in fadbad::F<double>
        FXval.resize( n );
        FFmul.resize( m );
        for( Ipopt::Index i=0; i<n; i++ ){
          FXval[i] = x[i];
          FXval[i].diff( i, n );
        }
        FCmul = obj_factor;
        for( Ipopt::Index j=0; j<m; j++ )
          FFmul[j] = lambda[j];
        // Initialize participating variables in fadbad::B<fadbad::F<double>>
        BFXval.resize( n );
        BFFmul.resize( m );
        for( Ipopt::Index i=0; i<n; i++ )
          BFXval[i] = FXval[i];
        BFCmul = FCmul;
        for( Ipopt::Index j=0; j<m; j++ )
          BFFmul[j] = FFmul[j];
        if( nP ){
          FPval.resize( nP );
          BFPval.resize( nP );
          // Initialize parameters in fadbad::B<double>
          for( size_t iP=0; iP<nP; ++iP ){
            FPval[iP]  = Pval[iP];
            BFPval[iP] = FPval[iP];
          }
          dag.eval( op_L, BFwk, 1, Lvar.data(), &BFLval, n, Xvar.data(), BFXval.data(), 1, Fmul.data(), &BFCmul, m, Fmul.data()+1, BFFmul.data(), nP, Pvar.data(), BFPval.data() );
        }
        else{
          dag.eval( op_L, BFwk, 1, Lvar.data(), &BFLval, n, Xvar.data(), BFXval.data(), 1, Fmul.data(), &BFCmul, m, Fmul.data()+1, BFFmul.data() );
        }
        BFwk.clear();
        BFLval.diff( 0, 1 );
        // Gather derivatives
        for( Ipopt::Index ie=0; ie<nele_hess; ++ie )
          d2L[ie] = BFXval[ jLvar[ie] ].d( 0 ).d( iLvar[ie] );
        break;

      // Other derivative method - error
      default:
        throw typename NLPSLV_IPOPT::Exceptions( NLPSLV_IPOPT::Exceptions::INTERN );
    }

#ifdef MC__NLPSLV_IPOPT_DEBUG
    for( Ipopt::Index ie=0; ie<nele_hess; ++ie )
       std::cout << "  d2L[" << iLvar[ie] << ", " << jLvar[ie] << "] = " << d2L[ie] << std::endl;
#endif
  }

  catch(...){
    return false;
  }
  return true;
}

inline
void
WORKER_IPOPT::finalize_solution
( Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* p,
  const Ipopt::Number* upL, const Ipopt::Number* upU, Ipopt::Index m,
  const Ipopt::Number* g, const Ipopt::Number* ug, Ipopt::Number f,
  const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
    std::cout << "  WORKER_IPOPT::finalize_solution\n";
#endif
  solution.stat    = status;
  solution.p       = Pval;
  solution.x.assign( p, p+n );
  solution.ux.resize( n );
  for( int i=0; i<n; i++ ) solution.ux[i] = upL[i] - upU[i];  
  solution.f.assign( 1, f );
  solution.f.insert( solution.f.end(), g, g+m );
  solution.uf.assign( 1, -1. );
  solution.uf.resize( m+1 );
  for( int j=0; j<m; j++ ) solution.uf[1+j] = - ug[j];
}

inline
bool
WORKER_IPOPT::intermediate_callback
( Ipopt::AlgorithmMode mode, Ipopt::Index iter, Ipopt::Number obj_value,
  Ipopt::Number inf_pr, Ipopt::Number inf_du, Ipopt::Number mu,
  Ipopt::Number d_norm, Ipopt::Number regularization_size,
  Ipopt::Number alpha_du, Ipopt::Number alpha_pr, Ipopt::Index ls_trials,
  const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq )
{
#ifdef MC__NLPSLV_IPOPT_TRACE
    std::cout << "  WORKER_IPOPT::intermediate_callback\n";
#endif
  return true;
}

inline
bool
WORKER_IPOPT::feasible
( double const CTRTOL, int const nP, int const nX, int const nF )
{
  double maxinfeas = 0.;
  for( int i=0; i<nX; i++ ){
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "X[" << i << "]: " << Xlow[i] << " <= " << solution.x[i] << " <= " << Xupp[i] << std::endl;
#endif
    maxinfeas = Xlow[i] - solution.x[i];
    if( maxinfeas > CTRTOL ) return false;
    maxinfeas = solution.x[i] - Xupp[i];
    if( maxinfeas > CTRTOL ) return false;
  }

  try{
    solution.f.assign( nF, 0. );
    if( nP ){
      dag.eval( op_f, dwk, 1, Fvar.data(), solution.f.data(), nX, Xvar.data(), solution.x.data(), nP, Pvar.data(), solution.p.data() );
      dag.eval( op_g, dwk, nF-1, Fvar.data()+1, solution.f.data()+1, nX, Xvar.data(), solution.x.data(), nP, Pvar.data(), solution.p.data() );
    }
    else{
      dag.eval( op_f, dwk, 1, Fvar.data(), solution.f.data(), nX, Xvar.data(), solution.x.data() );
      dag.eval( op_g, dwk, nF-1, Fvar.data()+1, solution.f.data()+1, nX, Xvar.data(), solution.x.data() );
    }
  }
  catch(...){
    return false;
  }
  for( int i=1; i<nF; i++ ){
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "F[" << i << "]: " << Flow[i] << " <= " << solution.f[i] << " <= " << Fupp[i] << std::endl;
#endif
    maxinfeas = Flow[i] - solution.f[i];
    if( maxinfeas > CTRTOL ) return false;
    maxinfeas = solution.f[i] - Fupp[i];
    if( maxinfeas > CTRTOL ) return false;
  }
  return true;
}

inline
bool
WORKER_IPOPT::stationary
( double const GRADTOL, int const nP, int const nX, int const nF, int const nG )
{
  Cval.resize( nX );
  Gval.resize( nG );
  try{
    switch( Gmeth ){
      // Compute symbolic derivative
      case NLPSLV_IPOPT::Options::FSYM:
      case NLPSLV_IPOPT::Options::BSYM:
        if( nP ){
          dag.eval( op_df, dwk, nX, Cvar.data(), Cval.data(), nX, Xvar.data(), solution.x.data(), nP, Pvar.data(), solution.p.data() );
          dag.eval( op_dg, dwk, nG, Gvar.data(), Gval.data(), nX, Xvar.data(), solution.x.data(), nP, Pvar.data(), solution.p.data() );
        }
        else{
          dag.eval( op_df, dwk, nX, Cvar.data(), Cval.data(), nX, Xvar.data(), solution.x.data() );
          dag.eval( op_dg, dwk, nG, Gvar.data(), Gval.data(), nX, Xvar.data(), solution.x.data() );
        }
        break;

      // Compute forward numeric derivative
      case NLPSLV_IPOPT::Options::FAD:
        FXval.resize( nX );
        // Initialize participating variables in fadbad::F<double>
        for( int i=0; i<nX; i++ ){
          FXval[i] = solution.x.data()[i];
          FXval[i].diff( i, nX );
        }

        if( nP ){
          FPval.resize( nP );
          // Initialize parameters in fadbad::F<double>
          for( int iP=0; iP<nP; ++iP ) FPval[iP] = solution.p[iP];
          dag.eval( op_f, Fwk, 1, Fvar.data(), &FCval, nX, Xvar.data(), FXval.data(), nP, Pvar.data(), FPval.data() );
        }
        else{
          dag.eval( op_f, Fwk, 1, Fvar.data(), &FCval, nX, Xvar.data(), FXval.data() );
        }
        // Gather derivatives
        for( int i=0; i<nX; i++ ) Cval[i] = FCval.d(i);

        FFval.resize( nF-1 );
        if( nP ){
          dag.eval( op_g, Fwk, nF-1, Fvar.data()+1, FFval.data(), nX, Xvar.data(), FXval.data(), nP, Pvar.data(), FPval.data() );
        }
        else{
          dag.eval( op_g, Fwk, nF-1, Fvar.data()+1, FFval.data(), nX, Xvar.data(), FXval.data() );
        }
        // Gather derivatives
        for( int i=0; i<nG; ++i )
          Gval[i] = FFval[ iGfun[i] ].d( jGvar[i] );
        break;

      // Compute backward numeric derivative
      case NLPSLV_IPOPT::Options::BAD:
        BXval.resize( nX );
        // Initialize participating variables in fadbad::B<double>
        for( int i=0; i<nX; i++ ) BXval[i] = solution.x.data()[i];
        if( nP ){
          BPval.resize( nP );
          // Initialize parameters in fadbad::B<double>
          for( int iP=0; iP<nP; ++iP ) BPval[iP] = solution.p[iP];
          dag.eval( op_f, Bwk, 1, Fvar.data(), &BCval, nX, Xvar.data(), BXval.data(), nP, Pvar.data(), BPval.data() );
        }
        else{
          dag.eval( op_f, Bwk, 1, Fvar.data(), &BCval, nX, Xvar.data(), BXval.data() );
        }
        Bwk.clear();
        BCval.diff( 0, 1 );
        // Gather derivatives
        for( int i=0; i<nX; i++ ) Cval[i] = BXval[i].d(0);

        for( int i=0; i<nX; i++ ) BXval[i] = solution.x.data()[i];
        BFval.resize( nF-1 );
        if( nP ){
          BPval.resize( nP );
          // Initialize parameters in fadbad::B<double>
          for( int iP=0; iP<nP; ++iP ) BPval[iP] = solution.p[iP];
          dag.eval( op_g, Bwk, nF-1, Fvar.data()+1, BFval.data(), nX, Xvar.data(), BXval.data(), nP, Pvar.data(), BPval.data() );
        }
        else{
          dag.eval( op_g, Bwk, nF-1, Fvar.data()+1, BFval.data(), nX, Xvar.data(), BXval.data() );
        }
        Bwk.clear();
        for( int j=0; j<nF-1; j++ )
          BFval[j].diff( j, nF-1 );
        // Gather derivatives
        for( int i=0; i<nG; ++i )
          Gval[i] = BXval[ jGvar[i] ].d( iGfun[i] );
        break;

      // Other derivative method - error
      default:
        throw typename NLPSLV_IPOPT::Exceptions( NLPSLV_IPOPT::Exceptions::INTERN );
    }
#ifdef MC__NLPSLV_IPOPT_DEBUG
    for( int i=0; i<nX; ++i )
      std::cout << "  Cval[" << i << "] = " << Cval[i] << std::endl;
    for( int ie=0; ie<nG; ++ie )
      std::cout << "  Gval[" << iGfun[ie] << ", " << jGvar[ie] << "] = " << Gval[ie] << std::endl;
#endif
  }
  catch(...){
    return false;
  }

  std::vector<double> gradL = solution.ux;
  for( int i=0; i<nX; i++ )
    gradL[i] += Cval[i] * solution.uf[0];
  for( int ie=0; ie<nG; ie++ )
    gradL[jGvar[ie]] += Gval[ie] * solution.uf[1+iGfun[ie]];
  for( int i=0; i<nX; i++ ){
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "  gradL[" << i << "] : " << gradL[i] << " = 0" << std::endl;
#endif
    if( std::fabs( gradL[i] ) > GRADTOL ) return false;
  }
  return true;
}

inline
double
WORKER_IPOPT::correction
( int const nP, int const nX, int const nF )
{
  double costcorr = 0.;
  for( int i=0; i<nX; i++ ){
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "X[" << i << "]: " << Xlow[i] << " <= " << solution.x[i] << " <= " << Xupp[i] << std::endl;
#endif
    costcorr += std::max( Xlow[i] - solution.x[i], 0. ) * solution.ux[i];
    costcorr -= std::max( solution.x[i] - Xupp[i], 0. ) * solution.ux[i];
  }

  try{
    solution.f.assign( nF, 0. );
    if( nP ){
      dag.eval( op_f, dwk, 1, Fvar.data(), solution.f.data(), nX, Xvar.data(), solution.x.data(), nP, Pvar.data(), solution.p.data() );
      dag.eval( op_g, dwk, nF-1, Fvar.data()+1, solution.f.data()+1, nX, Xvar.data(), solution.x.data(), nP, Pvar.data(), solution.p.data() );
    }
    else{
      dag.eval( op_f, dwk, 1, Fvar.data(), solution.f.data(), nX, Xvar.data(), solution.x.data() );
      dag.eval( op_g, dwk, nF-1, Fvar.data()+1, solution.f.data()+1, nX, Xvar.data(), solution.x.data() );
    }
  }
  catch(...){
    return costcorr;
  }
  for( int i=1; i<nF; i++ ){
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "F[" << i << "]: " << Flow[i] << " <= " << solution.f[i] << " <= " << Fupp[i] << std::endl;
#endif
    costcorr += std::max( Flow[i] - solution.f[i], 0. ) * solution.uf[i];
    costcorr -= std::max( solution.f[i] - Fupp[i], 0. ) * solution.uf[i];
  }

  return costcorr;
}

inline
void
NLPSLV_IPOPT::_set_options
( Ipopt::SmartPtr<Ipopt::IpoptApplication> & IpoptApp )
{
  IpoptApp->Options()->SetNumericValue( "constr_viol_tol",      options.FEASTOL<0.?  0.: options.FEASTOL );
  IpoptApp->Options()->SetNumericValue( "tol",                  options.OPTIMTOL<0.? 0.: options.OPTIMTOL );
  IpoptApp->Options()->SetNumericValue( "dual_inf_tol",         options.OPTIMTOL<0.? 0.: options.OPTIMTOL );
  IpoptApp->Options()->SetNumericValue( "compl_inf_tol",        options.OPTIMTOL<0.? 0.: options.OPTIMTOL );
  IpoptApp->Options()->SetIntegerValue( "max_iter",             options.MAXITER );
  IpoptApp->Options()->SetNumericValue( "max_cpu_time",         options.TIMELIMIT);
  IpoptApp->Options()->SetNumericValue( "nlp_upper_bound_inf",  BASE_OPT::INF );
  IpoptApp->Options()->SetNumericValue( "obj_scaling_factor",   _obj_dir > 0? -1.: 1. );
  switch( options.HESSMETH ){
   case Options::EXACT:
    IpoptApp->Options()->SetStringValue( "hessian_approximation", "exact" );
    break;
   case Options::LBFGS:
    IpoptApp->Options()->SetStringValue( "hessian_approximation", "limited-memory" );
    break;
  }
  switch( options.LINMETH ){
   case Options::MA27:
    IpoptApp->Options()->SetStringValue( "linear_solver", "ma27" );
    break;
   case Options::MA57:
    IpoptApp->Options()->SetStringValue( "linear_solver", "ma57" );
    break;
   case Options::MA77:
    IpoptApp->Options()->SetStringValue( "linear_solver", "ma77" );
    break;
   case Options::MA86:
    IpoptApp->Options()->SetStringValue( "linear_solver", "ma86" );
    break;
   case Options::MA97:
    IpoptApp->Options()->SetStringValue( "linear_solver", "ma97" );
    break;
   case Options::PARDISO:
    IpoptApp->Options()->SetStringValue( "linear_solver", "pardiso" );
    break;
   case Options::WSMP:
    IpoptApp->Options()->SetStringValue( "linear_solver", "wsmp" );
    break;
   case Options::MUMPS:
    IpoptApp->Options()->SetStringValue( "linear_solver", "mumps" );
    break;
  }
  IpoptApp->Options()->SetStringValue( "derivative_test", options.GRADCHECK? "second-order": "none" );
  IpoptApp->Options()->SetIntegerValue( "print_level",    options.DISPLEVEL<0? 0: (options.DISPLEVEL>12? 12: options.DISPLEVEL ) );
}

inline
bool
NLPSLV_IPOPT::setup
()
{
  // full set of parameters
  _Pvar = _par;
  _nP = _Pvar.size();

  // set dependencies of parameters
  _Pdep.assign( _nP, 0. );

  // full set of decision variables
  _Xvar = _var;
  _nX = _Xvar.size();

  // set dependencies of decision variables
  _Xdep.resize( _nX );
  for( int i=0; i<_nX; ++i )
    _Xdep[i].indep( _Xvar[i].id().second );

  // full set of variable initial values from GAMS
#if defined (MC__WITH_GAMS)
  _Xini = _varini;
#else
  _Xini.clear();
#endif
  
  // full set of variable bounds
  _Xlow = _varlb;
  _Xupp = _varub;

  // full set of nonlinear functions (cost, constraints & equations)
  _Fvar.clear();
  _Fmul.clear();
  _Flow.clear();
  _Fupp.clear();
  
  // first, cost function
  if( std::get<0>(_obj).size() ){
    _Fvar.push_back( std::get<1>(_obj)[0] );
    _Fmul.push_back( std::get<2>(_obj)[0] );
    _obj_dir = (std::get<0>(_obj)[0]==BASE_OPT::MIN? -1: 1 );
  }
  else{
    _Fvar.push_back( 0 );
    _Fmul.push_back( FFVar(_dag) );
    _obj_dir = 0;
  }
  _Flow.push_back( -BASE_OPT::INF );
  _Fupp.push_back(  BASE_OPT::INF );

  // then, regular constraints
  for( unsigned i=0; i<std::get<0>(_ctr).size(); i++ ){
    if( std::get<3>(_ctr)[i] ) continue; // ignore if redundant
    _Fvar.push_back( std::get<1>(_ctr)[i] );
    _Fmul.push_back( std::get<2>(_ctr)[i] );
    switch( std::get<0>(_ctr)[i] ){
      case BASE_OPT::EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );             break;
      case BASE_OPT::LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );             break;
      case BASE_OPT::GE: _Flow.push_back( 0. );             _Fupp.push_back(  BASE_OPT::INF ); break;
    }
  }
  _nF = _Fvar.size();

  _set_gradient();
  _set_hessian();
  _rec_model = false;

  return true;
}

inline
void
NLPSLV_IPOPT::_set_gradient
()
{
  // setup objective and constraint gradient evaluation
  _cleanup_grad();
  switch( options.GRADMETH ){
    case Options::FSYM: // Forward symbolic
      _obj_grad = _dag->FAD( 1,      _Fvar.data(),   _nX, _Xvar.data() ); 
      _ctr_grad = _dag->SFAD( _nF-1, _Fvar.data()+1, _nX, _Xvar.data() ); 
      break;
    case Options::BSYM: // Backward symbolic
      _obj_grad = _dag->BAD( 1,      _Fvar.data(),   _nX, _Xvar.data() ); 
      _ctr_grad = _dag->SBAD( _nF-1, _Fvar.data()+1, _nX, _Xvar.data() ); 
      break;
    default:
      break;
  }

  _iGfun.clear(); _jGvar.clear(); _Gvar.clear(); 
  for( unsigned k=0; k<std::get<0>(_ctr_grad); ++k ){
    _iGfun.push_back( std::get<1>(_ctr_grad)[k] );
    _jGvar.push_back( std::get<2>(_ctr_grad)[k] );
    _Gvar.push_back( std::get<3>(_ctr_grad)[k] );
#ifdef MC__NLPSLV_IPOPT_DEBUG
     std::cout << "  _Gvar[" << _iGfun.back() << "," << _jGvar.back() << "] = "
               << std::get<3>(_ctr_grad)[k] << std::endl;
#endif
  }

  // Gather derivative entries
  switch( options.GRADMETH ){
    case Options::FAD:
    case Options::BAD:
    case Options::FD:
      _Fdep.resize( _nF-1 );
      _dag->eval( _nF-1, _Fvar.data()+1, _Fdep.data(), _nX, _Xvar.data(), _Xdep.data(), _nP, _Pvar.data(), _Pdep.data() ); 
      for( int iF=0; iF<_nF-1; ++iF ){
        for( int iX=0; iX<_nX; ++iX ){
          if( !_Fdep[iF].dep( _Xvar[iX].id().second ).first ) continue;
          _iGfun.push_back( iF );
          _jGvar.push_back( iX );
#ifdef MC__NLPSLV_IPOPT_DEBUG
          std::cout << "  _Gvar[" << iF << "," << iX << "]" << std::endl;
#endif
        }
      }
      break;
    default:
      break;
  }
  _nG = _Gvar.size();
}

inline
void
NLPSLV_IPOPT::_set_hessian
()
{
  switch( options.HESSMETH ){
    // No beed to setup Lagrangian Hessian
    default:
    case Options::LBFGS:
      break;
    
    // setup Lagrangian Hessian evaluation
    case Options::EXACT:
      _cleanup_grad( false ); // Clean _ctr_grad, but not _obj_grad
      for( int i=0; i<_nF; i++ ){
        if( !i ) _lagr  = _Fvar[0] * _Fmul[0];
        else     _lagr += _Fvar[i] * _Fmul[i];
      }
#ifdef MC__NLPSLV_IPOPT_DEBUG
      _dag->output( _dag->subgraph( 1, &_lagr ) );
#endif

      switch( options.GRADMETH ){
        case Options::FSYM: // Forward symbolic
        case Options::BSYM: // Backward symbolic
          _ctr_grad  = _dag->SBAD( 1, &_lagr, _nX, _Xvar.data() ); 
          _lagr_hess = _dag->SFAD( std::get<0>(_ctr_grad), std::get<3>(_ctr_grad), _nX, _Xvar.data(), 1 ); // only keeps lower-triangular elements
          break;
        default:
          break;
      }
#ifdef MC__NLPSLV_IPOPT_DEBUG
      for( unsigned ie=0; ie<std::get<0>(_ctr_grad); ++ie )
        std::cout << "  _ctr_grad[" << std::get<1>(_ctr_grad)[ie] << "," << std::get<2>(_ctr_grad)[ie] << "] = "
                  << std::get<3>(_ctr_grad)[ie] << std::endl;
      _dag->output( _dag->subgraph( std::get<0>(_ctr_grad), std::get<3>(_ctr_grad) ) );
#endif

      _iLvar.clear(); _jLvar.clear(); _Lvar.clear(); 
      for( unsigned k=0; k<std::get<0>(_lagr_hess); ++k ){
        _iLvar.push_back( std::get<2>(_ctr_grad)[std::get<1>(_lagr_hess)[k]] );
        _jLvar.push_back( std::get<2>(_lagr_hess)[k] );
        _Lvar.push_back( std::get<3>(_lagr_hess)[k] );
#ifdef MC__NLPSLV_IPOPT_DEBUG
         std::cout << "  _Lvar[" << _iLvar.back() << "," << _jLvar.back() << "] =\n";
         _dag->output( _dag->subgraph( 1, &_Lvar.back() ) );
#endif
      }

      // Gather derivative entries
      switch( options.GRADMETH ){
        case Options::FAD:
        case Options::BAD:
        case Options::FD:
          _Lvar.push_back( _lagr );
          _Mdep.assign( _nF, 0 );
          _dag->eval( 1, _Lvar.data(), &_Ldep, _nX, _Xvar.data(), _Xdep.data(), _nF, _Fmul.data(), _Mdep.data(), _nP, _Pvar.data(), _Pdep.data() ); 
          for( int iX=0; iX<_nX; ++iX ){
            if( _Ldep.dep( _Xvar[iX].id().second ).second == FFDep::TYPE::L ) continue;
            for( int jX=0; jX<=iX; ++jX ){ // gather lower-triangular elements
              if( _Ldep.dep( _Xvar[jX].id().second ).second == FFDep::TYPE::L ) continue;
              _iLvar.push_back( iX );
              _jLvar.push_back( jX );
#ifdef MC__NLPSLV_IPOPT_DEBUG
              std::cout << "  _Lvar[" << iX << "," << jX << "]" << std::endl;
#endif
            }
          }
          break;
        default:
          break;
      }
      _nL = _Lvar.size();
  }  
}

inline
bool
NLPSLV_IPOPT::_add_gradient
( unsigned const ndxF )
{
  // setup objective and constraint gradient evaluation
  _cleanup_grad( false, true ); // reset constraint gradient only
  switch( options.GRADMETH ){
    case Options::FSYM: // Forward symbolic
      _ctr_grad = _dag->SFAD( 1, &_Fvar[ndxF], _nX, _Xvar.data() ); 
      break;
    case Options::BSYM: // Backward symbolic
      _ctr_grad = _dag->SBAD( 1, &_Fvar[ndxF], _nX, _Xvar.data() ); 
      break;
    default:
      break;
  }

  // Do *not* reinitialize _iGfun / _jGvar / _Gvar 
  for( unsigned k=0; k<std::get<0>(_ctr_grad); ++k ){
    _iGfun.push_back( ndxF-1 ); // subtracting cost function
    _jGvar.push_back( std::get<2>(_ctr_grad)[k] );
    _Gvar.push_back( std::get<3>(_ctr_grad)[k] );
#ifdef MC__NLPSLV_IPOPT_DEBUG
     std::cout << "  _Gvar[" << _iGfun.back() << "," << _jGvar.back() << "] = "
               << std::get<3>(_ctr_grad)[k] << std::endl;
#endif
  }

  // Gather derivative entries
  switch( options.GRADMETH ){
    case Options::FAD:
    case Options::BAD:
    case Options::FD:
      if( _Fdep.empty() ) _Fdep.resize( 1 );
      _dag->eval( 1, &_Fvar[ndxF], _Fdep.data(), _nX, _Xvar.data(), _Xdep.data(), _nP, _Pvar.data(), _Pdep.data() ); 
      for( int iX=0; iX<_nX; ++iX ){
        if( !_Fdep[0].dep( _Xvar[iX].id().second ).first ) continue;
        _iGfun.push_back( ndxF-1 ); // subtracting cost function
        _jGvar.push_back( iX );
#ifdef MC__NLPSLV_IPOPT_DEBUG
        std::cout << "  _Gvar[" << _iGfun.back() << "," << _jGvar.back() << "]" << std::endl;
#endif
      }
      break;
    default:
      break;
  }
  _nG = _Gvar.size();
  
  return true;
}

inline
bool
NLPSLV_IPOPT::_record_model
()
{
  if( _rec_model ) return false;

  _rec_obj_dir = _obj_dir;
  _rec_Fvar    = _Fvar;
  _rec_Fmul    = _Fmul;
  _rec_Flow    = _Flow;
  _rec_Fupp    = _Fupp;
  _rec_nF      = _nF;
  if( _obj_grad )
    _rec_obj_grad.assign( _obj_grad, _obj_grad+_nX );
  else
    _rec_obj_grad.clear();
  _rec_iGfun   = _iGfun;
  _rec_jGvar   = _jGvar;
  _rec_Gvar    = _Gvar;
  _rec_nG      = _nG;
  _rec_iLvar   = _iLvar;
  _rec_jLvar   = _jLvar;
  _rec_Lvar    = _Lvar; 
  _rec_nL      = _nL;

  _rec_model   = true;
  return true;
}

inline
bool
NLPSLV_IPOPT::restore_model
()
{
  if( !_rec_model ) return false;

  _obj_dir = _rec_obj_dir;
  _Fvar.swap( _rec_Fvar );
  _Fmul.swap( _rec_Fmul );
  _Flow.swap( _rec_Flow );
  _Fupp.swap( _rec_Fupp );
  _nF = _rec_nF;
  for( unsigned i=0; i<_rec_obj_grad.size(); i++ )
    _obj_grad[i] = _rec_obj_grad[i];
  _iGfun.swap( _rec_iGfun );
  _jGvar.swap( _rec_jGvar );
  _Gvar.swap( _rec_Gvar );
  _nG = _rec_nG;
  _iLvar.swap( _rec_iLvar );
  _jLvar.swap( _rec_jLvar );
  _Lvar.swap( _rec_Lvar ); 
  _nL = _rec_nL;

  _rec_model = false;
  return true;
}

inline
bool
NLPSLV_IPOPT::set_obj_lazy
( BASE_OPT::t_OBJ const& type, FFVar const& obj )
{
  // Keep track of original model 
  //if( _rec_model && _Fvar[0] == obj
  // && (type == BASE_OPT::MIN? _obj_dir == -1: _obj_dir == 1) ) return false;
  _record_model();
  
  // Change to new objective
  _obj_dir = (type==BASE_OPT::MIN? -1: 1 );
  _Fvar[0] = obj;
  
  // Update objective derivatives
  _cleanup_grad( true, false ); // reset objective gradient only
  switch( options.GRADMETH ){
    case Options::FSYM: // Forward symbolic
      _obj_grad = _dag->FAD( 1,      _Fvar.data(),   _nX, _Xvar.data() ); 
      break;
    case Options::BSYM: // Backward symbolic
      _obj_grad = _dag->BAD( 1,      _Fvar.data(),   _nX, _Xvar.data() ); 
      break;
    default:
      break;
  }

  // Update Lagrangian Hessian
  _set_hessian();

  return true;
}

inline
bool
NLPSLV_IPOPT::add_ctr_lazy
( BASE_OPT::t_CTR const type, FFVar const& ctr )
{
  // Keep track of original model 
  _record_model();

  // Append new constraint
  unsigned pos = _Fvar.size();
  _Fvar.push_back( ctr );
  _Fmul.push_back( FFVar(_dag) );
  switch( type ){
    case BASE_OPT::EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );             break;
    case BASE_OPT::LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );             break;
    case BASE_OPT::GE: _Flow.push_back( 0. );             _Fupp.push_back(  BASE_OPT::INF ); break;
  }
  _nF = _Fvar.size();
#ifdef MC__NLPSLV_IPOPT_DEBUG
  _dag->output( _dag->subgraph( _nX, _Xvar.data() ) );
  _dag->output( _dag->subgraph( 1, &_Fvar.back() ) );
#endif
  
  // Append new constraint derivatives
  if( !_add_gradient( pos ) ) return false;

  // Update Lagrangian Hessian
  _set_hessian();

  return true;
}

inline
void
NLPSLV_IPOPT::_set_worker
( Ipopt::SmartPtr<WORKER_IPOPT>& th )
{
  th->Pvar.resize( _nP );
  th->Xvar.resize( _nX );
  th->Fvar.resize( _nF );
  th->Fmul.resize( _nF );
  th->Cvar.resize( _nX );
  th->Gvar.resize( _nG );
  th->Lvar.resize( _nL );
  th->dag.insert( _dag, _nP, _Pvar.data(), th->Pvar.data() );
  th->dag.insert( _dag, _nX, _Xvar.data(), th->Xvar.data() );
  th->dag.insert( _dag, _nF, _Fvar.data(), th->Fvar.data() );
  th->dag.insert( _dag, _nF, _Fmul.data(), th->Fmul.data() );
  th->dag.insert( _dag, _obj_grad?_nX:0, _obj_grad, th->Cvar.data() );
  th->dag.insert( _dag, _Gvar.size(), _Gvar.data(), th->Gvar.data() );
  th->dag.insert( _dag, _Lvar.size(), _Lvar.data(), th->Lvar.data() );
  th->op_f.clear();
  th->op_df.clear();
  th->op_g.clear();
  th->op_dg.clear();
  th->op_L.clear();
  th->Gmeth = options.GRADMETH;
  th->iGfun = _iGfun;
  th->jGvar = _jGvar;
  th->iLvar = _iLvar;
  th->jLvar = _jLvar;
  th->Xlow  = _Xlow;
  th->Xupp  = _Xupp;
  th->Flow  = _Flow;
  th->Fupp  = _Fupp;
  th->tMAX  = userclock() + options.TIMELIMIT;
}

inline
void
NLPSLV_IPOPT::_resize_workers
( int const noth )
{
  while( (int)_worker.size() < noth ){
    _worker.push_back( new WORKER_IPOPT );
  }
}

template <typename T>
inline
int
NLPSLV_IPOPT::solve
( double const* Xini, T const* Xbnd, double const* Pval )
{
  std::vector<double> Xlow(_nX), Xupp(_nX);
  for( int i=0; i<_nX; i++ ){
    Xlow[i] = Op<T>::l( Xbnd[i] );
    Xupp[i] = Op<T>::u( Xbnd[i] );
  }
  return solve( Xini, Xlow.data(), Xupp.data(), Pval );
}

inline
int
NLPSLV_IPOPT::solve
( double const* Xini, double const* Xlow, double const* Xupp, double const* Pval )
{
  // Set worker
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _set_worker( _worker[th] );
  _worker[th]->update( _nX, Xlow, Xupp, _nP, Pval );
  
  // Run NLP solver
  Ipopt::SmartPtr<Ipopt::IpoptApplication> IpoptApp = new Ipopt::IpoptApplication();
  _set_options( IpoptApp );
  _worker[th]->initialize( _nX, !Xini && !_Xini.empty()? _Xini.data(): Xini );

  int stat = IpoptApp->Initialize();
  if( stat != Ipopt::Solve_Succeeded ){
    _solution.reset( stat );
    return stat;
  }

  stat = IpoptApp->OptimizeTNLP( _worker[th] );
  _solution = _worker[th]->solution;

  return stat;
}

#ifdef MC__USE_SOBOL
template <typename T>
inline
int
NLPSLV_IPOPT::solve
( unsigned const NSAM, T const* Xbnd, double const* Pval, bool const* logscal,
  bool const DISP )
{
  std::vector<double> Xlow(_nX), Xupp(_nX);
  for( int i=0; i<_nX; i++ ){
    Xlow[i] = Op<T>::l( Xbnd[i] );
    Xupp[i] = Op<T>::u( Xbnd[i] );
  }
  return solve( NSAM, Xlow.data(), Xupp.data(), Pval, logscal, DISP );
}

inline
int
NLPSLV_IPOPT::solve
( unsigned const NSAM, double const* Xlow, double const* Xupp, double const* Pval,
  bool const* logscal, bool const DISP )
{
  // Set all workers
  const unsigned NOTHREADS = ( options.MAXTHREAD>0? options.MAXTHREAD: std::thread::hardware_concurrency() );
  _resize_workers( NOTHREADS );
  for( unsigned th=0; th<NOTHREADS; th++ ){
    _set_worker( _worker[th] );
    _worker[th]->update( _nX, Xlow, Xupp, _nP, Pval );
  }
  std::vector<std::thread> vth( NOTHREADS-1 ); // Main threads also solves some NLPs

  // Initialize multistart
  if( DISP ) std::cout << "\nMULTISTART: ";
  std::vector<int> feasible( NOTHREADS, false );
  std::vector<SOLUTION_OPT> solution( NOTHREADS );

  // Run NLP solver on auxiliary threads
  for( unsigned th=1; th<NOTHREADS; th++ )
    vth[th-1] = std::thread( &NLPSLV_IPOPT::_mssolve, this, th, NOTHREADS, NSAM, logscal, DISP,
                             std::ref(feasible[th]), std::ref(solution[th]) );

  // Run NLP solver on main thread
  _mssolve( 0, NOTHREADS, NSAM, logscal, DISP, feasible[0], solution[0] ); 

  bool found = false;
  if( feasible[0] ){
    _solution = solution[0];
    if( !_obj_dir )
      return _solution.stat;
    found = true;
  }

  // Join all the threads to the main one
  for( unsigned th=1; th<NOTHREADS; th++ ){
    vth[th-1].join();
    if( feasible[th] ){
      if( !found ){
        _solution = solution[th];
        found = true;
        continue;
      }
      if( !_obj_dir ){
        _solution = solution[th];
        return _solution.stat;
      }
      else if( _obj_dir == -1 && solution[th].f[0] < _solution.f[0] ){
        _solution = solution[th];
      }
      else if( _obj_dir == 1  && solution[th].f[0] > _solution.f[0] ){
        _solution = solution[th];
      }
    }
  }

  if( DISP ) std::cout << std::endl;

  return _solution.stat;
}

inline
void
NLPSLV_IPOPT::_mssolve
( int const th, unsigned const NOTHREADS, unsigned const NSAM,
  bool const* logscal, bool const DISP, int& feasible, SOLUTION_OPT& solution )
{
  // Initialize multistart and seed at position th
  std::vector<double> vSAM(_nX), Xini(_nX);
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( _nX );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( th );
  
  // Run multistart
#ifdef MC__NLPSLV_IPOPT_DEBUG
  std::cout << "Thread #" << th << std::endl;
#endif
  Ipopt::SmartPtr<Ipopt::IpoptApplication> IpoptApp = new Ipopt::IpoptApplication();
  _set_options( IpoptApp );
  int stat = IpoptApp->Initialize();
  if( stat != Ipopt::Solve_Succeeded ){
    solution.reset( stat );
    return;
  }
  
  for( unsigned k=th; k<NSAM && userclock()<_worker[th]->tMAX; k+=NOTHREADS ){
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << "**** k = " << k << std::endl;
#endif

    // Sample variable domain
    for( int i=0; i<_nX; i++ ){
      vSAM[i] = gen();
      if( !logscal || !logscal[i] || _worker[th]->Xlow[i] <= 0. )
        Xini[i] = _worker[th]->Xlow[i] + ( _worker[th]->Xupp[i] - _worker[th]->Xlow[i] ) * vSAM[i];
      else
        Xini[i] = std::exp( std::log(_worker[th]->Xlow[i]) + ( std::log(_worker[th]->Xupp[i]) - std::log(_worker[th]->Xlow[i]) ) * vSAM[i] );
        //Xini[i] = std::exp( Op<T>::l(Op<T>::log(Xbnd[i])) + Op<T>::diam(Op<T>::log(Xbnd[i])) * vSAM[i] );
#ifdef MC__NLPSLV_IPOPT_DEBUG
      std::cout << "  " << Xini[i];
#endif
    }
#ifdef MC__NLPSLV_IPOPT_DEBUG
    std::cout << std::endl;
#endif

    // Set initial point and run NLP solver
    _worker[th]->initialize( _nX, Xini.data() );
    stat = IpoptApp->OptimizeTNLP( _worker[th] );

    // Test for feasibility and improvement
    if( !_worker[th]->feasible( options.FEASTOL, _nP, _nX, _nF ) ){
      if( DISP ) std::cout << "·";
    }
    // Solution point is feasible
    else{
      if( DISP ) std::cout << "*";
      if( !_obj_dir ){
        solution = _worker[th]->solution;
        return;
      }
      else if( !feasible ){
        solution = _worker[th]->solution;
        feasible = true;
      }
      else if( _obj_dir == -1 && _worker[th]->solution.f[0] < solution.f[0] ){
        solution = _worker[th]->solution;
      }
      else if( _obj_dir == 1  && _worker[th]->solution.f[0] > solution.f[0] ){
        solution = _worker[th]->solution;
      }
    }

    // Advance quasi-random counter
    gen.engine().discard( (NOTHREADS-1) * _nX );
  }
}
#endif

inline
bool
NLPSLV_IPOPT::is_feasible
( const double*x, const double CTRTOL )
{
  if( !x ) return false;

  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _set_worker( _worker[th] );
  _worker[th]->solution.x.assign( x, x+_nX );
  bool feas = _worker[th]->feasible( CTRTOL, _nP, _nX, _nF );
  _solution = _worker[th]->solution;
  return feas;
}

inline
bool
NLPSLV_IPOPT::is_feasible
( const double CTRTOL )
{
  if( _solution.x.empty() ) return false;
  
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _worker[th]->solution = _solution;
  bool feas = _worker[th]->feasible( CTRTOL, _nP, _nX, _nF );
  _solution = _worker[th]->solution;
  return feas;
}

inline
bool
NLPSLV_IPOPT::is_stationary
( const double*x, const double*ux, const double*uf, const double GRADTOL )
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _worker[th]->solution.x.assign( x, x+_nX );
  _worker[th]->solution.ux.assign( ux, ux+_nX );
  _worker[th]->solution.uf.assign( uf, uf+_nX );
  return _worker[th]->stationary( GRADTOL, _nP, _nX, _nF, _iGfun.size() );
}

inline
bool
NLPSLV_IPOPT::is_stationary
( const double GRADTOL )
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _worker[th]->solution = _solution;
  return _worker[th]->stationary( GRADTOL, _nP, _nX, _nF, _iGfun.size() );
}

inline
double
NLPSLV_IPOPT::cost_correction
( const double*x, const double*ux, const double*uf )
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _set_worker( _worker[th] );
  _worker[th]->solution.x.assign( x, x+_nX );
  _worker[th]->solution.ux.assign( ux, ux+_nX );
  _worker[th]->solution.uf.assign( uf, uf+_nF );
  return _worker[th]->correction( _nP, _nX, _nF );
}

inline
double
NLPSLV_IPOPT::cost_correction
()
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _worker[th]->solution = _solution;
  return _worker[th]->correction( _nP, _nX, _nF );
}

} // end namescape mc

#endif

// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MINLPSLV Local Mixed-Integer Nonlinear Optimization via Outer-Approximation using MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 2.0
\date 2023
\bug No known bugs.

Consider a mixed-integer nonlinear optimization problem (MINLP) in the form:
\f{align*}
\mathcal{P}:\quad & \min_{x_1,\ldots,x_n}\ f(x_1,\ldots,x_n)\\
& {\rm s.t.}\ \ g_j(x_1,\ldots,x_n)\ \leq,=,\geq\ 0,\ \ j=1,\ldots,m\\
& \qquad x_i^L\leq x_i\leq x_i^U,\ \ i=1,\ldots,n\\
& \qquad x_i \in \mathbb{Z},\ \ i\in I
\f}
where \f$f, g_1, \ldots, g_m\f$ are factorable, potentially nonlinear, real-valued functions; and \f$x_i, i=1\ldots n\f$ are either continuous (\f$i\notin I\f$) or binary/integer (\f$i\in I\f$) decision variables. The class mc::MINLPSLV tackles such problems using an outer-approximation algorithm (<A href="http://doi.org/10.1007/BF02592064">Duran & Grossmann, 1986</A>; <A href="http://doi.org/10.1007/BF01581153"> Fletcher & Leyffer, 1994</A>; <A href="http://doi.org/10.1007/BF01581153"> Bonami et al., 2009</A>), which alternates between solving nonlinear programs (NLPs) and mixed-integer linear programs (MILPs). The implementation follows the baseline of DICOPT (<A href="https://doi.org/10.1080/10556788.2019.1641498"> Bernal et al., 2020</A>).

\section sec_MINLPSLV_solve How to Solve an MINLP Model using mc::MINLPSLV?

Consider the following MINLP model:
\f{align*}
  \min_{x,y}\ & -6x-y \\
  \text{s.t.} \ & 0.3(x-8)^2+0.04(y-6)^4+0.1\frac{{\rm e}^{2x}}{y^4} \leq 56 \\
                & \frac{1}{x}+\frac{1}{y}-\sqrt{x}\sqrt{y}+4 \leq 0 \\
                & 2x-5y+1 \leq 0 \\
  & 1 \leq x \leq 20\\
  & 1 \leq y \leq 20,\ y\in\mathbb{Z}
\f}

Start by instantiating an mc::MINLPSLV class object, which is defined in the header file <tt>minlpslv.hpp</tt>:

\code
  mc::MINLPSLV MINLP;
\endcode

Next, set the variables and objective/constraint functions after creating a DAG of the problem: 

\code
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );
  MINLP.add_var( P[0], 1, 20, 0 );
  MINLP.add_var( P[1], 1, 20, 1 );
  MINLP.set_obj( mc::BASE_NLP::MIN, -6*P[0]-P[1] );
  MINLP.add_ctr( mc::BASE_NLP::LE, 0.3*pow(P[0]-8,2)+0.04*pow(P[1]-6,4)+0.1*exp(2*P[0])/pow(P[1],4)-56 );
  MINLP.add_ctr( mc::BASE_NLP::LE, 1/P[0]+1/P[1]-sqrt(P[0])*sqrt(P[1])+4 );
  MINLP.add_ctr( mc::BASE_NLP::LE, 2*P[0]-5*P[1]+1 );
\endcode

Possibly set options using the class member MINLPSLV::options:

\code
  MINLP.options.LINMETH   = mc::MINLPSLV<>::Options::CVX;
  MINLP.options.CVRTOL    = 1e-5;
  MINLP.options.CVATOL    = 1e-5;
  MINLP.options.DISPLEVEL = 1;
\endcode

Finally, set up the MINLP model and solve it using:

\code
  MINLP.setup();
  MINLP.optimize();
\endcode

The following display is obtained:

\verbatim
#  ITERATION     INCUMBENT    BEST BOUND    TIME
        0  r  1.000000e+30 -5.698551e+01      0s
        1  f  1.000000e+30 -5.698551e+01      0s
        2  * -5.698117e+01 -5.698551e+01      0s
        3    -5.698117e+01 -5.698117e+01      0s

#  TERMINATION AFTER 3 ITERATIONS: 0.029739 SEC
#  INCUMBENT VALUE: -5.698117e+01
#  INCUMBENT POINT:  7.663529e+00  1.100000e+01
\endverbatim

The return value of mc::MINLPSLV::optimize is per the enumeration mc::MINLPSLV::STATUS. The incumbent solution may be retrieved as an instance of <a>mc::SOLUTION_OPT</a> using the method <a>mc::MINLPSLV::incumbent</a>. A computational breakdown may be obtained from the internal class <a>mc::MINLPSLV::Stats</a>.
*/

//TO DO:
// [done] Allow pure integer problems - change NLP solve to simple feasibility test
// [done] Allow tailored rounding procedure for integer variables at root node
// [done] Improve setup of feasibility lazy objective in NLP subproblem - define constants in DAG
// - Add option to disable feasibility pump after root note
// - Add option for second-order information and MIQP master solve 
// - Test class with DAG externals, e.g. log det objective in MBDOE

#ifndef MC__MINLPSLV_HPP
#define MC__MINLPSLV_HPP

#include <chrono>

#include "interval.hpp"
#include "gamsio.hpp"
#if defined( MC__USE_SNOPT )
  #include "nlpslv_snopt.hpp"
#elif defined( MC__USE_IPOPT )
  #include "nlpslv_ipopt.hpp"
#endif
#include "mipslv_gurobi.hpp"
#include "sbbslv.hpp"

namespace mc
{

//! @brief C++ class for local optimization of MINLP using outer-approximation
////////////////////////////////////////////////////////////////////////
//! mc::MINLPSLV is a C++ class for local optimization of MINLP using
//! outer-approximation. Linearizations of the nonlinear objective or
//! constraints are generated using MC++. Further details can be found
//! at: \ref page_MINLPSLV
////////////////////////////////////////////////////////////////////////
template <typename T=Interval,
#if defined( MC__USE_SNOPT )
          typename NLP=NLPSLV_SNOPT,
#elif defined( MC__USE_IPOPT )
          typename NLP=NLPSLV_IPOPT,
#endif
          typename MIP=MIPSLV_GUROBI<T>>
class MINLPSLV
#if defined( MC__WITH_GAMS )
: protected virtual GAMSIO,
  protected SBBSLV<T>,
  public virtual BASE_NLP
#else
: protected SBBSLV<T>,
  public virtual BASE_NLP
#endif
{
public:

  using BASE_NLP::dag;
  using BASE_NLP::set_dag;
  using BASE_NLP::reset;
  
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
  using BASE_NLP::reset_ctr;

  typedef void (*ROUND)( unsigned const, unsigned const*, double* );

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

#if defined (MC__WITH_GAMS)
  using GAMSIO::_varini;
#endif

public:

  //! @brief NLP solution status
  enum STATUS{
     SUCCESSFUL=0,      //!< MINLP solution found (possibly suboptimal for nonconvex MINLP)
     INFEASIBLE,        //!< MINLP appears to be infeasible (nonconvex MINLP could still be feasible)
     UNBOUNDED,         //!< MIP subproblem returns an unbounded solution
     INTERRUPTED,       //!< MINLP algorithm was interrupted prior to convergence
     FAILURE,           //!< MINLP algorithm encountered numerical difficulties
     ABORTED            //!< MINLP algorithm aborted after critical error
  };

  //! @brief MINLPSLV options
  struct Options
  {
    //! @brief Constructor
    Options():
      SEARCHALG(OA),
      LINMETH(PENAL), FEASPUMP(true), CORRINC(true), ROOTCUT(true),
//      INCCUT(true),
      FEASTOL(1e-5), CVATOL(1e-3), CVRTOL(1e-3), MAXITER(20),
      CPMAX(10), CPTHRES(0.), 
      PENSOFT(1e3), MSLOC(8), TIMELIMIT(6e2), DISPLEVEL(1),
      NLPSLV(), POLIMG(), MIPSLV()
      { NLPSLV.DISPLEVEL = MIPSLV.DISPLEVEL = 0;
        NLPSLV.TIMELIMIT = MIPSLV.TIMELIMIT = TIMELIMIT;
        NLPSLV.GRADMETH  = NLP::Options::FSYM; }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        SEARCHALG     = options.SEARCHALG;
        LINMETH       = options.LINMETH;
        FEASPUMP      = options.FEASPUMP;
        CORRINC       = options.CORRINC;
//        INCCUT        = options.INCCUT;
        ROOTCUT       = options.ROOTCUT;
        FEASTOL       = options.FEASTOL;
        CVATOL        = options.CVATOL;
        CVRTOL        = options.CVRTOL;
        MAXITER       = options.MAXITER;
        CPMAX         = options.CPMAX;
        CPTHRES       = options.CPTHRES;
        PENSOFT       = options.PENSOFT;
        MSLOC         = options.MSLOC;
        TIMELIMIT     = options.TIMELIMIT;
        DISPLEVEL     = options.DISPLEVEL;
        NLPSLV        = options.NLPSLV;
        POLIMG        = options.POLIMG;
        MIPSLV        = options.MIPSLV;
        return *this ;
      }

    //! @brief Global search strategy
    enum ALGORITHM{
      OA=0,	//!< Outer-approximation (OA) algorithm
      BB	//!< Branch-and-bound (BB) algorithm
    };

    //! @brief Linearization method in OA algorithm
    enum LINEARIZATION{
      CVX=0,   //!< Direct linearization of cost and constraints at NLP solution point (assumes convexity)
      PENAL    //!< Softening and relaxation of constraints in MIP subproblem (does not assume convexity)
    };

    //! @brief Search algorithm
    int SEARCHALG;
    //! @brief Linearization method
    int LINMETH;
    //! @brief Whether or not to apply feasibility pump strategy in OA algorithm
    bool FEASPUMP;
    //! @brief Correct the incumbent for feasibility using multipliers
    bool CORRINC;
//    //! @brief Whether or not to add incumbent cut in master and feasibility OA subproblems
//    bool INCCUT;
    //! @brief Whether or not to add cut from root-node relaxation in master problem
    bool ROOTCUT;
    //! @brief Feasibility tolerance 
    double FEASTOL;
    //! @brief Convergence absolute tolerance
    double CVATOL;
    //! @brief Convergence relative tolerance
    double CVRTOL;
    //! @brief Maximum number of outer-approximation iterations (0-no limit)
    unsigned MAXITER;
    //! @brief Maximum rounds of constraint propagation
    unsigned CPMAX;
    //! @brief Threshold for repeating constraint propagation (minimum relative reduction in any variable)
    double CPTHRES;
    //! @brief Weight multiplying constraint marginal in soft constraints
    double PENSOFT;
    //! @brief Number of multistart local search
    unsigned MSLOC;
    //! @brief Maximum run time (seconds)
    double TIMELIMIT;
    //! @brief Display level for solver
    int DISPLEVEL;
    //! @brief NLP (nonlinear optimization) local solver options
    typename NLP::Options NLPSLV;
    //! @brief Polyhedral relaxation (PolImg) options
    typename PolImg<T>::Options POLIMG;
    //! @brief MIP (mixed-integer optimization) master solver options
    typename MIP::Options MIPSLV;
    //! @brief Display
    void display
      ( std::ostream&out=std::cout ) const;
  } options;

  //! @brief MINLPSLV exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for MINLPSLV exception handling
    enum TYPE{
      SETUP,		//!< Incomplete setup before a solve
      PARAM,	        //!< Undefined parameter values
      SEARCHALG,	//!< Invalid search strategy
      INTERN=-33	//!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr(){ return _ierr; }
    //! @brief Inline function returning the error description
    std::string what(){
      switch( _ierr ){
      case SETUP:
        return "MINLPSLV::Exceptions  Incomplete setup before a solve";
      case PARAM:
        return "MINLPSLV::Exceptions  Undefined parameter values";
      case SEARCHALG:
        return "MINLPSLV::Exceptions  Invalid search algorithm";
      case INTERN:
      default:
        return "MINLPSLV::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief MINLPSLV statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime_all = walltime_setup = walltime_slvnlp = walltime_slvmip =
        std::chrono::microseconds(0); }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "#  WALL-CLOCK TIMES" << std::endl
           << "#  SETUP:     " << std::setw(10) << to_time( walltime_setup )   << " SEC" << std::endl
           << "#  NLP SOLVE: " << std::setw(10) << to_time( walltime_slvnlp )  << " SEC" << std::endl
           << "#  MIP SOLVE: " << std::setw(10) << to_time( walltime_slvmip )  << " SEC" << std::endl
           << "#  TOTAL:     " << std::setw(10) << to_time( walltime_all )     << " SEC" << std::endl
           << std::endl; }
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime_all;
    //! @brief Cumulated wall-clock time used for problem setup (in microseconds)
    std::chrono::microseconds walltime_setup;
    //! @brief Cumulated wall-clock time used by local NLP solver (in microseconds)
    std::chrono::microseconds walltime_slvnlp;
    //! @brief Cumulated wall-clock time used by MIP solver (in microseconds)
    std::chrono::microseconds walltime_slvmip;
    //! @brief Get current time point
    std::chrono::time_point<std::chrono::system_clock> start
      () const
      { return std::chrono::system_clock::now(); }
    //! @brief Get current time lapse with respect to start time point
    std::chrono::microseconds walltime
      ( std::chrono::time_point<std::chrono::system_clock> const& start ) const
      { return std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start ); }    
    //! @brief Convert microsecond ticks to time
    double to_time
      ( std::chrono::microseconds t ) const
      { return t.count() * 1e-6; }
  } stats;


protected:

  //! @brief Current iteration
  STATUS                    _status;

  ROUND                     _roundf;

  //! @brief Current iteration
  unsigned                  _iter;

  //! @brief Flag for setup function
  bool                      _issetup;

  //! @brief Flag for MIP problem
  bool                      _ismip;

  //! @brief Flag for boundedness
  bool                      _isbnd;

  //! @brief objective scaling coefficient (1: min; -1: max)
  double                    _objscal;

  //! @brief Current relaxation value
  double                    _Zrel;

  //! @brief Current incumbent value
  double                    _Zinc;

  //! @brief Current incumbent correction
  double                    _Zcor;

  //! @brief Variable values at current relaxation
  std::vector<double>       _Xrel;
  
  //! @brief subset of continuous participating variables
  std::set<unsigned>        _Xcnt;
  
  //! @brief subset of integer participating variables
  std::set<unsigned>        _Xint;
  
  //! @brief Local solver for NLP subproblem
  NLP                       _NLPSLV;

  //! @brief Global solver for MIP subproblem
  MIP                       _MIPSLV;

  //! @brief size of parameters in MINLP model
  size_t                    _nP;
  
  //! @brief Parameters in MINLP model
  std::vector<FFVar>        _Pvar;

  //! @brief Parameter dependencies
  std::vector<FFDep>        _Pdep;

  //! @brief Parameter values
  std::vector<double>       _Pval;
  
  //!@brief vector of parameter values in fadbad::F<double> arithmetic
  std::vector<fadbad::F<double>> _FPval;

  //!@brief vector of parameter values in fadbad::B<double> arithmetic
  std::vector<fadbad::B<double>> _BPval;

  //! @brief Number of decision variables (independent and dependent) in MINLP model
  size_t                    _nX;

  //! @brief Reference variables in MINLP feasibility pump
  std::vector<FFVar>        _Xref;

  //! @brief Decision variables in MINLP model
  std::vector<FFVar>        _Xvar;

  //! @brief Decision variable dependencies
  std::vector<FFDep>        _Xdep;

  //! @brief Decision variable lower bounds
  std::vector<double>       _Xlow;

  //! @brief Decision variable upper bounds
  std::vector<double>       _Xupp;

  //! @brief Decision variable types
  std::vector<unsigned>     _Xtyp;

  //! @brief Decision variable bounds
  std::vector<T>            _Xbnd;

  //! @brief Decision variable bounds with integer fixing
  std::vector<T>            _Xbndi;

  //! @brief vector of zeros for constant term calculation in functions
  std::vector<FFVar>        _X0;

  //! @brief vector of forward derivatives for linear function gradients
  std::vector<fadbad::F<double>> _FXval;

  //! @brief vector of backward derivatives for linear function gradients
  std::vector<fadbad::B<double>> _BXval;

  //! @brief vector of decision variable levels
  std::vector<double>       _Xini;

  //! @brief number of functions (objective and constraints) in MINLP model
  size_t                    _nF;

  //! @brief Cost function in MINLP feasibility pump
  FFVar                     _Ffeas;

  //! @brief Functions in MINLP model
  std::vector<FFVar>        _Fvar;

  //! @brief Functions dependencies
  std::vector<FFDep>        _Fdep;

  //! @brief Function bounds
  std::vector<T>            _Fbnd;

  //! @brief vector of function offsets
  std::vector<FFVar>        _Foff;

  //! @brief Functions in MINLP model
  std::vector<unsigned>     _Ftyp;

  //! @brief index set of linear functions
  std::set<unsigned>        _Flin;

  //! @brief index set of nonlinear functions
  std::set<unsigned>        _Fnlin;

  //! @brief list of operations in nonlinear functions
  FFSubgraph                _Fop;
  
  //! @brief Function values
  std::vector<double>       _Fval;

  //! @brief vector of forward derivatives for linear function gradients
  std::vector<fadbad::F<double>> _FFval;

  //! @brief vector of backward derivatives for linear function gradients
  std::vector<fadbad::B<double>> _BFval;

  //!@brief number of nonzero elements in the linear part of each function
  size_t                    _nA;

  //!@brief row coordinates of nonzero elements in the linear part of each function
  std::vector<int>          _iAfun;

  //!@brief column coordinates of nonzero elements in the linear part of each function
  std::vector<int>          _jAvar;

  //!@brief Expressions of nonzero elements in the linear part of each function
  std::vector<FFVar>        _Avar;

  //!@brief values of nonzero elements in the linear part of each function
  std::vector<double>       _Aval;

  //!@brief number of nonzero elements in the derivative of the nonlinear part of each function
  size_t                    _nG;

  //!@brief row coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>          _iGfun;

  //!@brief column coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>          _jGvar;

  //! @brief derivatives of the nonlinear part of each function
  std::vector<FFVar>        _Gvar;

  //! @brief derivative values of the nonlinear part of each function
  std::vector<double>       _Gval;

  //! @brief list of operations in function derivatives
  FFSubgraph                _Gop;

  //! @brief Storage vector for DAG evaluation in double arithmetic
  std::vector<double>       _dwk;

  //! @brief Storage vector for DAG evaluation in fadbad::F<double> arithmetic
  std::vector<fadbad::F<double>> _Fwk;

  //! @brief Storage vector for DAG evaluation in fadbad::B<double> arithmetic
  std::vector<fadbad::B<double>> _Bwk;

  //! @brief Polyhedral image environment
  PolImg<T>                 _POLenv;

  //! @brief Polyhedral image decision variables
  std::vector< PolVar<T> >  _POLXvar;

  //! @brief Polyhedral image slack variables
  std::vector< PolVar<T> >  _POLSvar;

  //! @brief Cost coefficients for slack variables
  std::vector< double >     _POLScost;

  //! @brief Polyhedral image cut storage
  std::vector< PolCut<T>* > _POLcuts;

  //! @brief Structure holding NLP root relaxed solution
  SOLUTION_OPT              _rootrel;

  //! @brief Structure holding NLP intermediate solution
  SOLUTION_OPT              _solution;

  //! @brief Structure holding MINLP incumbent information
  SOLUTION_OPT              _incumbent;

  //! @brief Incumbent cut storage
  PolCut<T>*                _POLcutinc;

  //! @brief Root-node relaxation cut storage
  PolCut<T>*                _POLcutroot;

  //! @brief function sparse derivatives
  std::tuple< unsigned, unsigned const*, unsigned const*, FFVar const* > _Fgrad;

  //! @brief Set SBBSLV solver options
  void _set_options_sbbslv
    ();

  //! @brief Solve optimization model using outer-approximation algorithm
  int _optimize_oa
    ( double const* Xini,  T const* Xbnd, std::ostream& os );

  //! @brief Solve optimization model using branch-and-bound algorithm
  int _optimize_bb
    ( double const* Xini,  T const* Xbnd, std::ostream& os );

  //! @brief Apply constraint propagation
  int _propagate_bounds
    ( T* Xbnd );

  //! @brief Set function gradient storage
  void _set_gradient
    ();

  //! @brief Cleanup gradient storage
  void _cleanup_gradient
    ();

  //! @brief Add outer-approximation cuts to master MIP subproblem
  bool _add_outerapproximation_cuts
    ( std::vector<double> const& Xval, std::vector<double>& Fval,
      std::vector<double>& Fmul );
    
  //! @brief Add integer cut to master MIP subproblem
 bool _add_integer_cut
   ( std::vector<double> const& Xint );

  //! @brief Set integer cut in master MIP subproblem
  bool _set_integer_cut
    ( std::vector<double> const& Xint, std::vector<PolVar<T>>& linvar,
      std::vector<double>& linwei, double& cst );
      
  //! @brief Set anti-cycling cut in master MIP subproblem
  bool _add_anticycling_cut
    ( std::vector<double> const& Xloc, std::vector<double> const& Xrel );

  //! @brief Test whether a variable vector is integer feasible
  bool _is_integer_feasible
    ( double const* Xval, double const& feastol )
    const;

  //! @brief Test whether a variable vector is integer identical as a reference vector
  bool _is_integer_equal
    ( double const* Xval, double const* Xref )
    const;

  //! @brief Test feasibility
  bool _test_feasible
    ( double const* Xval, std::ostream& os );

  //! @brief Solve local NLP subproblem
  bool _solve_local
    ( std::chrono::time_point<std::chrono::system_clock> const& tstart,
      double const* Xini, T const* Xbnd, bool const pumpfeas,
      bool const inccut, std::ostream& os=std::cout );

  //! @brief Initialize master MIP subproblem
  void _init_master
    ();

  //! @brief Update master MIP subproblem with local NLP cuts
  bool _update_master
    ( bool const locfeas, bool const pumpfeas, bool const inccut, bool const intrel );

  //! @brief Solve master MIP subproblem
  int _solve_master
    ( std::chrono::time_point<std::chrono::system_clock> const& tstart );

  //! @brief Termination test for MINLP optimization
  bool _interrupted
    ( std::chrono::time_point<std::chrono::system_clock> const& tstart )
    const;

  //! @brief Convergence test for MINLP optimization
  bool _converged
    ()
    const;

  //! @brief Finalize optimization display and status
  int _finalize
    ( std::chrono::time_point<std::chrono::system_clock> const& tstart,
      STATUS const status, std::ostream& os=std::cout );

  //! @brief User-function to subproblems in SBB
  typename SBBSLV<T>::STATUS subproblems
    ( typename SBBSLV<T>::TASK const task, SBBNode<T>* node,
      std::vector<double>& p, double& f, double const& INC, std::ostream& os );

  //! @brief Select integer branching variable with largest range and closest to integrality
  static std::set<unsigned> _branch_subset
    ( SBBNode<T> const* node );

public:

  //! @brief Constructor
  MINLPSLV()
    : _status(FAILURE), _issetup(false), _nP(0), _nX(0), _nF(0), _nA(0), _nG(0),
      _rootrel(FAILURE), _solution(FAILURE), _incumbent(FAILURE)
    { stats.reset(); }

  //! @brief Destructor
  virtual ~MINLPSLV()
    { _cleanup_gradient(); }

  //! @brief Status after last NLP call
  STATUS get_status
    ()
    const
    { return _status; }

  //! @brief Load optimization model from GAMS file
#if defined (MC__WITH_GAMS)
  bool read
    ( std::string const& filename, bool const init=false );
#endif

  //! @brief Setup DAG for cost and constraint evaluation
  void setup
    ( std::ostream& os=std::cout );

  //! @brief Solve MINLP model to local optimality using outer-approximation
  int optimize
    ( double const* Xini=nullptr, T const* Xbnd=nullptr, double const* Pval=nullptr,
      ROUND const& f=nearest, std::ostream& os=std::cout );

  //! @brief Get incumbent info
  SOLUTION_OPT const& get_incumbent
    () 
    const
    { return _incumbent; }

  //! @brief Get reference to local NLP solver
  NLP& local_solver
    ()
    { return _NLPSLV; }
    
  //! @brief Get reference to master MIP solver
  MIP& master_solver
    ()
    { return _MIPSLV; }

  //! @brief Test domain boundedness
  bool is_bounded
    ( double const& maxdiam )
    { _isbnd = true;
      for( size_t i=0; _isbnd && i<_nX; i++ ){
        if( Op<T>::diam(_Xbnd[i]) < BASE_OPT::INF/10 ) continue;
        _isbnd = false;
      }
      return _isbnd; }

  //! @brief Test primal feasibility
  bool is_feasible
    ( double const* x, double const CTRTOL )
    { return x && _is_integer_feasible( x, CTRTOL ) && _NLPSLV.is_feasible( x, CTRTOL ) ?
             true : false; }

  //! @brief Test primal feasibility
  bool is_feasible
    ( double const CTRTOL )
    { return is_feasible( _incumbent.x.data(), CTRTOL ); }

  //! @brief Compute cost correction
  double cost_correction
    ()
    { return _NLPSLV.cost_correction( _incumbent.x.data(), _incumbent.ux.data(), _incumbent.uf.data() ); }

  //! @brief Compute cost correction
  double cost_correction
    ( double const* x, double const* ux, double const* uf )
    { return x && ux && uf ? _NLPSLV.cost_correction( x, ux, uf ) : 0.; }

  //! @brief Round relaxed integer variables to nearest integer
  static void nearest
    ( unsigned const n, unsigned const* typ, double* val )
    { for( unsigned i=0; i<n; ++i ) val[i] = std::round( val[i] ); }

private:

  //! @brief Private methods to block default compiler methods
  MINLPSLV( MINLPSLV<T,NLP,MIP> const& ) = delete;
  MINLPSLV<T,NLP,MIP>& operator=( MINLPSLV<T,NLP,MIP> const& ) = delete;

  //! @brief Interval representation of 'unbounded' variables
  static T _IINF;

  //! @brief Working array for bound propagation
  std::vector<T> _CPbnd;

  //! @brief storage for constant term in a cut
  double _auxcst;

  //! @brief storage for linear variables in a cut
  std::vector<PolVar<T>> _POLauxvar;

  //! @brief storage for linear weights in a cut
  std::vector<double> _POLauxwei;

  //! @brief maximum number of values displayed in a row
  static const unsigned int _LDISP = 4;

  //! @brief reserved space for integer variable display
  static const unsigned int _IPREC = 9;

  //! @brief reserved space for double variable display
  static const unsigned int _DPREC = 6;

  //! @brief reserved space for percentage display
  static const unsigned int _PPREC = 6;

  //! @brief stringstream for displaying results
  std::ostringstream _odisp;
  
  //! @brief Time point to enable TIMELIMIT option
  std::chrono::time_point<std::chrono::system_clock> _tstart;

  //! @brief Display setup info
  void _display_setup
    ( std::ostream& os=std::cout );

  //! @brief Initialize display
  void _display_init
    ( std::ostream& os=std::cout );
    
  //! @brief Final display
  void _display_final
    ( unsigned const iter, std::chrono::microseconds const& walltime,
      std::ostream& os=std::cout );

  //! @brief Add double to display
  void _display_add
    ( const double dval );

  //! @brief Add unsigned int to display
  void _display_add
    ( const unsigned ival );

  //! @brief Add string to display
  void _display_add
    ( const std::string &sval );

  //! @brief Add time to display
  void _display_add
    ( std::chrono::time_point<std::chrono::system_clock> const& tstart );

  //! @brief Display current buffer stream and reset it
  void _display_flush
    ( std::ostream& os=std::cout );

};

template <typename T, typename NLP, typename MIP>
inline T MINLPSLV<T,NLP,MIP>::_IINF = BASE_OPT::INF * T(-1,1);

#if defined (MC__WITH_GAMS)
template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::read
( std::string const& filename, bool const init )
{
  _tstart = stats.start();

  bool flag = this->GAMSIO::read( filename, init, options.DISPLEVEL>1? true: false );

  stats.walltime_setup += stats.walltime( _tstart );
  stats.walltime_all   += stats.walltime( _tstart );
  return flag;  
}
#endif

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::setup
( std::ostream& os )
{
  //stats.reset();
  _tstart = stats.start();

  assert( !std::get<0>(_obj).empty() );
  switch( std::get<0>(_obj)[0] ){
    case BASE_OPT::MIN: _objscal =  1e0; break;
    case BASE_OPT::MAX: _objscal = -1e0; break;
  }

#ifdef MC__MINLPSLV_DEBUG
  std::cout << "Setup NLP subproblem" << std::endl;
#endif
  _NLPSLV.options = options.NLPSLV;
  _NLPSLV.set( *this );
  _NLPSLV.setup();

#ifdef MC__NLGO_PREPROCESS_DEBUG
  std::cout << "Setup MIP subproblem" << std::endl;
#endif
  _MIPSLV.options = options.MIPSLV;

  // full set of parameters
  _Pvar = _par;
  _nP = _Pvar.size();

  // decision variables
  _Xvar = _var;
  _Xlow = _varlb;
  _Xupp = _varub;
  _Xtyp = _vartyp;
  _Xint.clear();
  _Xcnt.clear();
  for( size_t i=0; i<_var.size(); i++ )
    if( _Xtyp[i] ) _Xint.insert( i );
    else           _Xcnt.insert( i );
  _ismip = !_Xint.empty();
  _Xrel.clear();
  _Xbnd.clear();
  _nX = _Xvar.size();

  // full set of variable initial values from GAMS
#if defined (MC__WITH_GAMS)
  _Xini = _varini;
#else
  _Xini.clear();
#endif

  // set dependencies
  _Pdep.assign( _nP, 0. );
  _Xdep.resize( _nX );
  for( size_t i=0; i<_nX; ++i )
    _Xdep[i].indep( _Xvar[i].id().second );

  // full set of functions
  _Fvar.clear();
  _Fdep.clear();
  _Ftyp.clear();
  _Fbnd.clear();
  FFDep depF = 0.;

  // Cost
  if( std::get<0>(_obj).size() ){
    _Ftyp.push_back( std::get<0>(_obj)[0] );
    _Fvar.push_back( std::get<1>(_obj)[0] );
    _dag->eval( 1, &_Fvar.back(), &depF, _nX, _Xvar.data(), _Xdep.data(), _nP, _Pvar.data(), _Pdep.data() ); 
    _Fdep.push_back( depF );
  }
  else{
    _Ftyp.push_back( BASE_OPT::MIN );
    _Fvar.push_back( 0 );
    _Fdep.push_back( depF );
  }
  _Fbnd.push_back( _IINF );

  // Constraints
  for( size_t i=0; i<std::get<0>(_ctr).size(); i++ ){
    _Ftyp.push_back( std::get<0>(_ctr)[i] );
    _Fvar.push_back( std::get<1>(_ctr)[i] );
    switch( std::get<0>(_ctr)[i] ){
      case BASE_OPT::EQ: _Fbnd.push_back( T(0) );                break;
      case BASE_OPT::LE: _Fbnd.push_back( T(-BASE_OPT::INF,0) ); break;
      case BASE_OPT::GE: _Fbnd.push_back( T(0,BASE_OPT::INF) );  break;
    }
    _dag->eval( 1, &_Fvar.back(), &depF, _nX, _Xvar.data(), _Xdep.data(), _nP, _Pvar.data(), _Pdep.data() ); 
    _Fdep.push_back( depF );
  }
  _nF = _Fvar.size();
  assert( _Ftyp.size() == _nF );

#ifdef MC__MINLPSLV_DEBUG
  std::cout << "_dag = " << _dag << std::endl;
  std::cout << "_nF = " << _nF << std::endl;
  _dag->output( _dag->subgraph( _nF, _Fvar.data() ) );
#endif

  // linear and nonlinear functions
  _Flin.clear();
  _Fnlin.clear();
  _Foff.clear();
  _X0.resize( _nX, FFVar(_dag,0.) );
  for( size_t j=0; j<_nF; j++ ){
    if( _Fdep[j].worst() > FFDep::L )
      _Fnlin.insert( j );
    else
      _Flin.insert( j );
  }
#ifdef MC__MINLPSLV_DEBUG
  assert( _nF == _Flin.size() + _Fnlin.size() );
#endif

  // sparse function gradients
  _set_gradient();
  _Gval.resize( _nG );
  if( !_Gvar.empty() ){
    _Gop = _dag->subgraph( _nG, _Gvar.data() );
#ifdef MC__MINLPSLV_DEBUG
    _dag->output( _Gop );
#endif
  }
  else{
    _Fop = _dag->subgraph( _Fnlin, _Fvar.data() );
#ifdef MC__MINLPSLV_DEBUG
    _dag->output( _Fop );
#endif
  }

  // feasibility pump reference
  _Ffeas = 0.;
  _Xref.resize( _nX );
  for( auto i : _Xint ){
    _Xref[i].set( _dag );
    _Ffeas += sqr( _Xvar[i] - _Xref[i] );
  }

  stats.walltime_setup += stats.walltime( _tstart );
  stats.walltime_all   += stats.walltime( _tstart );
  _status = STATUS::SUCCESSFUL;
  _display_setup( os );
  _issetup = true;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_set_gradient
()
{
  // sparse linear function gradients
  _iAfun.clear(); _jAvar.clear(); _Aval.clear(); 
  for( auto const& iF : _Flin ){
    _cleanup_gradient();
    switch( options.NLPSLV.GRADMETH ){
      default:
      case NLP::Options::FSYM: _Fgrad = _dag->SFAD( 1, &_Fvar[iF], _nX, _Xvar.data() ); break;
      case NLP::Options::BSYM: _Fgrad = _dag->SBAD( 1, &_Fvar[iF], _nX, _Xvar.data() ); break;
    }

    // Gather derivative expressions
    for( size_t k=0; k<std::get<0>(_Fgrad); ++k ){
      _iAfun.push_back( iF );
      _jAvar.push_back( std::get<2>(_Fgrad)[k] );
      _Avar.push_back( std::get<3>(_Fgrad)[k] );
#ifdef MC__MINLPSLV_DEBUG
      std::cout << "  _Avar[" << _iAfun.back() << "," << _jAvar.back() << "] = " << _Avar.back() << std::endl;
#endif
    }
  }
  _nA = _Avar.size();

  // Compute constant offset expressions
  FFVar* Ftmp = _dag->compose( _Flin, _Fvar.data(), _nX, _Xvar.data(), _X0.data() );
  _Foff.assign( _nF, 0. );
  for( auto const& iF : _Flin ){
    _Foff[iF] = Ftmp[iF];
#ifdef MC__MINLPSLV_DEBUG
     std::cout << "  _Foff[" << iF << "] = " << _Foff[iF] << std::endl;
#endif
  }
  delete[] Ftmp;

  // sparse nonlinear function gradients
  _iGfun.clear(); _jGvar.clear(); _Gvar.clear(); 
  for( auto const& iF : _Fnlin ){
    _cleanup_gradient();
    switch( options.NLPSLV.GRADMETH ){
      case NLP::Options::FSYM: _Fgrad = _dag->SFAD( 1, &_Fvar[iF], _nX, _Xvar.data() ); break;
      case NLP::Options::BSYM: _Fgrad = _dag->SBAD( 1, &_Fvar[iF], _nX, _Xvar.data() ); break;
      default: break;
    }
    // Gather derivative expressions
    for( size_t k=0; k<std::get<0>(_Fgrad); ++k ){
      _iGfun.push_back( iF );
      _jGvar.push_back( std::get<2>(_Fgrad)[k] );
      _Gvar.push_back( std::get<3>(_Fgrad)[k] );
#ifdef MC__MINLPSLV_DEBUG
      std::cout << "  _Gvar[" << _iGfun.back() << "," << _jGvar.back() << "] = " << _Gvar.back() << std::endl;
#endif
    }

    // Gather derivative entries
    switch( options.NLPSLV.GRADMETH ){
      case NLP::Options::FAD:
      case NLP::Options::BAD:
        for( size_t iX=0; iX<_nX; ++iX ){
          if( !_Fdep[iF].dep( _Xvar[iX].id().second ).first ) continue;
          _iGfun.push_back( iF );
          _jGvar.push_back( iX );
#ifdef MC__MINLPSLV_DEBUG
          std::cout << "  _Gvar[" << iF << "," << iX << "]" << std::endl;
#endif
        }
        break;
      default:
        break;
    }
  }
  _nG = _jGvar.size();
  _cleanup_gradient();
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_cleanup_gradient
()
{
  std::get<0>(_Fgrad) = 0;
  delete[] std::get<1>(_Fgrad);  std::get<1>(_Fgrad) = nullptr;
  delete[] std::get<2>(_Fgrad);  std::get<2>(_Fgrad) = nullptr;
  delete[] std::get<3>(_Fgrad);  std::get<3>(_Fgrad) = nullptr;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_is_integer_feasible
( double const* Xval, double const& feastol )
const
{
  for( size_t i=0; i<_var.size(); i++ ){
    if( !_Xtyp[i] ) continue;
    if( std::fabs( Xval[i] - std::round(Xval[i]) ) > feastol )
      return false;
  }
  return true;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_is_integer_equal
( double const* Xval, double const* Xref )
const
{
  for( size_t i=0; i<_var.size(); i++ ){
    if( !_Xtyp[i] ) continue;
    if( std::fabs( Xval[i] - Xref[i] ) > options.FEASTOL )
      return false;
  }
  return true;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_interrupted
( std::chrono::time_point<std::chrono::system_clock> const& tstart )
const
{
  if( stats.to_time( stats.walltime_all + stats.walltime( tstart ) ) > options.TIMELIMIT
   || ( options.MAXITER && _iter >= options.MAXITER ) )
    return true;
  return false;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_converged
()
const
{
  if( std::fabs( _Zinc - _Zrel ) <= options.CVATOL 
   || std::fabs( _Zinc - _Zrel ) <= 0.5 * options.CVRTOL * std::fabs( _Zinc + _Zrel ) )
    return true;
  return false;
}

template <typename T, typename NLP, typename MIP>
inline int
MINLPSLV<T,NLP,MIP>::_finalize
( std::chrono::time_point<std::chrono::system_clock> const& tstart,
  STATUS const status, std::ostream& os )
{
  _status = status;
  stats.walltime_all += stats.walltime( tstart );
  _display_final( _iter, stats.walltime( tstart ), os );
  return _status;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_test_feasible
( double const* Xval, std::ostream& os )
{
  // Test integer feasibility first
  if( !_is_integer_feasible( Xval, options.FEASTOL ) )
    return false;

  // Test other constraint feasibility next
  auto tNLP = stats.start();
  _NLPSLV.restore_model();
  bool flag = _NLPSLV.is_feasible( Xval, options.FEASTOL );
  _solution = _NLPSLV.solution();
  stats.walltime_slvnlp += stats.walltime( tNLP );
  return flag;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_solve_local
( std::chrono::time_point<std::chrono::system_clock> const& tstart,
  double const* Xini, T const* Xbnd, bool const pumpfeas,
  bool const inccut, std::ostream& os )
{
  auto tNLP = stats.start();

  // Modify NLP model for feasibility pump
  double objscal = _objscal;
  if( pumpfeas ){
/*
    FFVar feasobj(0.);
    for( unsigned i=0; i<_var.size(); i++ ){
      if( !_vartyp[i] ) continue;
      feasobj += sqr( _Xvar[i] - _Xrel[i] );
    }
    _NLPSLV.set_obj_lazy( BASE_OPT::MIN, feasobj );
    if( inccut && options.INCCUT && !_incumbent.x.empty() )
      _NLPSLV.add_ctr_lazy( _Ftyp[0]==BASE_OPT::MIN? BASE_OPT::LE: BASE_OPT::GE, _Fvar[0] - _Zinc );
*/
    for( auto i : _Xint ) _Xref[i].set( _Xrel[i] );
#ifdef MC__MINLPSLV_DEBUG
  std::cout << "\nFeasibility pump reference: " << std::endl;
  for( auto i : _Xint )
    std::cout << "_Xref[" << i << "] = " << _Xref[i] << std::endl;
  _dag->output( _dag->subgraph( 1, &_Ffeas ) );
#endif
    objscal = 1;
    _NLPSLV.set_obj_lazy( BASE_OPT::MIN, _Ffeas );
//    if( inccut && options.INCCUT && !_incumbent.x.empty() )
    if( inccut && !_incumbent.x.empty() )
      _NLPSLV.add_ctr_lazy( _Ftyp[0]==BASE_OPT::MIN? BASE_OPT::LE: BASE_OPT::GE, _Fvar[0] - _Zinc );
  }
  else
    _NLPSLV.restore_model();

  // Local solve from provided initial point
  _solution.reset();
  _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( tstart ) );
#ifdef MC__MINLPSLV_DEBUG
  for( size_t iX=0; iX<_nX; ++iX )
    std::cout << "MINLPSLV::Xini[" << iX << "] = " << Xini[iX] << " in " << Xbnd[iX] << std::endl;
#endif
  _NLPSLV.solve( Xini, Xbnd, _Pval.data() );
  if( _NLPSLV.is_feasible( options.FEASTOL ) )
    _solution = _NLPSLV.solution();

  // Extra local solves from random starting points
  if( _isbnd && options.MSLOC > 1 ){
    _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( tstart ) );
    _NLPSLV.solve( options.MSLOC-1, Xbnd, _Pval.data() );
    if( _NLPSLV.is_feasible( options.FEASTOL )
     && (_solution.x.empty() || objscal*_NLPSLV.solution().f[0] < objscal*_solution.f[0]) )
      _solution = _NLPSLV.solution();
  }

  // Compute correction
  _Zcor = 0.;
  if( !_solution.x.empty() && options.CORRINC )
    _Zcor = _NLPSLV.cost_correction();

  stats.walltime_slvnlp += stats.walltime( tNLP );
  return !_solution.x.empty();
}
  
template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_init_master
()
{
  auto tMIP = stats.start();

  // Reset polyhedral image and MIP solver
  _POLenv.reset();
  _POLcutinc  = nullptr;
  _POLcutroot = nullptr;
  _POLenv.options = options.POLIMG;

  // Set polyhedral main variables
  _POLXvar.clear();
  auto itX = _Xvar.begin();
  for( size_t i=0; itX!=_Xvar.end(); ++itX, i++ )
    _POLXvar.push_back( PolVar<T>( &_POLenv, *itX, T(_Xlow[i],_Xupp[i]), (_Xtyp[i]? false: true) ) );

  // Set polyhedral slack variables
  _POLSvar.clear();
  _POLScost.clear();
  _POLSvar.push_back( PolVar<T>( &_POLenv, T(-BASE_OPT::INF,BASE_OPT::INF), true ) ); // cost variable
  _POLScost.push_back( 1. ); //_objscal ); // cost coefficient -> always minimize MIP
#ifdef MC__MINLPSLV_DEBUG
  std::cout << _POLenv;
#endif

  // Reinitialize MIP solver
  _MIPSLV.options = options.MIPSLV;
  //_MIPSLV.set_cuts( &_POLenv, true );

  stats.walltime_slvnlp += stats.walltime( tMIP );
}
  
template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_update_master
( bool const locfeas, bool const pumpfeas, bool const inccut, bool const intrel )
{
  auto tMIP = stats.start();

  // Append new outer-approximation cuts to polyhedral image
  if( locfeas ){
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "Adding outer-approximation cuts" << std::endl;
    std::cout << _solution;
#endif
    if( !_add_outerapproximation_cuts( _solution.x, _solution.f, _solution.uf ) ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
    }
  }

  // Append new integer cut
  if( !locfeas || (options.LINMETH != Options::CVX && intrel) ){
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "Adding integer cut" << std::endl;
#endif
    if( !_add_integer_cut( _Xrel ) ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
    }
  }

  // Append new anticyling cut
  if( pumpfeas ){
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "Adding anticycling cut" << std::endl;
#endif
    if( !_add_anticycling_cut( _solution.x, _Xrel )
     || (locfeas && !_set_integer_cut( _Xrel, _POLauxvar, _POLauxwei, _auxcst )) ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
    }
  }

//  // Update incumbent cut
//  if( inccut && options.INCCUT ){
//#ifdef MC__MINLPSLV_DEBUG
//    std::cout << "Adding incumbent cut" << std::endl;
//#endif
//    if( !_add_incumbent_cut() ){
//      stats.walltime_slvnlp += stats.walltime( tMIP );
//      return false;
//    }
//  }

  // Append new root-relaxation cut
  if( _iter == 1 && options.ROOTCUT ){
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "Adding root-node cut" << std::endl;
#endif
    if( !_add_outerapproximation_cuts( _rootrel.x, _rootrel.f, _rootrel.uf ) ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
    }
//    if( !_add_rootnode_cut() ){
//      stats.walltime_slvnlp += stats.walltime( tMIP );
//      return false;
//    }
  }
  
  // Update master MIP problem 
  _MIPSLV.set_cuts( &_POLenv, true );//false );
  if( pumpfeas )
    _MIPSLV.set_objective( _POLauxvar.size(), _POLauxvar.data(), _POLauxwei.data(), BASE_OPT::MIN );    
  else
    _MIPSLV.set_objective( _POLSvar.size(), _POLSvar.data(), _POLScost.data(), BASE_OPT::MIN );
#ifdef MC__MINLPSLV_DEBUG
    std::cout << _POLenv;
#endif

  stats.walltime_slvnlp += stats.walltime( tMIP );
  return true;
}

//template <typename T, typename NLP, typename MIP>
//inline bool
//MINLPSLV<T,NLP,MIP>::_add_rootnode_cut
//()
//{
//  double const Zrel = (_Ftyp[0]==BASE_OPT::MIN? _Zrel: -_Zrel );
//  _POLcutroot = *_POLenv.add_cut( nullptr, PolCut<T>::GE, Zrel, _POLSvar.front(), 1. );
//  return true;
//}

//template <typename T, typename NLP, typename MIP>
//inline bool
//MINLPSLV<T,NLP,MIP>::_add_incumbent_cut
//()
//{
//  double const Ztol = 0; //std::max( options.CVATOL, 0.5 * options.CVRTOL * std::fabs( _Zinc + _Zrel ) );
//  double const Zcor = (_Ftyp[0]==BASE_OPT::MIN? _Zinc-Ztol: -_Zinc+Ztol );
//  if( !_POLcutinc ) // create new incumbent cut
//    _POLcutinc = *_POLenv.add_cut( nullptr, PolCut<T>::LE, Zcor, _POLSvar.front(), 1. );
//  else              // update existing incumbent cut
//    _POLcutinc->rhs() = Zcor;
//  return true;
//}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_add_outerapproximation_cuts
( std::vector<double> const& Xval, std::vector<double>& Fval, std::vector<double>& Fmul )
{
  // Initialize cuts
  _POLcuts.assign( _nF, nullptr );
  for( size_t i=0; i<_nF; i++ ){
  
    // Only add linear cuts at first iteration
    bool islin = (_Fnlin.find(i) == _Fnlin.end()? true: false);
    if( _iter > 1 && islin ) continue;

    // Define cut type: objective
    if( !i ){
      if( islin ) switch( _Ftyp[0] ){
        case BASE_OPT::MIN: _POLcuts[0] = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -_Fval[0], _POLSvar.front(), -1. ); continue;
        case BASE_OPT::MAX: _POLcuts[0] = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -_Fval[0], _POLSvar.front(),  1. ); continue;
        default: return false;
      }
      else switch( _Ftyp[0] ){
        case BASE_OPT::MIN: _POLcuts[0] = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -Fval[0], _POLSvar.front(), -1. ); continue;
        case BASE_OPT::MAX: _POLcuts[0] = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -Fval[0], _POLSvar.front(),  1. ); continue;
        default: return false;
      }
    }

    // Define cut type: linear and convex constraints
    if( islin ) switch( _Ftyp[i] ){
      case BASE_OPT::EQ: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::EQ, -_Fval[i] ); continue;
      case BASE_OPT::LE: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -_Fval[i] ); continue;
      case BASE_OPT::GE: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -_Fval[i] ); continue;
      default: return false;
    }
    else if( options.LINMETH == Options::CVX ) switch( _Ftyp[i] ){
      case BASE_OPT::EQ: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::EQ, -Fval[i] ); continue;
      case BASE_OPT::LE: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -Fval[i] ); continue;
      case BASE_OPT::GE: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -Fval[i] ); continue;
      default: return false;
    }

    // Define cut type: nonlinear constraints
    if( _iter > 1 ){
      switch( _Ftyp[i] ){ // Active nonlinear constraints only
        case BASE_OPT::LE: if( Fval[i] < -options.FEASTOL ) continue;
        case BASE_OPT::GE: if( Fval[i] >  options.FEASTOL ) continue;
        default: break;
      }
    }
    if( options.LINMETH == Options::PENAL ){
      _POLSvar.push_back( PolVar<T>( &_POLenv, T(0.,BASE_OPT::INF), true ) ); // slack variable
      _POLScost.push_back( options.PENSOFT * std::fabs(Fmul[i]) ); // slack cost coefficient
      switch( _Ftyp[i] ){
        case BASE_OPT::LE: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -Fval[i], _POLSvar.back(), -1. ); continue;
        case BASE_OPT::GE: _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -Fval[i], _POLSvar.back(),  1. ); continue;
        case BASE_OPT::EQ:
          if( _objscal * Fmul[i] < 0 ) _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -Fval[i], _POLSvar.back(), -1. );
          else                         _POLcuts[i] = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -Fval[i], _POLSvar.back(),  1. );
          continue;
        default: return false;
      }
    }
  }

  // Populate linear cuts
  if( _iter == 1 ){
    for( size_t ie=0; ie<_nA; ie++ ){
      if( _Aval[ie] == 0. ) continue;
      _POLcuts[_iAfun[ie]]->append( _POLXvar[_jAvar[ie]], _Aval[ie] ).rhs();
    }
  }
  
  // Evaluate nonlinear function derivatives
  try{
    switch( options.NLPSLV.GRADMETH ){
      // Compute backward numeric derivative
      case NLP::Options::BAD:
        _BXval.resize( _nX );
        // Initialize participating variables in fadbad::B<double>
        for( size_t iX=0; iX<_nX; ++iX )
          _BXval[iX] = Xval[iX];
        _BFval.resize( _nF );
        if( _nP ){
          _BPval.resize( _nP );
          // Initialize parameters in fadbad::B<double>
          for( size_t iP=0; iP<_nP; ++iP ) _BPval[iP] = _Pval[iP];
          _dag->eval( _Fop, _Bwk, _Fnlin, _Fvar.data(), _BFval.data(), _nX, _Xvar.data(), _BXval.data(), _nP, _Pvar.data(), _BPval.data() );
        }
        else{
          _dag->eval( _Fop, _Bwk, _Fnlin, _Fvar.data(), _BFval.data(), _nX, _Xvar.data(), _BXval.data() );
        }
        _Bwk.clear();
        for( auto const& iF : _Fnlin )
          _BFval[iF].diff( iF, _nF );
        // Gather derivatives
        for( size_t ie=0; ie<_nG; ie++ ){
#ifdef MC__MINLPSLV_DEBUG_LINEARIZATION
          std::cout << "MINLPSLV::_Gval[" << _iGfun[ie] << "," << _jGvar[ie] << "] = "
                    << _BXval[ _jGvar[ie] ].d( _iGfun[ie] ) << std::endl;
#endif
          _Gval[ie] = _BXval[ _jGvar[ie] ].d( _iGfun[ie] );
        }
        break;
          
      // Compute forward numeric derivative
      case NLP::Options::FAD:
        _FXval.resize( _nX );
        // Initialize participating variables in fadbad::F<double>
        for( size_t iX=0; iX<_nX; ++iX ){
#ifdef MC__MINLPSLV_DEBUG_LINEARIZATION
          std::cout << "MINLPSLV::Xval[" << iX << "] = " << Xval[iX] << std::endl;
#endif
          _FXval[iX] = Xval[iX];
          _FXval[iX].diff( iX, _nX );
        }
        _FFval.resize( _nF );
        if( _nP ){
          _FPval.resize( _nP );
          // Initialize parameters in fadbad::F<double>
          for( size_t iP=0; iP<_nP; ++iP ) _FPval[iP] = _Pval[iP];
          _dag->eval( _Fop, _Fwk, _Fnlin, _Fvar.data(), _FFval.data(), _nX, _Xvar.data(), _FXval.data(), _nP, _Pvar.data(), _FPval.data() );
        }
        else{
          _dag->eval( _Fop, _Fwk, _Fnlin, _Fvar.data(), _FFval.data(), _nX, _Xvar.data(), _FXval.data() );
        }
#ifdef MC__MINLPSLV_DEBUG_LINEARIZATION
        for( auto const& iF : _Fnlin )
          std::cout << "MINLPSLV::Fval[" << iF << "] = " << _FFval[iF].x() << std::endl;
#endif
        // Gather derivatives
        for( size_t ie=0; ie<_nG; ie++ ){
#ifdef MC__MINLPSLV_DEBUG_LINEARIZATION
          std::cout << "MINLPSLV::_Gval[" << _iGfun[ie] << "," << _jGvar[ie] << "] = "
                    << _FFval[ _iGfun[ie] ].d( _jGvar[ie] ) << std::endl;
#endif
          _Gval[ie] = _FFval[ _iGfun[ie] ].d( _jGvar[ie] );
        }
        break;

      // Compute symbolic derivative
      case NLP::Options::BSYM:
      case NLP::Options::FSYM:
        if( _nP )
          _dag->eval( _Gop, _dwk, _nG, _Gvar.data(), _Gval.data(), _nX, _Xvar.data(), Xval.data(), _nP, _Pvar.data(), _Pval.data() );
        else
          _dag->eval( _Gop, _dwk, _nG, _Gvar.data(), _Gval.data(), _nX, _Xvar.data(), Xval.data() );
#ifdef MC__MINLPSLV_DEBUG_LINEARIZATION
        for( size_t ie=0; ie<_nG; ie++ )
          std::cout << "MINLPSLV::_Gval[" << _iGfun[ie] << "," << _jGvar[ie] << "] = " << _Gval[ie] << std::endl;
#endif
        break;

      // Other derivative method - error
      default:
        throw Exceptions( Exceptions::INTERN );
    }
  }
  catch(...){
    return false;
  }
  
  // Populate linearized nonlinear cuts
  for( size_t ie=0; ie<_nG; ie++ ){
    if( !_POLcuts[_iGfun[ie]] || _Gval[ie] == 0. ) continue;
    _POLcuts[_iGfun[ie]]->append( _POLXvar[_jGvar[ie]], _Gval[ie] ).rhs() += _Gval[ie] * Xval[_jGvar[ie]];
  }

#ifdef MC__MINLPSLV_DEBUG
  for( size_t i=0; i<_nF; i++ )
    if( _POLcuts[i] ) std::cout << " _POLcuts[" << i << "]: " << *_POLcuts[i] << std::endl;
  int dum; std::cout << "PRESS 1 TO CONTINUE"; std::cin >> dum;
#endif
  return true;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_add_integer_cut
( std::vector<double> const& Xint )
{
  // Add constraints for the linear cut: \|y-\bar{y}\|_1 \geq 1
  if( !_set_integer_cut( Xint, _POLauxvar, _POLauxwei, _auxcst ) ) return false;
  _POLenv.add_cut( nullptr, PolCut<T>::GE, 1-_auxcst, _POLauxvar.size(), _POLauxvar.data(), _POLauxwei.data() );
  return true;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_set_integer_cut
( std::vector<double> const& Xint, std::vector<PolVar<T>>& linvar,
  std::vector<double>& linwei, double& cst )
{
  // Add linear cut: \|y-\bar{y}\|_1 \geq 1
  cst = 0.;
  linvar.clear();
  linwei.clear();
  for( auto const& j: _Xint ){
    if( Xint[j] <= std::ceil(_Xlow[j]) + options.FEASTOL ){
      linvar.push_back( _POLXvar[j] );
      linwei.push_back( 1. );
      cst -= std::ceil(_Xlow[j]);
    }
    else if( Xint[j] >= std::floor(_Xupp[j]) - options.FEASTOL ){
      linvar.push_back( _POLXvar[j] );
      linwei.push_back( -1. );
      cst += std::ceil(_Xupp[j]);
    }
    else{
      PolVar<T> POLWvar( &_POLenv, T(0.,BASE_OPT::INF), true );
      linvar.push_back( POLWvar );
      linwei.push_back( 1. );
      _POLenv.add_cut( nullptr, PolCut<T>::GE, Xint[j], _POLXvar[j], 1., POLWvar,  1. );
      _POLenv.add_cut( nullptr, PolCut<T>::LE, Xint[j], _POLXvar[j], 1., POLWvar, -1. );
      PolVar<T> POLNvar( &_POLenv, T(0.,1.), false );
      double M1 = 2 * ( Xint[j] - std::ceil(_Xlow[j]) );
      double M2 = 2 * ( std::floor(_Xupp[j]) - Xint[j] );
      _POLenv.add_cut( nullptr, PolCut<T>::GE, Xint[j]-M1, _POLXvar[j], 1., POLWvar, -1., POLNvar, -M1 );
      _POLenv.add_cut( nullptr, PolCut<T>::LE, Xint[j],    _POLXvar[j], 1., POLWvar,  1., POLNvar, -M2 );
    }
  }

  return true;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_add_anticycling_cut
( std::vector<double> const& Xval, std::vector<double> const& Xrel )
{
  // Add linear cut: [\bar{y}-\hat{y}]^T[y-\hat{y}] \geq 0
  auto ACcut = *_POLenv.add_cut( nullptr, PolCut<T>::GE, 0. );
  for( auto const& j: _Xint )
    ACcut->append( _POLXvar[j], Xval[j]-Xrel[j] ).rhs() += Xval[j]*(Xval[j]-Xrel[j]);

  return true;
}

template <typename T, typename NLP, typename MIP>
inline int
MINLPSLV<T,NLP,MIP>::_solve_master
( std::chrono::time_point<std::chrono::system_clock> const& tstart )
{
  auto tMIP = stats.start();

  // Setup and solve master MIP problem
  _MIPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( tstart ) );
  _MIPSLV.solve();

  stats.walltime_slvmip += stats.walltime( tMIP );
  return _MIPSLV.get_status();
}

template <typename T, typename NLP, typename MIP>
inline int
MINLPSLV<T,NLP,MIP>::_propagate_bounds
( T* Xbnd )
{
  auto tCP = stats.start();
#ifdef MC__MINLPSLV_DEBUG
  _dag->output( _dag->subgraph( _Flin, _Fvar.data() ) );
  for( size_t i=0; i<_nX; i++ )
    std::cout << "X[" << i << "] = " << Xbnd[i] << std::endl;
  for( size_t i=0; i<_nF; i++ )
    std::cout << "F[" << i << "] = " << _Fbnd[i] << std::endl;
  std::cout << "FLin = { ";
  for( auto const& i: _Flin ) std::cout << i << " ";
  std::cout << "}" << std::endl;
#endif

  // Apply constraint propagation
  int flag = 0;
  for( size_t i=0; i<_nP; i++ )
    _Pvar[i].set( _Pval[i] );
  try{
    flag = _dag->reval( _CPbnd, _nF-1, _Fvar.data()+1, _Fbnd.data()+1, _nX, _Xvar.data(),
                        Xbnd, _IINF, options.CPMAX, options.CPTHRES );
    //flag = _dag->reval( _CPbnd, _Flin, _Fvar.data(), _Fbnd.data(), _nX, _Xvar.data(),
    //                    Xbnd, _IINF, options.CPMAX, options.CPTHRES );

    // Round binary and integer variables accordingly
    for( size_t ix=0; ix<_nX; ++ix )
      if( _Xtyp[ix] > 0 ) Xbnd[ix] = T( std::ceil( Op<T>::l(Xbnd[ix]) ), std::floor( Op<T>::u(Xbnd[ix]) ) );
  }
  catch(...){ std::cout << "Exception caught\n"; }
  for( size_t i=0; i<_nP; i++ )
    _Pvar[i].unset();
#ifdef MC__MINLPSLV_DEBUG
  std::cout << "\nVariable Domain: " << flag << std::endl;
  for( size_t i=0; i<_nX; i++ )
    std::cout << _Xvar[i] << " X[" << i << "] = " << Xbnd[i] << std::endl;
#endif

  stats.walltime_setup += stats.walltime( tCP );
  return flag;
}

template <typename T, typename NLP, typename MIP>
inline typename SBBSLV<T>::STATUS
MINLPSLV<T,NLP,MIP>::subproblems
( typename SBBSLV<T>::TASK const task, SBBNode<T>* node,
  std::vector<double>& p, double& f, double const& INC, std::ostream& os )
{
  typename SBBSLV<T>::STATUS status = SBBSLV<T>::FATAL;

  // Compute local solution
  if( (task == SBBSLV<T>::UPPERBD && _objscal > 0.) 
   || (task == SBBSLV<T>::LOWERBD && _objscal < 0.) ){

    if( _roundf ){
      _roundf( _Xtyp.size(), _Xtyp.data(), p.data() );
#ifdef MC__MINLPSLV_DEBUG
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Rounded point:\n@";
      for( auto const& pi : p ) std::cout << " " << pi;
      std::cout << std::endl;
#endif
      if( _test_feasible( p.data(), os ) ){
        f = _solution.f[0] + _Zcor;
        status = SBBSLV<T>::NORMAL;
#ifdef MC__MINLPSLV_DEBUG
        std::cout << "f = " << f << std::endl;
#endif
      }
      else
        status = SBBSLV<T>::FAILURE;
    }
    else
      status = SBBSLV<T>::FAILURE;
  }

  // compute relaxed solution
  else if( (task == SBBSLV<T>::UPPERBD && _objscal < 0.) 
        || (task == SBBSLV<T>::LOWERBD && _objscal > 0.) ){

#ifdef MC__MINLPSLV_DEBUG
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Initial point:\n@";
    for( auto const& pi : p ) std::cout << " " << pi;
    std::cout << std::endl;
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Range:\n";
    for( auto const& Xbndi : node->P() ) std::cout << " " << Xbndi << std::endl;
    std::cout << std::endl;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

    if( _solve_local( _tstart, p.data(), node->P().data(), false, false, os ) ){
#ifdef MC__MINLPSLV_DEBUG
      std::cout << "Relaxed solution:\n" << _solution;
      { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
      p = _solution.x;
      f = _solution.f[0] + _Zcor;
      status = SBBSLV<T>::NORMAL;
    }
    else if( _NLPSLV.get_status() == NLP::STATUS::INFEASIBLE )
      status = SBBSLV<T>::INFEASIBLE;
    else
      status = SBBSLV<T>::FAILURE;
    }

  // assess feasibility
  else if( task == SBBSLV<T>::FEASTEST ){
    if( _test_feasible( p.data(), os ) )
      status = SBBSLV<T>::NORMAL;
    else
      status = SBBSLV<T>::INFEASIBLE;
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "Feasibility test: " << status << std::endl;
#endif
  }

  // perform preprocessing
  else if( task == SBBSLV<T>::PREPROC ){
#ifdef MC__MINLPSLV_DEBUG
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Original range:\n";
    for( auto const& Xbndi : node->P() ) std::cout << " " << Xbndi << std::endl;
    std::cout << std::endl;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
    // propagate bounds
    if( options.CPMAX && _propagate_bounds( node->P().data() ) < 0 )
      status = SBBSLV<T>::INFEASIBLE;
    else{
      status = SBBSLV<T>::NORMAL;
      // round integer bounds
      for( auto const& i : _Xint ){
        if( std::ceil( Op<T>::l( node->P(i) ) ) > std::floor( Op<T>::u( node->P(i) ) ) ){
          status = SBBSLV<T>::INFEASIBLE;
          break;
        }
        node->P(i) = T( std::ceil( Op<T>::l( node->P(i) ) ), std::floor( Op<T>::u( node->P(i) ) ) );
      }
#ifdef MC__MINLPSLV_DEBUG
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Preprocessed range:\n";
      for( auto const& Xbndi : node->P() ) std::cout << " " << Xbndi << std::endl;
      std::cout << std::endl;
      { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
    }
  }

  // perform postprocessing
  else if( task == SBBSLV<T>::POSTPROC )
    status = SBBSLV<T>::NORMAL;

  // other
  else
    status = SBBSLV<T>::FATAL;

  return status;
}

template <typename T, typename NLP, typename MIP>
inline int
MINLPSLV<T,NLP,MIP>::optimize
(  double const* Xini, T const* Xbnd, double const* Pval, ROUND const& f, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _tstart = stats.start();
  _roundf = f;
  _iter = 0;
  _Zinc =  _objscal * BASE_OPT::INF;
  _Zrel = -_objscal * BASE_OPT::INF;
  _incumbent.reset();
 
  // Update parameter values
  if( !_Pvar.empty() && !Pval ) throw MINLPSLV::Exceptions( MINLPSLV::Exceptions::PARAM );
  if( Pval ) _Pval.assign( Pval, Pval+_nP );
#ifdef MC__MINLPSLV_DEBUG
  for( size_t i=0; !Pvar.empty() && P0 && i<_nP; i++ )
    std::cout << "  Pval[" << i << "] = " << Pval[i] << std::endl;
#endif

  // Update linear constraint coefficients
  _Aval.resize( _nA );
  _Fval.assign( _nF, 0. );
  _dag->eval( _dwk, _nA,   _Avar.data(), _Aval.data(), _nP, _Pvar.data(), _Pval.data() );
  _dag->eval( _dwk, _Flin, _Foff.data(), _Fval.data(), _nP, _Pvar.data(), _Pval.data() );

  // Search strategy
  int flag = 0;
  switch( options.SEARCHALG ){
    case Options::OA:
      flag = _optimize_oa( !Xini && !_Xini.empty()? _Xini.data(): Xini, Xbnd, os );
      break;

    case Options::BB:
      flag = _optimize_bb( !Xini && !_Xini.empty()? _Xini.data(): Xini, Xbnd, os );
      break;

    default:
      throw Exceptions( Exceptions::SEARCHALG );
  }

  return flag;
}

template <typename T, typename NLP, typename MIP>
inline int
MINLPSLV<T,NLP,MIP>::_optimize_bb
(  double const* Xini, T const* Xbnd, std::ostream& os )
{
  // Initialize and reduce variable bounds
  _Xbnd.resize( _nX );
  for( size_t i=0; i<_nX; i++ ){
    _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
    if( Xbnd && !Op<T>::inter( _Xbnd[i], Xbnd[i], _Xbnd[i] ) )
      return _finalize( _tstart, STATUS::INFEASIBLE );
  }
  if( options.CPMAX && _propagate_bounds( _Xbnd.data() ) < 0 )
    return _finalize( _tstart, STATUS::INFEASIBLE );
#ifdef MC__MINLPSLV_DEBUG
  std::cout << std::scientific << std::setprecision(4);
  std::cout << "Initial range:\n";
  for( auto const& Xbndi : _Xbnd ) std::cout << " " << Xbndi << std::endl;
  std::cout << std::endl;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

  // Call B&B solver
  is_bounded( BASE_OPT::INF/10 ); // set _isbnd flag
  _set_options_sbbslv();        // setting SBBSLV solver options
  int flag = SBBSLV<T>::solve( std::get<0>(_obj)[0], _nX, _Xbnd.data(), Xini, nullptr,
                               _Xtyp.data(), _Xcnt, os );
  stats.walltime_all += stats.walltime( _tstart );

  // RECOVER OPTIMAL SOLUTION VALUE AND SOLutION POINT
  return flag;
}

template <typename T, typename NLP, typename MIP>
inline int
MINLPSLV<T,NLP,MIP>::_optimize_oa
(  double const* Xini, T const* Xbnd, std::ostream& os )
{
  //auto tstart = stats.start();

  // Initialization
  _display_init( os );
  
  // Initial point
  //if( Xini ) _varini.assign( Xini, Xini+_var.size() );
  //else       _varini.clear(); // No default initial guess

  // Initialize and reduce variable bounds
  bool locfeas = true;
  _Xbnd.resize( _nX );
  for( size_t i=0; i<_nX; i++ ){
    _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
    if( Xbnd && !Op<T>::inter( _Xbnd[i], Xbnd[i], _Xbnd[i] ) ){
      locfeas = false;
      break;
    }
  }
  int cpflag = 0;
  if( locfeas && options.CPMAX ){
    cpflag = _propagate_bounds( _Xbnd.data() );
    if( cpflag < 0 ){
      locfeas = false;
      _Zrel = _objscal * BASE_OPT::INF;
    }
  }

  // Solve relaxed MINLP model
  if( locfeas ){
    is_bounded( BASE_OPT::INF/10 ); // set _isbnd flag
    //locfeas = _solve_local( _tstart, _varini.data(), _Xbnd.data(), false, false, os );
    locfeas = _solve_local( _tstart, Xini, _Xbnd.data(), false, false, os );
  }
#ifdef MC__MINLPSLV_DEBUG
  //std::cout << _solution;
  std::cout << "_Zloc = " << _solution.f[0] << std::endl;
  for( size_t i=0; i<_nX; i++ )
    if( std::fabs(_solution.x[i]) > 1e-5 )
      std::cout << "_Xloc[" << i << "] = " << _solution.x[i] << std::endl;
#endif

  // Update bounds
  bool intrel = false;
  if( locfeas ){
    _rootrel = _solution;
    _Zrel  = _solution.f[0];
    _Xrel  = _solution.x;
    _Xbndi = _Xbnd;
    if( _is_integer_feasible( _Xrel.data(), options.FEASTOL ) ){
      _incumbent = _solution;
      _Zinc = _incumbent.f[0];
      intrel = true;
    }
    else if( _roundf ){
      //SOLUTION_OPT soltmp = _solution; // temporary storage
      _roundf( _Xtyp.size(), _Xtyp.data(), _solution.x.data() );
      if( _is_integer_feasible( _solution.x.data(), options.FEASTOL ) ){
        if( _Xint.size() < _nX ){
          for( auto const& i: _Xint ) _Xbndi[i] = _solution.x[i];
          if( _solve_local( _tstart, _solution.x.data(), _Xbndi.data(), false, false, os ) ){
            _incumbent = _solution;
            _Zinc = _incumbent.f[0];
            intrel = true;
          }
        }
        else if( _test_feasible( _solution.x.data(), os ) ){
          _incumbent = _solution;
          _Zinc = _incumbent.f[0];
          intrel = true;      
        }
      }
      // Restore relaxed solution if rounding unsuccessful
      if( !intrel ){
        //_solution = soltmp;
        _solution = _rootrel; 
        _incumbent.reset();
      }
    }
  }
  else{
    _Zrel = _objscal * BASE_OPT::INF;
  }
  bool updinc = !_incumbent.x.empty();
  
  // Intermediate display
  _display_add( _iter );
  if( !locfeas )                _display_add( "i");
  else if( updinc && cpflag>0 ) _display_add( "r*");
  else if( updinc )             _display_add( "*" );
  else if( cpflag )             _display_add( "r" );
  else                          _display_add( " " );
  _display_add( _Zinc );
  _display_add( _Zrel );
  _display_add( _tstart );
  _display_flush( os );

  // Termination tests
  if( !locfeas )
    return _finalize( _tstart, STATUS::INFEASIBLE );
  if( !_ismip )
    return _finalize( _tstart, STATUS::SUCCESSFUL );
  if( _interrupted( _tstart ) )
    return _finalize( _tstart, STATUS::INTERRUPTED );
  
  // Initialize master MIP subproblem
  _init_master();

  // Main loop
  bool pumpfeas = false, stopiter = false;
  for( ++_iter; !stopiter && !_converged() ; ++_iter ){

    // Update master MIP subproblem
    if( !_update_master( locfeas, pumpfeas, updinc, intrel ) )
      return _finalize( _tstart, STATUS::ABORTED );

    // Solve master MIP subproblem
    switch( _solve_master( _tstart ) ){
      case MIP::OPTIMAL:
        break;
      case MIP::INFEASIBLE:
        _Zrel = _objscal * BASE_OPT::INF;
        return _finalize( _tstart, STATUS::INFEASIBLE );
      case MIP::UNBOUNDED:
        return _finalize( _tstart, STATUS::UNBOUNDED );
      case MIP::TIMELIMIT:
        return _finalize( _tstart, STATUS::INTERRUPTED );
      default:
        return _finalize( _tstart, STATUS::FAILURE );
    }

    // Retrieve MIP solution
    if( !pumpfeas )
      _Zrel = _objscal * _MIPSLV.get_variable( _POLSvar.front() );
    for( size_t i=0; i<_nX; i++ ){
      //std::cout << "Retrieve " << _POLXvar[i] << "(DAG: " << _POLXvar[i].var() << ")" << std::endl;
      try{
        _Xrel[i] = _MIPSLV.get_variable( _POLXvar[i] );
      }
      catch( const std::runtime_error& e){
//#ifdef MC__MINLPSLV_DEBUG
        std::cerr << "**MINLPSLV: Variable " << _POLXvar[i] << "(DAG: " << _POLXvar[i].var() << ") not in master MIP" << std::endl; 
//#endif
        _Xrel[i] = 0.;
      }
    }
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "_Zrel = " << _Zrel << std::endl;
    for( size_t i=0; i<_nX; i++ )
      if( std::fabs(_Xrel[i]) > 1e-5 )
        std::cout << "_Xrel[" << i << "] = " << _Xrel[i] << std::endl;
#endif
    locfeas = true; // reinitialize to not add integer cut to master MIP subproblem during feasibility pump
    intrel  = true; // reinitialize to enable integer cut to master MIP subproblem
    
    // Interrupt if master and local bounds cross each other
    if( !_incumbent.x.empty() && _objscal*_Zrel >= _objscal*_Zinc ){
      updinc   = false;
      stopiter = true;
    }
    
    // Apply feasibility pump if NLP model found infeasible at current MIP integer fixing
    else if( pumpfeas ){
      if( !_solve_local( _tstart, _Xrel.data(), _Xbnd.data(), true, !locfeas, os ) )
        return _finalize( _tstart, STATUS::FAILURE ); // may not be infeasible unless MINLP is infeasible
#ifdef MC__MINLPSLV_DEBUG
      std::cout << _solution;
#endif

      // Interrupt feasibilitity pump and solve local NLP model
      if( _is_integer_equal( _solution.x.data(), _Xrel.data() ) ){
        pumpfeas = false;
        if( _Xint.size() < _nX ){
          for( auto const& i: _Xint ) _Xbndi[i] = _Xrel[i];
          if( !_solve_local( _tstart, _solution.x.data(), _Xbndi.data(), false, false, os ) )
            return _finalize( _tstart, STATUS::FAILURE ); // may not be infeasible after feasibility pump
        }
        else if( !_test_feasible( _solution.x.data(), os ) )
          return _finalize( _tstart, STATUS::FAILURE ); // may not be infeasible after feasibility pump
#ifdef MC__MINLPSLV_DEBUG
        std::cout << _solution;
#endif

        // Update incumbent
        updinc = false;
        if( locfeas && _objscal*_Zinc > _objscal*_solution.f[0] ){
          updinc = true;
          _Zinc = _solution.f[0];
          _incumbent = _solution;
        }
      }
    }
    
    // Solve local NLP model at current MIP integer fixing outside feasibility pump
    else{
      if( _Xint.size() < _nX ){
        for( auto const& i: _Xint ) _Xbndi[i] = _Xrel[i];
        locfeas = _solve_local( _tstart, _Xrel.data(), _Xbndi.data(), false, false, os );
      }
      else
        locfeas = _test_feasible( _Xrel.data(), os );
#ifdef MC__MINLPSLV_DEBUG
      //std::cout << _solution;
      std::cout << "_Zloc = " << _solution.f[0] << std::endl;
      for( size_t i=0; i<_nX; i++ )
        if( std::fabs(_solution.x[i]) > 1e-5 )
          std::cout << "_Xloc[" << i << "] = " << _solution.x[i] << std::endl;
#endif
 
      // Enable feasbility pump
      if( !locfeas && options.FEASPUMP ) pumpfeas = true;
 
      // Update incumbent
      updinc = false;
      if( locfeas && _objscal*_Zinc > _objscal*_solution.f[0] ){
        updinc = true;
        _Zinc = _solution.f[0];
        _incumbent = _solution;
      }
    }

    // Intermediate display
    _display_add( _iter );
    if( updinc )        _display_add( "*");
    else if( pumpfeas ) _display_add( "f" );
    else if( !locfeas ) _display_add( "i" );
    else                _display_add( " " );
    _display_add( _Zinc );
    _display_add( _Zrel );
    _display_add( _tstart );
    _display_flush( os );

    // Termination tests
    if( stopiter )
      break;
    if( _interrupted( _tstart ) )
      return _finalize( _tstart, STATUS::INTERRUPTED );
  }

  return _finalize( _tstart, STATUS::SUCCESSFUL );
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::Options::display
( std::ostream & out ) const
{
  // Display MINLPSLV Options
  out << std::left;
  out << std::setw(60) << "#  LINEARIZATION METHOD";
  switch( LINMETH ){
   case CVX:   out << "CVX"   << std::endl; break;
   case PENAL: out << "PENAL" << std::endl; break;
  }
  out << std::setw(60) << "#  CONVERGENCE ABSOLUTE TOLERANCE"
      << std::scientific << std::setprecision(1)
      << CVATOL << std::endl;
  out << std::setw(60) << "#  CONVERGENCE RELATIVE TOLERANCE"
      << std::scientific << std::setprecision(1)
      << CVRTOL << std::endl;
  out << std::setw(60) << "#  FEASIBILITY PUMP"
      << ( FEASPUMP? "Y\n": "N\n" );
  out << std::setw(60) << "#  TIME LIMIT (SEC)"
      << std::scientific << std::setprecision(1)
      << TIMELIMIT << std::endl;
  out << std::setw(60) << "#  DISPLAY LEVEL"
      << DISPLEVEL << std::endl;
}

template <typename T, typename NLP, typename MIP>
inline std::ostream&
operator <<
( std::ostream & out, MINLPSLV<T,NLP,MIP> const& MINLP )
{
  out << std::left << std::endl
      << std::setfill('_') << std::setw(72) << "#" << std::endl << "#" << std::endl << std::setfill(' ')
      << "#  LOCAL MIXED-INTEGER NONLINEAR OPTIMIZATION IN CANON\n"
      << std::setfill('_') << std::setw(72) << "#" << std::endl << "#" << std::endl << std::setfill(' ');

  // Display MINLPSLV Options
  MINLP.options.display( out );

  out << std::left
      << std::setfill('_') << std::setw(72) << "#" << std::endl << std::endl << std::setfill(' ');
  return out;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_setup
( std::ostream& os )
{
  if( options.DISPLEVEL < 2 ) return;
  _odisp << "#  CONTINUOUS / DISCRETE VARIABLES:  " << _nX-_Xint.size()  << " / " << _Xint.size() << std::endl
         << "#  LINEAR / NONLINEAR FUNCTIONS:     " << _Flin.size() << " / " << _Fnlin.size() << std::endl;
  _display_flush( os ); 
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_init
( std::ostream& os)
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::endl
         << "#  " << std::right
  	 << std::setw(_IPREC) << "ITERATION"
  	 << std::setw(_DPREC+8) << "INCUMBENT"
  	 << std::setw(_DPREC+8) << "BEST BOUND"
  	 << std::setw(8) << "TIME";
  _display_flush( os ); 
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_final
( unsigned const iter, std::chrono::microseconds const& walltime,
  std::ostream& os )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::endl << "#  TERMINATION AFTER ";
  if( iter ) _odisp << _iter << " ITERATIONS: ";
  else       _odisp << "0 ITERATION: ";
  _odisp << std::fixed << std::setprecision(3) << walltime.count()*1e-6 << " SEC"
         << std::endl;

  // No feasible solution found
  if( _incumbent.x.empty() )
    _odisp << "#  NO FEASIBLE SOLUTION FOUND";

  // Feasible solution found
  else{
    // Incumbent
    _odisp << "#  INCUMBENT VALUE:" << std::scientific
           << std::setprecision(_DPREC) << std::setw(_DPREC+8) << _Zinc
           << std::endl;
    _odisp << "#  INCUMBENT POINT:";
    unsigned i(0);
    for( auto const& xi : _incumbent.x ){
      if( i++ == _LDISP ){
        _odisp << std::endl << std::left << std::setw(19) << "#";
        i = 1;
      }
      _odisp << std::right << std::setw(_DPREC+8) << xi;
    }
  }

  _display_flush( os );
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_add
( std::chrono::time_point<std::chrono::system_clock> const& tstart )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::fixed << std::setprecision(2)
         << std::setw(7) << stats.to_time( stats.walltime( tstart ) ) << "s";
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_add
( const double dval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::scientific << std::setprecision(_DPREC)
         << std::setw(_DPREC+8) << dval;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_add
( const unsigned ival )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(_IPREC) << ival;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_add
( const std::string &sval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(3) << sval;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_flush
( std::ostream &os )
{
  if( _odisp.str() == "" ) return;
  os << _odisp.str() << std::endl;
  _odisp.str("");
  return;
}

template <typename T, typename NLP, typename MIP>
inline std::set<unsigned>
MINLPSLV<T,NLP,MIP>::_branch_subset
( SBBNode<T> const* node )
{
  // Find maximal magnitude
  double mag = 0.;
  for( unsigned i=0; i<node->sbb()->get_variable_size(); ++i ){
    if( !node->sbb()->get_variable_type()[i] || std::round( Op<T>::diam( node->P(i) ) ) < mag ) continue;
    mag = std::round( Op<T>::diam( node->P(i) ) );
  }
#ifdef MC__MINLPSLV_DEBUG
  std::cout << "MINLPSLV::_branch_subset: maximal range: " << mag << std::endl;
#endif
  
  // Map all variables with maximal magnitude and corresponding rounding gap
  std::multimap<double,unsigned> branchmap;
  for( unsigned i=0; i<node->sbb()->get_variable_size(); ++i ){
    if( !node->sbb()->get_variable_type()[i] || std::round( Op<T>::diam( node->P(i) ) ) < mag ) continue;
    double pi = node->sbb()->get_problem_type()==BASE_OPT::MIN? node->pLB(i): node->pUB(i);
    double dist = std::fabs( pi - std::round( pi ) );
    branchmap.insert( std::make_pair( dist, i ) );
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "MINLPSLV::_branch_subset: <var,dist>: " << i << "  " << dist << std::endl;
#endif
  }

  // Return subset of variables with maximal range and rounding gap
  auto it = branchmap.lower_bound( branchmap.rbegin()->first );
  std::set<unsigned> branchset;
#ifdef MC__MINLPSLV_DEBUG
  std::cout << "MINLPSLV::_branch_subset: selection: {";
#endif
  for( ; it!=branchmap.cend(); ++it ){
    branchset.insert( it->second );
#ifdef MC__MINLPSLV_DEBUG
    std::cout << " " << it->second;
#endif
  }
#ifdef MC__MINLPSLV_DEBUG
  std::cout << " }" << std::endl;
#endif
  return branchset;
}
    
template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_set_options_sbbslv
()
{
  SBBSLV<T>::options.BRANCHING_USERFUNCTION = MINLPSLV<T,NLP,MIP>::_branch_subset;
  SBBSLV<T>::options.STOPPING_ABSTOL        = options.CVATOL;
  SBBSLV<T>::options.STOPPING_RELTOL        = options.CVRTOL;
  SBBSLV<T>::options.DISPLAY_LEVEL          = options.DISPLEVEL?2:0;
  SBBSLV<T>::options.MAX_NODES              = options.MAXITER;
  SBBSLV<T>::options.MAX_WALLTIME           = options.TIMELIMIT;
}

} // end namescape mc

#endif

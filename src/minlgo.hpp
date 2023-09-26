// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MINLGO Global Mixed-Integer Nonlinear Optimization using MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 1.0
\date 2020
\bug No known bugs.

Consider a mixed-integer nonlinear optimization problem (MINLP) in the form:
\f{align*}
\mathcal{P}:\quad & \min_{x_1,\ldots,x_n}\ f(x_1,\ldots,x_n)\\
& {\rm s.t.}\ \ g_j(x_1,\ldots,x_n)\ \leq,=,\geq\ 0,\ \ j=1,\ldots,m\\
& \qquad x_i^L\leq x_i\leq x_i^U,\ \ i=1,\ldots,n\\
& \qquad x_i \in \mathbb{Z},\ \ i\in I
\f}
where \f$f, g_1, \ldots, g_m\f$ are factorable, potentially nonlinear, real-valued functions; and \f$x_i, i=1\ldots n\f$ are either continuous (\f$i\notin I\f$) or binary/integer (\f$i\in I\f$) decision variables. The class mc::MINLGO solves seeks a certificate of global optimality for such problems using complete-search algorithms. One approach involves reformulating the MINLP into an equivalent problem for which existing global optimization technology can be used, such as mixed-integer quadratically constrained quadratic programming (MIQCQP) which is available in the commercial solver <A href="http://www.gurobi.com/">Gurobi</A>. Another approach entails the construction of a converging hierarchy of MIP or MIQCQP relaxations via the introduction of binary variables to approximate (some of) the nonconvexities. The combination of both approaches is also possible, for instance via the reformulation of polynomial or rational subexpressions as quadratic forms and piecewise-linear approximation of transcendental terms such as exp, log, sin, etc. The required reformulations and relaxations for the nonlinear / nonconvex participating functions are generated using various arithmetics in <A href="https://projects.coin-or.org/MCpp">MC++</A>.

\section sec_MINLGO_setup How do I setup my optimization model?

Consider the following NLP model:
\f{align*}
  \max_{\bf p}\ & p_1\,p_4\,(p_1+p_2+p_3)+p_3 \\
  \text{s.t.} \ & p_1\,p_2\,p_3\,p_4 \geq 25 \\
  & p_1^2+p_2^2+p_3^2+p_4^2 = 40 \\
  & 1 \leq p_1,p_2,p_3,p_4 \leq 5\,.
\f}

Start by instantiating an mc::MINLGO class object, which is defined in the header file <tt>minlgo.hpp</tt>:

\code
  mc::MINLGO MINLP;
\endcode

Next, set the variables and objective/constraint functions after creating a DAG of the model: 

\code
  mc::FFGraph DAG;
  const unsigned NP = 4; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, 1, 5, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_NLP::MIN, (P[0]*P[3])*(P[0]+P[1]+P[2])+P[2] ); // objective
  MINLP.add_ctr( mc::BASE_NLP::GE,  (P[0]*P[3])*P[1]*P[2]-25 );          // constraints
  MINLP.add_ctr( mc::BASE_NLP::EQ,  sqr(P[0])+sqr(P[1])+sqr(P[2])+sqr(P[3])-40 );
\endcode

Finally, set up the NLP model, presolve the model, then solve it using:

\code
  MINLP.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  MINLP.setup();
  MINLP.presolve();
  MINLP.optimize();
  MINLP.stats.display();
\endcode

The options can be modified through the public member mc::MINLPBND::options. The return value of mc::MINLGO is per the enumeration mc::MINLGO::STATUS. A computational breakdown may be obtained from the internal class <a>mc::MINLGO::Stats</a> or displayed using the method <a>mc::MINLGO::Stats::display</a>. The incumbent solution may be retrieved as an instance of <a>mc::SOLUTION_OPT</a> using the method <a>mc::MINLGO::incumbent</a>. In this instance, the following result is displayed:

\verbatim
#              |  VARIABLES      FUNCTIONS
# -------------+---------------------------
#  LINEAR      |         0              0
#  QUADRATIC   |         0              1
#  POLYNOMIAL  |         4              2
#  GENERAL     |         0              0
#
# GENERATING EXPRESSION TREES...

#  ITERATION     INCUMBENT    BEST BOUND    TIME
        0 r*  1.701402e+01  1.701402e+01      0s

#  TERMINATION AFTER 0 ITERATIONS: 0.008918 SEC
#  INCUMBENT VALUE:  1.701402e+01
#  INCUMBENT POINT:  1.000000e+00  4.743001e+00  3.821149e+00  1.379408e+00

#  ITERATION      INCUMBENT    BEST BOUND    TIME
        0  P*  1.701402e+01 -1.000000e+30      0s
        1   *  1.701402e+01  1.701401e+01      0s

#  TERMINATION AFTER 0 REFINEMENTS: 0.044984 SEC
#  INCUMBENT VALUE:  1.701402e+01
#  INCUMBENT POINT:  1.000000e+00  4.743001e+00  3.821149e+00  1.379408e+00
#  OPTIMALITY GAP:   2.60e-06 (ABS)
                     1.53e-07 (REL)


# WALL-CLOCK TIMES
# SETUP:              0.00 SEC
# PREPROCESSOR:       0.01 SEC
# LOCAL SOLVER:       0.01 SEC
# MIP SOLVER:         0.02 SEC
# TOTAL:              0.04 SEC
\endverbatim
*/

//TODO: 
//- Detect the class of problems to determine the need for refinements/iterations

#ifndef MC__MINLGO_HPP
#define MC__MINLGO_HPP

#include <filesystem>
#include <boost/program_options.hpp> 
namespace opt = boost::program_options; 

#include "interval.hpp"
#include "gamsio.hpp"
#include "minlpslv.hpp"
#include "nlpslv_snopt.hpp"
#include "mipslv_gurobi.hpp"
#include "minlpbnd.hpp"
#include "sbbslv.hpp"

namespace mc
{

//! @brief C++ class for global optimization of MINLP using complete search
////////////////////////////////////////////////////////////////////////
//! mc::MINLGO is a C++ class for global optimization of NLP and
//! MINLP using complete search. Relaxations for the nonlinear or
//! nonconvex participating terms are generated using MC++. Further
//! details can be found at: \ref page_MINLGO
////////////////////////////////////////////////////////////////////////
template < typename T=Interval,
           typename NLP=NLPSLV_SNOPT<>,
           typename MIP=MIPSLV_GUROBI<T>,
           typename... ExtOps >
class MINLGO
#if defined (MC__WITH_GAMS)
: protected virtual GAMSIO<ExtOps...>,
  protected SBBSLV<T>,
  public virtual BASE_NLP<ExtOps...>
#else
: protected SBBSLV<T>,
  public virtual BASE_NLP<ExtOps...>
#endif
{
protected:

  // Do not use BASE_AE<ExtOps...>::_dag since redefined locally
  using BASE_AE<ExtOps...>::_var;
  using BASE_AE<ExtOps...>::_vartyp;
  using BASE_AE<ExtOps...>::_varlb;
  using BASE_AE<ExtOps...>::_varlm;
  using BASE_AE<ExtOps...>::_varub;
  using BASE_AE<ExtOps...>::_varum;
  using BASE_AE<ExtOps...>::_dep;
  using BASE_AE<ExtOps...>::_deplb;
  using BASE_AE<ExtOps...>::_deplm;
  using BASE_AE<ExtOps...>::_depub;
  using BASE_AE<ExtOps...>::_depum;
  using BASE_AE<ExtOps...>::_sys;
  using BASE_AE<ExtOps...>::_sysm;
  using BASE_AE<ExtOps...>::_par;

  using BASE_NLP<ExtOps...>::_obj;
  using BASE_NLP<ExtOps...>::_ctr;
  using BASE_NLP<ExtOps...>::_nco;
  using BASE_NLP<ExtOps...>::_dag; // Make sure _dag is from BASE_NLP, not GAMSIO

#if defined (MC__WITH_GAMS)
  using GAMSIO<ExtOps...>::_varini;
#endif

public:

  using BASE_AE<ExtOps...>::set;
  using BASE_AE<ExtOps...>::dag;
  using BASE_AE<ExtOps...>::set_dag;
  using BASE_AE<ExtOps...>::par;
  using BASE_AE<ExtOps...>::set_par;
  using BASE_AE<ExtOps...>::add_par;
  using BASE_AE<ExtOps...>::reset_par;
  using BASE_AE<ExtOps...>::var;
  using BASE_AE<ExtOps...>::set_var;
  using BASE_AE<ExtOps...>::add_var;
  using BASE_AE<ExtOps...>::reset_var;
  using BASE_AE<ExtOps...>::update_vartyp;
  using BASE_AE<ExtOps...>::dep;
  using BASE_AE<ExtOps...>::set_dep;
  using BASE_AE<ExtOps...>::add_dep;
  using BASE_AE<ExtOps...>::reset_dep;
  using BASE_AE<ExtOps...>::sys;
  using BASE_AE<ExtOps...>::add_sys;
  using BASE_AE<ExtOps...>::reset_sys;

  using BASE_NLP<ExtOps...>::set;
  using BASE_NLP<ExtOps...>::set_obj;
  using BASE_NLP<ExtOps...>::add_ctr;

  //! @brief NLP solution status
  enum STATUS{
     SUCCESSFUL=0,      //!< MINLP global solution found within tolerances
     INFEASIBLE,        //!< MINLP is infeasible
     UNBOUNDED,         //!< MINLP has unbounded variables participating
     INTERRUPTED,       //!< MINLP algorithm was interrupted prior to convergence
     FAILED,            //!< MINLP algorithm encountered numerical difficulties
     ABORTED            //!< MINLP algorithm aborted after critical error
  };

  //! @brief MINLGO options
  class Options
  {
   public:
    //! @brief Constructor
    Options
      ();

    //! @brief Assignment operator
    Options& operator=
      ( Options const& other );

    //! @brief Global search strategy
    enum METHOD{
      PWR=0,	//!< Piecewise relaxation hierarchy
      SBB	//!< Spatial branch-and-bound
    };

    //! @brief Global search strategy
    int         STRATEGY;
    //! @brief Export GAMS model after preprocessing
    std::string GAMSEXPORT;
    //! @brief Level of preprocessing
    int         PRESOLVE;
    //! @brief Level of reformulation
    int         REFORM;
    //! @brief Search and append redundant polynomial cuts
    int         REDCUTS;
//    //! @brief Append NCO cuts
//    bool        NCOCUTS;
    //! @brief Correct the incumbent for feasibility using multipliers
    bool        CORRINC;
    //! @brief Initialize relaxed problem at incumbent point
    bool        INIINC;
    //! @brief Add a cut at incumbent value in relaxed problem
    bool        CUTINC;
    //! @brief Add a break-point at incumbent in relaxed problem
    bool        BKPTINC;
    //! @brief Feasibility tolerance 
    double      FEASTOL;
    //! @brief Convergence absolute tolerance
    double      CVATOL;
    //! @brief Convergence relative tolerance
    double      CVRTOL;
    //! @brief Maximum number of iterations in complete-search algorithim (0-no limit)
    unsigned    MAXITER;
    //! @brief Maximum run time (seconds)
    double      TIMELIMIT;
    //! @brief Overall display level
    int         DISPLEVEL;
    //! @brief Maximum run time for preprocessing (seconds)
    double      PRETIMELIMIT;

    //! @brief MINLP local solver options
    typename MINLPSLV<T,NLP,MIP,ExtOps...>::Options  MINLPSLV;
    //! @brief MINLP global bounder options
    typename MINLPBND<T,MIP,ExtOps...>::Options      MINLPBND;
    //! @brief MINLP global bounder options for presolve
    typename MINLPBND<T,MIP,ExtOps...>::Options      MINLPPRE;

    //! @brief Load option file
    bool read
      ( std::string const& optionfilename, std::ofstream& logfile,
        std::ostream& out=std::cout );
    bool read
      ( std::string const& optionfilename, std::ostream& out=std::cout );
    //! @brief user options
    opt::options_description const& user_options
      () const
      { return _USROPT; }
    //! @brief Display
    void display
      ( std::ostream& out=std::cout ) const;

   private:
    //! @brief Option description from file
    opt::options_description _USROPT;
    //! @brief Option map
    opt::variables_map _USRMAP; 
    //! @brief Log file
    std::string _LOGFILENAME;
    //! @brief Relaxation approach
    unsigned    _MINLPBND_ALLOW_NLIN;
    bool        _MINLPBND_ALLOW_DISJ;
    unsigned    _MINLPBND_QUADOPTIM;
  } options;

  //! @brief Class managing exceptions for MINLGO
  class Exceptions
  {
  public:
    //! @brief Enumeration type for MINLGO exception handling
    enum TYPE{
      SETUP,		//!< Incomplete setup before a solve
      STRATEGY,		//!< Invalid global search strategy
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
        return "MINLGO::Exceptions  Incomplete setup before a solve";
      case STRATEGY:
        return "MINLGO::Exceptions  Invalid global search strategy";
      case INTERN:
      default:
        return "MINLGO::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief Structure holding solve statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime_all = walltime_setup = walltime_preproc = walltime_slvloc = walltime_slvrel =
        std::chrono::microseconds(0); }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "# WALL-CLOCK TIMES" << std::endl
           << "# SETUP:        " << std::setw(10) << to_time( walltime_setup )   << " SEC" << std::endl
           << "# PREPROCESSOR: " << std::setw(10) << to_time( walltime_preproc ) << " SEC" << std::endl
           << "# LOCAL SOLVER: " << std::setw(10) << to_time( walltime_slvloc )  << " SEC" << std::endl
           << "# MIP SOLVER:   " << std::setw(10) << to_time( walltime_slvrel )  << " SEC" << std::endl
           << "# TOTAL:        " << std::setw(10) << to_time( walltime_all )     << " SEC" << std::endl
           << std::endl; }
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime_all;
    //! @brief Cumulated wall-clock time used for problem setup (in microseconds)
    std::chrono::microseconds walltime_setup;
    //! @brief Cumulated wall-clock time used by preprocessing (in microseconds)
    std::chrono::microseconds walltime_preproc;
    //! @brief Cumulated wall-clock time used by local solver (in microseconds)
    std::chrono::microseconds walltime_slvloc;
    //! @brief Cumulated wall-clock time used by relaxed solver (in microseconds)
    std::chrono::microseconds walltime_slvrel;
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

  //! @brief Current status
  STATUS                         _status;

  //! @brief Current iteration
  unsigned                       _iter;

  //! @brief Rounds of domain reduction
  unsigned                       _nred;

  //! @brief Flag for setup function
  bool                           _issetup;

  //! @brief Flag for presolve function
  bool                           _ispresolved;

  //! @brief Flag for MIP problem
  bool                           _ismip;

  //! @brief Flag for boundedness
  bool                           _isbnd;

  //! @brief objective scaling coefficient (1: min; -1: max)
  double                         _objscal;

  //! @brief Current relaxation value
  double                         _Zrel;

  //! @brief Current incumbent value
  double                         _Zinc;

  //! @brief Current incumbent correction
  double                         _Zcor;

  //! @brief Variable values at current relaxation
  std::vector<double>            _Xrel;

  //! @brief Structure holding incumbent information
  SOLUTION_OPT                   _incumbent;

  //! @brief Variable bounds
  std::vector<T>                 _Xbnd;
  
  //! @brief Decision variable bounds with integer fixing
  std::vector<T>                 _Xbndi;

  //! @brief Local solver for factorable MINLP
  MINLPSLV<T,NLP,MIP,ExtOps...>  _MINLPSLV;

  //! @brief Global bounder for factorable NLP
  MINLPBND<T,MIP,ExtOps...>      _MINLPBND;

  //! @brief Structure holding NLP intermediate solution
  SOLUTION_OPT                   _solution;

  //! @brief maximum number of values displayed in a row
  static const unsigned int      _LDISP = 4;

  //! @brief reserved space for integer variable display
  static const unsigned int      _IPREC = 9;

  //! @brief reserved space for double variable display
  static const unsigned int      _DPREC = 6;

  //! @brief stringstream for displaying results
  std::ostringstream             _odisp;

  //! @brief Time point to enable TIMELIMIT option
  std::chrono::time_point<std::chrono::system_clock> _tstart;

  //! @brief Set SBBSLV solver options
  void _set_options_sbbslv
    ();

  //! @brief Solve optimization model using piecewise relaxation hierarchy
  int _optimize_pwr
    ( std::ostream& os );

    //! @brief Test whether a variable vector is integer feasible
  bool _is_integer_feasible
    ( double const* Xval, double const& feastol )
    const;

  //! @brief Initialize display
  void _display_init
    ( std::ostream& os=std::cout );
    
  //! @brief Final display
  void _display_final
    ( std::chrono::microseconds const& walltime,
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

  //! @brief Add wall-time to display
  void _display_time
    ();

  //! @brief Display current buffer stream and reset it
  void _display_flush
    ( std::ostream& os );

  //! @brief Finalize optimization display and status
  int _finalize
    ( STATUS const status, std::ostream& os );

  //! @brief Test feasibility
  bool _test_feasible
    ( double const* Xini, std::ostream& os );

  //! @brief Solve local NLP subproblem
  bool _solve_local
    ( double const* Xini, T const* Xbnd, bool const mstart, std::ostream& os );

  //! @brief Bound reduction subproblem
  int _reduce_bounds
    ( T* Xbnd, double const* Zinc, bool const reset, bool const reinit, std::ostream& os );

  //! @brief Solve relaxed MIP subproblem   
  int _solve_relax
    ( T const* Xbnd, double const* Zinc, double const* pinc, bool const reinit, std::ostream& os );
    //( std::ostream& os );
      
  //! @brief Export relaxed MIP subproblem   
  bool _export_relax
    ( std::ostream& os );
      
  //! @brief Convergence test for piecewise-linear relaxation approach 
  bool _converged
    ()
    const;
    
  //! @brief Termination test for piecewise-linear relaxation approach 
  bool _interrupted
    ()
    const;

  //! @brief User-function to subproblems in SBB
  typename SBBSLV<T>::STATUS subproblems
    ( typename SBBSLV<T>::TASK const task, SBBNode<T>* node,
      std::vector<double>& p, double& f, double const& INC, std::ostream& os );

public:

  //! @brief Constructor
  MINLGO()
    : _issetup(false)
    { stats.reset(); }

  //! @brief Destructor
  virtual ~MINLGO()
    {}

  //! @brief Status after last NLP call
  STATUS get_status
    ()
    const
    { return _status; }

  //! @brief Load optimization model from GAMS file
#if defined (MC__WITH_GAMS)
  bool read
    ( std::string const& filename, bool const init=false, bool const disp=false );
#endif

  //! @brief Setup DAG for cost and constraint evaluation
  void setup
    ();

  //! @brief Preprocess optimization model - return value is false if model is provably infeasible
  int presolve
    (  T* Xbnd=nullptr, double* Xini=nullptr, std::ostream& os=std::cout );

  //! @brief Export relaxed optimization model to GAMS after preprocessing
  bool GAMSexport
    ( bool const relax=false, std::ostream& os=std::cout );

  //! @brief Solve optimization model to global optimality after preprocessing
  int optimize
    ( std::ostream& os=std::cout );

  //! @brief Get incumbent info
  SOLUTION_OPT const& get_incumbent
    () 
    const
    { return _incumbent; }

  //! @brief Interrupt solve process
  void interrupt
    ()
    { _MINLPSLV.master_solver().terminate();
      _MINLPBND.relax_solver()->terminate(); }

private:
  //! @brief Private methods to block default compiler methods
  MINLGO
    ( MINLGO<T,NLP,MIP,ExtOps...> const& ) =delete;
  MINLGO<T,NLP,MIP,ExtOps...>& operator=
    ( MINLGO<T,NLP,MIP,ExtOps...> const& ) =delete;
};

#if defined (MC__WITH_GAMS)
template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::read
( std::string const& filename, bool const init, bool const disp )
{
  _tstart = stats.start();

  bool flag = this->GAMSIO<ExtOps...>::read( filename, init, disp );

  stats.walltime_setup += stats.walltime( _tstart );
  stats.walltime_all   += stats.walltime( _tstart );
  return flag;
}
#endif

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::setup
()
{
  //stats.reset();
  _tstart = stats.start();

  _ismip = false;
  for( auto const& typ : _vartyp ){
    if( !typ ) continue;
    _ismip = true;
    break;
  }

  assert( !std::get<0>(_obj).empty() );
  switch( std::get<0>(_obj)[0] ){
    case BASE_OPT::MIN: _objscal =  1e0; break;
    case BASE_OPT::MAX: _objscal = -1e0; break;
  }

#ifdef MC__MINLGO_SETUP_DEBUG
  std::cout << "MINLPBND set-up" << std::endl;
#endif
  _MINLPBND.options = options.MINLPBND;
  _MINLPBND.set( *this );
  _MINLPBND.setup();
  _Xbnd.clear();
  _Xbndi.clear();
  
#ifdef MC__MINLGO_SETUP_DEBUG
  std::cout << "MINLPSLV set-up" << std::endl;
#endif
  _MINLPSLV.options = options.MINLPSLV;
  _MINLPSLV.set( *this );
  _MINLPSLV.setup();
  _incumbent.reset();

  _issetup = true;

  stats.walltime_setup += stats.walltime( _tstart );
  stats.walltime_all   += stats.walltime( _tstart );
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::_is_integer_feasible
( double const* Xval, double const& feastol )
const
{
  // Check integer feasibility
  for( unsigned i=0; i<_var.size(); i++ ){
    if( !_vartyp[i] ) continue;
    if( std::fabs( Xval[i] - std::round(Xval[i]) ) > feastol )
      return false;
  }
  return true;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline int
MINLGO<T,NLP,MIP,ExtOps...>::presolve
(  T* Xbnd, double* Xini, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _ispresolved = false;
  _tstart = stats.start();
  if( options.PRETIMELIMIT > options.TIMELIMIT ) options.PRETIMELIMIT = options.TIMELIMIT; 

  // Reset incumbent
  _incumbent.reset();
  _Zcor =  0.;
  _Zinc =  _objscal * BASE_OPT::INF;
  _Zrel = -_objscal * BASE_OPT::INF;
  _Xrel.resize( _var.size() );

  // User-supplied initial point
  if( Xini )
    _varini.assign( Xini, Xini+_var.size() );
#ifdef MC__MINLGO_PREPROCESS_DEBUG
  std::cout << std::scientific << std::setprecision(4);
  std::cout << "Initial point\n@";
  for( auto const& Xi : _varini ) std::cout << " " << Xi;
  std::cout << std::endl;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

  // User-supplied variable bounds
  if( Xbnd )
    _Xbnd.assign( Xbnd, Xbnd+_var.size() );
  else
    _Xbnd.assign( _var.size(), T( -BASE_OPT::INF, BASE_OPT::INF ) );

  // Check feasibility of user-supplied point
  if( _MINLPSLV.is_feasible( _varini.data(), options.CORRINC? 0.: options.FEASTOL ) ){
    _incumbent = _MINLPSLV.local_solver().solution();
    _Zinc = _MINLPSLV.local_solver().solution().f[0];
  }

  // Apply bounder reformulations
  _MINLPBND.options = options.MINLPPRE;
  _MINLPBND.options.TIMELIMIT = options.PRETIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( options.DISPLEVEL > 1 )
    _MINLPBND.options.SQUAD.MIPDISPLEVEL = _MINLPBND.options.SRED.MIPDISPLEVEL = 1;

  if( options.REFORM ){
    _MINLPBND.lift_polynomial_subexpressions( true );
    _MINLPBND.flatten_linear_functions( true );
    if( options.REFORM > 1 )
      _MINLPBND.quadratize_polynomial_functions( true );
    else{
      _MINLPBND.flatten_quadratic_functions( true );
      _MINLPBND.flatten_polynomial_functions( true );
    }
  }

  // Search for redundant constraints
  _MINLPBND.options.TIMELIMIT = options.PRETIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( options.REDCUTS )
    _MINLPBND.append_reduction_constraints( true );

  // Simple bound propagation if presolve is turned off
  if( !options.PRESOLVE ){
    // Set presolve bounder options
    //_MINLPBND.options = options.MINLPPRE;
    _MINLPBND.options.TIMELIMIT = options.PRETIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );

    // Apply domain contraction for both linear and nonlinear constraints
    if( !_MINLPBND.propagate_bounds( _Xbnd.data(), !_incumbent.x.empty()? &_Zinc: nullptr, true ) ){
      stats.walltime_preproc += stats.walltime( _tstart );
      stats.walltime_all     += stats.walltime( _tstart );
      return STATUS::INFEASIBLE;
    }
    _Xbnd = _MINLPBND.variable_bounds();
    for( unsigned i=0; Xbnd && i<_var.size(); i++ )
      Xbnd[i] = _Xbnd[i];

#ifdef MC__MINLGO_PREPROCESS_DEBUG
    std::cout << "Reduced bounds:" << std::endl;
    for( auto const& Xi : _Xbnd ) std::cout << " " << Xi;
    std::cout << std::endl;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

    // Check boundedness of general variables
    _isbnd = _MINLPBND.bounded_domain( BASE_OPT::INF/10, FFDep::N );
  
    stats.walltime_preproc += stats.walltime( _tstart );
    stats.walltime_all     += stats.walltime( _tstart );
    _ispresolved = true;
    return( _isbnd? STATUS::SUCCESSFUL: STATUS::UNBOUNDED );
  }

  // Apply MINLP feasibility pump
  _MINLPSLV.options = options.MINLPSLV;
  _MINLPSLV.options.TIMELIMIT = options.PRETIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( options.DISPLEVEL )
    os << "# PERFORMING LOCAL SEARCH" << std::endl;
  _MINLPSLV.optimize( _varini.data(), _Xbnd.data(), _MINLPSLV.nearest, os );
  if( _MINLPSLV.is_feasible( options.FEASTOL ) ){
    _Zcor = options.CORRINC? _MINLPSLV.cost_correction(): 0.;
    if( _objscal*(_MINLPSLV.get_incumbent().f[0]+_Zcor) < _objscal*_Zinc ){
      _incumbent = _MINLPSLV.get_incumbent();
      _Zinc = _MINLPSLV.get_incumbent().f[0] + _Zcor;
    }
  }

#ifdef MC__MINLGO_PREPROCESS_DEBUG
  if( _incumbent.x.empty() )
    std::cout << "Incumbent found: -" << std::endl;
  else{
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Incumbent found: " << _Zinc << std::endl
              << "@";
    for( auto const& Xi : _incumbent.x ) std::cout << " " << Xi;
    std::cout << _incumbent << std::endl;
  }
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  
  // Apply domain contraction for both linear and nonlinear constraints
  //_MINLPBND.options = options.MINLPPRE;
  _MINLPBND.options.TIMELIMIT = options.PRETIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( options.DISPLEVEL )
    os << "# PERFORMING DOMAIN CONTRACTION" << std::endl;
  switch( _MINLPBND.reduce_bounds( _nred, _Xbnd.data(), !_incumbent.x.empty()? &_Zinc: nullptr, true, true ) ){
    case MIP::STATUS::INFEASIBLE:
      stats.walltime_preproc += stats.walltime( _tstart );
      stats.walltime_all     += stats.walltime( _tstart );
      _ispresolved = true;
      return STATUS::INFEASIBLE;
    default:
      break;
  }
  _Xbnd = _MINLPBND.variable_bounds();
  for( unsigned i=0; Xbnd && i<_var.size(); i++ )
    Xbnd[i] = _Xbnd[i];

#ifdef MC__MINLGO_PREPROCESS_DEBUG
  std::cout << "Reduced bounds:" << std::endl;
  for( auto const& Xi : _Xbnd ) std::cout << " " << Xi;
  std::cout << std::endl;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

  // Compute relaxation -- COULD EXIT HERE, BENEFIT OF EXTRA FEASIBILITY PUMP?!?
  if( options.PRESOLVE > 1 ){
    _MINLPBND.options.TIMELIMIT = options.PRETIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
    _MINLPBND.relax_model( nullptr, nullptr, nullptr, 0, false, false, "", os ); // reset cuts but not variables or bounds 
    switch( _MINLPBND.relax_solver()->get_status() ){
      case MIP::STATUS::OPTIMAL:
      case MIP::STATUS::SUBOPTIMAL:
        _Zrel = _MINLPBND.relax_solver()->get_objective_bound();
        for( unsigned i=0; i<_var.size(); i++ )
          _Xrel[i] = _MINLPBND.relax_solver()->get_variable( _var[i] );
        break;
      case MIP::STATUS::INFEASIBLE:
      case MIP::STATUS::INFORUNBND:
        stats.walltime_preproc += stats.walltime( _tstart );
        stats.walltime_all     += stats.walltime( _tstart );
        _ispresolved = true;
        return STATUS::INFEASIBLE;
      case MIP::STATUS::UNBOUNDED:
        stats.walltime_preproc += stats.walltime( _tstart );
        stats.walltime_all     += stats.walltime( _tstart );
        _ispresolved = true;
        return STATUS::UNBOUNDED;
      case MIP::STATUS::TIMELIMIT:
        stats.walltime_preproc += stats.walltime( _tstart );
        stats.walltime_all     += stats.walltime( _tstart );
        _ispresolved = true;
        return STATUS::INTERRUPTED;
      case MIP::STATUS::OTHER:
      default:
        stats.walltime_preproc += stats.walltime( _tstart );
        stats.walltime_all     += stats.walltime( _tstart );
        _ispresolved = true;
        return STATUS::FAILED;
    }

#ifdef MC__MINLGO_PREPROCESS_DEBUG
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Relaxation bound: " << _Zrel << std::endl
              << "@";
    for( unsigned i=0; i<_var.size(); i++ ) std::cout << " " << _Xrel[i];
    std::cout << std::endl;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

    // Apply MINLP feasibility pump from relaxation optimum and using tightened bounds
    if( _varini.empty() )
      _varini.resize( _var.size() );
    for( unsigned i=0; i<_var.size(); i++ )
      _varini[i] = _MINLPBND.relax_solver()->get_variable( _var[i] );
    _MINLPSLV.options.TIMELIMIT = options.PRETIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
    _MINLPSLV.optimize( _varini.data(), _Xbnd.data(), _MINLPSLV.nearest, os );
    if( _MINLPSLV.is_feasible( options.FEASTOL ) ){
      _Zcor = options.CORRINC? _MINLPSLV.cost_correction(): 0.;
      if( _objscal*(_MINLPSLV.get_incumbent().f[0]+_Zcor) < _objscal*_Zinc ){
        _incumbent = _MINLPSLV.get_incumbent();
        _Zinc = _MINLPSLV.get_incumbent().f[0] + _Zcor;
      }
    }

#ifdef MC__MINLGO_PREPROCESS_DEBUG
    if( _incumbent.x.empty() )
      std::cout << "Incumbent found: -" << std::endl;
    else{
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Incumbent found: " << _Zinc << std::endl
                << "@";
      for( auto const& Xi : _incumbent.x ) std::cout << " " << Xi;
      std::cout << std::endl;
    }
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }
  
  // Check boundedness of general variables
  _isbnd = _MINLPBND.bounded_domain( BASE_OPT::INF/10, FFDep::N );
  
  stats.walltime_preproc += stats.walltime( _tstart );
  stats.walltime_all     += stats.walltime( _tstart );
  _ispresolved = true;
  return( _isbnd? STATUS::SUCCESSFUL: STATUS::UNBOUNDED );
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::GAMSexport
( bool const relax, std::ostream& os )
{
  if( !_issetup || !_ispresolved ) throw Exceptions( Exceptions::SETUP );
  //if( !_isbnd ) return _finalize( STATUS::UNBOUNDED, os );

  // Check GAMS export filename
  std::string extfile = std::filesystem::path(options.GAMSEXPORT).extension();
  if( extfile != ".gms" ) return false;

  // Export relaxed preprocessed model
  if( relax ){
    if( options.DISPLEVEL )
      os << "# EXPORTING RELAXED PREPROCESSED MODEL TO GAMS" << std::endl;
    return _export_relax( os );
  }

  // Export preprocessed model
  if( options.DISPLEVEL )
    os << "# EXPORTING PREPROCESSED MODEL TO GAMS" << std::endl;
  return _MINLPBND.export_model( options.GAMSEXPORT, options.INIINC? _incumbent.x.data(): nullptr, os );
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline int
MINLGO<T,NLP,MIP,ExtOps...>::optimize
( std::ostream& os )
{
  if( !_issetup || !_ispresolved ) throw Exceptions( Exceptions::SETUP );
  //if( !_isbnd ) return _finalize( STATUS::UNBOUNDED, os );
  if( options.DISPLEVEL )
    os << "# PERFORMING GLOBAL SEARCH" << std::endl;

  // Initialize solve
  _tstart = stats.start();
  _iter = 0;
  _nred = 0;
  
  // Set bounder options
  _MINLPBND.options = options.MINLPBND;
  if( options.DISPLEVEL > 1 )
    _MINLPBND.options.MIPSLV.DISPLEVEL = 1;

  // Search strategy
  int flag = 0;
  switch( options.STRATEGY ){
    case Options::PWR:
      flag = _optimize_pwr( os );
      break;

    case Options::SBB:
      _set_options_sbbslv();        // setting SBBSLV solver options
      _Xbndi.resize( _var.size() ); // storing bounds for local NLP solver
      _MINLPBND.init_polrelax();    // reinitialising polyhedral relaxation for relaxed MIP solver
      flag = SBBSLV<T>::solve( std::get<0>(_obj)[0], _var.size(), _Xbnd.data(),
                               _incumbent.x.data(), !_incumbent.x.empty()? &_Zinc: nullptr,
                               _vartyp.data(), std::set<unsigned>(), os );
      stats.walltime_all += stats.walltime( _tstart );
      break;

    default:
      throw Exceptions( Exceptions::STRATEGY );
  }
  
  return flag;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline typename SBBSLV<T>::STATUS
MINLGO<T,NLP,MIP,ExtOps...>::subproblems
( typename SBBSLV<T>::TASK const task, SBBNode<T>* node,
  std::vector<double>& p, double& f, double const& INC, std::ostream& os )
{
  typename SBBSLV<T>::STATUS status = SBBSLV<T>::FATAL;

  //if( SBBSLV<T>::_node_index == 207 )
  //  std::cout << "iteration 207\n";

  // Compute local solution
  if( (task == SBBSLV<T>::UPPERBD && _objscal > 0.) 
   || (task == SBBSLV<T>::LOWERBD && _objscal < 0.) ){

    // Solve local NLP model (integer variable bounds fixed to relaxed solution if MIP)
    for( unsigned i=0; i<_var.size(); i++ ){
      if( _vartyp[i] ) _Xbndi[i] = p[i];
      else             _Xbndi[i] = node->P(i);
    }
    try{
      if( _solve_local( p.data(), _Xbndi.data(), false, os ) ){
        f = _solution.f[0] + _Zcor;
        p = _solution.x;
#ifdef MC__MINLGO_DEBUG_SBB
        std::cout << "Local solution: " << f << std::endl;
#endif
        status = SBBSLV<T>::NORMAL;
      }
      else
        status = SBBSLV<T>::FAILURE;
    }
    catch(...){
     status = SBBSLV<T>::FAILURE;
    }
  }

  // compute relaxed solution
  else if( (task == SBBSLV<T>::UPPERBD && _objscal < 0.) 
        || (task == SBBSLV<T>::LOWERBD && _objscal > 0.) ){
    try{
#ifdef MC__MINLGO_DEBUG_SBB
      std::cout << "Initial box: " << std::endl;
      for( auto const& Pi : node->P() ) std::cout << " " << Pi;
      std::cout << std::endl;
      //{ int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
      // Not testing for infeasibility here, because contraction problem may
      // become infeasible due to round-off in LP solver
      _reduce_bounds( node->P().data(), &INC, true, false, os );
      //if( _reduce_bounds( node->P().data(), &INC, true, false, os ) == MIP::INFEASIBLE )
      //  return SBBSLV<T>::INFEASIBLE;
#ifdef MC__MINLGO_DEBUG_SBB
      std::cout << "Reduced box (" << _nred << "):" << std::endl;
      for( auto const& Pi : node->P() ) std::cout << " " << Pi;
      std::cout << std::endl;
      //{ int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

      // Setup and solve relaxed MINLP model
      switch( _solve_relax( node->P().data(), &INC, p.data(), false, os ) ){
        case MIP::OPTIMAL:
        case MIP::SUBOPTIMAL:
          f = _MINLPBND.relax_solver()->get_objective_bound();
          for( unsigned i=0; i<_var.size(); i++ )
            p[i] = _MINLPBND.relax_solver()->get_variable( _var[i] );
#ifdef MC__MINLGO_DEBUG_SBB
          std::cout << "Relaxed bound: " << f << std::endl;
#endif
          status = SBBSLV<T>::NORMAL;
          break;
        case MIP::INFEASIBLE:
          f = _objscal * BASE_OPT::INF;
          status = SBBSLV<T>::INFEASIBLE;
          break;
        case MIP::UNBOUNDED:
        case MIP::TIMELIMIT:
        default:
          status = SBBSLV<T>::FAILURE;
          break;
      }
    }
    catch(...){
      status = SBBSLV<T>::FAILURE;
    }
  }

  // assess feasibility
  else if( task == SBBSLV<T>::FEASTEST ){
    if( _test_feasible( p.data(), os ) )
      status = SBBSLV<T>::NORMAL;
    else
      status = SBBSLV<T>::INFEASIBLE;
  }

  // perform preprocessing/postprocessing
  else if( task == SBBSLV<T>::PREPROC
        || task == SBBSLV<T>::POSTPROC )
    status = SBBSLV<T>::NORMAL;

  // other
  else
    status = SBBSLV<T>::FATAL;

  return status;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline int
MINLGO<T,NLP,MIP,ExtOps...>::_optimize_pwr
( std::ostream& os )
{
  // Display presolve results
  _display_init( os );
  _display_add( _iter );
  std::ostringstream oflag;
  oflag << "P";
  if( !_incumbent.x.empty() ) oflag << "*";
  _display_add( oflag.str() );
  _display_add( _Zinc );
  _display_add( _Zrel );
  _display_time();
  _display_flush( os );

  // Iterative relaxation solution and refinement
  _Xbndi.resize( _var.size() ); // storing bounds for local NLP solver
  bool locfeas = false, updinc = false;
  for( ++_iter; options.MAXITER; ++_iter ){

    // Set-up and solve MIP relaxation
    switch( _solve_relax( _Xbnd.data(), options.CUTINC && !_incumbent.x.empty()? &_Zinc: nullptr,
                          options.INIINC? _incumbent.x.data(): nullptr, _iter>1? false: true, os ) ){
      case MIP::OPTIMAL:
        break;
      case MIP::INFEASIBLE:
        _Zrel = _objscal * BASE_OPT::INF;
        return _finalize( STATUS::INFEASIBLE, os );
      case MIP::UNBOUNDED:
        return _finalize( STATUS::UNBOUNDED, os );
      case MIP::TIMELIMIT:
        _Zrel = _MINLPBND.relax_solver()->get_objective_bound();
        return _finalize( STATUS::INTERRUPTED, os );
      default:
        return _finalize( STATUS::FAILED, os );
    }

    // Retrieve MIP solution - use bound on objective, not incumbent!
    _Zrel = _MINLPBND.relax_solver()->get_objective_bound();
    for( unsigned i=0; i<_var.size(); i++ )
      _Xrel[i] = _MINLPBND.relax_solver()->get_variable( _var[i] );
#ifdef MC__MINLGO_DEBUG
    std::cout << "_Zrel = " << _Zrel << std::endl;
    for( unsigned i=0; i<_var.size(); i++ )
      std::cout << "_Xrel[" << i << "] = " << _Xrel[i] << std::endl;
#endif

    // Solve local NLP model (integer variable bounds fixed to relaxed solution if MIP)
    for( unsigned i=0; i<_var.size(); i++ ){
      if( _vartyp[i] ) _Xbndi[i] = _Xrel[i];
      else             _Xbndi[i] = _Xbnd[i];
    }
    locfeas = _solve_local( _Xrel.data(), _Xbndi.data(), false, os );

    // Update incumbent
    updinc = false;
    if( locfeas && _objscal*_Zinc > _objscal*(_solution.f[0]+_Zcor) ){
      updinc = true;
      _Zinc = _solution.f[0] + _Zcor;
      _incumbent = _solution;
    }

    // Intermediate display
    _display_add( _iter );
    std::ostringstream oflag;
    if( _nred )    oflag << "R" << _nred;
    if( updinc )   oflag << "*";
    if( !locfeas ) oflag << "i";
    _display_add( oflag.str() );
    _display_add( _Zinc );
    _display_add( _Zrel );
    _display_time();
    _display_flush( os );

    // Termination tests
    if( _converged() )
      break;
    if( _interrupted() )
      return _finalize( STATUS::INTERRUPTED, os );

    // Refine relaxation via additional breakpoints
    _MINLPBND.refine_polrelax( options.BKPTINC && updinc? _incumbent.x.data(): nullptr );

    // Apply domain contraction
    // Do NOT test for infeasibility here, because contraction problem may
    // become infeasible due to round-off in LP solver
    auto tMIP = stats.start();
    _MINLPBND.reduce_bounds( _nred, _Xbnd.data(), !_incumbent.x.empty()? &_Zinc: nullptr, false, false );
    stats.walltime_slvrel += stats.walltime( tMIP );
#ifdef MC__MINLGO_DEBUG
    std::cout << "Reduced bounds: x" << _nred << std::endl;
    for( auto const& Xi : _Xbnd ) std::cout << " " << Xi;
    std::cout << std::endl;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }

  return _finalize( STATUS::SUCCESSFUL, os );
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::_test_feasible
( double const* Xini, std::ostream& os )
{
  auto tNLP = stats.start();
  auto& _NLPSLV = _MINLPSLV.local_solver();
  _NLPSLV.restore_model();
  bool flag = _NLPSLV.is_feasible( Xini, options.FEASTOL );
  stats.walltime_slvloc += stats.walltime( tNLP );
  return flag;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::_solve_local
( double const* Xini, T const* Xbnd, bool const mstart, std::ostream& os )
{
  auto tNLP = stats.start();
  auto& _NLPSLV = _MINLPSLV.local_solver();
  _NLPSLV.restore_model();
      
  // Local solve from provided initial point
  _solution.reset();
  _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( _NLPSLV.options.TIMELIMIT > 0 ){
    _NLPSLV.solve( Xini, Xbnd );
    if( _NLPSLV.is_feasible( options.FEASTOL ) )
      _solution = _NLPSLV.solution();
  }
  
  // Extra local solves from random starting points
  _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  bool dombnd = true;
  for( unsigned i=0; dombnd && i<_var.size(); i++ ){
    if( Op<T>::diam(Xbnd[i]) < BASE_OPT::INF/10 ) continue;
    dombnd = false;
  }
  if( mstart && dombnd && options.MINLPSLV.MSLOC > 1 && _NLPSLV.options.TIMELIMIT > 0 ){
    _NLPSLV.solve( options.MINLPSLV.MSLOC-1, Xbnd );
    if( _NLPSLV.is_feasible( options.FEASTOL )
     && (_solution.x.empty() || _objscal*_NLPSLV.solution().f[0] < _objscal*_solution.f[0]) )
      _solution = _NLPSLV.solution();
  }

  // Compute correction
  _Zcor = 0.;
  if( !_solution.x.empty() && options.CORRINC )
    _Zcor = _NLPSLV.cost_correction();

  stats.walltime_slvloc += stats.walltime( tNLP );
  return !_solution.x.empty();
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::_export_relax
( std::ostream& os )
{
  // Check GAMS export filename
  std::string extfile = std::filesystem::path(options.GAMSEXPORT).extension();
  if( extfile != ".gms" ) return false;
  
  // Call master MIP problem - do NOT reset bounds, otherwise reinitializing lifted variable bounds
  int flag = _MINLPBND.relax_model( _Xbnd.data(), options.CUTINC && !_incumbent.x.empty()? &_Zinc: nullptr,
                                    options.INIINC? _incumbent.x.data(): nullptr, 0, false, _iter>1? false: true,
                                    options.GAMSEXPORT, os );

  return( flag==MIP::OTHER? true: false );
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline int
MINLGO<T,NLP,MIP,ExtOps...>::_solve_relax
( T const* Xbnd, double const* Zinc, double const* pinc, bool const reinit, std::ostream& os )
//( std::ostream& os )
{
  // Update time limit
  _MINLPBND.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( _MINLPBND.options.TIMELIMIT <= 0 )
    return MIP::TIMELIMIT;
  
  // Solve master MIP problem - do NOT reset bounds, otherwise reinitializing lifted variable bounds
  auto tMIP = stats.start();
  int flag = _MINLPBND.relax_model( Xbnd, Zinc, pinc, 0, false, reinit, "", os );
  //int flag = _MINLPBND.relax_model( _Xbnd.data(), options.CUTINC && !_incumbent.x.empty()? &_Zinc: nullptr,
  //                                  options.INIINC? _incumbent.x.data(): nullptr, 0, false, _iter>1? false: true,
  //                                  "", os );
  stats.walltime_slvrel += stats.walltime( tMIP );

  return flag;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline int
MINLGO<T,NLP,MIP,ExtOps...>::_reduce_bounds
( T* Xbnd, double const* Zinc, bool const reset, bool const reinit, std::ostream& os )
{
  // Update time limit
  _MINLPBND.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( _MINLPBND.options.TIMELIMIT <= 0 )
    return MIP::TIMELIMIT;
  
  // Solve master MIP problem - do NOT reset bounds, otherwise reinitializing lifted variable bounds
  auto tMIP = stats.start();
  int flag = _MINLPBND.reduce_bounds( _nred, Xbnd, Zinc, reset, reinit );
  stats.walltime_slvrel += stats.walltime( tMIP );

  return flag;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::_converged
()
const
{
  if( _incumbent.x.empty() )
    return false;
  if( _objscal*_Zrel >= _objscal*_Zinc
   || std::fabs( _Zinc - _Zrel ) <= options.CVATOL 
   || std::fabs( _Zinc - _Zrel ) <= 0.5 * options.CVRTOL * std::fabs( _Zinc + _Zrel ) )
    return true;
  return false;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::_interrupted
()
const
{
  if( stats.to_time( stats.walltime_all + stats.walltime( _tstart ) ) > options.TIMELIMIT
   || ( options.MAXITER && _iter >= options.MAXITER ) )
//   || ( _iter && _MINLPBND.problem_class() <= FFDep::Q ) )
    return true;
  return false;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline int
MINLGO<T,NLP,MIP,ExtOps...>::_finalize
( STATUS const status, std::ostream& os )
{
  _status = status;
  stats.walltime_all += stats.walltime( _tstart );
  _display_final( stats.walltime_all, os );
  return _status;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_display_init
( std::ostream& os )
{
  _odisp.str("");
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::endl
         << "#  " << std::right
  	 << std::setw(_IPREC) << "ITERATION "
  	 << std::setw(_DPREC+8) << "INCUMBENT"
  	 << std::setw(_DPREC+8) << "BEST BOUND"
  	 << std::setw(8) << "TIME";
  _display_flush( os ); 
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_display_final
( std::chrono::microseconds const& walltime,
  std::ostream& os )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::endl << "#  TERMINATION AFTER " << (_iter?_iter-1:0) << " REFINEMENTS: "
         << std::fixed << std::setprecision(3) << walltime.count()*1e-6 << " SEC"
         << std::endl;

  // No feasible solution found
  if( _incumbent.x.empty() )
    _odisp << "#  NO FEASIBLE SOLUTION FOUND" << std::endl;

  // Feasible solution found
  else{
    // Incumbent
    _odisp << "#  INCUMBENT VALUE:" << std::scientific
           << std::setprecision(_DPREC) << std::setw(_DPREC+8) << _Zinc
           << std::endl;
    _odisp << "#  INCUMBENT POINT:";
    unsigned i(0);
    for( auto const& Xi : _incumbent.x ){
      if( i++ == _LDISP ){
        _odisp << std::endl << std::left << std::setw(19) << "#";
        i = 1;
      }
      _odisp << std::right << std::setw(_DPREC+8) << Xi;
    }
    _odisp << std::endl;
    _odisp << "#  OPTIMALITY GAP:   " << std::scientific << std::setprecision(2)
           << std::fabs( _Zinc - _Zrel ) << " (ABS)" << std::endl
           << "                     "
           << 2. * std::fabs( _Zinc - _Zrel )
                 / std::fabs( _Zinc + _Zrel ) << " (REL)"  << std::endl;
  }

  _display_flush( os );
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_display_add
( const double dval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::scientific << std::setprecision(_DPREC)
         << std::setw(_DPREC+8) << dval;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_display_add
( const unsigned ival )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(_IPREC) << ival;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_display_add
( const std::string &sval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(4) << sval;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_display_time
()
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::fixed << std::setprecision(0) << std::setw(7) 
         << stats.to_time( stats.walltime_all + stats.walltime( _tstart ) ) << "s";
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_display_flush
( std::ostream &os )
{
  if( _odisp.str() == "" ) return;
  os << _odisp.str() << std::endl;
  _odisp.str("");
  return;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::_set_options_sbbslv
()
{
  SBBSLV<T>::options.STOPPING_ABSTOL = options.CVATOL;
  SBBSLV<T>::options.STOPPING_RELTOL = options.CVRTOL;
  SBBSLV<T>::options.DISPLAY_LEVEL   = options.DISPLEVEL?2:0;
  SBBSLV<T>::options.MAX_NODES       = options.MAXITER;
  SBBSLV<T>::options.MAX_WALLTIME    = options.TIMELIMIT;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline
MINLGO<T,NLP,MIP,ExtOps...>::Options::Options()
: STRATEGY( PWR ),
  GAMSEXPORT( "" ),
  PRESOLVE( 1 ),
  REFORM( 2 ),
  REDCUTS( 0 ),
  CORRINC( 1 ),
  INIINC( 1 ),
  CUTINC( 0 ),
  BKPTINC( 0 ),
  FEASTOL( 1e-5 ),
  CVATOL( 1e-5 ),
  CVRTOL( 1e-3 ),
  MAXITER( 1 ),
  TIMELIMIT( 72e2 ),
  DISPLEVEL( 1 ),
  PRETIMELIMIT( 72e2 ),
  MINLPSLV(),
  MINLPBND(),
  MINLPPRE(),
  _USROPT( "User-defined solver options" )
{
  MINLPPRE.LINCTRSEP              = MINLPBND.LINCTRSEP              = 1;
  MINLPPRE.RELAXMETH                                                = { MINLPBND.DRL };
  MINLPBND.RELAXMETH                                                = { MINLPBND.DRL };
  MINLPPRE.SUBSETDRL              = MINLPPRE.SUBSETSCM              = 0;
  MINLPBND.SUBSETDRL              = MINLPBND.SUBSETSCM              = 0;
  MINLPPRE.POLIMG.ALLOW_QUAD                                        = 0;
  MINLPPRE.POLIMG.ALLOW_NLIN                                        = {};
  MINLPPRE.POLIMG.ALLOW_DISJ                                        = {};
  MINLPBND.POLIMG.ALLOW_QUAD                                        = 1;
  MINLPBND.POLIMG.ALLOW_NLIN                                        = { FFOp::IPOW, FFOp::DPOW, FFOp::CHEB, FFOp::SQRT, FFOp::EXP,  FFOp::LOG,
                                                                        FFOp::COS,  FFOp::SIN,  FFOp::TAN,  FFOp::ACOS, FFOp::ASIN, FFOp::ATAN, 
                                                                        FFOp::TANH }; //FFOp::FABS, FFOp::MINF, FFOp::MAXF };
  MINLPBND.POLIMG.ALLOW_DISJ                                        = { FFOp::FABS, FFOp::FSTEP, FFOp::MINF, FFOp::MAXF };
  MINLPPRE.POLIMG.AGGREG_LQ       = MINLPBND.POLIMG.AGGREG_LQ       = 1;
  MINLPPRE.POLIMG.SANDWICH_RTOL   = MINLPBND.POLIMG.SANDWICH_RTOL   = 1e-3;
  MINLPPRE.POLIMG.SANDWICH_MAXCUT = MINLPBND.POLIMG.SANDWICH_MAXCUT = 5;
  MINLPPRE.BCHPRIM                = MINLPBND.BCHPRIM                = 0;
  MINLPPRE.OBBTLIN                                                  = 1;
  MINLPBND.OBBTLIN                                                  = 2;
  MINLPPRE.OBBTCONT                                                 = 1;
  MINLPPRE.OBBTMAX                                                  = 2;
  MINLPBND.OBBTCONT                                                 = 1;
  MINLPBND.OBBTMAX                                                  = 0;
  MINLPPRE.OBBTTHRES              = MINLPBND.OBBTTHRES              = 5e-2;
  MINLPPRE.OBBTBKOFF              = MINLPBND.OBBTBKOFF              = 1e-7;
  MINLPPRE.OBBTMIG                = MINLPBND.OBBTMIG                = 1e-6;
  MINLPPRE.CPMAX                  = MINLPBND.CPMAX                  = 10;
  MINLPPRE.CPTHRES                = MINLPBND.CPTHRES                = 0.;
  MINLPPRE.REDELIM                = MINLPBND.REDELIM                = 0;
  MINLPPRE.CMODPROP               = MINLPBND.CMODPROP               = 15;
  MINLPPRE.CMODEL.MIG_ATOL        = MINLPBND.CMODEL.MIG_ATOL        = 1e-10;
  MINLPPRE.CMODEL.MIG_RTOL        = MINLPBND.CMODEL.MIG_RTOL        = 1e-10;
  MINLPPRE.SQUAD.BASIS            = MINLPBND.SQUAD.BASIS            = MINLPBND.SQUAD.MONOM;
  MINLPPRE.SQUAD.ORDER            = MINLPBND.SQUAD.ORDER            = MINLPBND.SQUAD.DEC;
  MINLPPRE.SQUAD.REDUC            = MINLPBND.SQUAD.REDUC            = 0;
  MINLPPRE.QUADOPTIM              = MINLPBND.QUADOPTIM              = 0;
  MINLPPRE.MONSCALE               = MINLPBND.MONSCALE               = 0;
  MINLPPRE.PSDQUADCUTS            = MINLPBND.PSDQUADCUTS            = 0;
  MINLPPRE.DCQUADCUTS             = MINLPBND.DCQUADCUTS             = 0;
  MINLPPRE.NCOCUTS                = MINLPBND.NCOCUTS                = 0;
  MINLPPRE.NCOADIFF               = MINLPBND.NCOADIFF               = MINLPBND.FSA;
  MINLPPRE.DISPLEVEL              = MINLPBND.DISPLEVEL              = 1;
  MINLPPRE.MIPSLV.PRESOLVE        = MINLPBND.MIPSLV.PRESOLVE        = -1;
  MINLPBND.MIPSLV.LPWARMSTART                                       = 1;
  MINLPPRE.MIPSLV.LPWARMSTART                                       = 2;
  MINLPPRE.MIPSLV.MIPRELGAP       = MINLPBND.MIPSLV.MIPRELGAP       = 1e-3;
  MINLPPRE.MIPSLV.MIPABSGAP       = MINLPBND.MIPSLV.MIPABSGAP       = 1e-5;
  MINLPPRE.MIPSLV.PWLRELGAP       = MINLPBND.MIPSLV.PWLRELGAP       = 1e-3;
  MINLPPRE.MIPSLV.HEURISTICS      = MINLPBND.MIPSLV.HEURISTICS      = 5e-2;
  MINLPPRE.MIPSLV.NUMERICFOCUS    = MINLPBND.MIPSLV.NUMERICFOCUS    = 0;
  MINLPPRE.MIPSLV.SCALEFLAG       = MINLPBND.MIPSLV.SCALEFLAG       = -1;
  MINLPPRE.MIPSLV.DISPLEVEL                                         = 0;
  MINLPBND.MIPSLV.DISPLEVEL                                         = 1;
  MINLPPRE.MIPSLV.OUTPUTFILE      = MINLPBND.MIPSLV.OUTPUTFILE      = "";
  MINLPPRE.MIPSLV.THREADS         = MINLPBND.MIPSLV.THREADS         = 0;

  MINLPSLV.MAXITER                = 10;
  MINLPSLV.CPMAX                  = 10;
  MINLPSLV.CPTHRES                = 0.;
  MINLPSLV.MSLOC                  = 16;
  MINLPSLV.DISPLEVEL              = 1;
  MINLPSLV.NLPSLV.FEASTOL         = 1e-7;
  MINLPSLV.NLPSLV.OPTIMTOL        = 1e-5;
  MINLPSLV.NLPSLV.MAXITER         = 200;
  MINLPSLV.NLPSLV.DISPLEVEL       = 0;
  MINLPSLV.NLPSLV.MAXTHREAD       = 0;
  MINLPSLV.MIPSLV.PRESOLVE        = -1;
  MINLPSLV.MIPSLV.LPWARMSTART     = 1;
  MINLPSLV.MIPSLV.MIPRELGAP       = 1e-3;
  MINLPSLV.MIPSLV.MIPABSGAP       = 1e-5;
  MINLPSLV.MIPSLV.HEURISTICS      = 5e-2;
  MINLPSLV.MIPSLV.NUMERICFOCUS    = 0;
  MINLPSLV.MIPSLV.SCALEFLAG       = -1;
  MINLPSLV.MIPSLV.DISPLEVEL       = 0;
  MINLPSLV.MIPSLV.OUTPUTFILE      = "";
  MINLPSLV.MIPSLV.THREADS         = 0;

  _USROPT.add_options()
    ( "GAMSEXPORT",       opt::value<std::string>(&GAMSEXPORT),   "export GAMS model after preprocessing" )
    ( "PRESOLVE",         opt::value<int>(&PRESOLVE),             "level of preprocessing" )
    ( "REFORM",           opt::value<int>(&REFORM),               "level of reformulation" )
    ( "REDCUTS",          opt::value<int>(&REDCUTS),              "level of redundant polynomial cuts" )
    ( "CVATOL",           opt::value<double>(&CVATOL),            "convergence absolute tolerance" )
    ( "CVRTOL",           opt::value<double>(&CVRTOL),            "convergence relative tolerance" )
    ( "FEASTOL",          opt::value<double>(&FEASTOL),           "feasibility tolerance" )
    ( "CORRINC",          opt::value<bool>(&CORRINC),             "feasibility correction of incumbent using KKT multipliers" )
    ( "INIINC",           opt::value<bool>(&INIINC),              "initialize relaxation at current incumbent point" )
    ( "CUTINC",           opt::value<bool>(&CUTINC),              "add cut at current incumbent in relaxation" )
    ( "BKPTINC",          opt::value<bool>(&BKPTINC),             "add breakpoint at current incumbent in piecewise relaxation" )
    ( "MAXITER",          opt::value<unsigned>(&MAXITER),         "maximal number of iterations" )
    ( "TIMELIMIT",        opt::value<double>(&TIMELIMIT),         "overall runtime limit" )
    ( "DISPLEVEL",        opt::value<int>(&DISPLEVEL),            "overall display level" )
    ( "PRETIMELIMIT",     opt::value<double>(&PRETIMELIMIT),      "runtime limit of preprocessing" )
    ( "LOGFILE",          opt::value<std::string>(&_LOGFILENAME), "log file" )
//
    ( "MINLPBND.PREOBBTLIN",       opt::value<unsigned>(&MINLPPRE.OBBTLIN),              "optimization-based bounds tighteneting approach during presolve" )
    ( "MINLPBND.PREOBBTCONT",      opt::value<bool>(&MINLPPRE.OBBTCONT),                 "continuous relaxation for optimization-based bounds tighteneting during presolve" )
    ( "MINLPBND.PREOBBTMAX",       opt::value<unsigned>(&MINLPPRE.OBBTMAX),              "maximal number of optimization-based bounds tighteneting iterations during presolve" )
    ( "MINLPBND.PREOBBTTHRES",     opt::value<double>(&MINLPPRE.OBBTTHRES),              "threshold for optimization-based bounds tighteneting repeats during presolve" )
    ( "MINLPBND.PREOBBTBKOFF",     opt::value<double>(&MINLPPRE.OBBTBKOFF),              "backoff for optimization-based bounds tighteneting during presolve" )
    ( "MINLPBND.PREOBBTMIG",       opt::value<double>(&MINLPPRE.OBBTMIG),                "minimal variable range for optimization-based bounds tighteneting during presolve" )
    ( "MINLPBND.PREMIPPRESOLVE",   opt::value<int>(&MINLPPRE.MIPSLV.PRESOLVE),           "presolve level in MIP solver during presolve" )
    ( "MINLPBND.PREMIPWARMSTART",  opt::value<int>(&MINLPPRE.MIPSLV.LPWARMSTART),        "use of warm start information for LP optimization during presolve" )
    ( "MINLPBND.PREMIPRELGAP",     opt::value<double>(&MINLPPRE.MIPSLV.MIPRELGAP),       "convergence relative tolerance of MIP solver during presolve" )
    ( "MINLPBND.PREMIPABSGAP",     opt::value<double>(&MINLPPRE.MIPSLV.MIPABSGAP),       "convergence absolute tolerance of MIP solver during presolve" )
    ( "MINLPBND.PREMIPHEURISTICS", opt::value<double>(&MINLPPRE.MIPSLV.HEURISTICS),      "fraction of time spent in MIP heuristics during presolve" )
    ( "MINLPBND.PREMIPNUMERIC",    opt::value<int>(&MINLPPRE.MIPSLV.NUMERICFOCUS),       "control of numerical issues by MIP solver during presolve" )
    ( "MINLPBND.PREMIPSCALE",      opt::value<int>(&MINLPPRE.MIPSLV.SCALEFLAG),          "control of model scaling by MIP solver during presolve" )
    ( "MINLPBND.PREMIPDISPLEVEL",  opt::value<int>(&MINLPPRE.MIPSLV.DISPLEVEL),          "display level of MIP solver during presolve" )
    ( "MINLPBND.PREMIPOUTPUTFILE", opt::value<std::string>(&MINLPPRE.MIPSLV.OUTPUTFILE), "output file for MIP model during presolve" )
    ( "MINLPBND.PREMIPMAXTHREAD",  opt::value<unsigned>(&MINLPPRE.MIPSLV.THREADS),       "number of threads used by MIP solver during presolve" )
//
    ( "MINLPBND.RETAINQUAD",    opt::value<bool>(&MINLPBND.POLIMG.ALLOW_QUAD),          "retain quadratic terms in MIP relaxation" )
    ( "MINLPBND.RETAINDISJ",    opt::value<bool>(&_MINLPBND_ALLOW_DISJ),                "retain disjunctive terms (abs,min,max) in MIP relaxation" )
    ( "MINLPBND.RETAINNLIN",    opt::value<unsigned>(&_MINLPBND_ALLOW_NLIN),            "retain nonlinear terms (sqrt,pow,exp,log,cos,sin,tan,tanh,...) in MIP relaxation" )
    ( "MINLPBND.BCHPRIM",       opt::value<unsigned>(&MINLPBND.BCHPRIM),                "Set higher branch priority to original variables" )
    ( "MINLPBND.OBBTLIN",       opt::value<unsigned>(&MINLPBND.OBBTLIN),                "optimization-based bounds tighteneting approach" )
    ( "MINLPBND.OBBTCONT",      opt::value<bool>(&MINLPBND.OBBTCONT),                   "continuous relaxation for optimization-based bounds tighteneting" )
    ( "MINLPBND.OBBTMAX",       opt::value<unsigned>(&MINLPBND.OBBTMAX),                "maximal number of optimization-based bounds tighteneting iterations" )
    ( "MINLPBND.OBBTTHRES",     opt::value<double>(&MINLPBND.OBBTTHRES),                "threshold for optimization-based bounds tighteneting repeats" )
    ( "MINLPBND.OBBTBKOFF",     opt::value<double>(&MINLPBND.OBBTBKOFF),                "backoff for optimization-based bounds tighteneting" )
    ( "MINLPBND.OBBTMIG",       opt::value<double>(&MINLPBND.OBBTMIG),                  "minimal variable range for optimization-based bounds tighteneting" )
    ( "MINLPBND.CPMAX",         opt::value<unsigned>(&MINLPBND.CPMAX),                  "maximal number of constraint propagation iterations" )
    ( "MINLPBND.CPTHRES",       opt::value<double>(&MINLPBND.CPTHRES),                  "threshold for constraint propagation repeats" )
    ( "MINLPBND.MONORDER",      opt::value<int>(&MINLPBND.SQUAD.ORDER),                 "monomial processing order in sparse quadratic form" )
    ( "MINLPBND.MONOPTIM",      opt::value<unsigned>(&_MINLPBND_QUADOPTIM),             "monomial minimisation in sparse quadratic form" )
    ( "MINLPBND.REDQUADCUTS",   opt::value<bool>(&MINLPBND.SQUAD.REDUC),                "add redundant cuts within quadratisation" )
    ( "MINLPBND.PSDQUADCUTS",   opt::value<unsigned>(&MINLPBND.PSDQUADCUTS),            "add PSD cuts within quadratisation" )
    ( "MINLPBND.LINCTRSEP",     opt::value<bool>(&MINLPBND.LINCTRSEP),                  "separate linear constraints during relaxation" )
    ( "MINLPBND.AGGREGLQ",      opt::value<bool>(&MINLPBND.POLIMG.AGGREG_LQ),           "keep linear and quadratic expressions aggregated" )
    ( "MINLPBND.SANDWICHRTOL",  opt::value<double>(&MINLPBND.POLIMG.SANDWICH_RTOL),     "relative tolerance in outer-approximation of univariate terms" )
    ( "MINLPBND.SANDWICHMAX",   opt::value<unsigned>(&MINLPBND.POLIMG.SANDWICH_MAXCUT), "maximal number of cuts in outer-approximation of univariate terms" )
    ( "MINLPBND.MIPPRESOLVE",   opt::value<int>(&MINLPBND.MIPSLV.PRESOLVE),             "presolve level in MIP solver" )
    ( "MINLPBND.MIPRELGAP",     opt::value<double>(&MINLPBND.MIPSLV.MIPRELGAP),         "convergence relative tolerance of MIP solver" )
    ( "MINLPBND.MIPABSGAP",     opt::value<double>(&MINLPBND.MIPSLV.MIPABSGAP),         "convergence absolute tolerance of MIP solver" )
    ( "MINLPBND.MIPPWLRELGAP",  opt::value<double>(&MINLPBND.MIPSLV.PWLRELGAP),         "tolerance in piecewise-linear approximation of nonlinear univariate terms" )
    ( "MINLPBND.MIPHEURISTICS", opt::value<double>(&MINLPBND.MIPSLV.HEURISTICS),        "fraction of time spent in MIP heuristics" )
    ( "MINLPBND.MIPNUMERIC",    opt::value<int>(&MINLPBND.MIPSLV.NUMERICFOCUS),         "control of numerical issues by MIP solver" )
    ( "MINLPBND.MIPSCALE",      opt::value<int>(&MINLPBND.MIPSLV.SCALEFLAG),            "control of model scaling by MIP solver" )
    ( "MINLPBND.MIPDISPLEVEL",  opt::value<int>(&MINLPBND.MIPSLV.DISPLEVEL),            "display level of MIP solver" )
    ( "MINLPBND.MIPOUTPUTFILE", opt::value<std::string>(&MINLPBND.MIPSLV.OUTPUTFILE),   "output file for MIP model" )
    ( "MINLPBND.MIPMAXTHREAD",  opt::value<unsigned>(&MINLPBND.MIPSLV.THREADS),         "set number of threads used by MIP solver" )
//
    ( "MINLPSLV.MAXITER",       opt::value<unsigned>(&MINLPSLV.MAXITER),              "maximal number of iterations by local MINLP solver" )
    ( "MINLPSLV.CPMAX",         opt::value<unsigned>(&MINLPSLV.CPMAX),                "maximum rounds of constraint propagation by local MINLP solver" )
    ( "MINLPSLV.CPTHRES",       opt::value<double>(&MINLPSLV.CPTHRES),                "threshold for constraint propagation repeats by local MINLP solver" )
    ( "MINLPSLV.MSLOC",         opt::value<unsigned>(&MINLPSLV.MSLOC),                "multistart local search repeats by local MINLP solver" )
    ( "MINLPSLV.DISPLEVEL",     opt::value<int>(&MINLPSLV.DISPLEVEL),                 "display level of local MINLP solver" )
    ( "MINLPSLV.NLPFEASTOL",    opt::value<double>(&MINLPSLV.NLPSLV.FEASTOL),         "feasibility tolerance of local NLP solver" )
    ( "MINLPSLV.NLPOPTIMTOL",   opt::value<double>(&MINLPSLV.NLPSLV.OPTIMTOL),        "optimality tolerance of local NLP solver" )
    ( "MINLPSLV.NLPMAXITER",    opt::value<int>(&MINLPSLV.NLPSLV.MAXITER),            "maximal number of iterations of local NLP solver" )
    ( "MINLPSLV.NLPDISPLEVEL",  opt::value<int>(&MINLPSLV.NLPSLV.DISPLEVEL),          "display level of local NLP solver" )
    ( "MINLPSLV.NLPMAXTHREAD",  opt::value<unsigned>(&MINLPSLV.NLPSLV.MAXTHREAD),     "set number of threads used by local NLP solver" )
    ( "MINLPSLV.MIPPRESOLVE",   opt::value<int>(&MINLPSLV.MIPSLV.PRESOLVE),           "presolve level in MIP solver called by local MINLP solver" )
    ( "MINLPSLV.MIPRELGAP",     opt::value<double>(&MINLPSLV.MIPSLV.MIPRELGAP),       "convergence relative tolerance of MIP solver called by local MINLP solver" )
    ( "MINLPSLV.MIPABSGAP",     opt::value<double>(&MINLPSLV.MIPSLV.MIPABSGAP),       "convergence absolute tolerance of MIP solver called by local MINLP solver" )
    ( "MINLPSLV.MIPHEURISTICS", opt::value<double>(&MINLPSLV.MIPSLV.HEURISTICS),      "fraction of time spent in MIP heuristics by local MINLP solver" )
    ( "MINLPSLV.MIPNUMERIC",    opt::value<int>(&MINLPSLV.MIPSLV.NUMERICFOCUS),       "control of numerical issues by MIP solver called by local MINLP solver" )
    ( "MINLPSLV.MIPSCALE",      opt::value<int>(&MINLPSLV.MIPSLV.SCALEFLAG),          "control of model scaling by MIP solver called by local MINLP solver" )
    ( "MINLPSLV.MIPDISPLEVEL",  opt::value<int>(&MINLPSLV.MIPSLV.DISPLEVEL),          "display level of MIP solver called by local MINLP solver" )
    ( "MINLPSLV.MIPOUTPUTFILE", opt::value<std::string>(&MINLPSLV.MIPSLV.OUTPUTFILE), "output file for MIP model called by local MINLP solver" )
    ( "MINLPSLV.MIPMAXTHREAD",  opt::value<unsigned>(&MINLPSLV.MIPSLV.THREADS),       "set number of threads used by MIP solver called by local MINLP solver" )
    ;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline typename MINLGO<T,NLP,MIP,ExtOps...>::Options&
MINLGO<T,NLP,MIP,ExtOps...>::Options::operator=
( Options const& other )
{
  GAMSEXPORT       = other.GAMSEXPORT;
  PRESOLVE         = other.PRESOLVE;
  REFORM           = other.REFORM;
  REDCUTS          = other.REDCUTS;
  CORRINC          = other.CORRINC;
  INIINC           = other.INIINC;
  CUTINC           = other.CUTINC;
  BKPTINC          = other.BKPTINC;
  FEASTOL          = other.FEASTOL;
  CVATOL           = other.CVATOL;
  CVRTOL           = other.CVRTOL;
  MAXITER          = other.MAXITER;
  TIMELIMIT        = other.TIMELIMIT;
  DISPLEVEL        = other.DISPLEVEL;
  PRETIMELIMIT     = other.PRETIMELIMIT;
  MINLPSLV         = other.MINLPSLV;
  MINLPBND         = other.MINLPBND;
  MINLPPRE         = other.MINLPPRE;       
  return *this;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::Options::read
( std::string const& optionfilename, std::ofstream&logfile, std::ostream&os )
{
  if( !read( optionfilename, os ) ) return false;

  if( _USRMAP.count( "LOGFILE" ) ){
    logfile.open( _LOGFILENAME, std::ofstream::out | std::ofstream::app );
    MINLPBND.MIPSLV.LOGFILE = _LOGFILENAME;
    MINLPSLV.NLPSLV.LOGFILE = _LOGFILENAME;
  }
  return true;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline bool
MINLGO<T,NLP,MIP,ExtOps...>::Options::read
( std::string const& optionfilename, std::ostream&os )
{
  std::ifstream optionfile( optionfilename.c_str() );
  if( optionfile.fail() )
  {
    os << "# Error: cannot open option file " << optionfilename << std::endl;
    return false;
  }

  try{
    opt::store( opt::parse_config_file<char>( optionfile, _USROPT ), _USRMAP );
    opt::notify( _USRMAP );
  }
  catch( const opt::reading_file& e ){
    os << "# Error: " << e.what() << std::endl;
    return false;
  }
  catch( const opt::required_option& e ){
    os << "# Error: " << e.what() << std::endl;
    return false;
  }

  if( _USRMAP.count( "REDCUTS" ) ){
    MINLPBND.SRED.ORDER = (REDCUTS>=0? REDCUTS: -REDCUTS);
    MINLPBND.SRED.NODIV = (REDCUTS>=0? 1: 0 );
  }

  if( _USRMAP.count( "MINLPBND.RETAINDISJ" ) )
    MINLPBND.POLIMG.ALLOW_DISJ.clear();
    switch( _MINLPBND_ALLOW_DISJ ){
      case 1: MINLPBND.POLIMG.ALLOW_DISJ.insert( {FFOp::FABS, FFOp::FSTEP, FFOp::MINF, FFOp::MAXF} ); // no break
      default: break;
    }

  if( _USRMAP.count( "MINLPBND.RETAINNLIN" ) )
    MINLPBND.POLIMG.ALLOW_NLIN.clear();
    switch( _MINLPBND_ALLOW_NLIN ){
      case 2: MINLPBND.POLIMG.ALLOW_NLIN.insert( {FFOp::FABS, FFOp::FSTEP, FFOp::MINF, FFOp::MAXF} ); // no break
      case 1: MINLPBND.POLIMG.ALLOW_NLIN.insert( {FFOp::IPOW, FFOp::DPOW, FFOp::CHEB, FFOp::SQRT, FFOp::EXP,  FFOp::LOG,
                                                  FFOp::COS,  FFOp::SIN,  FFOp::TAN,  FFOp::ACOS, FFOp::ASIN, FFOp::ATAN, 
                                                  FFOp::TANH} ); // no break
      default: break;
    }

  if( _USRMAP.count( "MINLPBND.MONOPTIM" ) )
    switch( _MINLPBND_QUADOPTIM ){
      case 2:  MINLPBND.QUADOPTIM = 1; MINLPBND.SQUAD.MIPFIXEDBASIS = 0; break;
      case 1:  MINLPBND.QUADOPTIM = 1; MINLPBND.SQUAD.MIPFIXEDBASIS = 1; break;
      default: MINLPBND.QUADOPTIM = 0; break;
    }

  if( _USRMAP.count( "MINLPBND.CPMAX"        ) ) MINLPPRE.CPMAX                  = MINLPBND.CPMAX;
  if( _USRMAP.count( "MINLPBND.CPTHRES"      ) ) MINLPPRE.CPTHRES                = MINLPBND.CPTHRES;
  if( _USRMAP.count( "MINLPBND.MONORDER"     ) ) MINLPPRE.SQUAD.ORDER            = MINLPBND.SQUAD.ORDER;
  if( _USRMAP.count( "MINLPBND.MONOPTIM"     ) ) MINLPPRE.QUADOPTIM              = MINLPBND.QUADOPTIM;
  if( _USRMAP.count( "MINLPBND.LINCTRSEP"    ) ) MINLPPRE.LINCTRSEP              = MINLPBND.LINCTRSEP;
  if( _USRMAP.count( "MINLPBND.AGGREGLQ"     ) ) MINLPPRE.POLIMG.AGGREG_LQ       = MINLPBND.POLIMG.AGGREG_LQ;
  if( _USRMAP.count( "MINLPBND.SANDWICHRTOL" ) ) MINLPPRE.POLIMG.SANDWICH_RTOL   = MINLPBND.POLIMG.SANDWICH_RTOL;
  if( _USRMAP.count( "MINLPBND.SANDWICHMAX"  ) ) MINLPPRE.POLIMG.SANDWICH_MAXCUT = MINLPBND.POLIMG.SANDWICH_MAXCUT;

  return true;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline void
MINLGO<T,NLP,MIP,ExtOps...>::Options::display
( std::ostream&os ) const
{
  // Display MINLGO Options
  os << std::left;
  os << std::setw(60) << "  GAMS MODEL EXPORT FILE"
     << (GAMSEXPORT.empty()? "-": GAMSEXPORT) << std::endl;
  os << std::setw(60) << "  CONVERGENCE ABSOLUTE TOLERANCE"
     << std::scientific << std::setprecision(1)
     << CVATOL << std::endl;
  os << std::setw(60) << "  CONVERGENCE RELATIVE TOLERANCE"
     << std::scientific << std::setprecision(1)
     << CVRTOL << std::endl;
  os << std::setw(60) << "  FEASIBILITY TOLERANCE"
     << std::scientific << std::setprecision(1)
     << FEASTOL << std::endl;
  os << std::setw(60) << "  FEASIBILITY CORRECTION"
     << (CORRINC?'Y':'N') << std::endl;
  os << std::setw(60) << "  PRESOLVE LEVEL"
     << PRESOLVE << std::endl;
  os << std::setw(60) << "  PRESOLVE TIME LIMIT (SEC)"
     << std::scientific << std::setprecision(1)
     << PRETIMELIMIT << std::endl;
  os << std::setw(60) << "  REFORMULATION LEVEL"
     << REFORM << std::endl;
  os << std::setw(60) << "  REDUNDANT CUTS"
     << REDCUTS << std::endl;
  os << std::setw(60) << "  INCUMBENT INITALIZATION"
     << (INIINC?'Y':'N') << std::endl;
  os << std::setw(60) << "  INCUMBENT CUT"
     << (CUTINC?'Y':'N') << std::endl;
  os << std::setw(60) << "  INCUMBENT BREAKPOINT"
     << (BKPTINC?'Y':'N') << std::endl;
  os << std::setw(60) << "  MAXIMAL ITERATIONS"
     << MAXITER << std::endl;
  os << std::setw(60) << "  TIME LIMIT (SEC)"
     << std::scientific << std::setprecision(1)
     << TIMELIMIT << std::endl;
}

template <typename T, typename NLP, typename MIP, typename... ExtOps>
inline std::ostream&
operator <<
( std::ostream & os, MINLGO<T,NLP,MIP,ExtOps...> const& MINLP )
{
  os << std::right << std::endl
     << std::setfill('_') << std::setw(72) << " " << std::endl << std::endl << std::setfill(' ')
     << std::setw(55) << "GLOBAL MIXED-INTEGER NONLINEAR OPTIMIZATION IN CANON\n"
     << std::setfill('_') << std::setw(72) << " " << std::endl << std::endl << std::setfill(' ');

  // Display MINLGO Options
  MINLP.options.display( os );

  os << std::setfill('_') << std::setw(72) << " " << std::endl << std::endl << std::setfill(' ');
  return os;
}

} // end namescape mc

#endif

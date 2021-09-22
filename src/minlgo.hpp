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

Consider the following NLP:
\f{align*}
  \max_{\bf p}\ & p_1\,p_4\,(p_1+p_2+p_3)+p_3 \\
  \text{s.t.} \ & p_1\,p_2\,p_3\,p_4 \geq 25 \\
  & p_1^2+p_2^2+p_3^2+p_4^2 = 40 \\
  & 1 \leq p_1,p_2,p_3,p_4 \leq 5\,.
\f}

We start by instantiating an mc::MINLGO class object, which is defined in the header file <tt>minlgo.hpp</tt>:

\code
  mc::MINLGO MINLP;
\endcode

Next, we set the variables and objective/constraint functions by creating a DAG of the problem: 

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

The MINLP model is solved using default options as:

\code
  MINLP.setup();
  MINLP.presolve();
  MINLP.optimize();
\endcode

The return value of mc::MINLGO is per the enumeration mc::MINLGO::STATUS. The following result is displayed (with the option mc::MINLGO::Options::DISPLEVEL defaulting to 1):

\verbatim
#  CONTINUOUS / DISCRETE VARIABLES:  4 / 0
#  LINEAR / NONLINEAR FUNCTIONS:     0 / 3

#  ITERATION     INCUMBENT    BEST BOUND    TIME
        0 r*  1.701402e+01  1.701402e+01      0s

#  TERMINATION AFTER 0 ITERATIONS: 0.008299 SEC
#  INCUMBENT VALUE:  1.701402e+01
#  INCUMBENT POINT:  1.000000e+00  4.743001e+00  3.821149e+00  1.379408e+00

#  ITERATION      INCUMBENT    BEST BOUND    TIME
        0  P*  1.701402e+01  1.633734e+01      0s
        1   *  1.701402e+01  1.700345e+01      0s

#  TERMINATION AFTER 0 REFINEMENTS: 0.059628 SEC
#  INCUMBENT VALUE:  1.701402e+01
#  INCUMBENT POINT:  1.000000e+00  4.743000e+00  3.821150e+00  1.379408e+00
#  OPTIMALITY GAP:   1.06e-02 (ABS)
                     6.21e-04 (REL)
\endverbatim

The incumbent solution may be retrieved as an instance of <a>mc::SOLUTION_OPT</a> using the method <a>mc::MINLGO::incumbent</a>. A computational breakdown may be obtained from the internal class <a>mc::MINLGO::Stats</a>. And the (many) options of the algorithm can be modified using the internal class mc::MINLGO::Options.
*/

//TODO: 
//- Detect the class of problems to determine the need for refinements/iterations
//- Debug quadratisation in Chebyshev basis during preprocessing

#ifndef MC__MINLGO_HPP
#define MC__MINLGO_HPP

#include <chrono>

#include <boost/program_options.hpp> 
namespace opt = boost::program_options; 

#include "interval.hpp"
#include "gamsio.hpp"
#include "minlpslv.hpp"
#include "nlpslv_snopt.hpp"
#include "mipslv_gurobi.hpp"
#include "minlpbnd.hpp"

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
           typename NLP=NLPSLV_SNOPT,
           typename MIP=MIPSLV_GUROBI<T> >
class MINLGO:
  public    virtual BASE_NLP,
  protected virtual GAMSIO
{
  using BASE_NLP::_dag; // Make sure _dag is from BASE_NLP, not GAMSIO

public:

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

    //! @brief Level of preprocessing
    int         PRESOLVE;
    //! @brief Correct the incumbent for feasibility using multipliers
    bool        CORRINC;
    //! @brief Add a cut at incumbent value
    bool        CUTINC;
    //! @brief Add a break-point at incumbent
    bool        BKPTINC;
    //! @brief Feasibility tolerance 
    double      FEASTOL;
    //! @brief Convergence absolute tolerance
    double      CVATOL;
    //! @brief Convergence relative tolerance
    double      CVRTOL;
    //! @brief Maximum number of outer-approximation iterations (0-no limit)
    unsigned    MAXITER;
    //! @brief Maximum run time (seconds)
    double      TIMELIMIT;
    //! @brief Display level for solver
    int         DISPLEVEL;

    //! @brief MINLP local solver options
    typename MINLPSLV<T,NLP,MIP>::Options MINLPSLV;
    //! @brief MINLP global bounder options
    typename MINLPBND<T,MIP>::Options     MINLPBND;
    //! @brief MINLP global bounder options for presolve
    typename MINLPBND<T,MIP>::Options     MINLPPRE;

    //! @brief Load option file
    bool read
      ( std::string const& optionfilename, std::ofstream&logfile,
        std::ostream&out=std::cout );
    bool read
      ( std::string const& optionfilename, std::ostream&out=std::cout );
    //! @brief user options
    opt::options_description const& user_options
      () const
      { return _USROPT; }
    //! @brief Display
    void display
      ( std::ostream&out=std::cout ) const;

   private:
    //! @brief Option description from file
    opt::options_description _USROPT;
    //! @brief Option map
    opt::variables_map _USRMAP; 
    //! @brief Log file
    std::string _LOGFILENAME;
    //! @brief Reformulation approach
    unsigned    _MINLPBND_REFORMMETH;
    unsigned    _MINLPPRE_REFORMMETH;
    //! @brief Relaxation approach
    unsigned    _MINLPBND_RELAXMETH;
    unsigned    _MINLPPRE_RELAXMETH;
  } options;

  //! @brief Class managing exceptions for MINLGO
  class Exceptions
  {
  public:
    //! @brief Enumeration type for MINLGO exception handling
    enum TYPE{
      SETUP,		//!< Incomplete setup before a solve
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
      case INTERN: default:
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
  STATUS                    _status;

  //! @brief Current iteration
  unsigned                  _iter;

  //! @brief Rounds of domain reduction
  unsigned                  _nred;

  //! @brief Flag for setup function
  bool                      _issetup;

  //! @brief Flag for presolve function
  bool                      _ispresolved;

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

  //! @brief Structure holding incumbent information
  SOLUTION_OPT              _incumbent;

  //! @brief Variable bounds
  std::vector<T>            _Xbnd;
  
  //! @brief Decision variable bounds with integer fixing
  std::vector<T>            _Xbndi;

  //! @brief Local solver for factorable MINLP
  MINLPSLV<T,NLP,MIP>       _MINLPSLV;

  //! @brief Global bounder for factorable NLP
  MINLPBND<T,MIP>           _MINLPBND;

  //! @brief Structure holding NLP intermediate solution
  SOLUTION_OPT              _solution;

  //! @brief maximum number of values displayed in a row
  static const unsigned int _LDISP = 4;

  //! @brief reserved space for integer variable display
  static const unsigned int _IPREC = 9;

  //! @brief reserved space for double variable display
  static const unsigned int _DPREC = 6;

  //! @brief stringstream for displaying results
  std::ostringstream        _odisp;

  //! @brief Time point to enable TIMELIMIT option
  std::chrono::time_point<std::chrono::system_clock> _tstart;

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
    ( std::ostream& os = std::cout );

  //! @brief Finalize optimization display and status
  int _finalize
    ( STATUS const status, std::ostream& os=std::cout );

  //! @brief Solve local NLP subproblem
  bool _solve_local
    ( double const* Xini, T const* Xbnd );
      
  //! @brief Solve relaxed MIP subproblem   
  int _solve_relax
    ();
      
  //! @brief Convergence test for piecewise-linear relaxation approach 
  bool _converged
    ()
    const;
    
  //! @brief Termination test for piecewise-linear relaxation approach 
  bool _interrupted
    ()
    const;


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
    ( std::string const& filename, bool const disp=false );
#endif

  //! @brief Setup DAG for cost and constraint evaluation
  void setup
    ();

  //! @brief Preprocess optimization model - return value is false if model is provably infeasible
  int presolve
    (  T* Xbnd=nullptr, double* Xini=nullptr, std::ostream& os=std::cout );

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
    { _MINLPSLV.MIPsolver().terminate();
      _MINLPBND.solver()->terminate(); }

private:
  //! @brief Private methods to block default compiler methods
  MINLGO
    ( MINLGO const& );
  MINLGO& operator=
    ( MINLGO const& );
};

#if defined (MC__WITH_GAMS)
template <typename T, typename NLP, typename MIP>
inline bool
MINLGO<T,NLP,MIP>::read
( std::string const& filename, bool const disp )
{
  _tstart = stats.start();

  bool flag = this->GAMSIO::read( filename, disp );

  stats.walltime_setup += stats.walltime( _tstart );
  stats.walltime_all   += stats.walltime( _tstart );
  return flag;
}
#endif

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::setup
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
    case MIN: _objscal =  1e0; break;
    case MAX: _objscal = -1e0; break;
  }
  
#ifdef MC__MINLGO_SETUP_DEBUG
  std::cout << "MINLPSLV set-up" << std::endl;
#endif
  _MINLPSLV.options = options.MINLPSLV;
  _MINLPSLV.set( *this );
  _MINLPSLV.setup();
  _incumbent.reset();

#ifdef MC__MINLGO_SETUP_DEBUG
  std::cout << "MINLPBND set-up" << std::endl;
#endif
  _MINLPBND.options = options.MINLPBND;
  _MINLPBND.set( *this );
  _MINLPBND.setup();
  _Xbnd.clear();
  _Xbndi.clear();

  _issetup = true;

  stats.walltime_setup += stats.walltime( _tstart );
  stats.walltime_all   += stats.walltime( _tstart );
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLGO<T,NLP,MIP>::_is_integer_feasible
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

template <typename T, typename NLP, typename MIP>
inline int
MINLGO<T,NLP,MIP>::presolve
(  T* Xbnd, double* Xini, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _ispresolved = false;
  _tstart = stats.start();

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
    _incumbent = _MINLPSLV.NLPsolver().solution();
    _Zinc = _MINLPSLV.NLPsolver().solution().f[0];
  }

  // Simple bound propagation if presolve is turned off
  if( !options.PRESOLVE ){
    // Set presolve bounder options
    _MINLPBND.options = options.MINLPPRE;
    _MINLPBND.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );

    // Apply domain contraction for both linear and nonlinear constraints
    switch( _MINLPBND.propagate( _Xbnd.data(), !_incumbent.x.empty()? &_Zinc: nullptr, true ) ){
      case MIP::STATUS::INFEASIBLE:
        stats.walltime_preproc += stats.walltime( _tstart );
        stats.walltime_all     += stats.walltime( _tstart );
        return STATUS::INFEASIBLE;
      default:
        break;
    }
    _Xbnd.assign( _MINLPBND.varbnd(), _MINLPBND.varbnd()+_var.size() );
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
  _MINLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  _MINLPSLV.optimize( _varini.data(), _Xbnd.data(), os );
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

  // Set presolve bounder options
  _MINLPBND.options = options.MINLPPRE;
  _MINLPBND.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );

  // Apply domain contraction for both linear and nonlinear constraints
  switch( _MINLPBND.reduce( _nred, _Xbnd.data(), !_incumbent.x.empty()? &_Zinc: nullptr, true, true ) ){
//  switch( _MINLPBND.solver()->get_status() ){
    case MIP::STATUS::INFEASIBLE:
      stats.walltime_preproc += stats.walltime( _tstart );
      stats.walltime_all     += stats.walltime( _tstart );
      _ispresolved = true;
      return STATUS::INFEASIBLE;
    default:
      break;
  }
  _Xbnd.assign( _MINLPBND.varbnd(), _MINLPBND.varbnd()+_var.size() );
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
    _MINLPBND.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
    _MINLPBND.relax( nullptr, nullptr, nullptr, 0, false, false ); // reset cuts but not variables or bounds 
    switch( _MINLPBND.solver()->get_status() ){
      case MIP::STATUS::OPTIMAL:
      case MIP::STATUS::SUBOPTIMAL:
        _Zrel = _MINLPBND.solver()->get_objective_bound();
        for( unsigned i=0; i<_var.size(); i++ )
          _Xrel[i] = _MINLPBND.solver()->get_variable( _var[i] );
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
      _varini[i] = _MINLPBND.solver()->get_variable( _var[i] );
    _MINLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
    _MINLPSLV.optimize( _varini.data(), _Xbnd.data(), os );
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

template <typename T, typename NLP, typename MIP>
inline int
MINLGO<T,NLP,MIP>::optimize
( std::ostream& os )
{
  if( !_issetup || !_ispresolved ) throw Exceptions( Exceptions::SETUP );
  if( !_isbnd ) return _finalize( STATUS::UNBOUNDED );

  // Initialize solve
  _tstart = stats.start();
  _iter = 0;
  _nred = 0;
  
  // Set bounder options
  _MINLPBND.options = options.MINLPBND;

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
    switch( _solve_relax() ){
      case MIP::OPTIMAL:
        break;
      case MIPSLV_GUROBI<T>::INFEASIBLE:
        _Zrel = _objscal * BASE_OPT::INF;
        return _finalize( STATUS::INFEASIBLE );
      case MIP::UNBOUNDED:
        return _finalize( STATUS::UNBOUNDED );
      case MIP::TIMELIMIT:
        _Zrel = _MINLPBND.solver()->get_objective_bound();
        return _finalize( STATUS::INTERRUPTED );
      default:
        return _finalize( STATUS::FAILED );
    }

    // Retrieve MIP solution - use bound on objective, not incumbent!
    _Zrel = _MINLPBND.solver()->get_objective_bound();
    for( unsigned i=0; i<_var.size(); i++ )
      _Xrel[i] = _MINLPBND.solver()->get_variable( _var[i] );
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
    locfeas = _solve_local( _Xrel.data(), _Xbndi.data() );

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
      return _finalize( STATUS::INTERRUPTED );

    // Refine relaxation via additional breakpoints
    _MINLPBND.refine_polrelax( options.BKPTINC && updinc? _incumbent.x.data(): nullptr );

    // Apply domain contraction
    // Do NOT test for infeasibility here, because contraction problem may
    // become infeasible due to round-off in LP solver
    auto tMIP = stats.start();
    _MINLPBND.reduce( _nred, _Xbnd.data(), !_incumbent.x.empty()? &_Zinc: nullptr, false, false );
    stats.walltime_slvrel += stats.walltime( tMIP );
#ifdef MC__MINLGO_DEBUG
    std::cout << "Reduced bounds: x" << _nred << std::endl;
    for( auto const& Xi : _Xbnd ) std::cout << " " << Xi;
    std::cout << std::endl;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }

  return _finalize( STATUS::SUCCESSFUL );
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLGO<T,NLP,MIP>::_solve_local
( double const* Xini, T const* Xbnd )
{
  auto tNLP = stats.start();
  auto& _NLPSLV = _MINLPSLV.NLPsolver();
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
  if( dombnd && options.MINLPSLV.MSLOC > 1 && _NLPSLV.options.TIMELIMIT > 0 ){
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

template <typename T, typename NLP, typename MIP>
inline int
MINLGO<T,NLP,MIP>::_solve_relax
()
{
  // Update time limit
  _MINLPBND.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( _tstart ) );
  if( _MINLPBND.options.TIMELIMIT <= 0 )
    return MIP::TIMELIMIT;
  
  // Solve master MIP problem - do NOT reset bounds, otherwise reinitializing lifted variable bounds
  auto tMIP = stats.start();
  int flag = _MINLPBND.relax( _Xbnd.data(), options.CUTINC && !_incumbent.x.empty()? &_Zinc: nullptr,
                              _incumbent.x.data(), 0, false, _iter>1? false: true );
  stats.walltime_slvrel += stats.walltime( tMIP );

  return flag;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLGO<T,NLP,MIP>::_converged
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

template <typename T, typename NLP, typename MIP>
inline bool
MINLGO<T,NLP,MIP>::_interrupted
()
const
{
  if( stats.to_time( stats.walltime_all + stats.walltime( _tstart ) ) > options.TIMELIMIT
   || ( options.MAXITER && _iter >= options.MAXITER ) )
//   || ( _iter && _MINLPBND.problem_class() <= FFDep::Q ) )
    return true;
  return false;
}

template <typename T, typename NLP, typename MIP>
inline int
MINLGO<T,NLP,MIP>::_finalize
( STATUS const status, std::ostream& os )
{
  _status = status;
  stats.walltime_all += stats.walltime( _tstart );
  _display_final( stats.walltime_all, os );
  return _status;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::_display_init
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

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::_display_final
( std::chrono::microseconds const& walltime,
  std::ostream& os )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::endl << "#  TERMINATION AFTER " << (_iter?_iter-1:0) << " REFINEMENTS: "
         << std::fixed << std::setprecision(6) << walltime.count()*1e-6 << " SEC"
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

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::_display_add
( const double dval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::scientific << std::setprecision(_DPREC)
         << std::setw(_DPREC+8) << dval;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::_display_add
( const unsigned ival )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(_IPREC) << ival;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::_display_add
( const std::string &sval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(4) << sval;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::_display_time
()
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::fixed << std::setprecision(0) << std::setw(7) 
         << stats.to_time( stats.walltime_all + stats.walltime( _tstart ) ) << "s";
}

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::_display_flush
( std::ostream &os )
{
  if( _odisp.str() == "" ) return;
  os << _odisp.str() << std::endl;
  _odisp.str("");
  return;
}

template <typename T, typename NLP, typename MIP>
inline
MINLGO<T,NLP,MIP>::Options::Options()
: PRESOLVE( 1 ),
  CORRINC( 1 ),
  CUTINC( 0 ),
  BKPTINC( 0 ),
  FEASTOL( 1e-5 ),
  CVATOL( 1e-5 ),
  CVRTOL( 1e-3 ),
  MAXITER( 10 ),
  TIMELIMIT( 9e2 ),
  DISPLEVEL( 1 ),
  MINLPSLV(),
  MINLPBND(),
  MINLPPRE(),
  _USROPT( "User-defined solver options" )
{
  MINLPPRE.REFORMMETH                                               = { MINLPBND.NPOL };
  MINLPBND.REFORMMETH                                               = { MINLPBND.NPOL, MINLPBND.QUAD };
  MINLPPRE.LINCTRSEP              = MINLPBND.LINCTRSEP              = 0;
  MINLPPRE.RELAXMETH                                                = { MINLPBND.DRL };
  MINLPBND.RELAXMETH                                                = { MINLPBND.DRL };
  MINLPPRE.SUBSETDRL              = MINLPPRE.SUBSETSCQ              = 0;
  MINLPBND.SUBSETDRL                                                = 0;
  MINLPBND.SUBSETSCQ                                                = 0;
  MINLPPRE.POLIMG.RELAX_QUAD                                        = 1;
  MINLPBND.POLIMG.RELAX_QUAD                                        = 0;
  MINLPPRE.POLIMG.RELAX_MONOM     = MINLPBND.POLIMG.RELAX_MONOM     = 0;
  MINLPPRE.POLIMG.RELAX_NLIN      = MINLPBND.POLIMG.RELAX_NLIN      = 0;
  MINLPPRE.POLIMG.AGGREG_LQ       = MINLPBND.POLIMG.AGGREG_LQ       = 0;
  MINLPPRE.POLIMG.SANDWICH_RTOL   = MINLPBND.POLIMG.SANDWICH_RTOL   = 1e-3;
  MINLPPRE.POLIMG.SANDWICH_MAXCUT = MINLPBND.POLIMG.SANDWICH_MAXCUT = 5;
  MINLPPRE.BCHPRIM                = MINLPBND.BCHPRIM                = 0;
  MINLPPRE.OBBTLIN                = MINLPBND.OBBTLIN                = 2;
  MINLPPRE.OBBTCONT                                                 = 1;
  MINLPPRE.OBBTMAX                                                  = 10;
  MINLPBND.OBBTCONT                                                 = 1;
  MINLPBND.OBBTMAX                                                  = 10;
  MINLPPRE.OBBTTHRES              = MINLPBND.OBBTTHRES              = 5e-2;
  MINLPPRE.OBBTBKOFF              = MINLPBND.OBBTBKOFF              = 1e-7;
  MINLPPRE.OBBTMIG                = MINLPBND.OBBTMIG                = 1e-6;
  MINLPPRE.CPMAX                  = MINLPBND.CPMAX                  = 10;
  MINLPPRE.CPTHRES                = MINLPBND.CPTHRES                = 0.;
  MINLPPRE.CMODPROP               = MINLPBND.CMODPROP               = 15;
  MINLPPRE.CMODEL.MIN_FACTOR      = MINLPBND.CMODEL.MIN_FACTOR      = 1e-13;
  MINLPPRE.SQUAD.BASIS            = MINLPBND.SQUAD.BASIS            = SQuad::Options::MONOM;
  MINLPPRE.SQUAD.ORDER            = MINLPBND.SQUAD.ORDER            = SQuad::Options::INC;
  MINLPPRE.MONSCALE               = MINLPBND.MONSCALE               = 0;
  MINLPPRE.RRLTCUTS               = MINLPBND.RRLTCUTS               = 0;
  MINLPPRE.SQUAD.REDUC            = MINLPBND.SQUAD.REDUC            = 1;
  MINLPPRE.PSDQUADCUTS            = MINLPBND.PSDQUADCUTS            = 1;
  MINLPPRE.DCQUADCUTS             = MINLPBND.DCQUADCUTS             = 0;
  MINLPPRE.NCOCUTS                = MINLPBND.NCOCUTS                = 0;
  MINLPPRE.NCOADIFF               = MINLPBND.NCOADIFF               = MINLPBND.ASA;
  MINLPPRE.DISPLEVEL              = MINLPBND.DISPLEVEL              = 0;
  MINLPPRE.MIPSLV.MIPRELGAP       = MINLPBND.MIPSLV.MIPRELGAP       = 1e-3;
  MINLPPRE.MIPSLV.MIPABSGAP       = MINLPBND.MIPSLV.MIPABSGAP       = 1e-5;
  MINLPPRE.MIPSLV.PWLRELGAP       = MINLPBND.MIPSLV.PWLRELGAP       = 1e-3;
  MINLPPRE.MIPSLV.HEURISTICS      = MINLPBND.MIPSLV.HEURISTICS      = 5e-2;
  MINLPPRE.MIPSLV.NUMERICFOCUS    = MINLPBND.MIPSLV.NUMERICFOCUS    = 0;
  MINLPPRE.MIPSLV.SCALEFLAG       = MINLPBND.MIPSLV.SCALEFLAG       = -1;
  MINLPPRE.MIPSLV.DISPLEVEL                                         = 0;
  MINLPBND.MIPSLV.DISPLEVEL                                         = 0;
  MINLPPRE.MIPSLV.OUTPUTFILE      = MINLPBND.MIPSLV.OUTPUTFILE      = "";
  MINLPPRE.MIPSLV.THREADS         = MINLPBND.MIPSLV.THREADS         = 0;

  MINLPSLV.MAXITER                = 40;
  MINLPSLV.CPMAX                  = 10;
  MINLPSLV.CPTHRES                = 0.;
  MINLPSLV.MSLOC                  = 16;
  MINLPSLV.DISPLEVEL              = 1;
  MINLPSLV.NLPSLV.FEASTOL         = 1e-7;
  MINLPSLV.NLPSLV.OPTIMTOL        = 1e-5;
  MINLPSLV.NLPSLV.MAXITER         = 200;
  MINLPSLV.NLPSLV.DISPLEVEL       = 0;
  MINLPSLV.NLPSLV.MAXTHREAD       = 0;
  MINLPSLV.MIPSLV.MIPRELGAP       = 1e-3;
  MINLPSLV.MIPSLV.MIPABSGAP       = 1e-5;
  MINLPSLV.MIPSLV.HEURISTICS      = 5e-2;
  MINLPSLV.MIPSLV.NUMERICFOCUS    = 0;
  MINLPSLV.MIPSLV.SCALEFLAG       = -1;
  MINLPSLV.MIPSLV.DISPLEVEL       = 0;
  MINLPSLV.MIPSLV.OUTPUTFILE      = "";
  MINLPSLV.MIPSLV.THREADS         = 0;

  _USROPT.add_options()
    ( "PRESOLVE",  opt::value<int>(&PRESOLVE),             "level of preprocessing" )
    ( "CVATOL",    opt::value<double>(&CVATOL),            "convergence absolute tolerance" )
    ( "CVRTOL",    opt::value<double>(&CVRTOL),            "convergence relative tolerance" )
    ( "FEASTOL",   opt::value<double>(&FEASTOL),           "feasibility tolerance" )
    ( "CORRINC",   opt::value<bool>(&CORRINC),             "feasibility correction of incumbent using KKT multipliers" )
    ( "CUTINC",    opt::value<bool>(&CUTINC),              "add cut at current incumbent in relaxation" )
    ( "BKPTINC",   opt::value<bool>(&BKPTINC),             "add breakpoint at current incumbent in piecewise relaxation" )
    ( "MAXITER",   opt::value<unsigned>(&MAXITER),         "maximal number of iterations" )
    ( "TIMELIMIT", opt::value<double>(&TIMELIMIT),         "runtime limit" )
    ( "DISPLEVEL", opt::value<int>(&DISPLEVEL),            "general display level" )
    ( "LOGFILE",   opt::value<std::string>(&_LOGFILENAME), "log file" )
//
    ( "MINLPBND.PRERELAXMETH",     opt::value<unsigned>(&_MINLPPRE_RELAXMETH),           "polyhedral relaxation approach during presolve" )
    ( "MINLPBND.PREOBBTLIN",       opt::value<unsigned>(&MINLPPRE.OBBTLIN),              "optimization-based bounds tighteneting approach during presolve" )
    ( "MINLPBND.PREOBBTCONT",      opt::value<bool>(&MINLPPRE.OBBTCONT),                 "continuous relaxation for optimization-based bounds tighteneting during presolve" )
    ( "MINLPBND.PREOBBTMAX",       opt::value<unsigned>(&MINLPPRE.OBBTMAX),              "maximum rounds of optimization-based bounds tighteneting during presolve" )
    ( "MINLPBND.PREOBBTTHRES",     opt::value<double>(&MINLPPRE.OBBTTHRES),              "threshold for optimization-based bounds tighteneting repeats during presolve" )
    ( "MINLPBND.PREOBBTBKOFF",     opt::value<double>(&MINLPPRE.OBBTBKOFF),              "backoff for optimization-based bounds tighteneting during presolve" )
    ( "MINLPBND.PREOBBTMIG",       opt::value<double>(&MINLPPRE.OBBTMIG),                "minimum range for optimization-based bounds tighteneting during presolve" )
    ( "MINLPBND.PRERELAXQUAD",     opt::value<bool>(&MINLPPRE.POLIMG.RELAX_QUAD),        "linearize quadratic terms during presolve" )
    ( "MINLPBND.PRERELAXMONOM",    opt::value<int>(&MINLPPRE.POLIMG.RELAX_MONOM),        "linearize monomial terms during presolve" )
    ( "MINLPBND.PRERELAXNLIN",     opt::value<bool>(&MINLPPRE.POLIMG.RELAX_NLIN),        "linearize nonlinear terms during presolve" )
    ( "MINLPBND.PREMIPRELGAP",     opt::value<double>(&MINLPPRE.MIPSLV.MIPRELGAP),       "convergence relative tolerance of MIP solver during presolve" )
    ( "MINLPBND.PREMIPABSGAP",     opt::value<double>(&MINLPPRE.MIPSLV.MIPABSGAP),       "convergence absolute tolerance of MIP solver during presolve" )
    ( "MINLPBND.PREMIPHEURISTICS", opt::value<double>(&MINLPPRE.MIPSLV.HEURISTICS),      "fraction of time spent in MIP heuristics during presolve" )
    ( "MINLPBND.PREMIPNUMERIC",    opt::value<int>(&MINLPPRE.MIPSLV.NUMERICFOCUS),       "control of numerical issues by MIP solver during presolve" )
    ( "MINLPBND.PREMIPSCALE",      opt::value<int>(&MINLPPRE.MIPSLV.SCALEFLAG),           "control of model scaling by MIP solver during presolve" )
    ( "MINLPBND.PREMIPDISPLEVEL",  opt::value<int>(&MINLPPRE.MIPSLV.DISPLEVEL),          "display level of MIP solver during presolve" )
    ( "MINLPBND.PREMIPOUTPUTFILE", opt::value<std::string>(&MINLPPRE.MIPSLV.OUTPUTFILE), "output file for MIP model during presolve" )
    ( "MINLPBND.PREMIPMAXTHREAD",  opt::value<unsigned>(&MINLPPRE.MIPSLV.THREADS),       "number of threads used by MIP solver during presolve" )
//
    ( "MINLPBND.REFORMMETH",    opt::value<unsigned>(&_MINLPBND_REFORMMETH),            "reformulation approach prior to relaxation" )
    ( "MINLPBND.RELAXMETH",     opt::value<unsigned>(&_MINLPBND_RELAXMETH),             "polyhedral relaxation approach" )
    ( "MINLPBND.RELAXQUAD",     opt::value<bool>(&MINLPBND.POLIMG.RELAX_QUAD),          "linearize quadratic terms" )
    ( "MINLPBND.RELAXMONOM",    opt::value<int>(&MINLPBND.POLIMG.RELAX_MONOM),          "linearize monomial terms" )
    ( "MINLPBND.RELAXNLIN",     opt::value<bool>(&MINLPBND.POLIMG.RELAX_NLIN),          "linearize nonlinear terms" )
    ( "MINLPBND.SUBSETDRL",     opt::value<unsigned>(&MINLPBND.SUBSETDRL),              "exclude functions from decomposition-relaxation-linearization" )
    ( "MINLPBND.SUBSETSCQ",     opt::value<unsigned>(&MINLPBND.SUBSETSCQ),              "exclude functions from quadratization" )
    ( "MINLPBND.BCHPRIM",       opt::value<unsigned>(&MINLPBND.BCHPRIM),                "Set higher branch priority to original variables" )
    ( "MINLPBND.OBBTLIN",       opt::value<unsigned>(&MINLPBND.OBBTLIN),                "optimization-based bounds tighteneting approach" )
    ( "MINLPBND.OBBTCONT",      opt::value<bool>(&MINLPBND.OBBTCONT),                   "continuous relaxation for optimization-based bounds tighteneting" )
    ( "MINLPBND.OBBTMAX",       opt::value<unsigned>(&MINLPBND.OBBTMAX),                "maximum rounds of optimization-based bounds tighteneting" )
    ( "MINLPBND.OBBTTHRES",     opt::value<double>(&MINLPBND.OBBTTHRES),                "threshold for optimization-based bounds tighteneting repeats" )
    ( "MINLPBND.OBBTBKOFF",     opt::value<double>(&MINLPBND.OBBTBKOFF),                "backoff for optimization-based bounds tighteneting" )
    ( "MINLPBND.OBBTMIG",       opt::value<double>(&MINLPBND.OBBTMIG),                  "minimum range for optimization-based bounds tighteneting" )
    ( "MINLPBND.CPMAX",         opt::value<unsigned>(&MINLPBND.CPMAX),                  "maximum rounds of constraint propagation" )
    ( "MINLPBND.CPTHRES",       opt::value<double>(&MINLPBND.CPTHRES),                  "threshold for constraint propagation repeats" )
    ( "MINLPBND.CMODPROP",      opt::value<unsigned>(&MINLPBND.CMODPROP),               "maximum order of sparse polynomial model" )
    ( "MINLPBND.MONMIG",        opt::value<double>(&MINLPBND.CMODEL.MIN_FACTOR),        "monomial minimal coefficient in sparse polynomial model" )
    ( "MINLPBND.MONBASIS",      opt::value<int>(&MINLPBND.SQUAD.BASIS),                 "monomial basis in sparse quadratic form" )
    ( "MINLPBND.MONORDER",      opt::value<int>(&MINLPBND.SQUAD.ORDER),                 "monomial processing order in sparse quadratic form" )
    ( "MINLPBND.MONSCALE",      opt::value<bool>(&MINLPBND.MONSCALE),                   "monomial scaling in sparse quadratic form" )
    ( "MINLPBND.REDQUADCUTS",   opt::value<bool>(&MINLPBND.SQUAD.REDUC),                "add redundant cuts within quadratisation" )
    ( "MINLPBND.PSDQUADCUTS",   opt::value<unsigned>(&MINLPBND.PSDQUADCUTS),            "add PSD cuts within quadratisation" )
    ( "MINLPBND.DCQUADCUTS",    opt::value<bool>(&MINLPBND.DCQUADCUTS),                 "add DC cuts within quadratisation" )
    ( "MINLPBND.RRLTCUTS",      opt::value<bool>(&MINLPBND.RRLTCUTS),                   "add reduced RLT cuts" )
    ( "MINLPBND.NCOCUTS",       opt::value<bool>(&MINLPBND.NCOCUTS),                    "add NCO cuts" )
    ( "MINLPBND.NCOADIFF",      opt::value<unsigned>(&MINLPBND.NCOADIFF),               "NCO cut generation method" )
    ( "MINLPBND.LINCTRSEP",     opt::value<bool>(&MINLPBND.LINCTRSEP),                  "separate linear constraints during relaxation" )
    ( "MINLPBND.AGGREGLQ",      opt::value<bool>(&MINLPBND.POLIMG.AGGREG_LQ),           "keep linear and quadratic expressions aggregated" )
    ( "MINLPBND.SANDWICHRTOL",  opt::value<double>(&MINLPBND.POLIMG.SANDWICH_RTOL),     "relative tolerance in outer-approximation of univariate terms" )
    ( "MINLPBND.SANDWICHMAX",   opt::value<unsigned>(&MINLPBND.POLIMG.SANDWICH_MAXCUT), "maximal number of cuts in outer-approximation of univariate terms" )
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

template <typename T, typename NLP, typename MIP>
inline typename MINLGO<T,NLP,MIP>::Options&
MINLGO<T,NLP,MIP>::Options::operator=
( Options const& other )
{
  PRESOLVE   = other.PRESOLVE;
  CORRINC    = other.CORRINC;
  CUTINC     = other.CUTINC;
  BKPTINC    = other.BKPTINC;
  FEASTOL    = other.FEASTOL;
  CVATOL     = other.CVATOL;
  CVRTOL     = other.CVRTOL;
  MAXITER    = other.MAXITER;
  TIMELIMIT  = other.TIMELIMIT;
  DISPLEVEL  = other.DISPLEVEL;
  MINLPSLV   = other.MINLPSLV;
  MINLPBND   = other.MINLPBND;
  MINLPPRE   = other.MINLPPRE;       
  return *this;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLGO<T,NLP,MIP>::Options::read
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

template <typename T, typename NLP, typename MIP>
inline bool
MINLGO<T,NLP,MIP>::Options::read
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

  if( _USRMAP.count( "MINLPBND.REFORMMETH" ) )
    MINLPBND.REFORMMETH.clear();
    switch( _MINLPBND_REFORMMETH ){
      default:
      case 2: MINLPBND.REFORMMETH.insert( MINLPBND.QUAD ); // no break
      case 1: MINLPBND.REFORMMETH.insert( MINLPBND.NPOL );
              MINLPPRE.REFORMMETH.insert( MINLPBND.NPOL ); break;
      case 0: break;
    }
    
  if( _USRMAP.count( "MINLPBND.RELAXMETH" ) )
    MINLPBND.RELAXMETH.clear();
    switch( _MINLPBND_RELAXMETH ){
      default:
      case 0: MINLPBND.RELAXMETH.insert( MINLPBND.DRL ); break;
      case 1: MINLPBND.RELAXMETH.insert( MINLPBND.SCQ ); break;
      case 2: MINLPBND.RELAXMETH.insert( { MINLPBND.DRL, MINLPBND.SCQ } ); break;
    }

  if( _USRMAP.count( "MINLPPRE.RELAXMETH" ) )
    MINLPPRE.RELAXMETH.clear();
    switch( _MINLPPRE_RELAXMETH ){
      default:
      case 0: MINLPPRE.RELAXMETH.insert( MINLPPRE.DRL ); break;
      case 1: MINLPPRE.RELAXMETH.insert( MINLPPRE.SCQ ); break;
      case 2: MINLPPRE.RELAXMETH.insert( { MINLPPRE.DRL, MINLPPRE.SCQ } ); break;
    }

  if( _USRMAP.count( "MINLPBND.BCHPRIM"      ) ) MINLPPRE.BCHPRIM                = MINLPBND.BCHPRIM;
  if( _USRMAP.count( "MINLPBND.MIPPWLRELGAP" ) ) MINLPPRE.MIPSLV.PWLRELGAP       = MINLPBND.MIPSLV.PWLRELGAP;
  if( _USRMAP.count( "MINLPBND.CPMAX"        ) ) MINLPPRE.CPMAX                  = MINLPBND.CPMAX;
  if( _USRMAP.count( "MINLPBND.CPTHRES"      ) ) MINLPPRE.CPTHRES                = MINLPBND.CPTHRES;
  if( _USRMAP.count( "MINLPBND.CMODPROP"     ) ) MINLPPRE.CMODPROP               = MINLPBND.CMODPROP;
  if( _USRMAP.count( "MINLPBND.MONMIG"       ) ) MINLPPRE.CMODEL.MIN_FACTOR      = MINLPBND.CMODEL.MIN_FACTOR;
  if( _USRMAP.count( "MINLPBND.MONBASIS"     ) ) MINLPPRE.SQUAD.BASIS            = MINLPBND.SQUAD.BASIS;
  if( _USRMAP.count( "MINLPBND.MONORDER"     ) ) MINLPPRE.SQUAD.ORDER            = MINLPBND.SQUAD.ORDER;
  if( _USRMAP.count( "MINLPBND.MONSCALE"     ) ) MINLPPRE.MONSCALE               = MINLPBND.MONSCALE;
  if( _USRMAP.count( "MINLPBND.REDQUADCUTS"  ) ) MINLPPRE.SQUAD.REDUC            = MINLPBND.SQUAD.REDUC;
  if( _USRMAP.count( "MINLPBND.PSDQUADCUTS"  ) ) MINLPPRE.PSDQUADCUTS            = MINLPBND.PSDQUADCUTS;
  if( _USRMAP.count( "MINLPBND.DCQUADCUTS"   ) ) MINLPPRE.DCQUADCUTS             = MINLPBND.DCQUADCUTS;
  if( _USRMAP.count( "MINLPBND.RRLTCUTS"     ) ) MINLPPRE.RRLTCUTS               = MINLPBND.RRLTCUTS;
  if( _USRMAP.count( "MINLPBND.NCOCUTS"      ) ) MINLPPRE.NCOCUTS                = MINLPBND.NCOCUTS;
  if( _USRMAP.count( "MINLPBND.NCOADIFF"     ) ) MINLPPRE.NCOADIFF               = MINLPBND.NCOADIFF;
  if( _USRMAP.count( "MINLPBND.LINCTRSEP"    ) ) MINLPPRE.LINCTRSEP              = MINLPBND.LINCTRSEP;
  if( _USRMAP.count( "MINLPBND.AGGREGLQ"     ) ) MINLPPRE.POLIMG.AGGREG_LQ       = MINLPBND.POLIMG.AGGREG_LQ;
  if( _USRMAP.count( "MINLPBND.SANDWICHRTOL" ) ) MINLPPRE.POLIMG.SANDWICH_RTOL   = MINLPBND.POLIMG.SANDWICH_RTOL;
  if( _USRMAP.count( "MINLPBND.SANDWICHMAX"  ) ) MINLPPRE.POLIMG.SANDWICH_MAXCUT = MINLPBND.POLIMG.SANDWICH_MAXCUT;

  return true;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLGO<T,NLP,MIP>::Options::display
( std::ostream&os ) const
{
  // Display MINLGO Options
  os << std::left;
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
  os << std::setw(60) << "  INCUMBENT CUT"
     << (CUTINC?'Y':'N') << std::endl;
  os << std::setw(60) << "  INCUMBENT BREAKPOINT"
     << (BKPTINC?'Y':'N') << std::endl;
  os << std::setw(60) << "  MAXIMAL ITERATIONS"
     << MAXITER << std::endl;
  os << std::setw(60) << "  PRESOLVE LEVEL"
     << PRESOLVE << std::endl;
  os << std::setw(60) << "  TIME LIMIT (SEC)"
     << std::scientific << std::setprecision(1)
     << TIMELIMIT << std::endl;
}

template <typename T, typename NLP, typename MIP>
inline std::ostream&
operator <<
( std::ostream & os, MINLGO<T,NLP,MIP> const& MINLP )
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

//template <typename T, typename NLP, typename MIP>
//inline const double*
//MINLGO<T,NLP,MIP>::_get_SLVLOC
//( const SOLUTION_OPT&locopt )
//{
//  //std::cout << locopt.n << " =?= " << BASE_NLP::_var.size() << " + " << BASE_NLP::_dep.size() << std::endl; 
//  assert( locopt.p.size() == BASE_NLP::_var.size() + BASE_NLP::_dep.size() );
//  assert( locopt.g.size() == std::get<0>(BASE_NLP::_ctr).size() + (int)BASE_NLP::_sys.size() );
//  _Dvar = locopt.p;
//  if( options.NCOCUTS ){
//    //_Dvar.insert( _Dvar.end(), _nvar-locopt.n, 0. ); // <- MODIFY TO USE ACTUAL FJ MULTIPLIER VALUES (AFTER RESCALING)?
//    _Dvar.push_back( 1. );
//    double mult_ineq = 1., mult_eq = 0.;
//    unsigned ig = 0;
//    for( auto&& t_ctr : std::get<0>(BASE_NLP::_ctr) ){
//      switch( t_ctr ){
//        case BASE_OPT::EQ: 
//          _Dvar.push_back( locopt.ug[ig] );
//          mult_eq += mc::sqr( locopt.ug[ig] );
//          break;
//        case BASE_OPT::LE:
//        case BASE_OPT::GE:
//          _Dvar.push_back( std::fabs( locopt.ug[ig] ) );
//          mult_ineq += std::fabs( locopt.ug[ig] );
//          break;
//      }
//      ig++;
//    }
//    for( auto it=_sys.begin(); it!=_sys.end(); ++it ){
//      _Dvar.push_back( locopt.ug[ig] );
//      mult_eq += sqr( locopt.ug[ig] );
//      ig++;
//    }
//    for( unsigned ivar=0; ivar<locopt.p.size(); ivar++ ){
//      if( !_tvar.empty() && _tvar[ivar] ) continue;
//      _Dvar.push_back( std::fabs( locopt.upL[ivar] ) );
//      _Dvar.push_back( std::fabs( locopt.upU[ivar] ) );
//      mult_ineq += std::fabs( locopt.upL[ivar] ) + std::fabs( locopt.upU[ivar] );
//    }
//    double mult_scal = ( - mult_ineq + std::sqrt( mult_ineq * mult_ineq + 4. * mult_eq ) )
//                      / ( 2. * mult_eq );
//    for( unsigned i=locopt.p.size(); i<_nvar; i++ ){
//      _Dvar[i] *= mult_scal;
//      //std::cout << "Dvar[" << i << "] = " << _Dvar[i] << std::endl;
//    }
//    //for( int i=0; i<locopt.m; i++ )
//    //  std::cout << "ug[" << i << "] = " << locopt.ug[i] << std::endl;
//    //for( int i=0; i<locopt.n; i++ ){
//    //  std::cout << "upL[" << i << "] = " << locopt.upL[i] << std::endl;
//    //  std::cout << "upU[" << i << "] = " << locopt.upU[i] << std::endl;
//    //}
//    //std::cout << "mult_eq = " << mult_eq << std::endl;
//    //std::cout << "mult_ineq = " << mult_ineq << std::endl;
//    //std::cout << "mult_scal = " << mult_scal << std::endl;
//  }
//  return _Dvar.data();
//}

} // end namescape mc

#endif

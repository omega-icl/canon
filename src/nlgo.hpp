// Copyright (C) 2015-2016 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_NLGO Nonlinear Global Optimization using Gurobi, SNOPT and MC++
\author Benoit C. Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt> and OMEGA Research Group (http://www3.imperial.ac.uk/environmentenergyoptimisation)
\version 1.0
\date 2015
\bug No known bugs.

Consider a nonlinear optimization problem in the form:
\f{align*}
\mathcal{P}:\quad & \min_{x_1,\ldots,x_n}\ f(x_1,\ldots,x_n)\\
& {\rm s.t.}\ \ g_j(x_1,\ldots,x_n)\ \leq,=,\geq\ 0,\ \ j=1,\ldots,m\\
& \qquad x_i^L\leq x_i\leq x_i^U,\ \ i=1,\ldots,n\,,
\f}
where \f$f, g_1, \ldots, g_m\f$ are factorable, potentially nonlinear, real-valued functions; and \f$x_1, \ldots, x_n\f$ can be either continuous or integer decision variables. The class mc::NLGO solves such NLP or MINLP problems to global optimality using complete search. Two main methods are implemented in mc::NLGO:
- spatial branch-and-bound search
- hierarchy of semi-linear relaxations
.
In both methods, relaxations are generated for the nonlinear or nonconvex participating terms using various arithmetics in <A href="https://projects.coin-or.org/MCpp">MC++</A>.

\section sec_NLGO_setup How do I setup my optimization model?

Consider the following NLP:
\f{align*}
  \max_{\bf p}\ & p_1\,p_4\,(p_1+p_2+p_3)+p_3 \\
  \text{s.t.} \ & p_1\,p_2\,p_3\,p_4 \geq 25 \\
  & p_1^2+p_2^2+p_3^2+p_4^2 = 40 \\
  & 1 \leq p_1,p_2,p_3,p_4 \leq 5\,.
\f}

First, we define an mc::NLGO class as below:

\code
  mc::NLGO NLP;
\endcode

Next, we set the variables and objective/constraint functions by creating a direct acyclic graph (DAG) of the problem: 

\code
  #include "NLGO.hpp"
  
  mc::FFGraph DAG;
  const unsigned NP = 4; mc::FFVar p[NP];
  for( unsigned i=0; i<NP; i++ ) p[i].set( &DAG );

  NLP.set_dag( &DAG );  // DAG
  NLP.set_var( NP, p ); // decision variables
  NLP.set_obj( mc::BASE_NLP::MIN, (p[0]*p[3])*(p[0]+p[1]+p[2])+p[2] ); // ojective
  NLP.add_ctr( mc::BASE_NLP::GE,  (p[0]*p[3])*p[1]*p[2]-25 );          // constraints
  NLP.add_ctr( mc::BASE_NLP::EQ,  sqr(p[0])+sqr(p[1])+sqr(p[2])+sqr(p[3])-40 );
  NLP.setup();
\endcode

The variable bounds and types are passed to mc::NLGO in invoking the various methods, as described below.


\section sec_NLGO_methods What are the methods available?


Given initial bounds \f$P\f$ and initial guesses \f$p_0\f$ on the decision variables, the NLP model is solved using branch-and-bound search (default) as follows:

\code
  #include "interval.hpp"

  typedef mc::Interval I;
  I Ip[NP] = { I(1,5), I(1,5), I(1,5), I(1,5) };

  std::cout << NLP;
  int status = NLP.solve( Ip );
\endcode

The following result is produced:

\verbatim
______________________________________________________________________

               NONLINEAR GLOBAL OPTIMIZATION IN CRONOS
______________________________________________________________________

  COMPLETE SEARCH METHOD                                    SBB
  CONVERGENCE ABSOLUTE TOLERANCE                            1.0e-03
  CONVERGENCE RELATIVE TOLERANCE                            1.0e-03
  ROOT NODE PROPROCESSING                                   Y
  OPTIMIZATION-BASED REDUCTION MAX LOOPS                    10
  OPTIMIZATION-BASED REDUCTION THRESHOLD LOOP               20%
  MAXIMUM ITERATION COUNT                                   -
  MAXIMUM CPU TIME (SEC)                                    7.2e+03
  DISPLAY LEVEL                                             2
 _______________________________________________________________________

 INDEX STACK    CUMUL TIME        RELAX         INC   PARENT        LBD           UBD         ACTION  

     1     1  0.000000e+00 -1.000000e+20  1.701402e+01     0  1.511037e+01  1.701402e+01       BRANCH2
     2     2  2.446400e-02  1.511037e+01  1.701402e+01     1  1.659980e+01  1.701402e+01       BRANCH3
     3     3  3.611400e-02  1.511037e+01  1.701402e+01     1  1.614164e+01  1.701402e+01       BRANCH2
     4     4  5.255800e-02  1.614164e+01  1.701402e+01     3  1.669923e+01  1.701402e+01       BRANCH3
     5     5  6.833200e-02  1.614164e+01  1.701402e+01     3  1.713529e+01       SKIPPED        FATHOM
     6     4  8.233300e-02  1.659980e+01  1.701402e+01     2  1.664991e+01  1.701402e+01       BRANCH2
     7     5  9.279500e-02  1.659980e+01  1.701402e+01     2  1.701402e+01       SKIPPED        FATHOM
     8     4  1.237580e-01  1.664991e+01  1.701402e+01     6  1.703133e+01       SKIPPED        FATHOM
     9     3  1.286230e-01  1.664991e+01  1.701402e+01     6  1.701402e+01       SKIPPED        FATHOM
    10     2  1.553660e-01  1.669923e+01  1.701402e+01     4  1.672776e+01  1.701402e+01       BRANCH3
    11     3  1.638520e-01  1.669923e+01  1.701402e+01     4  1.701406e+01       SKIPPED        FATHOM
    12     2  1.661390e-01  1.672776e+01  1.701402e+01    10  1.676720e+01  1.704213e+01       BRANCH2
    13     3  1.748900e-01  1.672776e+01  1.701402e+01    10  1.701402e+01       SKIPPED        FATHOM
    14     2  2.044000e-01  1.676720e+01  1.701402e+01    12    INFEASIBLE       SKIPPED        FATHOM
    15     1  2.149610e-01  1.676720e+01  1.701402e+01    12  1.703627e+01       SKIPPED        FATHOM

#  NORMAL TERMINATION: 0.222576 CPU SEC  (LBD:80.6%  UBD:19.3%)
#  INCUMBENT VALUE:  1.701402e+01
#  INCUMBENT POINT:  1.000000e+00  4.742955e+00  3.821208e+00  1.379400e+00
#  INCUMBENT FOUND AT NODE: 10
#  TOTAL NUMBER OF NODES:   15
#  MAXIMUM NODES IN STACK:  5
\endverbatim

Integrality of certain variables can be enforced by passing an additional argument as follows, here enforcing integrality of the second and third variables:

\code
  unsigned Tp[NP] = { 0, 1, 1, 0 };

  int status = NLP.solve( Ip, Tp );
\endcode

The following result is now produced:

\verbatim
 INDEX STACK    CUMUL TIME        RELAX         INC   PARENT        LBD           UBD         ACTION  

     1     1  0.000000e+00 -1.000000e+20  1.000000e+20     0  1.224000e+01  2.312461e+01       BRANCH0
     2     2  2.387200e-02  1.224000e+01  2.312461e+01     1  1.224000e+01  2.312461e+01       BRANCH1
     3     3  3.766500e-02  1.224000e+01  2.312461e+01     1  3.200000e+01       SKIPPED        FATHOM
     4     2  4.145500e-02  1.224000e+01  2.312461e+01     2  2.501911e+01       SKIPPED        FATHOM
     5     1  7.219900e-02  1.224000e+01  2.312461e+01     2  1.488304e+01  2.312461e+01       BRANCH2
     6     2  1.082510e-01  1.488304e+01  2.312461e+01     5  2.312460e+01       SKIPPED        FATHOM
     7     1  1.395070e-01  1.488304e+01  2.312461e+01     5  2.012448e+01  2.312461e+01       BRANCH1
     8     2  1.729230e-01  2.012448e+01  2.312461e+01     7  2.511437e+01       SKIPPED        FATHOM
     9     1  1.866310e-01  2.012448e+01  2.312461e+01     7  2.312458e+01       SKIPPED        FATHOM

#  NORMAL TERMINATION: 0.215300 CPU SEC  (LBD:94.3%  UBD:5.7%)
#  INCUMBENT VALUE:  2.312461e+01
#  INCUMBENT POINT:  1.000000e+00  5.000000e+00  3.000000e+00  2.236068e+00
#  INCUMBENT FOUND AT NODE: 7
#  TOTAL NUMBER OF NODES:   9
#  MAXIMUM NODES IN STACK:  3
\endverbatim

The complete search algorithm based on piecewise linearization can be selected by modifying the options as follows:

\code
  NLP.options.CSALGO = mc::NLGO<I>::Options::PWL;
  NLP.options.POLIMG.BREAKPOINT_TYPE = mc::PolImg<I>::Options::BIN;
  int status = NLP.solve( Ip, Tp );
\endcode

The following result is now produced:

\verbatim
     ITER  PREPROC      RELAX          BEST        CPU TOT          STATUS
        1        -  1.224000e+01  2.312461e+01  5.066300e-02        REFINE
        2        -  1.410526e+01  2.312461e+01  1.267340e-01        REFINE
        3       R3  1.425993e+01  2.312461e+01  2.750010e-01        REFINE
        4       R1  1.488303e+01  2.312461e+01  3.852940e-01        REFINE
        5       R3  2.312456e+01  2.312461e+01  5.360410e-01       OPTIMAL

#  TERMINATION: 0.536041 CPU SEC
#  INCUMBENT VALUE:  2.312461e+01
#  INCUMBENT POINT:  1.000000e+00  5.000000e+00  3.000000e+00  2.236068e+00
#  NUMBER OF RELAXATION REFINEMENTS:   5
#  OPTIMALITY GAP:   4.746061e-05 (ABS)
                     2.052387e-06 (REL)
\endverbatim

Other options can be modified to tailor the search, including output level, maximum number of iterations, tolerances, maximum CPU time, etc. These options can be modified through the public member mc::NLGO::options. 
*/

//TODO: 
//- [OK]    Implement SBB method
//- [OK]    Enable use of polynomial models in relaxations
//- [OK]    Enable multiple rounds of PWR refinement
//- [OK]    Support MINLP
//- [OK]    Exclude linear variables from bound contraction
//- [TO DO] Enable KKT cuts and reduction constraints
//- [TO DO] Make it possible to add/remove a constraint from the model?
//- [TO DO] Exploit the degree of separability to introduce auxiliary variables
//- [TO DO] Enable constraint propagation before generating the cuts in PolImg
//- [TO DO] Enable multi-start in local solver

#ifndef MC__NLGO_HPP
#define MC__NLGO_HPP

#include <chrono>
#include "gamsio.hpp"
#include "nlpslv_snopt.hpp"
#include "nlpbnd.hpp"

namespace mc
{

//! @brief C++ class for global optimization of MINLP using complete search
////////////////////////////////////////////////////////////////////////
//! mc::NLGO is a C++ class for global optimization of NLP and
//! MINLP using complete search. Relaxations for the nonlinear or
//! nonconvex participating terms are generated using MC++. Further
//! details can be found at: \ref page_NLGO
////////////////////////////////////////////////////////////////////////
template < typename T >
class NLGO:
  protected virtual GAMSIO
{
public:

  //! @brief NLGO options
  struct Options
  {
    //! @brief Constructor
    Options():
      CSALGO(QUAD), PREPROC(true), FEASTOL(1e-7), CVATOL(1e-3), CVRTOL(1e-3),
      MSLOC(10), TIMELIMIT(6e2), DISPLEVEL(1),
      NLPSLV(), NLPBND()
      { NLPBND.CMODPROP = 16;
        //NLPBND.MIPSLV.MIPRELGAP = CVRTOL;
        //NLPBND.MIPSLV.MIPABSGAP = CVATOL;        
        NLPSLV.DISPLEVEL  =  NLPBND.DISPLEVEL  = 0;
        NLPSLV.TIMELIMIT  =  NLPBND.TIMELIMIT  = TIMELIMIT; }
    //! @brief Assignment operator
    Options& operator= ( Options&options ){
        CSALGO        = options.CSALGO;
        PREPROC       = options.PREPROC;
        FEASTOL       = options.FEASTOL;
        CVATOL        = options.CVATOL;
        CVRTOL        = options.CVRTOL;
        MSLOC         = options.MSLOC;
        TIMELIMIT     = options.TIMELIMIT;
        DISPLEVEL     = options.DISPLEVEL;
        NLPSLV        = options.NLPSLV;
        NLPBND        = options.NLPBND;
        return *this ;
      }
    //! @brief Complete search method
    enum CS{
      MS=0,     //!< Multistart local optimization
      PWL,	    //!< Hierarchy of piecewise-linear relaxations
      QUAD      //!< Reformulation as MIQCQP
    };
    //! @brief Complete search algorithm
    unsigned CSALGO;
    //! @brief Whether or not to preprocess the root node (local optimization, domain reduction)
    bool PREPROC;
    //! @brief Feasibility tolerance 
    double FEASTOL;
    //! @brief Convergence absolute tolerance
    double CVATOL;
    //! @brief Convergence relative tolerance
    double CVRTOL;
    //! @brief Maximum number of multistart local search
    unsigned MSLOC;
    //! @brief Maximum run time (seconds)
    double TIMELIMIT;
    //! @brief Display level for solver
    int DISPLEVEL;
    //! @brief Set options of NLPSLV_SNOPT (local solution of factorable NLP)
    typename NLPSLV_SNOPT::Options NLPSLV;
    //! @brief Set options of NLPBND (global bounding of factorable NLP)
    typename NLPBND<T>::Options NLPBND;
    //! @brief Display
    void display
      ( std::ostream&out=std::cout ) const;
  } options;

  //! @brief Class managing exceptions for NLGO
  class Exceptions
  {
  public:
    //! @brief Enumeration type for NLGO exception handling
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
        return "NLGO::Exceptions  Incomplete setup before a solve";
      case INTERN: default:
        return "NLGO::Exceptions  Internal error";
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
           << "WALL-CLOCK TIMES" << std::endl
           << " SETUP:        " << std::setw(10) << walltime_setup.count()*1e-6   << " SEC" << std::endl
           << " PREPROCESSOR: " << std::setw(10) << walltime_preproc.count()*1e-6 << " SEC" << std::endl
           << " LOCAL SOLVER: " << std::setw(10) << walltime_slvloc.count()*1e-6  << " SEC" << std::endl
           << " MIP SOLVER:   " << std::setw(10) << walltime_slvrel.count()*1e-6  << " SEC" << std::endl
           << " TOTAL:        " << std::setw(10) << walltime_all.count()*1e-6     << " SEC" << std::endl
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
  } stats;

protected:

  //! @brief Flag for setup function
  bool _issetup;

  //! @brief Flag for MIP problem
  bool _ismip;

  //! @brief Flag for boundedness
  bool _isbnd;

  //! @brief objective scaling coefficient (1: min; -1: max)
  double _objscal;

  //! @brief Current incumbent value
  double _Finc;
  
  //! @brief Variable bounds
  std::vector<T> _Xbnd;
  
  //! @brief Variable values at current incumbent
  std::vector<double> _Xinc;

  //! @brief Local solver for factorable NLP
  NLPSLV_SNOPT _NLPSLV;

  //! @brief Global bounder for factorable NLP
  NLPBND<T> _NLPBND;

  //! @brief maximum number of values displayed in a row
  static const unsigned int _LDISP = 4;

  //! @brief reserved space for integer variable display
  static const unsigned int _IPREC = 9;

  //! @brief reserved space for double variable display
  static const unsigned int _DPREC = 6;

  //! @brief stringstream for displaying results
  std::ostringstream _odisp;
  
  //! @brief Initialize display
  void _pwl_display_init
    ();
    
  //! @brief Final display
  void _pwl_display_final
    ( unsigned const iter, std::chrono::microseconds const& walltime );

  //! @brief Add double to display
  void _pwl_display_add
    ( const double dval );

  //! @brief Add unsigned int to display
  void _pwl_display_add
    ( const int ival );

  //! @brief Add string to display
  void _pwl_display_add
    ( const std::string &sval );

  //! @brief Display current buffer stream and reset it
  void _pwl_display_flush
    ( std::ostream& os = std::cout );

  //! @brief Solve preprocessed optimization model using piecewise-linear relaxation approach
  int _pwl_optimize
    ( std::ostream& os = std::cout );

  //! @brief Solve preprocessed optimization model using quadratic reformulation approach
  int _quad_optimize
    ();

public:

  //! @brief Constructor
  NLGO()
    : _issetup(false)
    { stats.reset(); }

  //! @brief Destructor
  virtual ~NLGO()
    {}

  //! @brief Load optimization model from GAMS file
#if defined (MC__WITH_GAMS)
  bool read
    ( std::string const& filename );
#endif

  //! @brief Setup DAG for cost and constraint evaluation
  void setup
    ();

  //! @brief Preprocess optimization model - return value is false if model is provably infeasible
  bool preprocess
    (  T* Xbnd=nullptr, double* Xini=nullptr, std::ostream& os=std::cout );

  //! @brief Solve optimization model to global optimality after preprocessing
  int optimize
    ();

  //! @brief Get incumbent after last optimization
  std::pair<double,double const*> get_incumbent
    () const
    { return std::make_pair(_Finc,_Xinc.data()); }

private:
  //! @brief Private methods to block default compiler methods
  NLGO
    ( NLGO const& );
  NLGO& operator=
    ( NLGO const& );
};

#if defined (MC__WITH_GAMS)
template <typename T>
inline bool
NLGO<T>::read
( std::string const& filename )
{
  auto tstart = stats.start();

  bool flag = this->GAMSIO::read( filename, options.DISPLEVEL>1? true: false );

  stats.walltime_setup += stats.walltime( tstart );
  stats.walltime_all   += stats.walltime( tstart );
  return flag;  
}
#endif

template <typename T>
inline void
NLGO<T>::setup
()
{
  //stats.reset();
  auto tstart = stats.start();

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
  
#ifdef MC__NLGO_PREPROCESS_DEBUG
  std::cout << "NLPSLV set-up" << std::endl;
#endif
  _NLPSLV.options = options.NLPSLV;
  _NLPSLV.set( *this );
  _NLPSLV.setup();
  _Xinc.clear();

#ifdef MC__NLGO_PREPROCESS_DEBUG
  std::cout << "NLPBND set-up" << std::endl;
#endif
  _NLPBND.options = options.NLPBND;
  _NLPBND.set( *this );
  _NLPBND.setup();
  _Xbnd.clear();

  _issetup = true;

  stats.walltime_setup += stats.walltime( tstart );
  stats.walltime_all   += stats.walltime( tstart );
}

template <typename T>
inline bool
NLGO<T>::preprocess
(  T* Xbnd, double* Xini, std::ostream& os )
{
  auto tstart = stats.start();

  _Xinc.clear();
  if( Xini )
    _varini.assign( Xini, Xini+_var.size() );
  if( options.DISPLEVEL > 1 ){
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Initial point\n@";
    for( auto const& Xi : _varini ) std::cout << " " << Xi;
    std::cout << std::endl;
#ifdef MC__NLGO_PREPROCESS_DEBUG
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }

  if( Xbnd )
    _Xbnd.assign( Xbnd, Xbnd+_var.size() );
  else
    _Xbnd.assign( _var.size(), T( -BASE_OPT::INF, BASE_OPT::INF ) );

  // Check feasibility of user-supplied point - INTEGRALITY CHECK?!?
  if( _NLPSLV.is_feasible( _varini.data(), options.FEASTOL ) ){
    _Xinc = _varini;
    _Finc = _NLPSLV.solution().f[0];// * ( 1 + _objscal * options.CVRTOL );
  }

  // Solve local optimization from user-supplied point
  //bool FEASPBSAVE = _NLPSLV.options.FEASPB;
  //_NLPSLV.options.FEASPB = true;
  _NLPSLV.solve( _varini.data(), _Xbnd.data() );
  //_NLPSLV.options.FEASPB = FEASPBSAVE;
  if( _NLPSLV.is_feasible( options.FEASTOL ) ){
    bool feas = true;
    if( _ismip && _NLPSLV.options.RELAXINT ){ // Perform rounding
      std::vector<T> Xround(_var.size());
      for( unsigned i=0; i<_var.size(); i++ ){
        if( _vartyp[i] )
          Xround[i] = std::round( _NLPSLV.solution().x[i] );
        else
          Xround[i] = _Xbnd[i];
      }
      _NLPSLV.solve( _varini.data(), Xround.data() );
      feas = _NLPSLV.is_feasible( options.FEASTOL );
    }
    if( feas && (_Xinc.empty() || _objscal*_NLPSLV.solution().f[0] < _objscal*_Finc) ){
      _Xinc = _NLPSLV.solution().x;
      _Finc = _NLPSLV.solution().f[0];// * ( 1 + _objscal * options.CVRTOL );
    }
  }

  if( options.DISPLEVEL > 1 ){
    if( _Xinc.empty() )
      std::cout << "Incumbent found: -" << std::endl;
    else{
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Incumbent found: " << _Finc << std::endl
                << "@";
      for( auto const& Xi : _Xinc ) std::cout << " " << Xi;
      std::cout << std::endl;
    }
#ifdef MC__NLGO_PREPROCESS_DEBUG
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }

  // Apply domain contraction with constraint propagation and optimization-based bound tightnening on linear constraints only
  unsigned nred;
  _NLPBND.options.RELMETH = { NLPBND<T>::Options::DRL };
  _NLPBND.options.OBBTLIN = 0;
  if( _NLPBND.reduce( nred, _Xbnd.data(), !_Xinc.empty()? &_Finc: nullptr ) == MIPSLV_GUROBI<T>::INFEASIBLE ){
    stats.walltime_preproc += stats.walltime( tstart );
    stats.walltime_all     += stats.walltime( tstart );
    return false;
  }
  _Xbnd.assign( _NLPBND.varbnd(), _NLPBND.varbnd()+_var.size() );
  for( unsigned i=0; Xbnd && i<_var.size(); i++ ) Xbnd[i] = _Xbnd[i];

  if( options.DISPLEVEL > 1 ){
    std::cout << "Reduced bounds:" << std::endl;
    for( auto const& Xi : _Xbnd ) std::cout << " " << Xi;
    std::cout << std::endl;
#ifdef MC__NLGO_PREPROCESS_DEBUG
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }

  _isbnd = true;
  for( auto const& Xi : _Xbnd ){
    if( Op<T>::diam(Xi) < BASE_OPT::INF/2 ) continue;
    _isbnd = false;
    break;
  }

  // Solve local optimization using multistart search
  if( options.MSLOC && _isbnd ){
    _NLPSLV.solve( options.MSLOC, _Xbnd.data() );
    if( _NLPSLV.is_feasible( options.FEASTOL ) ){
      bool feas = true;
      if( _ismip && _NLPSLV.options.RELAXINT ){ // Perform rounding
        if( options.DISPLEVEL > 1 ){
          std::cout << std::scientific << std::setprecision(4);
          std::cout << "Incumbent found: " << _NLPSLV.solution().f[0] << std::endl
                    << "@";
          for( auto const& Xi : _NLPSLV.solution().x ) std::cout << " " << Xi;
          std::cout << std::endl;
#ifdef MC__NLGO_PREPROCESS_DEBUG
          { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
        }
//        _NLPSLV.options.RELAXINT = false;
//        _NLPSLV.setup();
//        _NLPSLV.solve( _NLPSLV.solution().x.data(), _Xbnd.data() );
        std::vector<T> Xround(_var.size());
        for( unsigned i=0; i<_var.size(); i++ ){
          if( _vartyp[i] )
            Xround[i] = std::round( _NLPSLV.solution().x[i] );
          else
            Xround[i] = _Xbnd[i];
//          else if( Xbnd )
//            Xround[i] = Xbnd[i];
//          else
//            Xround[i] = T( -BASE_OPT::INF, BASE_OPT::INF );
        }
        _NLPSLV.solve( _NLPSLV.solution().x.data(), Xround.data() );
        feas = _NLPSLV.is_feasible( options.FEASTOL );
      }
      if( feas && (_Xinc.empty() || _objscal*_NLPSLV.solution().f[0] < _objscal*_Finc) ){
        _Xinc = _NLPSLV.solution().x;
        _Finc = _NLPSLV.solution().f[0];// * ( 1 + _objscal * options.CVRTOL );
      }
    }
  }

  if( options.DISPLEVEL > 1 || (options.DISPLEVEL == 1 && options.CSALGO == 0) ){
    if( _Xinc.empty() )
      std::cout << "Incumbent found: -" << std::endl;
    else{
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Incumbent found: " << _Finc << std::endl
                << "@";
      for( auto const& Xi : _Xinc ) std::cout << " " << Xi;
      std::cout << std::endl;
    }
#ifdef MC__NLGO_PREPROCESS_DEBUG
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }
  if( options.CSALGO == 0 ){
    stats.walltime_preproc += stats.walltime( tstart );
    stats.walltime_all     += stats.walltime( tstart );
    return true;
  }

  // Apply domain contraction for both linear and nonlinear constraints
  _NLPBND.options.OBBTLIN = 2;
  _NLPBND.options.RELMETH = { NLPBND<T>::Options::DRL };
  //_NLPBND.options.RELMETH = { NLPBND<T>::Options::SCQ };
  //_NLPBND.options.RELMETH = { NLPBND<T>::Options::DRL, NLPBND<T>::Options::SCQ };
  _NLPBND.options.POLIMG.RELAX_QUAD = true;
  if( _NLPBND.reduce( nred, _Xbnd.data(), !_Xinc.empty()? &_Finc: nullptr ) == MIPSLV_GUROBI<T>::INFEASIBLE ){
    stats.walltime_preproc += stats.walltime( tstart );
    stats.walltime_all     += stats.walltime( tstart );
    return false;
  }
  _Xbnd.assign( _NLPBND.varbnd(), _NLPBND.varbnd()+_var.size() );
  for( unsigned i=0; Xbnd && i<_var.size(); i++ ) Xbnd[i] = _Xbnd[i];

  if( options.DISPLEVEL > 1 ){
    std::cout << "Reduced bounds:" << std::endl;
    for( auto const& Xi : _Xbnd ) std::cout << " " << Xi;
    std::cout << std::endl;
#ifdef MC__NLGO_PREPROCESS_DEBUG
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }

  // Compute relaxation
  //_NLPBND.relax( nullptr, nullptr, nullptr, 0, false );
  _NLPBND.relax( nullptr, nullptr, nullptr, 3, true );
  if( options.DISPLEVEL > 1 ){
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Relaxation bound: " << _NLPBND.solver()->get_objective() << std::endl
              << "@";
    for( auto const& Xi : _var ) std::cout << " " << _NLPBND.solver()->get_variable( Xi );
    std::cout << std::endl;
  #ifdef MC__NLGO_PREPROCESS_DEBUG
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }
  // Solve local optimization from relaxed optimum
  std::vector<T> Xround(_var.size());
  for( unsigned i=0; i<_var.size(); i++ ){
    _varini[i] = _NLPBND.solver()->get_variable( _var[i] );
    if( _vartyp[i] )
      Xround[i] = _varini[i];
    else
      Xround[i] = _Xbnd[i];
  }
  _NLPSLV.solve( _varini.data(), Xround.data() );
  if( _NLPSLV.is_feasible( options.FEASTOL )
   && (_Xinc.empty() || _objscal*_NLPSLV.solution().f[0] < _objscal*_Finc) ){
    _Xinc = _NLPSLV.solution().x;
    _Finc = _NLPSLV.solution().f[0];// * ( 1 + _objscal * options.CVRTOL );
  }
  if( options.DISPLEVEL >= 1 ){
    if( _Xinc.empty() )
      std::cout << "Incumbent found: -" << std::endl;
    else{
      std::cout << std::scientific << std::setprecision(4);
      std::cout << "Incumbent found: " << _Finc << std::endl;
      if( options.DISPLEVEL > 1 ){
        std::cout << "@";
        for( auto const& Xi : _Xinc ) std::cout << " " << Xi;
        std::cout << std::endl;
      }
    }
#ifdef MC__NLGO_PREPROCESS_DEBUG
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }
  
  stats.walltime_preproc += stats.walltime( tstart );
  stats.walltime_all     += stats.walltime( tstart );
  return true;
}

template <typename T>
inline int
NLGO<T>::optimize
()
{
  // Multistart search performed as part of preprocessing
  if( options.CSALGO == Options::MS )
    return MIPSLV_GUROBI<T>::UNDEFINED;

  // Check domain boundedness
  _isbnd = _Xbnd.empty()? false: true;
  for( auto const& Xi : _Xbnd ){
    if( Op<T>::diam(Xi) < BASE_OPT::INF/2 ) continue;
    _isbnd = false;
    break;
  }
  if( !_isbnd ){
    if( options.DISPLEVEL >= 1 ){
      std::cout << "Exit: Unbounded variable domain" << std::endl;
    }
    return MIPSLV_GUROBI<T>::UNBOUNDED;
  }

  // Call optimization algorithm
  if( options.CSALGO == Options::QUAD )
    return _quad_optimize();
  else
    return _pwl_optimize();
}

template <typename T>
inline int
NLGO<T>::_quad_optimize
()
{
  _NLPBND.options.MIPSLV.MIPRELGAP  = options.CVRTOL;
  _NLPBND.options.MIPSLV.MIPABSGAP  = options.CVATOL;
  _NLPBND.options.MIPSLV.TIMELIMIT  = options.TIMELIMIT;
  _NLPBND.options.MIPSLV.DISPLEVEL  = ( options.DISPLEVEL>=1? 1: 0 );
  _NLPBND.options.RELMETH = { NLPBND<T>::Options::SCQ };
  //_NLPBND.options.RELMETH = { NLPBND<T>::Options::DRL, NLPBND<T>::Options::SCQ };
  _NLPBND.options.POLIMG.RELAX_QUAD = false;
  
  auto tstart = stats.start();
  auto flag = _NLPBND.relax( _Xbnd.data(), !_Xinc.empty()? &_Finc: nullptr, _Xinc.data(), 0, true );
  stats.walltime_slvrel += stats.walltime( tstart );
  stats.walltime_all    += stats.walltime( tstart );

  return flag;
}

template <typename T>
inline int
NLGO<T>::_pwl_optimize
( std::ostream& os )
{
  _NLPBND.options.OBBTLIN = 2;
  _NLPBND.options.POLIMG.RELAX_QUAD = true;
  _NLPBND.options.RELMETH = { NLPBND<T>::Options::DRL };
  //_NLPBND.options.RELMETH = { NLPBND<T>::Options::SCQ };
  //_NLPBND.options.RELMETH = { NLPBND<T>::Options::DRL, NLPBND<T>::Options::SCQ };

  // Initialize solve and preprocess
  auto tstart = stats.start();
  std::vector<T> Xbndloc(_var.size());
  _pwl_display_init();

  // Iterative relaxation solution and refinement
  unsigned iter = 1, nred = 0;
  for( ; ; ++iter ){
  
    // iteration and preprocessing display
    std::ostringstream odisp, onred;
    odisp << std::right << std::setw(_IPREC) << iter;
    if( nred ) onred << "  R" << nred;
    else       onred << "-";
    odisp << std::right << std::setw(_IPREC) << onred.str();
    _pwl_display_add( odisp.str() );

    // Set-up relaxation and solve relaxed model
    switch( _NLPBND.relax( _Xbnd.data(), !_Xinc.empty()? &_Finc: nullptr, _Xinc.data(), 0, iter>1? false: true ) ){
     case MIPSLV_GUROBI<T>::OPTIMAL:
      _pwl_display_add( _NLPBND.solver()->get_objective() );
      break;
     case MIPSLV_GUROBI<T>::INFEASIBLE:
      _pwl_display_add( "-" );
      _pwl_display_add( "-" );
      _pwl_display_add( stats.walltime( tstart ).count()*1e-6 );
      _pwl_display_add( "INFEASIBLE" );
      _pwl_display_flush( os );
      _pwl_display_final( iter, stats.walltime( tstart ) );
      _pwl_display_flush( os );
      return _NLPBND.solver()->get_status();
     default:
      _pwl_display_add( "-" );
      _pwl_display_add( "-" );
      _pwl_display_add( stats.walltime( tstart ).count()*1e-6 );
      _pwl_display_add( "FAILURE" );
      _pwl_display_flush( os );
      _pwl_display_final( iter, stats.walltime( tstart ) );
      _pwl_display_flush( os );
      return _NLPBND.solver()->get_status();
    }
    
    // Solve local NLP model (integer variable bounds fixed to relaxed solution if MIP)
    for( unsigned i=0; i<_var.size(); i++ ){
      _varini[i] = _NLPBND.solver()->get_variable( _var[i] );
      if( _vartyp[i] )
        Xbndloc[i] = _varini[i];
      else
        Xbndloc[i] = _Xbnd[i];
    }
    _NLPSLV.solve( _varini.data(), Xbndloc.data() );

    // Update incumbent
    if( _NLPSLV.is_feasible( options.FEASTOL )
     && (_Xinc.empty() || _objscal*_NLPSLV.solution().f[0] < _objscal*_Finc) ){
      _Xinc = _NLPSLV.solution().x;
      _Finc = _NLPSLV.solution().f[0];// * ( 1 + _objscal * options.CVRTOL );
    }

    // Test termination criteria
    if( !_Xinc.empty() ){
      _pwl_display_add( _Finc );
      _pwl_display_add( stats.walltime( tstart ).count()*1e-6 );
      if( std::fabs( _Finc - _NLPBND.solver()->get_objective() ) <= options.CVATOL 
       || std::fabs( _Finc - _NLPBND.solver()->get_objective() ) <= 0.5 * options.CVRTOL
          * std::fabs( _Finc + _NLPBND.solver()->get_objective() ) ){
        _pwl_display_add( "OPTIMAL" );
        _pwl_display_flush( os );
        break;
      }
    }
    else{
      _pwl_display_add( "-" );
      _pwl_display_add( stats.walltime( tstart ).count()*1e-6 );
    }
    _pwl_display_add( "REFINE" );
    _pwl_display_flush( os );

    //if( options.MAXITER && iter >= options.MAXITER ){
    //  _pwl_display_add( options, "INTERRUPT" );
    //  _pwl_display_flush( os );
    //  break;
    //}

    // Refine relaxation via additional breakpoints
    _NLPBND.refine_polrelax(); // _Xinc.data() );

    // Apply domain contraction
    // Do NOT test for infeasibility here, because contraction problem may
    // become infeasible due to round-off in LP solver
    _NLPBND.reduce( nred, _Xbnd.data(), !_Xinc.empty()? &_Finc: nullptr );
    if( options.DISPLEVEL > 1 ){
      std::cout << "Reduced bounds:" << std::endl;
      for( auto const& Xi : _Xbnd ) std::cout << " " << Xi;
      std::cout << std::endl;
#ifdef MC__NLGO_PREPROCESS_DEBUG
      { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
    }
  }

  // Final display
  _pwl_display_final( iter, stats.walltime( tstart ) );
  _pwl_display_flush( os );

  return _NLPBND.solver()->get_status();
}

template <typename T> inline void
NLGO<T>::_pwl_display_init
()
{
  _odisp.str("");
  if( options.DISPLEVEL <= 1 ) return;
  _odisp << std::right
  	 << std::setw(_IPREC) << "ITER"
  	 << std::setw(_IPREC) << "PREPROC"
  	 << std::setw(_DPREC+8) << "RELAX   "
  	 << std::setw(_DPREC+8) << "BEST   "
  	 << std::setw(_DPREC+8) << "TIME     "
  	 << std::setw(_DPREC+8) << "STATUS"
  	 << std::endl;  
}

template <typename T>
inline void
NLGO<T>::_pwl_display_final
( unsigned const iter, std::chrono::microseconds const& walltime )
{
  if( options.DISPLEVEL <= 0 ) return;
  _odisp << std::endl << "#  TERMINATION: ";
  _odisp << std::fixed << std::setprecision(6) << walltime.count()*1e-6 << " SEC"
         << std::endl;

  // No feasible solution found
  if( _Xinc.empty() )
    _odisp << "#  NO FEASIBLE SOLUTION FOUND" << std::endl;

  // Feasible solution found
  else{
    // Incumbent
    _odisp << "#  INCUMBENT VALUE:" << std::scientific
           << std::setprecision(_DPREC) << std::setw(_DPREC+8) << _Finc
           << std::endl;
    _odisp << "#  INCUMBENT POINT:";
    unsigned i(0);
    for( auto const& Xi : _Xinc ){
      if( i == _LDISP ){
        _odisp << std::endl << std::left << std::setw(19) << "#";
        i = 0;
      }
      _odisp << std::right << std::setw(_DPREC+8) << Xi;
    }
    _odisp << std::endl;
  }

  _odisp << "#  NUMBER OF RELAXATION REFINEMENTS:   " << iter << std::endl;
  if( _NLPBND.solver()->get_status() != MIPSLV_GUROBI<T>::OPTIMAL )
    _odisp << "#  SOLUTION INTERRUPTED AFTER RELAXATION FAILURE" << std::endl;
  else{
    _odisp << "#  OPTIMALITY GAP:   "
           << std::fabs( _Finc - _NLPBND.solver()->get_objective() ) << " (ABS)" << std::endl
           << "                     "
           << 2. * std::fabs( _Finc - _NLPBND.solver()->get_objective() )
                 / std::fabs( _Finc + _NLPBND.solver()->get_objective() ) << " (REL)"  << std::endl;
  }
}

template <typename T>
inline void
NLGO<T>::_pwl_display_add
( const double dval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::scientific << std::setprecision(_DPREC)
         << std::setw(_DPREC+8) << dval;
}

template <typename T>
inline void
NLGO<T>::_pwl_display_add
( const int ival )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(_IPREC) << ival;
}

template <typename T>
inline void
NLGO<T>::_pwl_display_add
( const std::string &sval )
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << std::right << std::setw(_DPREC+8) << sval;
}

template <typename T>
inline void
NLGO<T>::_pwl_display_flush
( std::ostream &os )
{
  if( _odisp.str() == "" ) return;
  os << _odisp.str() << std::endl;
  _odisp.str("");
  return;
}

template <typename T>
inline void
NLGO<T>::Options::display
( std::ostream&out ) const
{
  // Display NLGO Options
  out << std::left;
  out << std::setw(60) << "  COMPLETE SEARCH METHOD";
  switch( CSALGO ){
   case MS:   out << "MS"  << std::endl; break;
   case PWL:  out << "PWL"  << std::endl; break;
   case QUAD: out << "QUAD" << std::endl; break;
  }
  out << std::setw(60) << "  CONVERGENCE ABSOLUTE TOLERANCE"
      << std::scientific << std::setprecision(1)
      << CVATOL << std::endl;
  out << std::setw(60) << "  CONVERGENCE RELATIVE TOLERANCE"
      << std::scientific << std::setprecision(1)
      << CVRTOL << std::endl;
  out << std::setw(60) << "  FEASIBILITY TOLERANCE"
      << std::scientific << std::setprecision(1)
      << FEASTOL << std::endl;
  out << std::setw(60) << "  PREPROCESSING"
      << (PREPROC?"Y\n":"N\n");
  out << std::setw(60) << "  TIME LIMIT (SEC)"
      << std::scientific << std::setprecision(1)
      << TIMELIMIT << std::endl;
  out << std::setw(60) << "  DISPLAY LEVEL"
      << DISPLEVEL << std::endl;
}

template <typename T>
inline std::ostream&
operator <<
( std::ostream & out, NLGO<T> const& MINLP )
{
  out << std::right << std::endl
      << std::setfill('_') << std::setw(72) << " " << std::endl << std::endl << std::setfill(' ')
      << std::setw(55) << "GLOBAL MIXED-INTEGER NONLINEAR OPTIMIZATION IN CANON\n"
      << std::setfill('_') << std::setw(72) << " " << std::endl << std::endl << std::setfill(' ');

  // Display NLGO Options
  MINLP.options.display( out );

  out << std::setfill('_') << std::setw(72) << " " << std::endl << std::endl << std::setfill(' ');
  return out;
}

//template <typename T>
//inline void
//NLGO<T>::_set_SLVLOC
//()
//{
//  _NLPSLV->options = options.NLPSLV;
//  _NLPSLV->set( *this );
//  _NLPSLV->setup();
//}

//template <typename T>
//inline const double*
//NLGO<T>::_get_SLVLOC
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

//template <typename T>
//inline bool
//NLGO<T>::_init
//( const T*P, const unsigned*tvar, const double*p0 )
//{
//  const unsigned n0var = BASE_NLP::_var.size(), n0dep = BASE_NLP::_dep.size();
//  // Form vector of parameter and state bounds
//  _Ivar.assign( P, P+n0var+n0dep );
//  // Extra multiplier variables for NCO cuts
//  if( options.NCOCUTS ){
//    //_Ivar.insert( _Ivar.end(), _nvar-n0, T(0.,1.) ); // multipliers are bounded in [0,1]
//    _Ivar.push_back( T(0.,1.) );
//    for( auto&& t_ctr : std::get<0>(BASE_NLP::_ctr) )
//      switch( t_ctr ){
//        case BASE_OPT::EQ: _Ivar.push_back( T(-1.,1.) ); break;
//        case BASE_OPT::LE:
//        case BASE_OPT::GE: _Ivar.push_back( T(0.,1.) ); break;
//      }
//    _Ivar.insert( _Ivar.end(), _sysm.size(), T(-1.,1.) );
//    for( unsigned ivar=0; ivar<n0var; ivar++ ){
//      if( tvar && tvar[ivar] ) continue;
//      _Ivar.push_back( T(0.,1.) );
//      _Ivar.push_back( T(0.,1.) );
//    }
//    for( unsigned idep=0; idep<n0dep; idep++ ){
//      if( tvar && tvar[_nrvar+idep] ) continue;
//      _Ivar.push_back( T(0.,1.) );
//      _Ivar.push_back( T(0.,1.) );
//    }
//  }
//  // Form vector of parameter and state types (continuous vs discrete)
//  _tvar.clear();
//  if( tvar ){
//    _tvar.assign( tvar, tvar+n0var+n0dep );
//    if( options.NCOCUTS )
//      _tvar.insert( _tvar.end(), _nvar-n0var-n0dep, false ); // multipliers are continuous variables
//  }
//  // Form vector of parameter and state values
//  _Dvar.clear();
//  if( p0 ){
//    _Dvar.assign( p0, p0+n0var+n0dep );
//    if( options.NCOCUTS )
//      _Dvar.insert( _Dvar.end(), _nvar-n0var-n0dep, 0. );
//  }
//  return true;
//}

} // end namescape mc

#endif

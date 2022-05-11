// Copyright (C) 2021 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MINLPSLV Local Mixed-Integer Nonlinear Optimization via Outer-Approximation using MC++
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

Possibly set options using the class member NLPSLV_SNOPT::options:

\code
  MINLP.options.CVRTOL = 1e-5;
  MINLP.options.CVATOL = 1e-5;
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
        0  r  1.000000e+20 -5.698551e+01      0s
        1  * -3.951361e+01 -5.698551e+01      0s
        2  * -5.416791e+01 -5.698551e+01      0s
        3  * -5.698117e+01 -5.698551e+01      0s
        4    -5.698117e+01 -5.698551e+01      0s
        5    -5.698117e+01 -5.698117e+01      0s

#  TERMINATION AFTER 4 ITERATIONS: 0.063044 SEC
#  INCUMBENT VALUE: -5.698117e+01
#  INCUMBENT POINT:  7.663529e+00  1.100000e+01
\endverbatim

The return value of mc::MINLPSLV::optimize is per the enumeration mc::MINLPSLV::STATUS. The incumbent solution may be retrieved as an instance of <a>mc::SOLUTION_OPT</a> using the method <a>mc::MINLPSLV::incumbent</a>. A computational breakdown may be obtained from the internal class <a>mc::MINLPSLV::Stats</a>.
*/

#ifndef MC__MINLPSLV_HPP
#define MC__MINLPSLV_HPP

#include <chrono>

#include "interval.hpp"
#include "gamsio.hpp"
#include "nlpslv_snopt.hpp"
#include "mipslv_gurobi.hpp"

namespace mc
{

//! @brief C++ class for local optimization of MINLP using outer-approximation
////////////////////////////////////////////////////////////////////////
//! mc::MINLPSLV is a C++ class for local optimization of MINLP using
//! outer-approximation. Linearizations of the nonlinear objective or
//! constraints are generated using MC++. Further details can be found
//! at: \ref page_MINLPSLV
////////////////////////////////////////////////////////////////////////
template < typename T=Interval,
           typename NLP=NLPSLV_SNOPT,
           typename MIP=MIPSLV_GUROBI<T> >
class MINLPSLV:
  public    virtual BASE_NLP,
  protected virtual GAMSIO
{
using BASE_NLP::_dag; // Make sure _dag is from BASE_NLP, not GAMSIO

public:

  //! @brief NLP solution status
  enum STATUS{
     SUCCESSFUL=0,      //!< MINLP solution found (possibly suboptimal for nonconvex MINLP)
     INFEASIBLE,        //!< MINLP appears to be infeasible (nonconvex MINLP could still be feasible)
     UNBOUNDED,         //!< MIP subproblem returns an unbounded solution
     INTERRUPTED,       //!< MINLP algorithm was interrupted prior to convergence
     FAILED,            //!< MINLP algorithm encountered numerical difficulties
     ABORTED            //!< MINLP algorithm aborted after critical error
  };

  //! @brief MINLPSLV options
  struct Options
  {
    //! @brief Constructor
    Options():
      LINMETH(PENAL), FEASPUMP(true), INCCUT(true),
      FEASTOL(1e-5), CVATOL(1e-3), CVRTOL(1e-3), MAXITER(20),
      CPMAX(10), CPTHRES(0.), 
      PENSOFT(1e3), MSLOC(8), TIMELIMIT(6e2), DISPLEVEL(1),
      NLPSLV(), POLIMG(), MIPSLV()
      { NLPSLV.DISPLEVEL = MIPSLV.DISPLEVEL = 0;
        NLPSLV.TIMELIMIT = MIPSLV.TIMELIMIT = TIMELIMIT;
        NLPSLV.GRADMETH  = NLP::Options::BAD; }
    //! @brief Assignment operator
    Options& operator= ( Options&options ){
        LINMETH       = options.LINMETH;
        FEASPUMP      = options.FEASPUMP;
        INCCUT        = options.INCCUT;
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
    //! @brief Linearization method
    enum LM{
      CVX=0,   //!< Direct linearization of cost and constraints at NLP solution point (assumes convexity)
      PENAL    //!< Softening and relaxation of constraints in MIP subproblem (does not assume convexity)
    };
    //! @brief Linearization method
    unsigned LINMETH;
    //! @brief Whether or not to apply feasibility pump strategy
    bool FEASPUMP;
    //! @brief Whether or not to add incumbent cuts in master problem
    bool INCCUT;
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

  //! @brief Structure holding solve statistics
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
  
  //! @brief Variable values at current relaxation
  std::vector<double>       _Xrel;
  
  //! @brief subset of integer participating variables
  std::set<unsigned>        _Xint;
  
  //! @brief Local solver for NLP subproblem
  NLP                       _NLPSLV;

  //! @brief Global solver for MIP subproblem
  MIP                       _MIPSLV;

  //! @brief Number of decision variables (independent and dependent) in MINLP model
  unsigned                  _nX;

  //! @brief Decision variables in MINLP model
  std::vector<FFVar>        _Xvar;

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

  //! @brief number of functions (objective and constraints) in MINLP model
  unsigned                  _nF;

  //! @brief Functions in MINLP model
  std::vector<FFVar>        _Fvar;

  //! @brief Function bounds
  std::vector<T>            _Fbnd;

  //! @brief Functions in MINLP model
  std::vector<unsigned>     _Ftyp;

  //! @brief index set of linear functions
  std::set<unsigned>        _Flin;

  //! @brief index set of nonlinear functions
  std::set<unsigned>        _Fnlin;

  //!@brief number of nonzero elements in the linear part of each function
  unsigned                  _nA;

  //!@brief row coordinates of nonzero elements in the linear part of each function
  std::vector<int>          _iAfun;

  //!@brief column coordinates of nonzero elements in the linear part of each function
  std::vector<int>          _jAvar;

  //!@brief values of nonzero elements in the linear part of each function
  std::vector<double>       _Aval;

  //!@brief number of nonzero elements in the derivative of the nonlinear part of each function
  unsigned                  _nG;

  //!@brief row coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>          _iGfun;

  //!@brief column coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>          _jGvar;

  //! @brief derivatives of the nonlinear part of each function
  std::vector<FFVar>        _Gvar;

  //! @brief derivative values of the nonlinear part of each function
  std::vector<double>       _Gval;

  //!@brief set of indices of nonlinear constraints 
  std::set<unsigned>        _Gndx;

  //! @brief list of operations in function derivatives
  FFSubgraph                _Gop;

  //! @brief Storage vector for DAG evaluation in double arithmetic
  std::vector<double>       _dwk;

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

  //! @brief Structure holding NLP intermediate solution
  SOLUTION_OPT              _solution;

  //! @brief Structure holding MINLP incumbent information
  SOLUTION_OPT              _incumbent;
  
  //! @brief function sparse derivatives
  std::tuple< unsigned, unsigned const*, unsigned const*, FFVar const* > _Fgrad;

  //! @brief Apply constraint propagation
  int _propagate_bounds
    ( T* Xbnd );

  //! @brief Cleanup gradient storage
  void _cleanup_grad
    ();

  //! @brief Add outer-approximation cuts to master MIP subproblem
  bool _add_outerapproximation_cuts
    ( std::vector<double> const& Xval, std::vector<double>& Fval,
      std::vector<double>& Fmul );

  //! @brief Add incumbent cut to master MIP subproblem
  bool _add_incumbent_cut
    ();
    
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
    ( bool const locfeas, bool const pumpfeas, bool const inccut );

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

public:

  //! @brief Constructor
  MINLPSLV()
    : _issetup(false)
    { stats.reset(); }

  //! @brief Destructor
  virtual ~MINLPSLV()
    { _cleanup_grad(); }

  //! @brief Status after last NLP call
  STATUS get_status
    ()
    const
    { return _status; }

  //! @brief Load optimization model from GAMS file
#if defined (MC__WITH_GAMS)
  bool read
    ( std::string const& filename );
#endif

  //! @brief Setup DAG for cost and constraint evaluation
  void setup
    ( std::ostream& os=std::cout );

  //! @brief Solve MINLP model to local optimality using outer-approximation
  int optimize
    ( double const* Xini=nullptr,  T const* Xbnd=nullptr, std::ostream& os=std::cout );

  //! @brief Get incumbent info
  SOLUTION_OPT const& get_incumbent
    () 
    const
    { return _incumbent; }
    
  //! @brief Get reference to local NLP solver
  NLP& NLPsolver
    ()
    { return _NLPSLV; }
    
  //! @brief Get reference to master MIP solver
  MIP& MIPsolver
    ()
    { return _MIPSLV; }

  //! @brief Test domain boundedness
  bool is_bounded
    ( double const& maxdiam )
    { _isbnd = true;
      for( unsigned i=0; _isbnd && i<_nX; i++ ){
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

private:

  //! @brief Private methods to block default compiler methods
  MINLPSLV
    ( MINLPSLV const& );
  MINLPSLV& operator=
    ( MINLPSLV const& );

  //! @brief Interval representation of 'unbounded' variables
  T _IINF;

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

#if defined (MC__WITH_GAMS)
template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::read
( std::string const& filename )
{
  auto tstart = stats.start();

  bool flag = this->GAMSIO::read( filename, options.DISPLEVEL>1? true: false );

  stats.walltime_setup += stats.walltime( tstart );
  stats.walltime_all   += stats.walltime( tstart );
  return flag;  
}
#endif

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::setup
( std::ostream& os )
{
  //stats.reset();
  auto tstart = stats.start();

  _IINF = BASE_OPT::INF * T(-1,1);
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

  // independent decision variables
  _Xvar = _var;
  _Xlow = _varlb;
  _Xupp = _varub;
  _Xtyp = _vartyp;
  _Xint.clear();
  for( unsigned i=0; i<_var.size(); i++ )
    if( _Xtyp[i] ) _Xint.insert( i );

  // dependent decision variables
  _Xvar.insert( _Xvar.end(), _dep.begin(), _dep.end() );
  _Xlow.insert( _Xlow.end(), _deplb.begin(), _deplb.end() );
  _Xupp.insert( _Xupp.end(), _depub.begin(), _depub.end() );
  _Xtyp.insert( _Xtyp.end(), _dep.size(), 0 );
  _Xrel.clear();
  _Xbnd.clear();
  _nX = _Xvar.size();
  
  // full set of functions
  _Fvar.clear();
  _Ftyp.clear();
  _Fbnd.clear();
  // Cost
  if( std::get<0>(_obj).size() ){
    _Ftyp.push_back( std::get<0>(_obj)[0] );
    _Fvar.push_back( std::get<1>(_obj)[0] );
  }
  else{
    _Ftyp.push_back( MIN );
    _Fvar.push_back( 0 );
  }
  _Fbnd.push_back( _IINF );
  // Constraints
  for( unsigned i=0; i<std::get<0>(_ctr).size(); i++ ){
    _Ftyp.push_back( std::get<0>(_ctr)[i] );
    _Fvar.push_back( std::get<1>(_ctr)[i] );
    switch( std::get<0>(_ctr)[i] ){
      case EQ: _Fbnd.push_back( T(0) );                break;
      case LE: _Fbnd.push_back( T(-BASE_OPT::INF,0) ); break;
      case GE: _Fbnd.push_back( T(0,BASE_OPT::INF) );  break;
    }
  }
  _Ftyp.insert( _Ftyp.end(), _sys.size(), EQ );
  _Fvar.insert( _Fvar.end(), _sys.begin(), _sys.end() );
  _Fbnd.insert( _Fbnd.end(), _sys.size(), T(0) );
  _nF = _Fvar.size();
  assert( _Ftyp.size() == _nF );

#ifdef MC__MINLPSLV_DEBUG
  std::cout << "_dag = " << _dag << std::endl;
  std::cout << "_nF = " << _nF << std::endl;
  _dag->output( _dag->subgraph( _nF, _Fvar.data() ) );
#endif

  // nonlinear functions
  _Flin.clear();
  _Fnlin.clear();
  for( unsigned j=0; j<_nF; j++ ){
    auto && Fjdep = _Fvar[j].dep();
    auto it = Fjdep.dep().cbegin();
    for( ; it != Fjdep.dep().cend(); ++it ){
      if( it->second > FFDep::L ){
        _Fnlin.insert( j );
        break;
      }
    }
    if( _Fnlin.find( j ) == _Fnlin.end() )
      _Flin.insert( j );
  }
  assert( _nF == _Flin.size() + _Fnlin.size() );
  
  // sparse function gradients
  _cleanup_grad();
  switch( options.NLPSLV.GRADMETH ){
    default:
    case NLP::Options::FAD: _Fgrad = _dag->SFAD( _nF, _Fvar.data(), _nX, _Xvar.data() ); break;
    case NLP::Options::BAD: _Fgrad = _dag->SBAD( _nF, _Fvar.data(), _nX, _Xvar.data() ); break;
  }

  _iAfun.clear(); _jAvar.clear(); _Aval.clear(); 
  _iGfun.clear(); _jGvar.clear(); _Gvar.clear(); 
  for( unsigned k=0; k<std::get<0>(_Fgrad); ++k ){
    // derivative term in nonlinear constraint
    if( _Fnlin.find( std::get<1>(_Fgrad)[k] ) != _Fnlin.end() ){
      _iGfun.push_back( std::get<1>(_Fgrad)[k] );
      _jGvar.push_back( std::get<2>(_Fgrad)[k] );
      _Gvar.push_back( std::get<3>(_Fgrad)[k] );
#ifdef MC__MINLPSLV_DEBUG
     std::cout << "  _Gvar[" << std::get<1>(_Fgrad)[k] << "," << std::get<2>(_Fgrad)[k]
               << "] = " << std::get<3>(_Fgrad)[k] << std::endl;
#endif
    }
    // derivative term in linear constraint
    else{
      _iAfun.push_back( std::get<1>(_Fgrad)[k] );
      _jAvar.push_back( std::get<2>(_Fgrad)[k] );
      assert( std::get<3>(_Fgrad)[k].cst() );
      _Aval.push_back( std::get<3>(_Fgrad)[k].num().val() );    
#ifdef MC__MINLPSLV_DEBUG
     std::cout << "  _Aval[" << std::get<1>(_Fgrad)[k] << "," << std::get<2>(_Fgrad)[k]
               << "] = " << std::get<3>(_Fgrad)[k].num().val() << std::endl;
#endif
    }
  }
  _nG = _Gvar.size();
  _nA = _Aval.size();
//  _Fop = _dag->subgraph( _Fnlin, _Fvar.data() );
  _Gop = _dag->subgraph( _nG,    _Gvar.data() );
  _Gval.resize( _nG );


  stats.walltime_setup += stats.walltime( tstart );
  stats.walltime_all   += stats.walltime( tstart );
  _status = STATUS::SUCCESSFUL;
  _display_setup( os );
  _issetup = true;
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_cleanup_grad
()
{
  delete[] std::get<1>(_Fgrad);  std::get<1>(_Fgrad) = 0;
  delete[] std::get<2>(_Fgrad);  std::get<2>(_Fgrad) = 0;
  delete[] std::get<3>(_Fgrad);  std::get<3>(_Fgrad) = 0;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_is_integer_feasible
( double const* Xval, double const& feastol )
const
{
  for( unsigned i=0; i<_var.size(); i++ ){
    if( !_vartyp[i] ) continue;
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
  for( unsigned i=0; i<_var.size(); i++ ){
    if( !_vartyp[i] ) continue;
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
  _display_final( _iter, stats.walltime_all, os );
  return _status;
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
  if( pumpfeas ){
    FFVar feasobj(0.);
    for( unsigned i=0; i<_var.size(); i++ ){
      if( !_vartyp[i] ) continue;
      feasobj += sqr( _Xvar[i] - _Xrel[i] );
    }
    _NLPSLV.set_obj_lazy( BASE_OPT::MIN, feasobj );
    if( inccut && options.INCCUT && !_incumbent.x.empty() )
      _NLPSLV.add_ctr_lazy( _Ftyp[0]==BASE_OPT::MIN? BASE_OPT::LE: BASE_OPT::GE, _Fvar[0] - _Zinc );
  }
  else
    _NLPSLV.restore_model();

  // Local solve from provided initial point
  _solution.reset();
  _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( tstart ) );
  _NLPSLV.solve( Xini, Xbnd );
  if( _NLPSLV.is_feasible( options.FEASTOL ) )
    _solution = _NLPSLV.solution();

  // Extra local solves from random starting points
  if( _isbnd && options.MSLOC > 1 ){
    _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( tstart ) );
    _NLPSLV.solve( options.MSLOC-1, Xbnd );
    if( _NLPSLV.is_feasible( options.FEASTOL )
     && (_solution.x.empty() || _objscal*_NLPSLV.solution().f[0] < _objscal*_solution.f[0]) )
      _solution = _NLPSLV.solution();
  }

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
  _POLenv.options = options.POLIMG;

  // Set polyhedral main variables
  _POLXvar.clear();
  auto itX = _Xvar.begin();
  for( unsigned i=0; itX!=_Xvar.end(); ++itX, i++ )
    _POLXvar.push_back( PolVar<T>( &_POLenv, *itX, T(_Xlow[i],_Xupp[i]), (_Xtyp[i]? false: true) ) );

  // Set polyhedral slack variables
  _POLSvar.clear();
  _POLScost.clear();
  _POLSvar.push_back( PolVar<T>( &_POLenv, T(-BASE_OPT::INF,BASE_OPT::INF), true ) ); // cost variable
  _POLScost.push_back( _objscal ); // cost coefficient -> minimize MIP
#ifdef MC__MINLPSLV_DEBUG
  std::cout << _POLenv;
#endif

  // Reinitialize MIP solver
  _MIPSLV.options = options.MIPSLV;
  _MIPSLV.set_cuts( &_POLenv, true );

  stats.walltime_slvnlp += stats.walltime( tMIP );
}
  
template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_update_master
( bool const locfeas, bool const pumpfeas, bool const inccut )
{
  auto tMIP = stats.start();

  // Append new cuts to polyhedral image
  _POLenv.erase_cuts();
  if( locfeas ){
    if( !_add_outerapproximation_cuts( _solution.x, _solution.f, _solution.uf ) ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
    }
  }
  else{
    if( !_add_integer_cut( _Xrel ) ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
    }
  }
  if( pumpfeas ){
    if( !_add_anticycling_cut( _solution.x, _Xrel )
     || (locfeas && !_set_integer_cut( _Xrel, _POLauxvar, _POLauxwei, _auxcst )) ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
    }
  }
  if( inccut && options.INCCUT && !_add_incumbent_cut() ){
      stats.walltime_slvnlp += stats.walltime( tMIP );
      return false;
  }
  
  // Update master MIP problem 
  _MIPSLV.set_cuts( &_POLenv, false );
  if( pumpfeas )
    _MIPSLV.set_objective( _POLauxvar.size(), _POLauxvar.data(), _POLauxwei.data(), BASE_OPT::MIN );    
  else
    _MIPSLV.set_objective( _POLSvar.size(), _POLSvar.data(), _POLScost.data(), BASE_OPT::MIN );

  stats.walltime_slvnlp += stats.walltime( tMIP );
  return true;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_add_incumbent_cut
()
{
  _POLenv.add_cut( _Ftyp[0]==BASE_OPT::MIN? PolCut<T>::LE: PolCut<T>::GE, _Zinc, _POLSvar.front(), 1. );
  return true;
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_add_outerapproximation_cuts
( std::vector<double> const& Xval, std::vector<double>& Fval, std::vector<double>& Fmul )
{
  // Initialize cuts
  _POLcuts.assign( _nF, nullptr );
  for( unsigned i=0; i<_nF; i++ ){
  
    // Only add linear cuts at first iteration
    bool islin = (_Fnlin.find(i) == _Fnlin.end()? true: false);
    if( _iter > 1 && islin ) continue;

    // Define cut type: objective
    if( !i ) switch( _Ftyp[0] ){
      case BASE_OPT::MIN: _POLcuts[0] = *_POLenv.add_cut( PolCut<T>::LE, -Fval[0], _POLSvar.front(), -1. ); continue;
      case BASE_OPT::MAX: _POLcuts[0] = *_POLenv.add_cut( PolCut<T>::GE, -Fval[0], _POLSvar.front(),  1. ); continue;
      default: return false;
    }

    // Define cut type: linear constraints
    if( islin || options.LINMETH == Options::CVX ) switch( _Ftyp[i] ){
      case BASE_OPT::EQ: _POLcuts[i] = *_POLenv.add_cut( PolCut<T>::EQ, -Fval[i] ); continue;
      case BASE_OPT::LE: _POLcuts[i] = *_POLenv.add_cut( PolCut<T>::LE, -Fval[i] ); continue;
      case BASE_OPT::GE: _POLcuts[i] = *_POLenv.add_cut( PolCut<T>::GE, -Fval[i] ); continue;
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
    _POLSvar.push_back( PolVar<T>( &_POLenv, T(0.,BASE_OPT::INF), true ) ); // slack variable
    _POLScost.push_back( options.PENSOFT * std::fabs(Fmul[i]) ); // slack cost coefficient
    switch( _Ftyp[i] ){
      case BASE_OPT::LE: _POLcuts[i] = *_POLenv.add_cut( PolCut<T>::LE, -Fval[i], _POLSvar.back(), -1. ); continue;
      case BASE_OPT::GE: _POLcuts[i] = *_POLenv.add_cut( PolCut<T>::GE, -Fval[i], _POLSvar.back(),  1. ); continue;
      case BASE_OPT::EQ:
        if( _objscal * Fmul[i] < 0 ) _POLcuts[i] = *_POLenv.add_cut( PolCut<T>::LE, -Fval[i], _POLSvar.back(), -1. );
        else                         _POLcuts[i] = *_POLenv.add_cut( PolCut<T>::GE, -Fval[i], _POLSvar.back(),  1. );
        continue;
      default: return false;
    }
  }

  // Populate linear cuts
  if( _iter == 1 ){
    for( unsigned ie=0; ie<_nA; ie++ )
      _POLcuts[_iAfun[ie]]->append( _POLXvar[_jAvar[ie]], _Aval[ie] ).rhs() += _Aval[ie] * Xval[_jAvar[ie]];
  }
  
  // Evaluate nonlinear function derivatives
  try{
    _dag->eval( _Gop, _dwk, _nG, _Gvar.data(), _Gval.data(), _nX, _Xvar.data(), Xval.data() );
#ifdef MC__MINLPSLV_DEBUG
    for( unsigned ie=0; ie<_nG; ie++ )
      std::cout << "  _Gval[" << _iGfun[ie] << "," << _jGvar[ie] << "] = " << _Gval[ie] << std::endl;
#endif
  }
  catch(...){
    return false;
  }
  
  // Populate linearized nonlinear cuts
  for( unsigned ie=0; ie<_nG; ie++ ){
    if( !_POLcuts[_iGfun[ie]] ) continue;
    _POLcuts[_iGfun[ie]]->append( _POLXvar[_jGvar[ie]], _Gval[ie] ).rhs() += _Gval[ie] * Xval[_jGvar[ie]];
  }
#ifdef MC__MINLPSLV_DEBUG
    std::cout << _POLenv;
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
  _POLenv.add_cut( PolCut<T>::GE, 1-_auxcst, _POLauxvar.size(), _POLauxvar.data(), _POLauxwei.data() );
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
  for( auto&& j: _Xint ){
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
      _POLenv.add_cut( PolCut<T>::GE, Xint[j], _POLXvar[j], 1., POLWvar,  1. );
      _POLenv.add_cut( PolCut<T>::LE, Xint[j], _POLXvar[j], 1., POLWvar, -1. );
      PolVar<T> POLNvar( &_POLenv, T(0.,1.), false );
      double M1 = 2 * ( Xint[j] - std::ceil(_Xlow[j]) );
      double M2 = 2 * ( std::floor(_Xupp[j]) - Xint[j] );
      _POLenv.add_cut( PolCut<T>::GE, Xint[j]-M1, _POLXvar[j], 1., POLWvar, -1., POLNvar, -M1 );
      _POLenv.add_cut( PolCut<T>::LE, Xint[j],    _POLXvar[j], 1., POLWvar,  1., POLNvar, -M2 );
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
  auto ACcut = *_POLenv.add_cut( PolCut<T>::GE, 0. );
  for( auto&& j: _Xint )
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
  auto tstart = stats.start();
#ifdef MC__MINLPSLV_DEBUG
  _dag->output( _dag->subgraph( _Flin, _Fvar.data() ) );
  for( unsigned i=0; i<_nX; i++ )
    std::cout << "X[" << i << "] = " << Xbnd[i] << std::endl;
  for( unsigned i=0; i<_nF; i++ )
    std::cout << "F[" << i << "] = " << _Fbnd[i] << std::endl;
  std::cout << "FLin = { ";
  for( auto const& i: _Flin ) std::cout << i << " ";
  std::cout << "}" << std::endl;
#endif
  
  // Apply constraint propagation
  int flag = 0;
  try{
    flag = _dag->reval( _CPbnd, _nF-1, _Fvar.data()+1, _Fbnd.data()+1, _nX, _Xvar.data(),
                        Xbnd, _IINF, options.CPMAX, options.CPTHRES );
    //flag = _dag->reval( _CPbnd, _Flin, _Fvar.data(), _Fbnd.data(), _nX, _Xvar.data(),
    //                    Xbnd, _IINF, options.CPMAX, options.CPTHRES );

    // Round binary and integer variables accordingly
    for( unsigned ix=0; ix<_nX; ++ix )
      if( _Xtyp[ix] > 0 ) Xbnd[ix] = T( std::ceil( Op<T>::l(Xbnd[ix]) ), std::floor( Op<T>::u(Xbnd[ix]) ) );
  }
  catch(...){ std::cout << "Exception caught\n"; }
#ifdef MC__MINLPSLV_DEBUG
  std::cout << "\nVariable Domain: " << flag << std::endl;
  for( unsigned i=0; i<_nX; i++ )
    std::cout << "X[" << i << "] = " << Xbnd[i] << std::endl;
#endif

  stats.walltime_setup += stats.walltime( tstart );
  return flag;
}

template <typename T, typename NLP, typename MIP>
inline int
MINLPSLV<T,NLP,MIP>::optimize
(  double const* Xini, T const* Xbnd, std::ostream& os )
{
  auto tstart = stats.start();

  // Initialization
  _display_init( os );
  _iter = 0;
  _Zinc =  _objscal * BASE_OPT::INF;
  _Zrel = -_objscal * BASE_OPT::INF;
  _incumbent.reset();
  
  // Initial point
  if( Xini ) _varini.assign( Xini, Xini+_var.size() );
  
  // Initialize and reduce variable bounds
  bool locfeas = true;
  _Xbnd.resize( _nX );
  for( unsigned i=0; i<_nX; i++ ){
    _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
    if( Xbnd && !Op<T>::inter( _Xbnd[i], Xbnd[i], _Xbnd[i] ) ){
      locfeas = false;
      break;
//      return _finalize( tstart, STATUS::INFEASIBLE );
    }
  }
  int cpflag = 0;
  if( locfeas && options.CPMAX ){
    cpflag = _propagate_bounds( _Xbnd.data() );
    if( cpflag < 0 ){
      locfeas = false;
      _Zrel = _objscal * BASE_OPT::INF;
//      return _finalize( tstart, STATUS::INFEASIBLE );
    }
  }

  // Solve relaxed MINLP model
  if( locfeas ){
    is_bounded( BASE_OPT::INF/10 );
    locfeas = _solve_local( tstart, _varini.data(), _Xbnd.data(), false, false, os );
  }
#ifdef MC__MINLPSLV_DEBUG
  std::cout << _solution;
#endif

  // Update bounds
  if( locfeas ){
    _Zrel = _NLPSLV.solution().f[0];
    _Xrel = _NLPSLV.solution().x;
    if( _is_integer_feasible( _solution.x.data(), options.FEASTOL ) ){
      _incumbent = _solution;
      _Zinc = _Zrel;
    }
    _Xbndi = _Xbnd;
  }
  else{
    _Zrel = _objscal * BASE_OPT::INF;
//    return _finalize( tstart, STATUS::INFEASIBLE );
  }
  
  // Intermediate display
  _display_add( _iter );
  bool updinc = !_incumbent.x.empty();
  if( !locfeas )                _display_add( "i");
  else if( updinc && cpflag>0 ) _display_add( "r*");
  else if( updinc )             _display_add( "*" );
  else if( cpflag )             _display_add( "r" );
  else                          _display_add( " " );
  _display_add( _Zinc );
  _display_add( _Zrel );
  _display_add( tstart );
  _display_flush( os );

  // Termination tests
  if( !locfeas )
    return _finalize( tstart, STATUS::INFEASIBLE );
  if( !_ismip )
    return _finalize( tstart, STATUS::SUCCESSFUL );
  if( _interrupted( tstart ) )
    return _finalize( tstart, STATUS::INTERRUPTED );
  
  // Initialize master MIP subproblem
  _init_master();

  // Main loop
  bool pumpfeas = false, stopiter = false;
  for( ++_iter; !stopiter && !_converged() ; ++_iter ){

    // Update master MIP subproblem
    if( !_update_master( locfeas, pumpfeas, updinc ) )
      return _finalize( tstart, STATUS::ABORTED );

    // Solve master MIP subproblem
    switch( _solve_master( tstart ) ){
      case MIP::OPTIMAL:
        break;
      case MIP::INFEASIBLE:
        _Zrel = _objscal * BASE_OPT::INF;
        return _finalize( tstart, STATUS::INFEASIBLE );
      case MIP::UNBOUNDED:
        return _finalize( tstart, STATUS::UNBOUNDED );
      case MIP::TIMELIMIT:
        return _finalize( tstart, STATUS::INTERRUPTED );
      default:
        return _finalize( tstart, STATUS::FAILED );
    }

    // Retrieve MIP solution
    if( !pumpfeas )
      _Zrel = _MIPSLV.get_objective();
    for( unsigned i=0; i<_nX; i++ )
      _Xrel[i] = _MIPSLV.get_variable( _POLXvar[i] );
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "_Zrel = " << _Zrel << std::endl;
    for( unsigned i=0; i<_nX; i++ )
      std::cout << "_Xrel[" << i << "] = " << _Xrel[i] << std::endl;
#endif
    locfeas = true; // reinitialize to not add integer cut to master MIP subproblem during feasibility pump

    // Interrupt if master and local bounds cross each other
    if( !_incumbent.x.empty() && _objscal*_Zrel >= _objscal*_Zinc ){
      updinc = false;
      stopiter = true;
    }
    
    // Apply feasibility pump if NLP model found infeasible at current MIP integer fixing
    else if( pumpfeas ){
      if( !_solve_local( tstart, _Xrel.data(), _Xbnd.data(), true, !locfeas, os ) )
        return _finalize( tstart, STATUS::FAILED ); // may not be infeasible unless MINLP is infeasible
#ifdef MC__MINLPSLV_DEBUG
      std::cout << _solution;
#endif

      // Interrupt feasibilitity pump and solve local NLP model
      if( _is_integer_equal( _solution.x.data(), _Xrel.data() ) ){
        pumpfeas = false;
        for( auto&& i: _Xint ) _Xbndi[i] = _Xrel[i];
        if( !_solve_local( tstart, _solution.x.data(), _Xbndi.data(), false, false, os ) )
          return _finalize( tstart, STATUS::FAILED ); // may not be infeasible after feasibility pump
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
      for( auto&& i: _Xint ) _Xbndi[i] = _Xrel[i];
      locfeas = _solve_local( tstart, _Xrel.data(), _Xbndi.data(), false, false, os );
#ifdef MC__MINLPSLV_DEBUG
      std::cout << _solution;
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
    _display_add( tstart );
    _display_flush( os );

    // Termination tests
    if( stopiter )
      break;
    if( _interrupted( tstart ) )
      return _finalize( tstart, STATUS::INTERRUPTED );
  }

  return _finalize( tstart, STATUS::SUCCESSFUL );
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
  _odisp << std::endl << "#  TERMINATION AFTER " << (_iter?_iter-1:0) << " ITERATIONS: "
         << std::fixed << std::setprecision(6) << walltime.count()*1e-6 << " SEC"
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
  _odisp << std::right << std::fixed << std::setprecision(0)
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

} // end namescape mc

#endif

// Copyright (C) 2021- Benoit Chachuat, Imperial College London.
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

We start by instantiating an mc::MINLPSLV class object, which is defined in the header file <tt>minlpslv.hpp</tt>:

\code
  mc::MINLPSLV MINLP;
\endcode

Next, we set the variables and objective/constraint functions by creating a DAG of the problem: 

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

The MINLP model is solved using:

\code
  MINLP.options.CVRTOL = MINLP.options.CVATOL = 1e-5;
  MINLP.setup();
  MINLP.optimize();
\endcode

The return value of mc::MINLPSLV is per the enumeration mc::MINLPSLV::STATUS. The following result is displayed (with the option mc::MINLPSLV::Options::DISPLEVEL defaulting to 1):

\verbatim
#  CONTINUOUS / INTEGER VARIABLES:   1 / 1
#  LINEAR / NONLINEAR FUNCTIONS:     2 / 2

#  ITERATION     INCUMBENT    BEST BOUND    TIME
        0     1.000000e+20 -5.698551e+01      0s
        1  i  1.000000e+20 -5.698551e+01      0s
        2  i  1.000000e+20 -5.698551e+01      0s
        3  i  1.000000e+20 -5.698551e+01      0s
        4  i  1.000000e+20 -5.698551e+01      0s
        5  i  1.000000e+20 -5.698551e+01      0s
        6  i  1.000000e+20 -5.698551e+01      0s
        7  i  1.000000e+20 -5.698551e+01      0s
        8  i  1.000000e+20 -5.698551e+01      0s
        9  * -5.266063e+01 -5.698551e+01      0s
       10  * -5.698117e+01 -5.698551e+01      0s
       11    -5.698117e+01 -5.698117e+01      0s

#  TERMINATION AFTER 11 ITERATIONS: 0.150152 SEC
#  INCUMBENT VALUE: -5.698117e+01
#  INCUMBENT POINT:  7.663529e+00  1.100000e+01
\endverbatim

The incumbent solution may be retrieved as an instance of <a>mc::SOLUTION_OPT</a> using the method <a>mc::MINLPSLV::incumbent</a>. A computational breakdown may be obtained from the internal class <a>mc::MINLPSLV::Stats</a>. And the options of the algorithm can be modified using the internal class mc::MINLPSLV::Options.
*/

#ifndef MC__MINLPSLV_HPP
#define MC__MINLPSLV_HPP

#include <chrono>

#include "interval.hpp"
#include "gamsio.hpp"
#include "nlpslv_snopt.hpp"
#include "mipslv_gurobi.hpp"
#include "nlpbnd.hpp"

namespace mc
{

//! @brief C++ class for local optimization of MINLP using outer-approximation
////////////////////////////////////////////////////////////////////////
//! mc::MINLPSLV is a C++ class for local optimization of MINLP using
//! outer-approximation. Linearizations of the nonlinear objective or
//! constraints are generated using MC++. Further details can be found
//! at: \ref page_MINLPSLV
////////////////////////////////////////////////////////////////////////
template < typename T=Interval, typename NLP=NLPSLV_SNOPT, typename MIP=MIPSLV_GUROBI<T> >
class MINLPSLV:
  public    virtual BASE_NLP,
  protected virtual GAMSIO
{
using BASE_NLP::_dag;

public:

  //! @brief NLP solution status
  enum STATUS{
     SUCCESSFUL=0,		//!< MINLP solution found (possibly suboptimal for nonconvex MINLP)
     INFEASIBLE,	    //!< MINLP appears to be infeasible (nonconvex MINLP could still be feasible)
     UNBOUNDED,	        //!< MIP subproblem returns an unbounded solution
     INTERRUPTED,       //!< MINLP algorithm was interrupted prior to convergence
     FAILED,            //!< MINLP algorithm encountered numerical difficulties
     ABORTED            //!< MINLP algorithm aborted after critical error
  };

  //! @brief MINLPSLV options
  struct Options
  {
    //! @brief Constructor
    Options():
      LINMETH(PENAL), FEASPUMP(false), FEASTOL(1e-7), CVATOL(1e-3), CVRTOL(1e-3),
      MAXITER(20), PENSOFT(1e3), MSLOC(10), TIMELIMIT(6e2), DISPLEVEL(1),
      NLPSLV(), POLIMG(), MIPSLV()
      { NLPSLV.DISPLEVEL  =  MIPSLV.DISPLEVEL  = 0;
        NLPSLV.TIMELIMIT  =  MIPSLV.TIMELIMIT  = TIMELIMIT; }
    //! @brief Assignment operator
    Options& operator= ( Options&options ){
        LINMETH       = options.LINMETH;
        FEASPUMP      = options.FEASPUMP;
        FEASTOL       = options.FEASTOL;
        CVATOL        = options.CVATOL;
        CVRTOL        = options.CVRTOL;
        MAXITER       = options.MAXITER;
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
    //! @brief Feasibility tolerance 
    double FEASTOL;
    //! @brief Convergence absolute tolerance
    double CVATOL;
    //! @brief Convergence relative tolerance
    double CVRTOL;
    //! @brief Maximum number of outer-approximation iterations (0-no limit)
    unsigned MAXITER;
    //! @brief Weight multiplying constraint marginal in soft constraints
    double PENSOFT;
    //! @brief Number of multistart local search
    unsigned MSLOC;
    //! @brief Maximum run time (seconds)
    double TIMELIMIT;
    //! @brief Display level for solver
    int DISPLEVEL;
    //! @brief Set options of NLPSLV_SNOPT (local solution of factorable NLP)
    typename NLPSLV_SNOPT::Options NLPSLV;
    //! @brief PolImg (polyhedral relaxation) options
    typename PolImg<T>::Options POLIMG;
    //! @brief MIPSLV_GUROBI (mixed-integer optimization) options
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
  
  //! @brief Variable values at current incumbent
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

  //! @brief number of functions (objective and constraints) in MINLP model
  unsigned                  _nF;

  //! @brief Functions in MINLP model
  std::vector<FFVar>        _Fvar;

  //! @brief Functions in MINLP model
  std::vector<unsigned>     _Ftyp;

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
  
  //! @brief list of operations in functions
//  FFSubgraph                _Fop;

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

  //! @brief Position in slack variable at current iteration
  unsigned                  _POLSpos;
  
  //! @brief Polyhedral image cut storage
  std::vector< PolCut<T>* > _POLcuts;

  //! @brief Structure holding NLP intermediate solution
  SOLUTION_OPT              _solution;

  //! @brief Structure holding MINLP incumbent information
  SOLUTION_OPT              _incumbent;
  
  //! @brief function sparse derivatives
  std::tuple< unsigned, unsigned const*, unsigned const*, FFVar const* > _Fgrad;

  //! @brief Cleanup gradient storage
  void _cleanup_grad
    ();

  //! @brief Add cuts to master MIP subproblem when local NLP is feasible
  bool _update_master_feas
    ( std::vector<double> const& Xval, std::vector<double>& Fval,
      std::vector<double>& Fmul );

  //! @brief Add cuts to master MIP subproblem when local NLP is infeasible
  bool _update_master_infeas
    ( std::vector<double> const& Xval );

  //! @brief Test whether a variable vector is integer feasible
  bool _is_integer_feasible
    ( std::vector<double> const& Xval )
    const;
  
  //! @brief Solve local NLP subproblem
  bool _solve_local
    ( std::chrono::time_point<std::chrono::system_clock> const& tstart,
      double const* Xini, T const* Xbnd, std::ostream& os=std::cout );

  //! @brief Initialize master MIP subproblem
  void _init_master
    ();

  //! @brief Update master MIP subproblem with local NLP cuts
  bool _update_master
    ( bool const locfeas );

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

//  //! @brief Get incumbent after last optimization
//  std::pair<double,double const*> get_incumbent
//    () const
//    { return std::make_pair(_Finc,_Xinc.data()); }

  //! @brief Get incumbent info
  SOLUTION_OPT const& incumbent() const
    {
      return _incumbent;
    }

private:

  //! @brief Private methods to block default compiler methods
  MINLPSLV
    ( MINLPSLV const& );
  MINLPSLV& operator=
    ( MINLPSLV const& );

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
  _NLPSLV.options.RELAXINT = true;
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
  for( unsigned i=0; i<_var.size(); i++ ) // 
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
  //if( std::get<0>(_obj).size() > 1 ) throw Exceptions( NLP::Exceptions::MULTOBJ );
  if( std::get<0>(_obj).size() ){
    _Ftyp.push_back( std::get<0>(_obj)[0] );
    _Fvar.push_back( std::get<1>(_obj)[0] );
  }
  else{
    _Ftyp.push_back( MIN );
    _Fvar.push_back( 0 );
  }
  for( unsigned i=0; i<std::get<0>(_ctr).size(); i++ ){
    _Ftyp.push_back( std::get<0>(_ctr)[i] );
    _Fvar.push_back( std::get<1>(_ctr)[i] );
  }
  _Ftyp.insert( _Ftyp.end(), _sys.size(), EQ );
  _Fvar.insert( _Fvar.end(), _sys.begin(), _sys.end() );
  _nF = _Fvar.size();
  assert( _Ftyp.size() == _nF );

#ifdef MC__MINLPSLV_DEBUG
  std::cout << "_dag = " << _dag << std::endl;
  std::cout << "_nF = " << _nF << std::endl;
  _dag->output( _dag->subgraph( _nF, _Fvar.data() ) );
#endif
      
  // nonlinear functions
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
  }

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
(  std::vector<double> const& Xval )
const
{
  for( unsigned i=0; i<_var.size(); i++ ){
    if( !_vartyp[i] ) continue;
    if( std::fabs( Xval[i] - std::round(Xval[i]) ) > options.FEASTOL )
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
   || _iter >= options.MAXITER )
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
  double const* Xini, T const* Xbnd, std::ostream& os )
{
  auto tNLP = stats.start();

  // Local solve from provided initial point
  _solution.reset();
  _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( tstart ) );
  _NLPSLV.solve( _varini.data(), _Xbnd.data() );
  if( _NLPSLV.is_feasible( options.FEASTOL ) )
    _solution = _NLPSLV.solution();

  // Extra local solves from random starting points
  if( _isbnd && options.MSLOC > 1 ){
    _NLPSLV.options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime_all + stats.walltime( tstart ) );
    _NLPSLV.solve( options.MSLOC-1, _Xbnd.data() );
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
  _POLSpos = 1;
#ifdef MC__MINLPSLV_DEBUG
  std::cout << _POLenv;
#endif

  // Reinitialize MIP solver
  _MIPSLV.options = options.MIPSLV;
  _MIPSLV.set_cuts( &_POLenv, true );
  _MIPSLV.set_objective( _POLSvar.size(), _POLSvar.data(), _POLScost.data(), BASE_OPT::MIN );

  stats.walltime_slvnlp += stats.walltime( tMIP );
}
  
template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_update_master
( bool const locfeas )
{
  auto tMIP = stats.start();

  // Append new cuts to polyhedral image
  _POLenv.erase_cuts();
  bool updflag = false;
  if( locfeas ) updflag = _update_master_feas( _solution.x, _solution.f, _solution.uf );
  else          updflag = _update_master_infeas( _Xrel ); // May not use _solution.x since empty

  // Update master MIP problem 
  if( updflag ){
    _MIPSLV.set_cuts( &_POLenv, false );
    _MIPSLV.update_objective( _POLSvar.size()-_POLSpos, _POLSvar.data()+_POLSpos, _POLScost.data()+_POLSpos );
  }

  stats.walltime_slvnlp += stats.walltime( tMIP );
  return updflag;
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
MINLPSLV<T,NLP,MIP>::optimize
(  double const* Xini, T const* Xbnd, std::ostream& os )
{
  auto tstart = stats.start();

  // Initialization
  _display_init( os );
  _iter = 0;
  _display_add( _iter );
  _Zinc =  _objscal * BASE_OPT::INF;
  _Zrel = -_objscal * BASE_OPT::INF;
  _incumbent.reset();
  
  // Initial point
  if( Xini ) _varini.assign( Xini, Xini+_var.size() );
  
  // Initial bounds
  _Xbnd.resize( _nX );
  _isbnd = true;
  for( unsigned i=0; i<_nX; i++ ){
    _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
    if( Xbnd && !Op<T>::inter( _Xbnd[i], Xbnd[i], _Xbnd[i] ) )
      return _finalize( tstart, STATUS::INFEASIBLE );
    if( !_isbnd || Op<T>::diam(_Xbnd[i]) < BASE_OPT::INF/2 ) continue;
    _isbnd = false;
  }

  // Solve relaxed MINLP model
  bool locfeas = _solve_local( tstart, _varini.data(), _Xbnd.data(), os );
#ifdef MC__MINLPSLV_DEBUG
  std::cout << _solution;
#endif

  // Update bounds
  if( locfeas ){
    _Zrel = _NLPSLV.solution().f[0];
    _Xrel = _NLPSLV.solution().x;
    if( _is_integer_feasible( _solution.x ) ){
      _incumbent = _solution;
      _Zinc = _Zrel;
    }
  }
  else{
    _Zrel = _objscal * BASE_OPT::INF;
    return _finalize( tstart, STATUS::INFEASIBLE );
  }
  
  // Intermediate display
  if( !_incumbent.x.empty() ) _display_add( "*");
  else                        _display_add( " " );
  _display_add( _Zinc );
  _display_add( _Zrel );
  _display_add( tstart );
  _display_flush( os );

  // Termination tests
  if( _interrupted( tstart ) )
    return _finalize( tstart, STATUS::INTERRUPTED );
  
  // Initialize master MIP subproblem
  _init_master();

  // Main loop
  for( ++_iter; !_converged() ; ++_iter ){
    _display_add( _iter );

    // Update master MIP subproblem
    if( !_update_master( locfeas ) ) return _finalize( tstart, STATUS::ABORTED );

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
    _Zrel = _MIPSLV.get_objective();
    for( unsigned i=0; i<_nX; i++ )
      _Xrel[i] = _MIPSLV.get_variable( _POLXvar[i] );
#ifdef MC__MINLPSLV_DEBUG
    std::cout << "_Zrel = " << _Zrel << std::endl;
    for( unsigned i=0; i<_nX; i++ )
      std::cout << "_Xrel[" << i << "] = " << _Xrel[i] << std::endl;
#endif

    // Solve local NLP model at current MIP integer fixing
    for( auto&& i: _Xint ) _Xbnd[i] = _Xrel[i];
    locfeas = _solve_local( tstart, _Xrel.data(), _Xbnd.data(), os );
#ifdef MC__MINLPSLV_DEBUG
    std::cout << _solution;
#endif
    
    // Update incumbent
    bool updinc = false;
    if( locfeas && _objscal*_Zinc > _objscal*_solution.f[0] ){
      updinc = true;
      _Zinc = _solution.f[0];
      _incumbent = _solution;
    }

    // Intermediate display
    if( updinc )        _display_add( "*");
    else if( !locfeas ) _display_add( "i" );
    else                _display_add( " " );
    _display_add( _Zinc );
    _display_add( _Zrel );
    _display_add( tstart );
    _display_flush( os );

    // Termination tests
    if( _interrupted( tstart ) )
      return _finalize( tstart, STATUS::INTERRUPTED );
  }

  return _finalize( tstart, STATUS::SUCCESSFUL );
}

template <typename T, typename NLP, typename MIP>
inline bool
MINLPSLV<T,NLP,MIP>::_update_master_feas
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
MINLPSLV<T,NLP,MIP>::_update_master_infeas
( std::vector<double> const& Xval )
{
  // Partition variable index set
  std::set<unsigned> JL, JU, JM;
  for( auto&& j: _Xint ){
    if( Xval[j] <= std::ceil(_Xlow[j]) + options.FEASTOL )
      JL.insert( j ); 
    else if( Xval[j] >= std::floor(_Xupp[j]) - options.FEASTOL )
      JU.insert( j ); 
    else
      JM.insert( j ); 
  }

  // Add constraints for the linear cut: \|y-\bar{y}\|_1 \geq 1
  auto NORM1cut = *_POLenv.add_cut( PolCut<T>::GE, 1. );
  for( auto&& j: JL ) NORM1cut->append( _POLXvar[j],  1. ).rhs() += std::ceil(_Xlow[j]);
  for( auto&& j: JU ) NORM1cut->append( _POLXvar[j], -1. ).rhs() -= std::floor(_Xupp[j]);
  for( auto&& j: JM ){
    PolVar<T> POLWvar( &_POLenv, T(0.,BASE_OPT::INF), true );
    NORM1cut->append( POLWvar,  1. );
    _POLenv.add_cut( PolCut<T>::GE, Xval[j], _POLXvar[j], 1., POLWvar,  1. );
    _POLenv.add_cut( PolCut<T>::LE, Xval[j], _POLXvar[j], 1., POLWvar, -1. );
    PolVar<T> POLNvar( &_POLenv, T(0.,1.), false );
    double M1 = 2 * ( Xval[j] - std::ceil(_Xlow[j]) );
    double M2 = 2 * ( std::floor(_Xupp[j]) - Xval[j] );
    _POLenv.add_cut( PolCut<T>::GE, Xval[j]-M1, _POLXvar[j], 1., POLWvar, -1., POLNvar, -M1 );
    _POLenv.add_cut( PolCut<T>::LE, Xval[j],    _POLXvar[j], 1., POLWvar,  1., POLNvar, -M2 );
  }

  return true;
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
  if( options.DISPLEVEL < 1 ) return;
  _odisp << "#  CONTINUOUS / INTEGER VARIABLES:   " << _nX-_Xint.size()  << " / " << _Xint.size() << std::endl
         << "#  LINEAR / NONLINEAR FUNCTIONS:     " << _nF-_Fnlin.size() << " / " << _Fnlin.size() << std::endl;
  _display_flush( os ); 
}

template <typename T, typename NLP, typename MIP>
inline void
MINLPSLV<T,NLP,MIP>::_display_init
( std::ostream& os)
{
  if( options.DISPLEVEL < 1 ) return;
  _odisp << "#  " << std::right
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
      if( i == _LDISP ){
        _odisp << std::endl << std::left << std::setw(19) << "#";
        i = 0;
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

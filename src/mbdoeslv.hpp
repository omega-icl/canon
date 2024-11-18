// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MBDOESLV Model-based Design of Experiments with MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 2.0
\date 2024
\bug No known bugs.

*/

#ifndef CANON__MBDOESLV_HPP
#define CANON__MBDOESLV_HPP

#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <algorithm>

#if defined( MC__USE_PROFIL )
 #include "mcprofil.hpp"
#elif defined( MC__USE_BOOST )
 #include "mcboost.hpp"
#elif defined( MC__USE_FILIB )
 #include "mcfilib.hpp"
#else
 #include "interval.hpp"
#endif

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
#elif  MC__USE_IPOPT
 #include "mipslv_cplex.hpp"
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
#endif

#include "minlpslv.hpp"

#include "base_mbdoe.hpp"
#include "fflin.hpp"
#include "ffdoe.hpp"
#include "ffode.hpp"
//#include "ffvect.hpp"

#define CANON__MBDOE_USE_OPFIM

namespace mc
{
//! @brief C++ class for MBDoE solution using MC++
////////////////////////////////////////////////////////////////////////
//! mc::MBDOESLV is a C++ class for solving problems in model-based
//! design of experiments using MC++
////////////////////////////////////////////////////////////////////////
class MBDOESLV
: public virtual BASE_MBDOE
{

protected:

#if defined( MC__USE_PROFIL )
 typedef ::INTERVAL I;
#elif defined( MC__USE_BOOST )
 typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
 typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
 typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
 typedef boost::numeric::interval<double,T_boost_policy> I;
#elif defined( MC__USE_FILIB )
 typedef filib::interval<double> I;
#else
 typedef Interval I;
#endif

  typedef FFGraph DAG;

#if defined( MC__USE_GUROBI )
  typedef MIPSLV_GUROBI<I> MIP;
#elif defined( MC__USE_CPLEX )
  typedef MIPSLV_CPLEX<I> MIP;
#endif
#if defined( MC__USE_SNOPT )
  typedef NLPSLV_SNOPT NLP;
#elif defined( MC__USE_IPOPT )
  typedef NLPSLV_IPOPT NLP;
#endif
  typedef MINLPSLV<I,NLP,MIP> MINLP;
  
private:

  //! @brief DAG of model
  DAG* _dag;

  //! @brief DAG of MBDOE problems
  DAG* _dagdoe;

  //! @brief local copy of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief local copy of experimental controls
  std::vector<FFVar> _vCON;

  //! @brief vector of experimental control samples
  std::vector<std::vector<double>> _vCONSAM;

  //! @brief local copy of model outputs
  std::vector<FFVar> _vOUT;
  
  //! @brief vector of FIM entries
  std::vector<FFVar> _vFIM;

  //! @brief output subgraph
  FFSubgraph _sgOUT;

  //! @brief work array for output evaluations
  std::vector<double> _wkOUT;

  //! @brief output values
  std::vector<std::vector<double>> _dOUT;
  
  // Vector of response vectors
  std::vector< std::vector< arma::vec > > _vOUTSAM;

  //! @brief FIM subgraph
  FFSubgraph _sgFIM;

  //! @brief work array for FIM evaluations
  std::vector<double> _wkFIM;

  //! @brief FIM values
  std::vector<std::vector<double>> _dFIM;
  
  // Vector of atom matrices
  std::vector< std::vector< arma::mat > > _vFIMSAM;

public:
  /** @defgroup MBDOESLV Model-based Design of Experiments using MC++
   *  @{
   */
   
  //! @brief Constructor
  MBDOESLV()
    : _dag(nullptr), _dagdoe(nullptr),
      _VOpt(0./0.)
    { stats.reset(); }

  //! @brief Destructor
  virtual ~MBDOESLV()
    {
      delete   _dag;
      delete   _dagdoe;
    }

  //! @brief MBDOE solver options
  struct Options
  {
    //! @brief Constructor
    Options():
      CRITERION(BASE_MBDOE::DOPT), RISK(NEUTRAL), CVARTHRES(0.25),
      UNCREDUC(1e-2), FIMSTOL(1e-7), 
      MINDIST(1e-6), MAXITER(4), TOLITER(1e-5), DISPLEVEL(0),
      MINLPSLV(), NLPSLV()
      {
#ifdef MC__USE_SNOPT
        NLPSLV.DISPLEVEL            = 0;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-6;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#elif  MC__USE_IPOPT
        NLPSLV.DISPLEVEL            = 0;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-5;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.HESSMETH             = NLP::Options::LBFGS;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#endif
        MINLPSLV.SEARCHALG          = MINLP::Options::OA;
        MINLPSLV.DISPLEVEL          = DISPLEVEL;
        MINLPSLV.CVRTOL             = 1e-6;
        MINLPSLV.CVATOL             = 1e-9;
        MINLPSLV.FEASTOL            = 1e-6;
        MINLPSLV.FEASPUMP           = 0;
        MINLPSLV.ROOTCUT            = 1;
        MINLPSLV.TIMELIMIT          = 36e2;
        MINLPSLV.LINMETH            = MINLP::Options::CVX;
        MINLPSLV.MAXITER            = 40;
        MINLPSLV.MSLOC              = 1;
        MINLPSLV.CPMAX              = 5;
        MINLPSLV.NLPSLV             = NLPSLV;
#ifdef MC__USE_GUROBI
        MINLPSLV.MIPSLV.DISPLEVEL   = 0;
        MINLPSLV.MIPSLV.THREADS     = 0;
        MINLPSLV.MIPSLV.MIPRELGAP   = 1e-6;
        MINLPSLV.MIPSLV.MIPABSGAP   = 1e-9;
        MINLPSLV.MIPSLV.OUTPUTFILE  = "";//"doe.lp";
#elif  MC__USE_CPLEX
        throw std::runtime_error("Error: CPLEX solver not yet implemented");
#endif
        FFDOEBase::type             = CRITERION;
      }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        CRITERION   = options.CRITERION;
        RISK        = options.RISK;
        CVARTHRES   = options.CVARTHRES;
        UNCREDUC    = options.UNCREDUC;
        FIMSTOL     = options.FIMSTOL;
        MINDIST     = options.MINDIST;
        MAXITER     = options.MAXITER;
        TOLITER     = options.TOLITER;
        DISPLEVEL   = options.DISPLEVEL;
        MINLPSLV    = options.MINLPSLV;
        NLPSLV      = options.NLPSLV;
        return *this;
      }
    //! @brief Enumeration type for risk attitude
    enum RISK_TYPE{
      NEUTRAL=0, //!< Perform a risk-neutral average design
      AVERSE     //!< Perform a risk-averse CVaR design
    };
    //! @brief Selected DOE criterion
    BASE_MBDOE::TYPE         CRITERION;
    //! @brief Selected risk attitude
    RISK_TYPE                RISK;
    //! @brief Percentile threshold for CVaR calculation
    double                   CVARTHRES;
    //! @brief Uncertainty scenario reduction within threshold (if >=0) or to k-nearest neighboors (<0)
    double                   UNCREDUC;
    //! @brief Tolerance for singular value in FIM
    double                   FIMSTOL;
    //! @brief Minimal relative mean-absolute distance between support points after refinement
    double                   MINDIST;
   //! @brief Maximal iteration of effort-based and gradient-based solves
    int                      MAXITER;
   //! @brief Stopping tolerance for effort-based and gradient-based iteration
    double                   TOLITER;
    //! @brief Verbosity level
    int                      DISPLEVEL;
    
    //! @brief MINLP effort-based solver options
    typename MINLP::Options  MINLPSLV;
    //! @brief NLP gradient-based solver options
    typename NLP::Options    NLPSLV;
  } options;

  //! @brief MBDOE solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for MBDOESLV exception handling
    enum TYPE{
      BADSIZE=0,    //!< Inconsistent dimensions
      BADIVP,       //!< Misspecified IVP-ODE
      NOMODEL,	    //!< unspecified model
      INTERN=-33    //!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr(){ return _ierr; }
    //! @brief Inline function returning the error description
    std::string what(){
      switch( _ierr ){
        case BADSIZE:
          return "MBDOESLV::Exceptions  Inconsistent dimensions";
        case BADIVP:
          return "MBDOESLV::Exceptions  Misspecified IVP-ODE model";
        case NOMODEL:
          return "MBDOESLV::Exceptions  Unspecified model";
        case INTERN:
        default:
          return "MBDOESLV::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief MBDOE solver statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime_all = walltime_setup = walltime_samgen = walltime_slvnlp = walltime_slvmip =
        std::chrono::microseconds(0); }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "#  WALL-CLOCK TIMES" << std::endl
           << "#  SOLVER SETUP:         " << std::setw(10) << to_time( walltime_setup )   << " SEC" << std::endl
           << "#  SCENARIO GENERATION:  " << std::setw(10) << to_time( walltime_samgen )  << " SEC" << std::endl
           << "#  EFFORT-BASED SOLVE:   " << std::setw(10) << to_time( walltime_slvmip )  << " SEC" << std::endl
           << "#  GRADIENT-BASED SOLVE: " << std::setw(10) << to_time( walltime_slvnlp )  << " SEC" << std::endl
           << "#  TOTAL:                " << std::setw(10) << to_time( walltime_all )     << " SEC" << std::endl
           << std::endl; }
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime_all;
    //! @brief Cumulated wall-clock time used for problem setup (in microseconds)
    std::chrono::microseconds walltime_setup;
    //! @brief Cumulated wall-clock time used by sample generation and scenario reduction (in microseconds)
    std::chrono::microseconds walltime_samgen;
    //! @brief Cumulated wall-clock time used by gradient-based NLP solver (in microseconds)
    std::chrono::microseconds walltime_slvnlp;
    //! @brief Cumulated wall-clock time used by effort-based MINLP solver (in microseconds)
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

  //! @brief Setup MBDOE problem before solution
  bool setup
    ();

  //! @brief Evaluate performance of experimental campaign
  std::pair<double,bool> evaluate_design
    ( std::list<std::pair<double,std::vector<double>>> const& Campaign, std::string const& type="",
      std::ostream& os=std::cout );

  //! @brief Generate FIM samples for <a>NSAM</a> initial supports
  bool sample_supports
    ( size_t const NSAM, std::ostream& os=std::cout );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports
  void effort_solve
    ( size_t const NEXP, std::map<size_t,double> const& EIni = std::map<size_t,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports 
  void gradient_solve
    ( std::map<size_t,double> const& EOpt, bool const update=true, std::ostream& os=std::cout );

  //! @brief Solve combined effort- and gradient-based experiment designwith <a>NEXP</a> supports 
  void combined_solve
    ( size_t const NEXP, std::ostream& os=std::cout );

  //! @brief Export effort, support and fim to file
  bool file_export
    ( std::string const& name );

  //! @brief Retrieve optimized efforts  
  std::map<size_t,double> const& efforts
    ()
    const
    { return _EOpt; }

  //! @brief Retrieve optimized supports
  std::map<size_t,std::vector<double>> const& supports
    ()
    const
    { return _SOpt; }
  
  //! @brief Retrieve optimized criterion
  double criterion
    ()
    const
    { return _VOpt; }

  //! @brief Retrieve optimized campaign
  std::list<std::pair<double,std::vector<double>>> campaign
    ()
    const;

protected:

  //! @brief map of current optimal efforts
  std::set<std::pair<size_t,size_t>> _sPARSEL;

  //! @brief current optimal criterion
  double _VOpt;

  //! @brief map of current optimal efforts
  std::map<size_t,double> _EOpt;
  
  //! @brief map of current optimal supports
  std::map<size_t,std::vector<double>> _SOpt;
  
  //! @brief vector of current optimal values for risk-averse variables
  std::vector<double> _ROpt;

  //! @brief Create local copy of output model for output prediction
  void _setup_out
    ();

  //! @brief Create local copy of output model for FIM prediction
  void _setup_fim
    ();

  //! @brief Generate output samples for <a>NSAM</a> initial supports
  bool _sample_out
    ( size_t const NSAM, std::ostream& os=std::cout );

  //! @brief Select uncertainty scenario subset based on highest value
  void _sample_rank
    ( std::set<std::pair<size_t,size_t>>& parsel,
      FFDOEBase const& BRCrit, std::vector<double> const& E0, std::ostream& os );

  //! @brief Select uncertainty scenario subset based on nearest-neighbors criterion
  void _sample_select
    ( size_t const start, size_t const inc, std::set<std::pair<size_t,size_t>>& parsel,
      FFDOEBase const& BRCrit, std::vector<double> const& E0, std::ostream& os );
  
  //! @brief Append output under given control scenario for all uncertainty scenarios
  bool _append_out
    ( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
      std::vector<std::vector<double>>& Output, std::vector<std::vector<arma::vec>>& Response,
      std::ostream& os=std::cout );

  //! @brief Generate FIM samples for <a>NSAM</a> initial supports
  bool _sample_fim
    ( size_t const NSAM, std::ostream& os=std::cout );

  //! @brief Append FIM (columnwise, lower-triangular) under given control scenario for all uncertainty scenarios
  bool _append_fim
    ( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
      std::vector<std::vector<double>>& FIM, std::vector<std::vector<arma::mat>>& Response,
      std::ostream& os=std::cout );

  //! @brief Set Bayes Risk criterion in effort-based MINLP model
  void _effort_set_BRisk
    ( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF )
    const;

  //! @brief Set FIM-based risk-neutral criterion in effort-based MINLP model
  void _effort_set_FIMNeutral
    ( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF )
    const;

  //! @brief Set FIM-based risk-averse criterion in effort-based MINLP model
  void _effort_set_FIMAverse
    ( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF, std::vector<double>& E0 )
    const;

  //! @brief Evaluate Bayes Risk criterion
  double _evaluate_BRisk
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
      std::vector<double> const& CTOT0, std::ostream& os );

  //! @brief Evaluate FIM-based risk-neutral criterion
  double _evaluate_FIMNeutral
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT, 
      std::vector<double> const& CTOT0, std::ostream& os );

  //! @brief Evaluate FIM-based risk-averse criterion
  double _evaluate_FIMAverse
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
      std::vector<double> const& CTOT0, std::ostream& os );

  //! @brief Set Bayes Risk criterion in gradient-based NLP model
  void _refine_set_BRisk
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
      std::vector<double> const& CTOT0, std::ostream& os );

  //! @brief Set FIM-based risk-neutral criterion in gradient-based NLP model
  void _refine_set_FIMNeutral
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
      std::vector<double> const& CTOT0, std::ostream& os );

  //! @brief Set FIM-based risk-averse criterion in gradient-based NLP model
  void _refine_set_FIMAverse
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
      std::vector<double>& CTOT0, std::ostream& os );

  //! @brief Generate samples for refined supports
  bool _update_supports
    ( std::map<size_t,double> const EOpt, std::map<size_t,std::vector<double>> const SOpt,
      std::ostream& os );

  //! @brief Determine if support <a>supp</a> is redundant with an existing support
  size_t _redundant_support
    ( std::vector<double> const& supp );

  //! @brief Mean-absolute error between two supports
  double _mae_support
    ( std::vector<double> const& s1, std::vector<double> const& s2 );

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit, std::map<size_t,double> const& eff,
      std::map<size_t,std::vector<double>> const& supp, std::ostream& os )
    const;

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit,
      std::list<std::pair<double,std::vector<double>>> const& campaign,
      std::ostream& os )
    const;
};

inline
bool
MBDOESLV::setup
()
{
  stats.reset();
  auto&& t_setup = stats.start();

  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  switch( options.CRITERION ){
   case BROPT:
    _setup_out();
    break;
   case AOPT:
   case DOPT:
   case EOPT:
   default:
    _setup_fim();
    break;
  }

  stats.walltime_setup += stats.walltime( t_setup );
  stats.walltime_all   += stats.walltime( t_setup );
  return true;
}

inline
void
MBDOESLV::_setup_out
()
{
  if( !_ny || _ny != BASE_MBDOE::_vOUT.size() || !BASE_MBDOE::_vCON.size() )
    throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;
  _dag->options = BASE_MBDOE::_dag->options;

  _vCON.resize( _nc );
  _dag->insert( BASE_MBDOE::_dag, _nc, BASE_MBDOE::_vCON.data(), _vCON.data() );
  _vPAR.resize( _np );
  _dag->insert( BASE_MBDOE::_dag, _np, BASE_MBDOE::_vPAR.data(), _vPAR.data() );
  _vOUT.resize( _ny );
  _dag->insert( BASE_MBDOE::_dag, _ny, BASE_MBDOE::_vOUT.data(), _vOUT.data() );

#ifdef CANON__MBDOE_SETUP_DEBUG
  _sgOUT = _dag->subgraph( _ny, _vOUT.data() );
  std::vector<FFExpr> exOUT = FFExpr::subgraph( _dag, _sgOUT ); 
  for( size_t i=0; i<_ny; ++i )
    std::cout << "OUT[" << i << "] = " << exOUT[i] << std::endl;
#endif
}

inline
void
MBDOESLV::_setup_fim
()
{
  if( !_ny || _ny != BASE_MBDOE::_vOUT.size() || !BASE_MBDOE::_vCON.size() )
    throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;
  _dag->options = BASE_MBDOE::_dag->options;

  _vCON.resize( _nc );
  _dag->insert( BASE_MBDOE::_dag, _nc, BASE_MBDOE::_vCON.data(), _vCON.data() );
  _vPAR.resize( _np );
  _dag->insert( BASE_MBDOE::_dag, _np, BASE_MBDOE::_vPAR.data(), _vPAR.data() );
  _vOUT.resize( _ny );
  _dag->insert( BASE_MBDOE::_dag, _ny, BASE_MBDOE::_vOUT.data(), _vOUT.data() );

  auto OPTION_DIFF_SAVE = FFODE::options.DIFF;
  FFODE::options.DIFF = FFODE::Options::SYM_C;
  FFVar* y_p = _dag->FAD( _ny, _vOUT.data(), _np, _vPAR.data(), true ); // Jacobian in dense format
  FFODE::options.DIFF = OPTION_DIFF_SAVE;

#ifdef CANON__MBDOE_USE_OPFIM
  FFFIM OpFIM;
  FFVar** ppFIM = OpFIM( _np, _ny, y_p, &_vOUTVAR );
  _vFIM.resize( _np*(_np+1)/2 );
  for( size_t i=0, ij=0; i<_np; ++i )
    for( size_t j=i; j<_np; ++j, ++ij )
      _vFIM[ij] = *ppFIM[ij];
#else
  _vFIM.assign( _np*(_np+1)/2, 0. );
  for( size_t k=0; k<_ny; k++ )
    for( size_t i=0, ij=0; i<_np; ++i )
      for( size_t j=i; j<_np; ++j, ++ij ){
        if( _vOUTVAR.size() == _ny )
          _vFIM[ij] += (y_p[_ny*i+k] * y_p[_ny*j+k]) / _vOUTVAR[k];
        else
          _vFIM[ij] += y_p[_ny*i+k] * y_p[_ny*j+k];
      }
#endif
  _dFIM.resize( _vFIM.size() );
  delete[] y_p;
  
#ifdef CANON__MBDOE_SETUP_DEBUG
  _sgFIM = _dag->subgraph( _vFIM.size(), _vFIM.data() );
  std::vector<FFExpr> exFIM = FFExpr::subgraph( _dag, _sgFIM ); 
  for( size_t i=0, ij=0; i<_np; ++i )
    for( size_t j=i; j<_np; ++j, ++ij )
      std::cout << "FIM[" << i << "][" << j << "] = " << exFIM[ij] << std::endl;
#endif
}

inline
bool
MBDOESLV::sample_supports
( size_t const NSAM, std::ostream& os )
{
  auto&& t_samgen = stats.start();

  // Control samples
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( _nc );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( 0 );

  _vCONSAM.clear();
  _vCONSAM.reserve( NSAM );
  for( size_t s=0; s<NSAM; ++s ){
    _vCONSAM.push_back( std::vector<double>( _nc ) );
    for( size_t i=0; i<_nc; i++ )
      _vCONSAM.back()[i] = _vCONLB[i] + ( _vCONUB[i] - _vCONLB[i] ) * gen();
  }

  // Observation samples
  bool flag = false;
  switch( options.CRITERION ){
    case BROPT:
      flag = _sample_out( NSAM, os );
      break;
      
    case AOPT:
    case DOPT:
    case EOPT:
    default:
      flag = _sample_fim( NSAM, os );
      break;
  }

  if( options.DISPLEVEL )
    os << std::endl;

  stats.walltime_samgen += stats.walltime( t_samgen );
  stats.walltime_all    += stats.walltime( t_samgen );
  return flag;
}

inline
bool
MBDOESLV::_sample_out
( size_t const NSAM, std::ostream& os )
{
  auto&& tstart = stats.start();
  if( options.DISPLEVEL )
    os << "** GENERATING SUPPORT SAMPLES " << std::flush;

  // Reset and resize output/intermediate containers
  size_t NUNC = _vPARVAL.size();
  _vOUTSAM.clear();
  _vOUTSAM.resize( NUNC );
  _dOUT.resize( NUNC );

  // Compute responses for every a priori control and uncertainty scenarios
  for( size_t s=0; s<_ne0; ++s ){
    if( !_append_out( _vCONAP[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
      return false;
    if( options.DISPLEVEL > 1  && !(s%20) )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  // Compute responses for every control samples and uncertainty scenarios
  for( size_t s=0; s<NSAM; ++s ){
    if( !_append_out( _vCONSAM[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
        return false;
    if( options.DISPLEVEL > 1  && !((s+_ne0)%20) )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL )
    os << "(" << NUNC*(_ne0+NSAM) << ")"
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.walltime( tstart ) ) << " SEC" << std::flush;

  if( !NSAM || options.UNCREDUC==0 || options.UNCREDUC >= 1. || -options.UNCREDUC >= NUNC-1 )
    return true;

  tstart = stats.start();
  if( options.DISPLEVEL )
    os << std::endl
       << "** REDUCING UNCERTAINTY SCENARIO PAIRS " << std::flush;

  FFDOEBase BRCrit;
  FFDOEBase::set_noise( _vOUTVAR );
  std::vector<double> E0( NSAM, 1./(double)NSAM );

  // Select sample pairs leading to given %age of BR full criterion
  if( options.UNCREDUC > 0. )
    _sample_rank( _sPARSEL, BRCrit, E0, os );

  // Select sample pairs as nearest neighbours 
  else{
#ifdef MC__USE_THREAD
    size_t const NOTHREADS = ( _dag->options.MAXTHREAD>0? _dag->options.MAXTHREAD: std::thread::hardware_concurrency() );
    std::vector<std::thread> vth( NOTHREADS-1 ); // Main thread also runs evaluations
    std::vector<std::set<std::pair<size_t,size_t>>> vsel( NOTHREADS-1 );

    // Dispatch neightbors selection on auxiliary thread
    for( size_t th=1; th<NOTHREADS; th++ )
      vth[th-1] = std::thread( &MBDOESLV::_sample_select, this, th, NOTHREADS, std::ref(vsel[th-1]),
                               std::cref(BRCrit), std::cref(E0), std::ref(os) );

    // Run evaluations on main thread as well
    _sPARSEL.clear();
    _sample_select( 0, NOTHREADS, _sPARSEL, BRCrit, E0, os ); 

    // Join all the threads to the main one
    for( size_t th=1; th<NOTHREADS; th++ ){
      vth[th-1].join();
      _sPARSEL.insert( vsel[th-1].cbegin(), vsel[th-1].cend() );
    }
#else
    _sample_select( 0, 1, _sPARSEL, BRCrit, E0, os ); 
#endif
  }

  if( options.DISPLEVEL )
    os << "(" << _sPARSEL.size() << ")"
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.walltime( tstart ) ) << " SEC" << std::flush;

#ifdef CANON__MBDOE_SAMPLE_DEBUG
  std::cout << "PARSEL[" << _sPARSEL.size() << "]: ";
  for( auto const& [j,k] : _sPARSEL )
    std::cout << " (" << j << "," << k << ")";
  std::cout << std::endl;
#endif

  return true;
}

inline
void
MBDOESLV::_sample_select
( size_t const start, size_t const inc, std::set<std::pair<size_t,size_t>>& parsel,
  FFDOEBase const& BRCrit, std::vector<double> const& E0, std::ostream& os )
{
  std::vector<std::pair<size_t,double>> BRall( _vOUTSAM.size()-1 );
  std::vector<std::pair<size_t,double>> BRtop( std::round(-options.UNCREDUC) );

  for( size_t j=start; j<_vOUTSAM.size(); j+=inc ){
  
    if( options.DISPLEVEL > 1 && !(j%20) )
      os << "." << std::flush;

    for( size_t k=0, kk=0; k<_vOUTSAM.size(); ++k )
      if( k != j ) BRall[kk++] = std::make_pair( k, BRCrit.atom_BR( _vOUTSAM.at(j), _vOUTSAM.at(k), E0 ) );
    auto BRgt = []( std::pair<size_t,double> const& a, std::pair<size_t,double> const& b )
                  { return a.second > b.second; };
    assert( std::partial_sort_copy( BRall.begin(), BRall.end(), BRtop.begin(), BRtop.end(), BRgt ) == BRtop.end() );

    //std::cerr << "BRtop[" << j << "]: ";
    for( auto const& [k,val] : BRtop ){
      //std::cerr << "(" << k << "," << val << ") ";
      parsel.insert( j<k? std::make_pair(j,k): std::make_pair(k,j) );
    }
    //std::cerr << std::endl;
  }
}

inline
void
MBDOESLV::_sample_rank
( std::set<std::pair<size_t,size_t>>& parsel,
  FFDOEBase const& BRCrit, std::vector<double> const& E0, std::ostream& os )
{
  size_t NUNC = _vPARVAL.size();
  std::vector<std::pair<std::pair<size_t,size_t>,double>> BRall( NUNC*(NUNC+1)/2 );
  double BRtot = 0., BRsum = 0.;
  
  for( size_t j=0, kk=0; j<NUNC; ++j ){
    for( size_t k=j+1; k<NUNC; ++k, ++kk ){
      BRall[kk] = std::make_pair( std::make_pair(j,k), BRCrit.atom_BR( _vOUTSAM.at(j), _vOUTSAM.at(k), E0 ) );
      BRtot += BRall[kk].second;
    }
  }
  
  auto BRgt = []( std::pair<std::pair<size_t,size_t>,double> const& a, std::pair<std::pair<size_t,size_t>,double> const& b )
                { return a.second > b.second; };
  std::sort( BRall.begin(), BRall.end(), BRgt );

  for( auto const& [jk,val] : BRall ){
    BRsum += val;
    parsel.insert( jk );
    //std::cerr << "(" << jk.first << "," << jk.second << "," << val << ") " << BRsum/BRtot*1e2 << "%\n";
    if( BRsum >= BRtot*(1-options.UNCREDUC) ) break;
  }
  //std::cerr << parsel.size() << " pairs\n";
}

inline
bool
MBDOESLV::_append_out
( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
  std::vector<std::vector<double>>& Output, std::vector<std::vector<arma::vec>>& Response,
  std::ostream& os )
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  try{
    _dag->veval( _sgOUT, _wkOUT, _vOUT, Output, _vPAR, Parameter, _vCON, Control );
  }
  catch(...){
    return false;
  }

  for( size_t k=0; k<Output.size(); ++k ){
    arma::vec vOut( Output[k] );
    if( Response[k].size() && arma::size( Response[k].back() ) != arma::size( vOut ) )
      throw Exceptions( Exceptions::BADSIZE );
    Response[k].push_back( vOut );
#ifdef CANON__MBDOE_SAMPLE_DEBUG
    std::cout << "OUT[" << k << "][" << Response[k].size() << "]:" << std::endl << arma::trans(vOut);
#endif
  }

  return true;
}

inline
bool
MBDOESLV::_sample_fim
( size_t const NSAM, std::ostream& os )
{
  auto&& tstart = stats.start();
  if( options.DISPLEVEL )
    os << "** GENERATING SUPPORT SAMPLES " << std::flush;

  // Reset and resize FIM/intermediate containers
  size_t NUNC = _vPARVAL.size();
  _vFIMSAM.clear();
  _vFIMSAM.resize( NUNC );
  _dFIM.resize( NUNC );

  // Compute FIMs for every a priori control and uncertainty scenarios
  for( size_t s=0; s<_ne0; ++s ){
    if( !_append_fim( _vCONAP[s], _vPARVAL, _dFIM, _vFIMSAM, os ) )
      return false;
    if( options.DISPLEVEL > 1  && !(s%20) )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  // Compute FIMs at every control samples and uncertainty scenarios
  for( size_t s=0; s<NSAM; ++s ){
    if( !_append_fim( _vCONSAM[s], _vPARVAL, _dFIM, _vFIMSAM, os ) )
      return false;
    if( options.DISPLEVEL > 1  && !((s+_ne0)%20) )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL )
    os << "(" << NUNC*(_ne0+NSAM) << ")"
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.walltime( tstart ) ) << " SEC" << std::flush;

#ifdef CANON__MBDOE_SAMPLE_DEBUG
  size_t s=0;
  for( auto const& FIMk : _vFIMSAM )
    std::cout << "_vFIMSAM[" << s++ << "]: " << FIMk.size() << std::endl;
#endif

  if( options.FIMSTOL <= 0e0 )
    return true;

  tstart = stats.start();
  if( options.DISPLEVEL )
    os << std::endl
       << "** CHECKING FIM REGULARITY " << std::flush;

  arma::mat FIM( _np, _np, arma::fill::zeros ); // average FIM
  for( size_t s=0; s<NUNC; ++s ){
    arma::mat FIMs( _np, _np, arma::fill::zeros ); // average FIM
    size_t e=0;
    for( auto const& eff : _vEFFAP ) // Atomic FIM of prior experiment
      FIMs += eff * _vFIMSAM.at(s).at(e++);
    for( size_t i=0; i<NSAM; ++i ) // Atomic FIM of new experiment
      FIMs += _vFIMSAM.at(s).at(e++) / NSAM;
#ifdef CANON__MBDOE_SAMPLE_DEBUG
    std::cout << "FIM[" << s << "]: rank = " << arma::rank( FIMs ) << std::endl << FIMs;
#endif
    if( _vPARWEI.size() == NUNC ) FIMs *= _vPARWEI[s]; // uncertainty scenario probability
    FIM += FIMs;
  }
  if( _mPARSCA.n_elem ) FIM = arma::trans(_mPARSCA) * FIM * _mPARSCA; // parameter scaling factors
#ifdef CANON__MBDOE_SAMPLE_DEBUG
  std::cout << "Average FIM: rank = " << arma::rank( FIM ) << std::endl << FIM;
#endif

  if( options.DISPLEVEL )
    os << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.walltime( tstart ) ) << " SEC" << std::flush;

  arma::mat PROJ = arma::orth( FIM, options.FIMSTOL );
#ifdef CANON__MBDOE_SAMPLE_DEBUG
  std::cout << "\nFIM image space: " << std::endl << PROJ;
#endif
  if( PROJ.n_rows > PROJ.n_cols )
    if( _mPARSCA.n_elem ) _mPARSCA *= PROJ;
    else                  _mPARSCA  = PROJ;

  if( options.DISPLEVEL )
    if( PROJ.n_rows > PROJ.n_cols )
      os << "\n   " << PROJ.n_cols << " OUT OF " << PROJ.n_rows << " SINGULAR VALUES ABOVE THRESHOLD ("
         << std::scientific << options.FIMSTOL << ")";
    else
      os << "\n   ALL " << PROJ.n_cols << " SINGULAR VALUES ABOVE THRESHOLD ("
         << std::scientific << options.FIMSTOL << ")";  
  
// USE ARMADILLO FUNCTIONS NULL AND ORTH: https://arma.sourceforge.net/docs.html#orth
// os << "  FIM is regular on the experimental space discretization" << std::endl; 
// os << "  FIM is rank-defficient on the experimental space discretization - Rank = " << std::endl; 
// os << "  x relationships between parameters... " << std::endl; 

  return true;
}

inline
bool
MBDOESLV::_append_fim
( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
  std::vector<std::vector<double>>& FIM, std::vector<std::vector<arma::mat>>& Response,
  std::ostream& os )
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  try{
    _dag->veval( _sgFIM, _wkFIM, _vFIM, FIM, _vPAR, Parameter, _vCON, Control );
  }
  catch(...){
    return false;
  }

  for( size_t k=0; k<FIM.size(); ++k ){
    arma::mat mFIM( _np, _np, arma::fill::none );
    for( size_t i=0, l=0; i<_np; ++i )
      for( size_t j=i; j<_np; ++j, ++l )
        if( i == j ) mFIM(i,i) = FIM[k][l]; 
        else         mFIM(i,j) = mFIM(j,i) = FIM[k][l];
    if( Response[k].size() && arma::size( Response[k].back() ) != arma::size( mFIM ) )
      throw Exceptions( Exceptions::BADSIZE );
    Response[k].push_back( mFIM );
#ifdef CANON__MBDOE_SAMPLE_DEBUG
    std::cout << "FIM[" << k << "][" << Response[k].size() << "]:" << std::endl << mFIM;
#endif
  }

  return true;
}

inline
size_t
MBDOESLV::_redundant_support
( std::vector<double> const& suppref )
{
  size_t pos=0;
  for( auto const& supp : _vCONSAM ){
    if( _mae_support( supp, suppref ) < options.MINDIST )
      return pos;
    ++pos;
  }
  return pos;
}

inline
double
MBDOESLV::_mae_support
( std::vector<double> const& s1, std::vector<double> const& s2 )
{
  if( s1.size() != s2.size() || s1.size() != _vCONLB.size() )
    return 0./0.; // NaN
  double mae=0;
  for( auto it1 = s1.cbegin(), it2 = s2.cbegin(), itLB = _vCONLB.cbegin(), itUB = _vCONUB.cbegin();
       it1 != s1.end();
       ++it1, ++it2, ++itLB, ++itUB )
    mae += std::fabs( *it1 - *it2 ) / std::fabs( *itUB - *itLB );
  mae /= s1.size();
#ifdef CANON__MBDOE_SAMPLE_DEBUG
  std::cout << "mae: " << std::scientific << std::setprecision(7) << mae << std::endl;
#endif
  return mae;
}

inline
bool
MBDOESLV::_update_supports
( std::map<size_t,double> const EOpt, std::map<size_t,std::vector<double>> const SOpt,
  std::ostream& os )
{
  auto&& t_samgen = stats.start();
  if( options.DISPLEVEL )
    os << "** REFINING SUPPORT SAMPLES" << std::endl;

  size_t posSupp = _vCONSAM.size(), newSupp = 0;
  auto itE = EOpt.cbegin();
  auto itS = SOpt.cbegin();
  _EOpt.clear();
  _SOpt.clear();
  for( ; itS != SOpt.cend(); ++itS, ++itE ){
    auto const& eff  = itE->second; 
    auto const& supp = itS->second;
    size_t pos = _redundant_support( supp );
    // Refined support is redundant
    if( pos < _vCONSAM.size() ){
      if( options.DISPLEVEL > 1 )
        os << "   REFINED SUPPORT REDUNDANT WITH #" << pos << std::endl;
      auto itR = _EOpt.find( pos );
      // Redundant support not present
      if( itR == _EOpt.end() ){
        _EOpt[pos] = eff;
        _SOpt[pos] = supp;
      }
      // Redundant support already present
      else
        _EOpt[pos] += eff;      
    }
    // Refined support is distinct
    else{
      _EOpt[_vCONSAM.size()] = eff;
      _SOpt[_vCONSAM.size()] = supp;
      _vCONSAM.push_back( supp );
      ++newSupp;
    }
  }

  // Add samples for refined controls
  for( size_t s=posSupp; s<posSupp+newSupp; ++s ){

    switch( options.CRITERION ){
      case BROPT:
        if( !_append_out( _vCONSAM[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
          return false;
        break;
      
      case AOPT:
      case DOPT:
      case EOPT:
      default:
        if( !_append_fim( _vCONSAM[s], _vPARVAL, _dFIM, _vFIMSAM, os ) )
          return false;
        break;
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  stats.walltime_samgen += stats.walltime( t_samgen );
  stats.walltime_all    += stats.walltime( t_samgen );
  return true;
}

inline
bool
MBDOESLV::file_export
( std::string const& name )
{
  auto itPARVAL = _vPARVAL.cbegin();
  for( size_t s=0; s<_vPARVAL.size(); ++s, ++itPARVAL ){
    std::ofstream ofile( name + "_" + std::to_string(s) + ".log" );
    if( !ofile ) return false;
    
    ofile << std::scientific << std::setprecision(6);
    for( size_t k=0; k<_vCONSAM.size(); ++k ){
      for( size_t i=0; i<itPARVAL->size(); ++i )
        ofile << (*itPARVAL)[i] << "  ";

      for( size_t i=0; i<_vCONSAM[k].size(); ++i )
        ofile << _vCONSAM[k][i] << "  ";

      ofile << ( _EOpt.count(k)? _EOpt[k]: 0 ) << "  ";

      switch( options.CRITERION ){
        case BROPT:
          for( size_t i=0; i<_vOUTSAM[s][k].n_rows; ++i )
            ofile << _vOUTSAM[s][k](i) << "  ";
          break;
          
        case AOPT:
        case DOPT:
        case EOPT:
        default:
          for( size_t i=0; i<_vFIMSAM[s][k].n_rows; ++i )
            for( size_t j=i; j<_vFIMSAM[s][k].n_cols; ++j )
              ofile << _vFIMSAM[s][k](i,j) << "  ";
          break;
      }
      ofile << std::endl;
    }
  }
  return true;
}

inline
void
MBDOESLV::combined_solve
( size_t const NEXP, std::ostream& os )
{
  _EOpt.clear();
  double VLast;
  for( int it=0; ; ){
    effort_solve( NEXP, _EOpt, os );
    if( it && std::fabs( VLast - _VOpt ) < options.TOLITER * std::fabs( VLast + _VOpt ) / 2 ){
      if( options.DISPLEVEL )
        os << "** CONVERGENCE TOLERANCE SATISFIED" << std::endl;
      break;
    }

    gradient_solve( _EOpt, true, os );
    VLast = _VOpt;
    if( ++it >= options.MAXITER ){
      if( options.DISPLEVEL )
        os << "** MAXIMUM ITERATION LIMIT REACHED" << std::endl;
      break;
    }
  }
}

inline
void
MBDOESLV::_effort_set_BRisk
( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF )
const
{
  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF ) - (int)NEXP );  

  FFBREff OpBRCrit;
  doe.set_obj( BASE_OPT::MIN, OpBRCrit( EFF, &_vOUTSAM, &_vEFFAP ) );
}

inline
void
MBDOESLV::_effort_set_FIMNeutral
( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF )
const
{

  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF ) - (int)NEXP );

  FFFIMEff OpFIMEff;
  size_t const NUNC  = _vPARVAL.size();
  doe.set_obj( BASE_OPT::MAX, Sum( NUNC, OpFIMEff( EFF, &_vFIMSAM, &_vEFFAP ), _vPARWEI.data() ) );
}

inline
void
MBDOESLV::_effort_set_FIMAverse
( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF, std::vector<double>& E0 )
const
{
  size_t const NUNC  = _vPARVAL.size();
  std::vector<FFVar> DELTA( NUNC );
  FFVar VaR( _dagdoe );
  for( auto& Dk : DELTA )
    Dk.set( _dagdoe );
  E0.resize( EFF.size()+NUNC+1, 0e0 );
  doe.add_var( NUNC, DELTA.data(), 0e0 );//, 1e2 );
  doe.add_var( VaR );//, -1e2, 1e2 );

  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF ) - (int)NEXP );

  FFFIMEff OpFIMEff;
  FFVar const*const* ppFIMEff = OpFIMEff( EFF, &_vFIMSAM, &_vEFFAP );
  for( size_t s=0; s<NUNC; s++ )
    doe.add_ctr( BASE_OPT::LE, VaR - DELTA[s] - *(ppFIMEff[s]) );

  doe.set_obj( BASE_OPT::MAX, VaR - Sum( DELTA, _vPARWEI ) / options.CVARTHRES );
}

inline
void
MBDOESLV::effort_solve
( size_t const NEXP, std::map<size_t,double> const& EIni, std::ostream& os )
{
  auto&& t_slvmip = stats.start();

  // Define MINLP DAG
  delete _dagdoe; _dagdoe = new DAG;

  size_t const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF( NSUPP );
  for( auto& Ek : EFF )
    Ek.set( _dagdoe );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, (double)NEXP/(double)NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [ndx,eff] : EIni )
      E0[ndx] = eff;
  }
  
  // Update external operations
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;

  // Convex MINLP optimization
  MINLP doe;
  doe.options = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( EFF, 0e0, NEXP, 1 );

  switch( options.CRITERION ){
    case BROPT:
      _effort_set_BRisk( doe, NEXP, EFF );
      break;
      
    case AOPT:
    case DOPT:
    case EOPT:
    default:
      switch( options.RISK){
        case Options::NEUTRAL:
          _effort_set_FIMNeutral( doe, NEXP, EFF );
          break;
        case Options::AVERSE:
          _effort_set_FIMAverse( doe, NEXP, EFF, E0 );
          break;
      }
      break;
  }
  
  doe.setup();
  //doe.optimize( E0.data() );
  //doe.optimize( E0.data(), nullptr, nullptr, effort_apportion );
  doe.optimize( E0.data(), nullptr, nullptr, effort_rounding );

  if( options.DISPLEVEL > 1 )
    doe.stats.display();

  _EOpt.clear();
  _SOpt.clear();
  _ROpt.clear();
  _VOpt = BASE_OPT::BASE_OPT::INF;
  if( doe.get_status() == MINLP::SUCCESSFUL ){
    size_t isupp = 0;
    for( auto const& Ek : doe.get_incumbent().x ){
      if( isupp >= NSUPP ){
        _ROpt.push_back( Ek );
      }
      else if( Ek > 1e-3 ){
        _EOpt[isupp] = Ek;
        _SOpt[isupp] = _vCONSAM[isupp];
      }
      ++isupp;
    }
    _VOpt = doe.get_incumbent().f[0];
  }

  if( options.DISPLEVEL )
    _display_design( "EFFORT-BASED EXACT DESIGN", _VOpt, _EOpt, _SOpt, os ); 

  stats.walltime_slvmip += stats.walltime( t_slvmip );
  stats.walltime_all    += stats.walltime( t_slvmip );
}

inline
std::list<std::pair<double,std::vector<double>>>
MBDOESLV::campaign
()
const
{
  assert( _EOpt.size() == _SOpt.size() );
  std::list<std::pair<double,std::vector<double>>> C;
  auto iteff = _EOpt.cbegin();
  auto itsup = _SOpt.cbegin();
  for( ; iteff != _EOpt.cend(); ++iteff, ++itsup )
    C.push_back( { iteff->second, itsup->second } );
  return C;
}

inline
double
MBDOESLV::_evaluate_BRisk
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
  std::vector<double> const& CTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() ){
    auto DISPLEVEL  = options.DISPLEVEL;
    auto UNCREDUC   = options.UNCREDUC;
    options.UNCREDUC  = 0.;
    options.DISPLEVEL = 0;
    _sample_out( 0, os );
    options.DISPLEVEL = DISPLEVEL;
    options.UNCREDUC  = UNCREDUC;
  }

  // Define Bayes risk criterion
  FFBRCrit OpBRCrit;
  FFVar FBR = OpBRCrit( CTOT.data(), _dag, &_vPAR, &_vCON, &_vOUT, &EOpt, &_vPARVAL, &_vOUTSAM, &_vEFFAP );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, &FBR ), " FBR" );
#endif

  // Evaluate Bayes risk criterion
  double DBR = 0./0.;
  _dagdoe->eval( 1, &FBR, &DBR, CTOT.size(), CTOT.data(), CTOT0.data() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  auto&& FGRADOUT  = _dagdoe->FAD( std::vector<FFVar>({FBR}), CTOT );
  std::vector<double> DGRADOUT;
  _dagdoe->eval( FGRADOUT, DGRADOUT, CTOT, CTOT0 );
  for( size_t i=0; i<CTOT.size(); i++ ){
    auto CTOT0_pert = CTOT0;
    CTOT0_pert[i] = CTOT0[i]*(1+1e-3)+1e-3;
    double DBR_pert;
    _dagdoe->eval( 1, &FBR, &DBR_pert, CTOT.size(), CTOT.data(), CTOT0_pert.data() );
    std::cout << "d_CTOT[" << i << "] = " << (CTOT0_pert[i]-CTOT0[i]) << std::endl;
    std::cout << "grad BR[" << i << "] = " << (DBR_pert-DBR)/(CTOT0_pert[i]-CTOT0[i]) << std::endl;
  }
#endif

  return DBR;
}

inline
double
MBDOESLV::_evaluate_FIMNeutral
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
  std::vector<double> const& CTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() ){
    auto DISPLEVEL = options.DISPLEVEL;
    auto FIMSTOL   = options.FIMSTOL;
    options.FIMSTOL   = 0.;
    options.DISPLEVEL = 0;
    _sample_fim( 0, os );
    options.DISPLEVEL = DISPLEVEL;
    options.FIMSTOL   = FIMSTOL;
  }
  
  // Define FIM criterion
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( CTOT.data(), _dag, &_vPAR, &_vCON, &_vFIM, &EOpt, &_vPARVAL, &_vFIMSAM, &_vEFFAP );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppFIMCrit[0] ), " DOECrit" );
#endif

  size_t const NUNC = _vPARVAL.size();
  FFLin<I> Sum;
  FFVar FFIM = Sum( NUNC, ppFIMCrit, _vPARWEI.data() );

  // Evaluate FIM criterion
  double DFIM = 0./0.;
  _dagdoe->eval( 1, &FFIM, &DFIM, CTOT.size(), CTOT.data(), CTOT0.data() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  auto&& FGRADFIM  = _dagdoe->FAD( std::vector<FFVar>({FFIM}), CTOT );
  std::vector<double> DGRADFIM;
  _dagdoe->eval( FGRADFIM, DGRADFIM, CTOT, CTOT0 );
#endif

  return DFIM;
}

inline
double
MBDOESLV::_evaluate_FIMAverse
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
  std::vector<double> const& CTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() ){
    auto DISPLEVEL = options.DISPLEVEL;
    auto FIMSTOL   = options.FIMSTOL;
    options.FIMSTOL   = 0.;
    options.DISPLEVEL = 0;
    _sample_fim( 0, os );
    options.DISPLEVEL = DISPLEVEL;
    options.FIMSTOL   = FIMSTOL;
  }

  // Define FIM criterion
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( CTOT.data(), _dag, &_vPAR, &_vCON, &_vFIM, &EOpt, &_vPARVAL, &_vFIMSAM, &_vEFFAP );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppFIMCrit[0] ), " DOECrit" );
#endif

  size_t const NUNC = _vPARVAL.size();
  std::vector<double> DFIMCrit( NUNC );
  std::vector<FFVar> FFIMCrit( NUNC );
  for( size_t s=0; s<NUNC; ++s )
    FFIMCrit[s] = *ppFIMCrit[s];

  // Evaluate cost function
  _dagdoe->eval( NUNC, FFIMCrit.data(), DFIMCrit.data(), CTOT.size(), CTOT.data(), CTOT0.data() );

  std::map<double,double> SA;
  for( size_t s=0; s<NUNC; ++s )
    SA[DFIMCrit[s]] = _vPARWEI[s];

  double prsum = 0., VaR = 0.;
  for( auto const& [crit,pr] : SA ){
    VaR = crit;
    if( prsum + pr > options.CVARTHRES ) break;
    prsum += pr;
  }
  //std::cout << "VaR = " << VaR << std::endl;

  double DFIM = VaR;
  for( auto const& [crit,pr] : SA ){
    if( crit > VaR ) break;
    DFIM -= ( VaR - crit ) * pr / options.CVARTHRES;
  }
  //std::cout << "CVaR = " << DFIM << std::endl;

  return DFIM;
}

inline
std::pair<double,bool>
MBDOESLV::evaluate_design
( std::list<std::pair<double,std::vector<double>>> const& Campaign, std::string const& type, std::ostream& os )
{
  // Define evaluation DAG
  delete _dagdoe; _dagdoe = new DAG;

  size_t const NSUP  = Campaign.size();
  size_t const NCTOT = _nc * NSUP;
  std::vector<FFVar> CTOT(NCTOT);  // Concatenate experimental controls
  for( size_t i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::vector<double> CTOT0;
  CTOT0.reserve(NCTOT);
  std::map<size_t,double> EOpt;
  size_t ieff = 0;
  for( auto const& [eff,supp] : Campaign ){
    EOpt[ieff++] = eff;
    CTOT0.insert( CTOT0.end(), supp.cbegin(), supp.cend() );
  }
  
  // Update external operations
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;

  // Evaluate design
  std::string header( type.empty()? "DESIGN PERFORMANCE": type + " DESIGN PERFORMANCE" );
  double crit = 0./0.;

  try{
    switch( options.CRITERION ){
      case BROPT:
        crit = _evaluate_BRisk( EOpt, CTOT, CTOT0, os );
        break;
      
      case AOPT:
      case DOPT:
      case EOPT:
      default:
        switch( options.RISK){
          case Options::NEUTRAL:
            crit = _evaluate_FIMNeutral( EOpt, CTOT, CTOT0, os );
            break;
          case Options::AVERSE:
            crit = _evaluate_FIMAverse( EOpt, CTOT, CTOT0, os );
            break;
        }
        break;
    }
  }

  catch(...){
    if( options.DISPLEVEL )
      _display_design( header, 0./0., std::list<std::pair<double,std::vector<double>>>(), os ); 
    return std::make_pair( 0./0., false ); // NaN
  }

  if( options.DISPLEVEL )
    _display_design( header, crit, Campaign, os ); 
  return std::make_pair( crit, true );
}

inline
void
MBDOESLV::_refine_set_BRisk
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
  std::vector<double> const& CTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() )
    _sample_out( 0, os );

  // Define Bayes risk objective
  FFBRCrit OpBRCrit;
  FFVar FBR = OpBRCrit( CTOT.data(), _dag, &_vPAR, &_vCON, &_vOUT, &EOpt, &_vPARVAL, &_vOUTSAM, &_vEFFAP );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, &FBR ), " FBR" );
#endif
  doeref.set_obj( BASE_OPT::MIN, FBR ); // minimize Bayesian risk

#ifdef CANON__MBDOE_SOLVE_DEBUG
  double DFBR = 0./0.;
  _dagdoe->eval( 1, &FBR, &DFBR, CTOT.size(), CTOT.data(), CTOT0.data() );
  std::cout << "BROPT = " << DFBR << std::endl;
  { int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
}

inline
void
MBDOESLV::_refine_set_FIMNeutral
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
  std::vector<double> const& CTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() )
    _sample_fim( 0, os );

  // Define FIM criterion
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( CTOT.data(), _dag, &_vPAR, &_vCON, &_vFIM, &EOpt, &_vPARVAL, &_vFIMSAM, &_vEFFAP );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppFIMCrit[0] ), " DOECrit" );
#endif

  size_t const NUNC = _vPARVAL.size();
  FFLin<I> Sum;
  FFVar FFIM = Sum( NUNC, ppFIMCrit, _vPARWEI.data() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  double DFIM;
  _dagdoe->eval( 1, &FFIM, &DFIM, CTOT.size(), CTOT.data(), CTOT0.data() );
  std::cout << "Crit = " << DFIM << std::endl;
  auto&& FGRADFIM  = _dagdoe->FAD( std::vector<FFVar>({FFIM}), CTOT );
  std::vector<double> DGRADFIM;
  _dagdoe->eval( FGRADFIM, DGRADFIM, CTOT, CTOT0 );
  std::cout << "Grad Cit AD = " << arma::trans( arma::vec( DGRADFIM.data(), DGRADFIM.size(), false ) ) << std::endl;
  double const TOL = 1e-3;
  for( size_t i=0; i<CTOT.size(); ++i ){
    std::vector<double> CTOT1 = CTOT0;
    CTOT1[i] = CTOT0[i] + std::fabs(CTOT0[i])*TOL + TOL;
    _dagdoe->eval( 1, &FFIM, &DFIM, CTOT.size(), CTOT.data(), CTOT1.data() );
    DGRADFIM[i] = DFIM;
    CTOT1[i] = CTOT0[i] - std::fabs(CTOT0[i])*TOL - TOL;
    _dagdoe->eval( 1, &FFIM, &DFIM, CTOT.size(), CTOT.data(), CTOT1.data() );
    DGRADFIM[i] -= DFIM;
    DGRADFIM[i] /= std::fabs(CTOT0[i])*2*TOL + 2*TOL;
  }
  std::cout << "Grad Cit FD = " << arma::trans( arma::vec( DGRADFIM.data(), DGRADFIM.size(), false ) ) << std::endl;
  { int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
  doeref.set_obj( BASE_OPT::MAX, FFIM ); // maximize average FIM-based criterion
}

inline
void
MBDOESLV::_refine_set_FIMAverse
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& CTOT,
  std::vector<double>& CTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() )
    _sample_fim( 0, os );

  // Define FIM criterion
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( CTOT.data(), _dag, &_vPAR, &_vCON, &_vFIM, &EOpt, &_vPARVAL, &_vFIMSAM, &_vEFFAP );
#ifdef CANON__MBDOE_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppFIMCrit[0] ), " DOECrit" );
#endif

  size_t const NUNC = _vPARVAL.size();
  std::vector<FFVar> DELTA( NUNC );
  FFVar VaR( _dagdoe );
  for( auto& Dk : DELTA )
    Dk.set( _dagdoe );
  if( _ROpt.size() == NUNC+1 )
    for( auto const& r0 : _ROpt ) CTOT0.push_back( r0 ); 
  else
    CTOT0.resize( CTOT.size()+NUNC+1, 0e0 );
  doeref.add_var( DELTA, 0e0 );//, 1e2 );
  doeref.add_var( VaR );//, -1e2, 1e2 );

  FFLin<I> Sum;
  doeref.set_obj( BASE_OPT::MAX, VaR - Sum( NUNC, DELTA.data(), _vPARWEI.data() ) / options.CVARTHRES );  // maximize risk-averse FIM-based criterion
  for( size_t s=0; s<NUNC; s++ )
    doeref.add_ctr( BASE_OPT::LE, VaR - DELTA[s] - *ppFIMCrit[s] );
}

inline
void
MBDOESLV::gradient_solve
( std::map<size_t,double> const& EOpt, bool const update, std::ostream& os )
{
  auto&& t_slvnlp = stats.start();

  // Define NLP DAG
  delete _dagdoe; _dagdoe = new DAG;
  
  size_t const NSUP = EOpt.size();
  size_t const NCTOT = _nc * NSUP;
  std::vector<FFVar> CTOT(NCTOT);  // Concatenated experimental controls
  for( size_t i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::vector<double> CTOT0, CTOTLB, CTOTUB;
  CTOT0.reserve(NCTOT);
  CTOTLB.reserve(NCTOT);
  CTOTUB.reserve(NCTOT);
  for( auto const& [ndx,eff] : EOpt ){
    CTOT0.insert( CTOT0.end(), _vCONSAM[ndx].cbegin(), _vCONSAM[ndx].cend() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
    std::cout << "C[" << ndx << "] = ";
    for( size_t i=0; i<_nc; i++ )
      std::cout << _vCONSAM[ndx][i] << "  ";
    std::cout << std::endl;
#endif
    CTOTLB.insert( CTOTLB.end(), _vCONLB.cbegin(), _vCONLB.cend() );
    CTOTUB.insert( CTOTUB.end(), _vCONUB.cbegin(), _vCONUB.cend() );
  }

  // Update external operations
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;

  // Local NLP optimization
  NLP doeref;
  doeref.options = options.NLPSLV;
  doeref.set_dag( _dagdoe );
  doeref.add_var( CTOT, CTOTLB, CTOTUB );

  switch( options.CRITERION ){
    case BROPT:
      _refine_set_BRisk( doeref, EOpt, CTOT, CTOT0, os );
      break;
      
    case AOPT:
    case DOPT:
    case EOPT:
    default:
      switch( options.RISK){
        case Options::NEUTRAL:
          _refine_set_FIMNeutral( doeref, EOpt, CTOT, CTOT0, os );
          break;
        case Options::AVERSE:
          _refine_set_FIMAverse( doeref, EOpt, CTOT, CTOT0, os );
          break;
      }
      break;
  }

  doeref.setup();
  doeref.solve( CTOT0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << doeref.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << doeref.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( update ){
    _SOpt.clear();
    _VOpt = 0./0.;
 
    if( doeref.get_status() == NLP::SUCCESSFUL
     || doeref.get_status() == NLP::FAILURE
     || doeref.get_status() == NLP::INTERRUPTED ){
      double const* dC = doeref.solution().x.data();
      for( auto const& [ndx,eff] : EOpt ){
        _SOpt[ndx] = std::vector<double>( dC, dC+_nc );
        dC += _nc;
      }
      _update_supports( _EOpt, _SOpt, os );
      _VOpt = doeref.solution().f[0];
      size_t const NEXTRA = doeref.solution().x.size() - NCTOT;
      if( NEXTRA > 0 )
        _ROpt.assign( dC, dC+NEXTRA );
    }
  }

  if( options.DISPLEVEL )
    _display_design( "GRADIENT-BASED REFINED DESIGN", _VOpt, EOpt, _SOpt, os ); 

  stats.walltime_slvnlp += stats.walltime( t_slvnlp );
  stats.walltime_all    += stats.walltime( t_slvnlp );
}

inline
void
MBDOESLV::_display_design
( std::string const& title, double const& crit, std::map<size_t,double> const& eff,
  std::map<size_t,std::vector<double>> const& supp, std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( eff.empty() ){
     os << " FAILED" << std::endl;
     return;
  } 
   
  os  << std::scientific << std::setprecision(5) << crit << std::endl;
  for( auto const& [i,s] : supp ){
    os << "   SUPPORT #" << i << ": " << std::fixed << std::setprecision(0) << eff.at(i) << " x [ "
       << std::scientific << std::setprecision(5);
      for( auto Ck : s )
        os << Ck << " ";
      os << "]" << std::endl;
  }
  os << std::endl;
}

inline
void
MBDOESLV::_display_design
( std::string const& title, double const& crit,
  std::list<std::pair<double,std::vector<double>>> const& campaign,
  std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( campaign.empty() ){
     os << " FAILED" << std::endl;
     return;
  } 
   
  os  << std::scientific << std::setprecision(5) << crit << std::endl;
  size_t i=0;
  for( auto const& [eff,supp] : campaign ){
    os << "   SUPPORT #" << i++ << ": " << std::fixed << std::setprecision(0) << eff << " x [ "
       << std::scientific << std::setprecision(5);
      for( auto Ck : supp )
        os << Ck << " ";
      os << "]" << std::endl;
  }
  os << std::endl;
}

} // end namespace mc

#endif

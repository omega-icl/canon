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
#include "ffvect.hpp"

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

  //! @brief local copy of IVP-ODE time variable
  std::vector<FFVar> _vT;

  //! @brief local copy of IVP-ODE state variable
  std::vector<FFVar> _vX;

  //! @brief local copy of IVP-ODE quadrature variables
  std::vector<FFVar> _vQ;

  //! @brief local copy of IVP-ODE differential equations
  std::vector<std::vector<FFVar>> _vDE;

  //! @brief local copy of IVP-ODE initial conditions
  std::vector<std::vector<FFVar>> _vIC;

  //! @brief local copy of IVP-ODE quadrature equations
  std::vector<std::vector<FFVar>> _vQUAD;

  //! @brief local copy of IVP-ODE state functions
  std::vector<std::vector<FFVar>> _vFCT;

public:
  /** @defgroup MBDOESLV Model-based Design of Experiments using MC++
   *  @{
   */
   
  //! @brief Constructor
  MBDOESLV()
    : _dag(nullptr), _dagdoe(nullptr),
      _VOpt(0./0.)
    {}

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
      CRITERION(FFDOEBase::DOPT), RISK(NEUTRAL), UNCREDUC(0), CVARTHRES(0.2),
      INISAMP(100), MINDIST(1e-6), MAXITER(4), TOLITER(1e-5), DISPLEVEL(0), MAXTHREAD(1), 
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
        UNCREDUC    = options.UNCREDUC;
        CVARTHRES   = options.CVARTHRES;
        INISAMP     = options.INISAMP;
        MINDIST     = options.MINDIST;
        MAXITER     = options.MAXITER;
        TOLITER     = options.TOLITER;
        DISPLEVEL   = options.DISPLEVEL;
        MAXTHREAD   = options.MAXTHREAD;
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
    FFDOEBase::TYPE          CRITERION;
    //! @brief Selected risk attitude
    RISK_TYPE                RISK;
    //! @brief Uncertainty scenario reduction to k nearest neighboors (0: no reduction)
    size_t                   UNCREDUC;
    //! @brief Percentile threshold for CVaR calculation
    double                   CVARTHRES;
    //! @brief Initial sampling size of experimental control space
    unsigned                 INISAMP;
    //! @brief Minimal relative mean-absolute distance between support points after refinement
    double                   MINDIST;
   //! @brief Maximal iteration of effort-based and gradient-based solves
    int                      MAXITER;
   //! @brief Stopping tolerance for effort-based and gradient-based iteration
    double                   TOLITER;
    //! @brief Verbosity level
    int                      DISPLEVEL;
    //! @brief Maximum number of threads for output/FIM evaluation
    size_t                   MAXTHREAD;
    
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

  //! @brief Setup MBDOE problem before solution
  bool setup
    ();

  //! @brief Evaluate performance of experimental campaign
  std::pair<double,bool> evaluate_design
    ( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type="",
      std::ostream& os=std::cout );

  //! @brief Generate FIM samples for <a>NSAM</a> initial supports
  bool sample_supports
    ( unsigned const NSAM, std::ostream& os=std::cout );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports
  void effort_solve
    ( unsigned const NEXP, std::map<unsigned,double> const& EIni = std::map<unsigned,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports 
  void gradient_solve
    ( std::map<unsigned,double> const& EOpt, bool const update=true, std::ostream& os=std::cout );

  //! @brief Solve combined effort- and gradient-based experiment designwith <a>NEXP</a> supports 
  void combined_solve
    ( unsigned const NEXP, std::ostream& os=std::cout );

  //! @brief Export effort, support and fim to file
  bool file_export
    ( std::string const& name );

  //! @brief Retrieve optimized efforts  
  std::map<unsigned,double> const& efforts
    ()
    const
    { return _EOpt; }

  //! @brief Retrieve optimized supports
  std::map<unsigned,std::vector<double>> const& supports
    ()
    const
    { return _SOpt; }
  
  //! @brief Retrieve optimized criterion
  double criterion
    ()
    const
    { return _VOpt; }

  //! @brief Retrieve optimized campaign
  std::multimap<double,std::vector<double>> campaign
    ()
    const;

protected:

  //! @brief map of current optimal efforts
  std::set<std::pair<unsigned,unsigned>> _sPARSEL;

  //! @brief current optimal criterion
  double _VOpt;

  //! @brief map of current optimal efforts
  std::map<unsigned,double> _EOpt;
  
  //! @brief map of current optimal supports
  std::map<unsigned,std::vector<double>> _SOpt;
  
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
    ( unsigned const NSAM, std::ostream& os=std::cout );

#ifdef MC__USE_THREAD
  //! @brief Append output under given control scenario for all uncertainty scenarios
  bool _append_out
    ( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
      std::vector<std::vector<double>>& Output, std::vector<std::vector<arma::vec>>& Response,
      std::ostream& os=std::cout );
#else
  //! @brief Append output under given control and uncertainty scenario
  bool _append_out
    ( std::vector<double> const& Control, std::vector<double> const& Parameter,
      std::vector<double>& Output, std::vector<arma::vec>& Response,
      std::ostream& os=std::cout );
#endif

  //! @brief Generate FIM samples for <a>NSAM</a> initial supports
  bool _sample_fim
    ( unsigned const NSAM, std::ostream& os=std::cout );

#ifdef MC__USE_THREAD
  //! @brief Append FIM (columnwise, lower-triangular) under given control scenario for all uncertainty scenarios
  bool _append_fim
    ( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
      std::vector<std::vector<double>>& FIM, std::vector<std::vector<arma::mat>>& Response,
      std::ostream& os=std::cout );
#else
  //! @brief Append FIM (columnwise, lower-triangular) under given control and uncertainty scenario
  bool _append_fim
    ( std::vector<double> const& Control, std::vector<double> const& Parameter,
      std::vector<double>& FIM, std::vector<arma::mat>& Response,
      std::ostream& os=std::cout );
#endif

  //! @brief Evaluate Bayesian risk of experimental campaign
  std::pair<double,bool> _evaluate_design_br
    ( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os );

  //! @brief Evaluate FIM-based criterion of experimental campaign
  std::pair<double,bool> _evaluate_design_fim
    ( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports to minimize Bayesian risk
  void _effort_minimize_br
    ( unsigned const NEXP, std::map<unsigned,double> const& EIni = std::map<unsigned,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports to maximize FIM
  void _effort_maximize_fim
    ( unsigned const NEXP, std::map<unsigned,double> const& EIni = std::map<unsigned,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports to minimize Bayesian risk
  void _gradient_minimize_br
    ( std::map<unsigned,double> const& EOpt, bool const update=true, std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports to maximize FIM
  void _gradient_maximize_fim
    ( std::map<unsigned,double> const& EOpt, bool const update=true, std::ostream& os=std::cout );

  //! @brief Build Bayesian risk for gradient-based search
#ifdef MC__USE_THREAD
  void _build_br
    ( std::vector<std::vector<FFVar>>& BROUT, std::vector<FFVar>& CTOT, std::vector<FFVar>& PTOT,
      std::map<unsigned,double> const& EOpt, std::ostream& os );
#else
  void _build_br
    ( std::vector<FFVar>& BROUT, std::vector<FFVar>& CTOT, std::vector<FFVar>& PTOT,
      std::map<unsigned,double> const& EOpt, std::ostream& os );
#endif

  //! @brief Build FIM for gradient-based search
  void _build_fim
    ( std::vector<FFVar>& vFIM, std::vector<FFVar>& CTOT, FFVar const* PREF,
      std::map<unsigned,double> const& EOpt, std::ostream& os=std::cout );

  //! @brief Generate samples for refined supports
  bool _update_supports
    ( std::map<unsigned,double> const EOpt, std::map<unsigned,std::vector<double>> const SOpt,
      std::ostream& os=std::cout );

  //! @brief Determine if support <a>supp</a> is redundant with an existing support
  unsigned _redundant_support
    ( std::vector<double> const& supp );

  //! @brief Mean-absolute error between two supports
  double _mae_support
    ( std::vector<double> const& s1, std::vector<double> const& s2 );

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit, std::map<unsigned,double> const& eff,
      std::map<unsigned,std::vector<double>> const& supp, std::ostream& os=std::cout )
    const;

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit, std::multimap<double,std::vector<double>> const& campaign,
      std::ostream& os=std::cout )
    const;
};

inline
bool
MBDOESLV::setup
()
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  switch( options.CRITERION ){
   case FFDOEBase::BROPT:
    _setup_out();
    break;
   case FFDOEBase::AOPT:
   case FFDOEBase::DOPT:
   case FFDOEBase::EOPT:
   default:
    _setup_fim();
    break;
  }

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
//  _dOUT.resize( _ny );

#ifdef CANON__MBDOE_SETUP_DEBUG
  _sgOUT = _dag->subgraph( _ny, _vOUT.data() );
  std::vector<FFExpr> exOUT = FFExpr::subgraph( _dag, _sgOUT ); 
  for( unsigned i=0; i<_ny; ++i )
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
  for( unsigned i=0, ij=0; i<_np; ++i )
    for( unsigned j=i; j<_np; ++j, ++ij )
      _vFIM[ij] = *ppFIM[ij];
#else
  _vFIM.assign( _np*(_np+1)/2, 0. );
  for( unsigned k=0; k<_ny; k++ )
    for( unsigned i=0, ij=0; i<_np; ++i )
      for( unsigned j=i; j<_np; ++j, ++ij ){
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
  for( unsigned i=0, ij=0; i<_np; ++i )
    for( unsigned j=i; j<_np; ++j, ++ij )
      std::cout << "FIM[" << i << "][" << j << "] = " << exFIM[ij] << std::endl;
#endif
}

inline
bool
MBDOESLV::sample_supports
( unsigned const NSAM, std::ostream& os )
{
  if( options.DISPLEVEL )
    os << "** GENERATING SUPPORT SAMPLES" << std::endl;

  // Control samples
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( _nc );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( 0 );

  _vCONSAM.clear();
  _vCONSAM.reserve( NSAM );
  for( unsigned s=0; s<NSAM; ++s ){
    _vCONSAM.push_back( std::vector<double>( _nc ) );
    for( unsigned i=0; i<_nc; i++ )
      _vCONSAM.back()[i] = _vCONLB[i] + ( _vCONUB[i] - _vCONLB[i] ) * gen();
  }

  // Observation samples
  switch( options.CRITERION ){
    case FFDOEBase::BROPT:
      return _sample_out( NSAM, os );
      
    case FFDOEBase::AOPT:
    case FFDOEBase::DOPT:
    case FFDOEBase::EOPT:
    default:
      return _sample_fim( NSAM, os );
  }
}

#ifdef MC__USE_THREAD
inline
bool
MBDOESLV::_sample_out
( unsigned const NSAM, std::ostream& os )
{
  // Compute responses at every control samples and uncertainty scenarios
  _vOUTSAM.clear();
  _vOUTSAM.resize( _vPARVAL.size() );
  _dOUT.resize( _vPARVAL.size() );

  for( unsigned s=0; s<NSAM; ++s ){
    if( !_append_out( _vCONSAM[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
        return false;
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  if( !options.UNCREDUC ) return true;
  if( options.DISPLEVEL )
    os << "** REDUCING UNCERTAINTY SCENARIO PAIRS" << std::endl;

  FFDOEBase BRCrit;
  FFDOEBase::set_noise( _vOUTVAR );
  std::vector<double> E0( NSAM, 1./(double)NSAM );

  _sPARSEL.clear();
  for( unsigned j=0; j<_vOUTSAM.size(); ++j ){
    std::multimap<double,unsigned> BRrank;
    for( unsigned k=0; k<_vOUTSAM.size(); ++k ){
      if( j == k ) continue;
      double const BRval = BRCrit.atom_BR( _vOUTSAM.at(j), _vOUTSAM.at(k), E0 );
      if( BRrank.size() < options.UNCREDUC || BRval >= BRrank.begin()->first )
        BRrank.insert( { BRval, k } );
      if( BRrank.size() > options.UNCREDUC )
        BRrank.erase( BRrank.cbegin() );
    }
    for( auto const& [val,k] : BRrank )
      _sPARSEL.insert( j<k? std::make_pair(j,k): std::make_pair(k,j) );
  }

#ifdef CANON__MBDOE_SAMPLE_DEBUG
  std::cout << "PARSEL[" << _sPARSEL.size() << "]: ";
  for( auto const& [j,k] : _sPARSEL )
    std::cout << " (" << j << "," << k << ")";
  std::cout << std::endl;
#endif

  return true;
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

  for( unsigned k=0; k<Output.size(); ++k ){
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
( unsigned const NSAM, std::ostream& os )
{
  // Compute FIMs at every control samples and uncertainty scenarios
  _vFIMSAM.clear();
  _vFIMSAM.resize( _vPARVAL.size() );
  _dFIM.resize( _vPARVAL.size() );
  
  for( unsigned s=0; s<NSAM; ++s ){
    if( !_append_fim( _vCONSAM[s], _vPARVAL, _dFIM, _vFIMSAM, os ) )
      return false;
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

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

  for( unsigned k=0; k<FIM.size(); ++k ){
    arma::mat mFIM( _np, _np, arma::fill::none );
    for( unsigned i=0, l=0; i<_np; ++i )
      for( unsigned j=i; j<_np; ++j, ++l )
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

#else
inline
bool
MBDOESLV::_sample_out
( unsigned const NSAM, std::ostream& os )
{
  // Compute responses at every control samples and uncertainty scenarios
  _vOUTSAM.clear();
  _vOUTSAM.reserve( _vPARVAL.size() );
  _dOUT.resize( _vPARVAL.size() );
  for( unsigned k=0; k<_vPARVAL.size(); ++k ){
    _vOUTSAM.push_back( std::vector< arma::vec >() );
    for( unsigned s=0; s<NSAM; ++s ){
      if( !_append_out( _vCONSAM[s], _vPARVAL[k], _dOUT[k], _vOUTSAM[k], os ) )
        return false;
#ifdef CANON__MBDOE_SAMPLE_DEBUG
      std::cout << "OUT[" << k << "][" << _vOUTSAM[k].size() << "]:" << std::endl << arma::trans(_vOUTSAM[k].back());
#endif
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  if( !options.UNCREDUC ) return true;
  FFDOEBase BRCrit;
  FFDOEBase::set_noise( _vOUTVAR );
  std::vector<double> E0( NSAM, 1./(double)NSAM );

  _sPARSEL.clear();
  for( unsigned j=0; j<_vOUTSAM.size(); ++j ){
    std::multimap<double,unsigned> BRrank;
    for( unsigned k=0; k<_vOUTSAM.size(); ++k ){
      if( j == k ) continue;
      double const BRval = BRCrit.atom_BR( _vOUTSAM.at(j), _vOUTSAM.at(k), E0 );
      if( BRrank.size() < options.UNCREDUC || BRval >= BRrank.begin()->first )
        BRrank.insert( { BRval, k } );
      if( BRrank.size() > options.UNCREDUC )
        BRrank.erase( BRrank.cbegin() );
    }
    for( auto const& [val,k] : BRrank )
      _sPARSEL.insert( j<k? std::make_pair(j,k): std::make_pair(k,j) );
  }

#ifdef CANON__MBDOE_SAMPLE_DEBUG
  std::cout << "PARSEL[" << _sPARSEL.size() << "]: ";
  for( auto const& [j,k] : _sPARSEL )
    std::cout << " (" << j << "," << k << ")";
  std::cout << std::endl;
#endif

  return true;
}

inline
bool
MBDOESLV::_append_out
( std::vector<double> const& Control, std::vector<double> const& Parameter,
  std::vector<double>& Output, std::vector<arma::vec>& Response,
  std::ostream& os )
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  try{
    _dag->eval( _sgOUT, _wkOUT, _vOUT, Output, _vCON, Control, _vPAR, Parameter );
  }
  catch(...){
    return false;
  }

  arma::vec OUT( Output );
  if( Response.size() && arma::size( Response.back() ) != arma::size( OUT ) )
    throw Exceptions( Exceptions::BADSIZE );
  Response.push_back( OUT );
  
  return true;
}

inline
bool
MBDOESLV::_sample_fim
( unsigned const NSAM, std::ostream& os )
{
  // Compute FIMs at every control samples and uncertainty scenarios
  _vFIMSAM.clear();
  _vFIMSAM.reserve( _vPARVAL.size() );
  for( unsigned k=0; k<_vPARVAL.size(); ++k ){
    _vFIMSAM.push_back( std::vector< arma::mat >() );
    for( unsigned s=0; s<NSAM; ++s ){
      if( !_append_fim( _vCONSAM[s], _vPARVAL[k], _dFIM[k], _vFIMSAM[k], os ) )
        return false;
#ifdef CANON__MBDOE_SAMPLE_DEBUG
      std::cout << "FIM[" << k << "][" << _vFIMSAM[k].size() << "]:" << std::endl << _vFIMSAM[k].back();
#endif
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  return true;
}

inline
bool
MBDOESLV::_append_fim
( std::vector<double> const& Control, std::vector<double> const& Parameter,
  std::vector<double>& FIM, std::vector<arma::mat>& Response,
  std::ostream& os )
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  try{
    _dag->eval( _sgFIM, _wkFIM, _vFIM, FIM, _vCON, Control, _vPAR, Parameter );
  }
  catch(...){
    return false;
  }

  arma::mat mFIM( _np, _np, arma::fill::none );
  for( unsigned i=0, l=0; i<_np; ++i )
    for( unsigned j=i; j<_np; ++j, ++l )
      if( i == j ) mFIM(i,i) = FIM[l]; 
      else         mFIM(i,j) = mFIM(j,i) = FIM[l];
  if( Response.size() && arma::size( Response.back() ) != arma::size( mFIM ) )
    throw Exceptions( Exceptions::BADSIZE );
  Response.push_back( mFIM );

  return true;
}
#endif

inline
unsigned
MBDOESLV::_redundant_support
( std::vector<double> const& suppref )
{
  unsigned pos=0;
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
( std::map<unsigned,double> const EOpt, std::map<unsigned,std::vector<double>> const SOpt,
  std::ostream& os )
{
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
    unsigned pos = _redundant_support( supp );
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

#ifdef MC__USE_THREAD
  // Add samples for refined controls
  for( unsigned s=posSupp; s<posSupp+newSupp; ++s ){

    switch( options.CRITERION ){
      case FFDOEBase::BROPT:
        if( !_append_out( _vCONSAM[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
          return false;
        break;
      
      case FFDOEBase::AOPT:
      case FFDOEBase::DOPT:
      case FFDOEBase::EOPT:
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

#else
  // Add samples for refined controls
  for( unsigned k=0; k<_vPARVAL.size(); ++k ){
    for( unsigned s=posSupp; s<posSupp+newSupp; ++s ){

      switch( options.CRITERION ){
        case FFDOEBase::BROPT:
          if( !_append_out( _vCONSAM[s], _vPARVAL[k], _dOUT[k], _vOUTSAM[k], os ) )
            return false;
#ifdef CANON__MBDOE_SAMPLE_DEBUG
          std::cout << "OUT[" << k << "][" << _vOUTSAM[k].size() << "]:" << std::endl << arma::trans(_vOUTSAM[k].back());
#endif
          break;
      
        case FFDOEBase::AOPT:
        case FFDOEBase::DOPT:
        case FFDOEBase::EOPT:
        default:
          if( !_append_fim( _vCONSAM[s], _vPARVAL[k], _dFIM[k], _vFIMSAM[k], os ) )
            return false;
#ifdef CANON__MBDOE_SAMPLE_DEBUG
          std::cout << "FIM[" << k << "][" << _vFIMSAM[k].size() << "]:" << std::endl << _vFIMSAM[k].back();
#endif
          break;
      }
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;
#endif

  return true;
}

inline
bool
MBDOESLV::file_export
( std::string const& name )
{
  auto itPARVAL = _vPARVAL.cbegin();
  for( unsigned s=0; s<_vPARVAL.size(); ++s, ++itPARVAL ){
    std::ofstream ofile( name + "_" + std::to_string(s) + ".log" );
    if( !ofile ) return false;
    
    ofile << std::scientific << std::setprecision(6);
    for( unsigned k=0; k<_vCONSAM.size(); ++k ){
      for( unsigned i=0; i<itPARVAL->size(); ++i )
        ofile << (*itPARVAL)[i] << "  ";

      for( unsigned i=0; i<_vCONSAM[k].size(); ++i )
        ofile << _vCONSAM[k][i] << "  ";

      ofile << ( _EOpt.count(k)? _EOpt[k]: 0 ) << "  ";

      switch( options.CRITERION ){
        case FFDOEBase::BROPT:
          for( unsigned i=0; i<_vOUTSAM[s][k].n_rows; ++i )
            ofile << _vOUTSAM[s][k](i) << "  ";
          break;
          
        case FFDOEBase::AOPT:
        case FFDOEBase::DOPT:
        case FFDOEBase::EOPT:
        default:
          for( unsigned i=0; i<_vFIMSAM[s][k].n_rows; ++i )
            for( unsigned j=i; j<_vFIMSAM[s][k].n_cols; ++j )
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
( unsigned const NEXP, std::ostream& os )
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

    //_evaluate_design_fim( campaign(), "Effort-based", os );
    //break;

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
MBDOESLV::effort_solve
( unsigned const NEXP, std::map<unsigned,double> const& EIni, std::ostream& os )
{
  // Observation samples
  switch( options.CRITERION ){
    case FFDOEBase::BROPT:
      return _effort_minimize_br( NEXP, EIni, os );
      
    case FFDOEBase::AOPT:
    case FFDOEBase::DOPT:
    case FFDOEBase::EOPT:
    default:
      return _effort_maximize_fim( NEXP, EIni, os );
  }
}

inline
void
MBDOESLV::_effort_minimize_br
( unsigned const NEXP, std::map<unsigned,double> const& EIni, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAG;
  FFBREff  OpDOECrit;
  FFLin<I> Sum;

  unsigned const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF( NSUPP );
  for( auto& Ek : EFF )
    Ek.set( _dagdoe );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, (double)NEXP/(double)NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [isupp,eff] : EIni )
      E0[isupp] = eff;
  }
  
  // Convex MINLP optimization
  MINLP doe;
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _vPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;
  doe.options   = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( NSUPP, EFF.data(), 0e0, NEXP, 1 ); 
  doe.set_obj( BASE_OPT::MIN, OpDOECrit( NSUPP, EFF.data(), &_vOUTSAM ) );
  doe.add_ctr( BASE_OPT::EQ, Sum( NSUPP, EFF.data() ) - (int)NEXP );  

  doe.setup();
  //doe.optimize( E0.data() );
  //doe.optimize( E0.data(), nullptr, nullptr, effort_apportion );
  doe.optimize( E0.data(), nullptr, nullptr, effort_rounding );

  if( options.DISPLEVEL > 1 )
    doe.stats.display();

  _EOpt.clear();
  _SOpt.clear();
  _VOpt = BASE_OPT::BASE_OPT::INF;
  if( doe.get_status() == MINLP::SUCCESSFUL ){
    unsigned isupp = 0;
    for( auto const& Ek : doe.get_incumbent().x ){
      if( Ek > 1e-3 ){
        _EOpt[isupp] = Ek;
        _SOpt[isupp] = _vCONSAM[isupp];
      }
      if( ++isupp >= NSUPP )
        break;
    }
    _VOpt = doe.get_incumbent().f[0];
  }

  if( options.DISPLEVEL )
    _display_design( "EFFORT-BASED EXACT DESIGN", _VOpt, _EOpt, _SOpt, os ); 
}

inline
void
MBDOESLV::_effort_maximize_fim
( unsigned const NEXP, std::map<unsigned,double> const& EIni, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAG;
  FFDOEEff OpDOECrit;
  FFLin<I> Sum;

  unsigned const NUNC  = _vPARVAL.size();
  unsigned const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF( NSUPP );
  for( auto& Ek : EFF )
    Ek.set( _dagdoe );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, (double)NEXP/(double)NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [isupp,eff] : EIni )
      E0[isupp] = eff;
  }
  
  // Convex MINLP optimization
  MINLP doe;
  FFDOEBase::type = options.CRITERION;
  FFDOEBase::set_scaling( _vPARSCA );
  doe.options   = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( NSUPP, EFF.data(), 0e0, NEXP, 1 );
 
  switch( options.RISK){
    case Options::NEUTRAL:
    {
      doe.set_obj( BASE_OPT::MAX, Sum( NUNC, OpDOECrit( NSUPP, EFF.data(), &_vFIMSAM ), _vPARWEI.data() ) );
      doe.add_ctr( BASE_OPT::EQ, Sum( NSUPP, EFF.data() ) - (int)NEXP );
      break;
    }
    case Options::AVERSE:
    {
      std::vector<FFVar> DELTA( NUNC );
      FFVar VaR( _dagdoe );
      for( auto& Dk : DELTA )
        Dk.set( _dagdoe );
      E0.resize( NSUPP+NUNC+1, 0e0 );
      doe.add_var( NUNC, DELTA.data(), 0e0 );//, 1e2 );
      doe.add_var( VaR );//, -1e2, 1e2 );
      doe.set_obj( BASE_OPT::MAX, VaR - Sum( NUNC, DELTA.data(), _vPARWEI.data() ) / options.CVARTHRES );
      doe.add_ctr( BASE_OPT::EQ, Sum( NSUPP, EFF.data() ) - (int)NEXP );
      for( unsigned s=0; s<NUNC; s++ )
        doe.add_ctr( BASE_OPT::LE, VaR - DELTA[s] - OpDOECrit( s, NSUPP, EFF.data(), &_vFIMSAM ) );
      break;
    }
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
    unsigned isupp = 0;
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
}

#ifdef MC__USE_THREAD
inline
void
MBDOESLV::_build_br
( std::vector<std::vector<FFVar>>& BROUT, std::vector<FFVar>& CTOT, std::vector<FFVar>& PTOT,
  std::map<unsigned,double> const& EOpt, std::ostream& os )
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  // Copy DAG variables and dependents
  std::vector<FFVar> vPref( _np ), vCref( _nc ), vYref( _ny );
  _dagdoe->insert( _dag, _np, _vPAR.data(), vPref.data() );
  _dagdoe->insert( _dag, _nc, _vCON.data(), vCref.data() );
  _dagdoe->insert( _dag, _ny, _vOUT.data(), vYref.data() );

  // Define outputs in each scenario and each support
  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();
  size_t const NTH  = std::min( NUNC, options.MAXTHREAD>0? options.MAXTHREAD: std::thread::hardware_concurrency() ); // no more subvectors than threads
  BROUT.clear();
  BROUT.resize( NTH );
  for( size_t th=0; th<NTH; ++th )
    BROUT[th].reserve( NUNC/NTH+1 );

  std::set<size_t> POSTH;
  size_t NBATCH = NUNC/NTH, NREM = NUNC-NBATCH*NTH;
  for( size_t i=0, pos=0; i<NTH; ++i ){
    POSTH.insert( pos += ( i<NREM? NBATCH+1: NBATCH ) );
//    std::cout << "pos #" << i << ": " << pos << std::endl;
  }

  FFVar* Pndx = PTOT.data();
  size_t s = 0, th = 0;
  for( auto const& undx : POSTH ){
    for( ; s < undx; ++s, Pndx+=_np ){
      FFVar* Cndx = CTOT.data();
      for( size_t k=0; k<NSUP; ++k, Cndx+=_nc ){
        FFVar* vYcomp = _dagdoe->compose( _ny, vYref.data(), _nc, vCref.data(), Cndx, _np, vPref.data(), Pndx );
        BROUT[th].insert( BROUT[th].end(), vYcomp, vYcomp+_ny );
        delete[] vYcomp;
#ifdef CANON__MBDOE_BUILDBR_DEBUG
        for( size_t i=0; i<_ny; ++i )
          std::cout << "BROUT[" << th << "][" << BROUT[th].size()-_ny+i << "] -> "
                    << BROUT[th][BROUT[th].size()-_ny+i] << std::endl;
#endif
      }
    }
    ++th;
  }
}

#else
inline
void
MBDOESLV::_build_br
( std::vector<FFVar>& BROUT, std::vector<FFVar>& CTOT, std::vector<FFVar>& PTOT,
  std::map<unsigned,double> const& EOpt, std::ostream& os )
{
  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();
  BROUT.clear();
  BROUT.reserve( NUNC*NSUP*_ny );

  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  // Copy DAG variables and dependents
  std::vector<FFVar> vPref( _np ), vCref( _nc ), vYref( _ny );
  _dagdoe->insert( _dag, _np, _vPAR.data(), vPref.data() );
  _dagdoe->insert( _dag, _nc, _vCON.data(), vCref.data() );
  _dagdoe->insert( _dag, _ny, _vOUT.data(), vYref.data() );

  // Define outputs in each scenario and each support
  FFVar* Pndx = PTOT.data();
  for( size_t s=0; s<NUNC; ++s, Pndx+=_np ){
    FFVar* Cndx = CTOT.data();
    for( size_t k=0; k<NSUP; ++k, Cndx+=_nc ){
      FFVar* vYcomp = _dagdoe->compose( _ny, vYref.data(), _nc, vCref.data(), Cndx, _np, vPref.data(), Pndx );
      BROUT.insert( BROUT.end(), vYcomp, vYcomp+_ny );
      delete[] vYcomp;
#ifdef CANON__MBDOE_BUILDBR_DEBUG
      for( size_t i=0; i<_ny; ++i )
        std::cout << "BROUT[" << s << "][" << k << "][" << i << "] -> " << BROUT[BROUT.size()-_ny+i] << std::endl;
#endif
    }
  }
}
#endif

inline
void
MBDOESLV::_build_fim
( std::vector<FFVar>& vFIM, std::vector<FFVar>& CTOT, FFVar const* PREF,
  std::map<unsigned,double> const& EOpt, std::ostream& os )
{
  unsigned const NELE = _np*(_np+1)/2;
  vFIM.assign( NELE, 0. );

  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  // Copy DAG variables and dependents
  std::vector<FFVar> vPref( _np ), vCref( _nc ), vFIMref( NELE );
  _dagdoe->insert( _dag, _np, _vPAR.data(), vPref.data() );
  _dagdoe->insert( _dag, _nc, _vCON.data(), vCref.data() );
  _dagdoe->insert( _dag, NELE, _vFIM.data(), vFIMref.data() );

  // Define atom matrices in current scenario for each support
  FFVar* Cndx = CTOT.data();
  for( auto const& [ndx,eff] : EOpt ){
    FFVar* pFIMndx = _dagdoe->compose( NELE, vFIMref.data(), _nc, vCref.data(), Cndx, _np, vPref.data(), PREF );
    for( unsigned int ij=0; ij<NELE; ++ij )
      vFIM[ij] += eff * pFIMndx[ij];
#ifdef CANON__MBDOE_SOLVE_DEBUG
    _dagdoe->output( _dagdoe->subgraph( NELE, pFIMndx ), " A["+std::to_string(ndx)+"]" );
    //{ int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
    Cndx += _nc;
    delete[] pFIMndx;
  }
#ifdef CANON__MBDOE_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( NELE, vFIM.data() ), " FIM" );
  //{ int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
}

inline
std::multimap<double,std::vector<double>>
MBDOESLV::campaign
()
const
{
  assert( _EOpt.size() == _SOpt.size() );
  std::multimap<double,std::vector<double>> C;
  auto iteff = _EOpt.cbegin();
  auto itsup = _SOpt.cbegin();
  for( ; iteff != _EOpt.cend(); ++iteff, ++itsup )
    C.insert( std::make_pair( iteff->second, itsup->second ) );
  return C;
}

inline
std::pair<double,bool>
MBDOESLV::evaluate_design
( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os )
{
  // Observation samples
  switch( options.CRITERION ){
    case FFDOEBase::BROPT:
      return _evaluate_design_br( Campaign, type, os );

    case FFDOEBase::AOPT:
    case FFDOEBase::DOPT:
    case FFDOEBase::EOPT:
    default:
      return _evaluate_design_fim( Campaign, type, os );
  }
}

inline
std::pair<double,bool>
MBDOESLV::_evaluate_design_br
( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAG;
  FFBRCrit OpBRCrit;

  size_t const NSUP = Campaign.size();
  size_t const NUNC = _vPARVAL.size();

  // Concatenate uncertainty scenatios
  size_t const NPTOT = _np * NUNC;
  std::vector<FFVar> PTOT(NPTOT);  // Parameter scenarios
  for( unsigned i=0; i<NPTOT; i++ )
    PTOT[i].set( _dagdoe );

  std::vector<double> PTOT0;
  PTOT0.reserve(NPTOT);
  for( unsigned s=0; s<NUNC; ++s )
    PTOT0.insert( PTOT0.end(), _vPARVAL[s].cbegin(), _vPARVAL[s].cend() );

  // Concatenate experimental controls
  size_t const NCTOT = _nc * NSUP;
  std::vector<FFVar> CTOT(NCTOT);  // Experimental controls
  for( unsigned i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::vector<double> CTOT0;
  CTOT0.reserve(NCTOT);
  std::map<unsigned,double> EOpt;
  unsigned ieff = 0;
  for( auto const& [eff,supp] : Campaign ){
    EOpt[ieff++] = eff;
    CTOT0.insert( CTOT0.end(), supp.cbegin(), supp.cend() );
  }

  // Evaluate cost function
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::type  = options.CRITERION;
#ifdef MC__USE_THREAD
  std::vector<std::vector<FFVar>> BROUT;
  _build_br( BROUT, CTOT, PTOT, EOpt, os );
  FFVar FBR;
  if( BROUT.size() > 1 ){
    FFVect<I> OpVect;
    Vect vBROUT( _dagdoe, CTOT, PTOT, BROUT );
    FFVar** ppBROUT = OpVect( &vBROUT );
    FBR = OpBRCrit( _ny*NSUP*NUNC, ppBROUT, &EOpt, NUNC, _ny );
  }
  else
    FBR = OpBRCrit( _ny*NSUP*NUNC, BROUT[0].data(), &EOpt, NUNC, _ny );
  //_dagdoe->output( _dagdoe->subgraph( 1, &FBR ) );
#else
  std::vector<FFVar> BROUT;
  _build_br( BROUT, CTOT, PTOT, EOpt, os );
  FFVar& FBR = OpBRCrit( _ny*NSUP*NUNC, BROUT.data(), &EOpt, NUNC, _ny );
#endif

  double DBR = 0./0.;
  std::string header( type.empty()? "DESIGN PERFORMANCE": type + " DESIGN PERFORMANCE" );
  try{
    _dagdoe->eval( 1, &FBR, &DBR, NCTOT, CTOT.data(), CTOT0.data(), NPTOT, PTOT.data(), PTOT0.data() );
  }
  catch(...){
    if( options.DISPLEVEL )
      _display_design( header, DBR, std::multimap<double,std::vector<double>>(), os ); 
    return std::make_pair( DBR, false ); // NaN
  }
/*
  FFGraph dagdoe_copy;
  std::vector<FFVar> CTOT_copy(NCTOT);
  dagdoe_copy.insert( _dagdoe, NCTOT, CTOT.data(), CTOT_copy.data() );
  std::vector<FFVar> PTOT_copy(NPTOT);
  dagdoe_copy.insert( _dagdoe, NPTOT, PTOT.data(), PTOT_copy.data() );

  std::vector<FFVar> vOpVect, vOpVect_copy(_ny*NSUP*NUNC);
//  FFVar** ppBROUT = OpVect( &vBROUT );
//  for( size_t i=0; i<_ny*NSUP*NUNC ; ++i ) vOpVect.push_back( *(ppBROUT[i]) );
//  dagdoe_copy.insert( _dagdoe, 1, vOpVect.data(), vOpVect_copy.data() );
//  dagdoe_copy.insert( _dagdoe, _ny*NSUP*NUNC, vOpVect.data(), vOpVect_copy.data() );
  FFVar FBR_copy;
  dagdoe_copy.insert( _dagdoe, 1, &FBR, &FBR_copy );
  dagdoe_copy.eval( 1, &FBR_copy, &DBR, NCTOT, CTOT_copy.data(), CTOT0.data(), NPTOT, PTOT_copy.data(), PTOT0.data() );
*/
  if( options.DISPLEVEL )
    _display_design( header, DBR, Campaign, os ); 
  return std::make_pair( DBR, true );
}

inline
std::pair<double,bool>
MBDOESLV::_evaluate_design_fim
( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAG;
  FFDOEBase::type = options.CRITERION;
  FFDOECrit OpDOECrit;
  FFLin<I>  Sum;

  size_t const NSUP = Campaign.size();
  size_t const NUNC = _vPARVAL.size();

  // Concatenate uncertainty scenatios
  size_t const NPTOT = _np * NUNC;
  std::vector<FFVar> PTOT(NPTOT);  // Parameter scenarios
  for( unsigned i=0; i<NPTOT; i++ )
    PTOT[i].set( _dagdoe );

  std::vector<double> PTOT0;//(NPTOT); 
  PTOT0.reserve(NPTOT);
  for( unsigned s=0; s<NUNC; ++s )
    PTOT0.insert( PTOT0.end(), _vPARVAL[s].cbegin(), _vPARVAL[s].cend() );

  // Concatenate experimental controls
  size_t const NCTOT = _nc * NSUP;
  std::vector<FFVar> CTOT(NCTOT);  // Experimental controls
  for( unsigned i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::vector<double> CTOT0;//(NCTOT); 
  CTOT0.reserve(NCTOT);
  std::map<unsigned,double> EOpt;
  unsigned ieff = 0;
  for( auto const& [eff,supp] : Campaign ){
    EOpt[ieff++] = eff;
    CTOT0.insert( CTOT0.end(), supp.cbegin(), supp.cend() );
  }

  // Define cost function
  std::vector<FFVar> DOECRIT( NUNC );
  std::vector<FFVar> FIMREF;
  FFVar const* PREF = PTOT.data();
  for( unsigned s=0; s<NUNC; ++s, PREF+=_np ){
    // Define FIM and DOE criterion in current scenario
    _build_fim( FIMREF, CTOT, PREF, EOpt, os );
    DOECRIT[s] = OpDOECrit( FIMREF.size(), FIMREF.data() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
    //_dagdoe->output( _dagdoe->subgraph( FIMREF.size(), FIMREF.data() ), " FIM" );
    _dagdoe->output( _dagdoe->subgraph( 1, &DOECRIT[s] ), " J["+std::to_string(s)+"]" );
    { int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
  }

  // Evaluate cost function
  FFVar FFIM;
  double DFIM;
  std::string header = ( type.empty()? "DESIGN PERFORMANCE": type + " DESIGN PERFORMANCE" );
  try{
    switch( options.RISK){
      case Options::NEUTRAL:
        FFIM = Sum( NUNC, DOECRIT.data(), _vPARWEI.data() );
        _dagdoe->eval( 1, &FFIM, &DFIM, NCTOT, CTOT.data(), CTOT0.data(), NPTOT, PTOT.data(), PTOT0.data() );
        break;

      case Options::AVERSE:
      {
        std::vector<double> DA( NUNC );
        _dagdoe->eval( NUNC, DOECRIT.data(), DA.data(), NCTOT, CTOT.data(), CTOT0.data(), NPTOT, PTOT.data(), PTOT0.data() );
        std::map<double,double> SA;
        for( unsigned s=0; s<NUNC; ++s )
          SA[DA[s]] = _vPARWEI[s];
        double prsum = 0., VaR = 0.;
        for( auto const& [crit,pr] : SA ){
          VaR = crit;
          if( prsum + pr > options.CVARTHRES ) break;
          prsum += pr;
        }
        //std::cout << "VaR = " << VaR << std::endl;
        DFIM = VaR;
        for( auto const& [crit,pr] : SA ){
          if( crit > VaR ) break;
          DFIM -= ( VaR - crit ) * pr / options.CVARTHRES;
        }
        //std::cout << "CVaR = " << DFIM << std::endl;
        break;
      }
    }
  }
  catch(...){
    if( options.DISPLEVEL )
      _display_design( header, DFIM, std::multimap<double,std::vector<double>>(), os ); 
    return std::make_pair( 0./0., false ); // NaN
  }

  if( options.DISPLEVEL )
    _display_design( header, DFIM, Campaign, os ); 
  return std::make_pair( DFIM, true );
}

inline
void
MBDOESLV::gradient_solve
( std::map<unsigned,double> const& EOpt, bool const update, std::ostream& os )
{
  // Observation samples
  switch( options.CRITERION ){
    case FFDOEBase::BROPT:
      return _gradient_minimize_br( EOpt, update, os );
      
    case FFDOEBase::AOPT:
    case FFDOEBase::DOPT:
    case FFDOEBase::EOPT:
    default:
      return _gradient_maximize_fim( EOpt, update, os );
  }
}

inline
void
MBDOESLV::_gradient_minimize_br
( std::map<unsigned,double> const& EOpt, bool const update, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAG;
  FFBRCrit OpBRCrit;

  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();

  // Concatenate uncertainty scenatios
  size_t const NPTOT = _np * NUNC;
  std::vector<FFVar> PTOT(NPTOT);  // Parameter scenarios
  for( unsigned i=0; i<NPTOT; i++ )
    PTOT[i].set( _dagdoe );

  std::vector<double> PTOT0;
  PTOT0.reserve(NPTOT);
  for( unsigned s=0; s<NUNC; ++s )
    PTOT0.insert( PTOT0.end(), _vPARVAL[s].cbegin(), _vPARVAL[s].cend() );

  // Concatenate experimental controls
  size_t const NCTOT = _nc * NSUP;
  std::vector<FFVar> CTOT(NCTOT);  // Experimental controls
  for( unsigned i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::vector<double> CTOT0, CTOTLB, CTOTUB;
  CTOT0.reserve(NCTOT);
  CTOTLB.reserve(NCTOT);
  CTOTUB.reserve(NCTOT);
  for( auto const& [ndx,eff] : EOpt ){
    CTOT0.insert( CTOT0.end(), _vCONSAM[ndx].cbegin(), _vCONSAM[ndx].cend() );
    CTOTLB.insert( CTOTLB.end(), _vCONLB.cbegin(), _vCONLB.cend() );
    CTOTUB.insert( CTOTUB.end(), _vCONUB.cbegin(), _vCONUB.cend() );
  }

  // Local NLP optimization
  NLP doeref;
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _vPARSCA );
  FFDOEBase::type  = options.CRITERION;
  doeref.options = options.NLPSLV;
  doeref.set_dag( _dagdoe ); // DAG
  doeref.add_par( NPTOT, PTOT.data() ); // parameters
  doeref.add_var( NCTOT, CTOT.data(), CTOTLB.data(), CTOTUB.data() ); // decision variables

#ifdef MC__USE_THREAD
  std::vector<std::vector<FFVar>> BROUT;
  _build_br( BROUT, CTOT, PTOT, EOpt, os );
  FFVar FBR;
  if( BROUT.size() > 1 ){
    FFVect<I> OpVect;
    Vect vBROUT( _dagdoe, CTOT, PTOT, BROUT );
    FBR = OpBRCrit( _ny*NSUP*NUNC, OpVect( &vBROUT ), const_cast<std::map<unsigned,double>*>(&EOpt), NUNC, _ny );
  }
  else{
//    std::cout << _ny*NSUP*NUNC << "=?" << BROUT[0].size() << std::endl;
    FBR = OpBRCrit( _ny*NSUP*NUNC, BROUT[0].data(), const_cast<std::map<unsigned,double>*>(&EOpt), NUNC, _ny );
  }
#else
  std::vector<FFVar> BROUT;
  _build_br( BROUT, CTOT, PTOT, EOpt, os );
  FFVar& FBR = OpBRCrit( BROUT.size(), BROUT.data(), const_cast<std::map<unsigned,double>*>(&EOpt), NUNC, _ny );
#endif
  doeref.set_obj( BASE_OPT::MIN, FBR ); // minimize Bayesian risk

//  double DFBR;
//  _dagdoe->eval( 1, &FBR, &DFBR, NCTOT, CTOT.data(), CTOT0.data(), NPTOT, PTOT.data(), PTOT0.data() );
//  std::cout << "BROPT = " << DFBR << std::endl;
//  { int dum; std::cout << "Paused"; std::cin >> dum; }

  doeref.setup();
  doeref.solve( CTOT0.data(), nullptr, nullptr, PTOT0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << doeref.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << doeref.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( update ){
    _SOpt.clear();
    _VOpt = 0./0.;//BASE_OPT::BASE_OPT::INF;
 
    if( doeref.get_status() == NLP::SUCCESSFUL || doeref.get_status() == NLP::FAILURE ){
      unsigned isupp = 0;
      for( auto const& [ndx,eff] : EOpt ){
        double const* dC = doeref.solution().x.data() + isupp*_nc;
        _SOpt[ndx] = std::vector<double>( dC, dC+_nc );
        ++isupp;
      }
      _update_supports( _EOpt, _SOpt, os );
      _VOpt = doeref.solution().f[0];
    }
  }

  if( options.DISPLEVEL )
    _display_design( "GRADIENT-BASED REFINED DESIGN", _VOpt, EOpt, _SOpt, os ); 
}

inline
void
MBDOESLV::_gradient_maximize_fim
( std::map<unsigned,double> const& EOpt, bool const update, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAG;
  FFDOECrit OpDOECrit;
  FFLin<I>  Sum;

  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();

  // Concatenate uncertainty scenarios
  size_t const NPTOT = _np * NUNC;
  std::vector<FFVar> PTOT(NPTOT);  // Parameter scenarios
  for( unsigned i=0; i<NPTOT; i++ )
    PTOT[i].set( _dagdoe );

  std::vector<double> PTOT0;
  PTOT0.reserve(NPTOT);
  for( unsigned s=0; s<NUNC; ++s ){
    PTOT0.insert( PTOT0.end(), _vPARVAL[s].cbegin(), _vPARVAL[s].cend() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
    std::cout << "P[" << s << "] = ";
    for( unsigned i=0; i<_np; i++ )
      std::cout << _vPARVAL[s][i] << "  ";
    std::cout << std::endl;
#endif
  }
  
  // Concatenate experimental controls
  size_t const NCTOT = _nc * NSUP;
  std::vector<FFVar> CTOT(NCTOT);  // Experimental controls
  for( unsigned i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::vector<double> CTOT0, CTOTLB, CTOTUB;
  CTOT0.reserve(NCTOT);
  CTOTLB.reserve(NCTOT);
  CTOTUB.reserve(NCTOT);
  for( auto const& [ndx,eff] : EOpt ){
    CTOT0.insert( CTOT0.end(), _vCONSAM[ndx].cbegin(), _vCONSAM[ndx].cend() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
    std::cout << "C[" << ndx << "] = ";
    for( unsigned i=0; i<_nc; i++ )
      std::cout << _vCONSAM[ndx][i] << "  ";
    std::cout << std::endl;
#endif
    CTOTLB.insert( CTOTLB.end(), _vCONLB.cbegin(), _vCONLB.cend() );
    CTOTUB.insert( CTOTUB.end(), _vCONUB.cbegin(), _vCONUB.cend() );
  }

  // Define cost function
  std::vector<FFVar> DOECRIT( NUNC );
  std::vector<FFVar> FIMREF;
  FFVar const* PREF = PTOT.data();
  for( unsigned s=0; s<NUNC; ++s, PREF+=_np ){
    // Define FIM and DOE criterion in current scenario
    _build_fim( FIMREF, CTOT, PREF, EOpt, os );
    DOECRIT[s] = OpDOECrit( FIMREF.size(), FIMREF.data() );
#ifdef CANON__MBDOE_SOLVE_DEBUG
    _dagdoe->output( _dagdoe->subgraph( 1, &DOECRIT[s] ), " J["+std::to_string(s)+"]" );
    { int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
  }

  // Local NLP optimization
  NLP doeref;
  FFDOEBase::type  = options.CRITERION;
  FFDOEBase::set_scaling( _vPARSCA );
  doeref.options = options.NLPSLV;
  doeref.set_dag( _dagdoe ); // DAG
  doeref.add_par( NPTOT, PTOT.data() ); // parameters
  doeref.add_var( NCTOT, CTOT.data(), CTOTLB.data(), CTOTUB.data() ); // decision variables
 
  switch( options.RISK){
    case Options::NEUTRAL:
    {
      //FFVar FFIM = Sum( NUNC, DOECRIT.data(), _vPARWEI.data() );
      //double DFIM;
      //_dagdoe->eval( 1, &FFIM, &DFIM, NCTOT, CTOT.data(), CTOT0.data(), NPTOT, PTOT.data(), PTOT0.data() );
      //std::cout << "DOPT = " << DFIM << std::endl;
      //{ int dum; std::cout << "Paused"; std::cin >> dum; }
      doeref.set_obj( BASE_OPT::MAX, Sum( NUNC, DOECRIT.data(), _vPARWEI.data() ) ); // objective
      break;
    }
    case Options::AVERSE:
    {
      std::vector<FFVar> DELTA( NUNC );
      FFVar VaR( _dagdoe );
      for( auto& Dk : DELTA )
        Dk.set( _dagdoe );
      if( _ROpt.size() == NUNC+1 )
        for( auto const& r0 : _ROpt ) CTOT0.push_back( r0 ); 
      else
        CTOT0.resize( NCTOT+NUNC+1, 0e0 );
      doeref.add_var( NUNC, DELTA.data(), 0e0 );//, 1e2 );
      doeref.add_var( VaR );//, -1e2, 1e2 );
      doeref.set_obj( BASE_OPT::MAX, VaR - Sum( NUNC, DELTA.data(), _vPARWEI.data() ) / options.CVARTHRES );
      for( unsigned s=0; s<NUNC; s++ )
        doeref.add_ctr( BASE_OPT::LE, VaR - DELTA[s] - DOECRIT[s] );
      break;
    }
  }

  doeref.setup();
  doeref.solve( CTOT0.data(), nullptr, nullptr, PTOT0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << doeref.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << doeref.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( update ){
    _SOpt.clear();
    _VOpt = 0./0.;
 
    if( doeref.get_status() == NLP::SUCCESSFUL || doeref.get_status() == NLP::FAILURE ){
      double const* dC = doeref.solution().x.data();
      for( auto const& [ndx,eff] : EOpt ){
        _SOpt[ndx] = std::vector<double>( dC, dC+_nc );
        dC += _nc;
      }
      _update_supports( _EOpt, _SOpt, os );
      _VOpt = doeref.solution().f[0];
      if( doeref.solution().x.size() == NCTOT+NUNC+1 )
        _ROpt.assign( dC, dC+NUNC+1 );
    }
  }

  if( options.DISPLEVEL )
    _display_design( "GRADIENT-BASED REFINED DESIGN", _VOpt, EOpt, _SOpt, os ); 
}

inline
void
MBDOESLV::_display_design
( std::string const& title, double const& crit, std::map<unsigned,double> const& eff,
  std::map<unsigned,std::vector<double>> const& supp, std::ostream& os )
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
( std::string const& title, double const& crit, std::multimap<double,std::vector<double>> const& campaign,
  std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( campaign.empty() ){
     os << " FAILED" << std::endl;
     return;
  } 
   
  os  << std::scientific << std::setprecision(5) << crit << std::endl;
  unsigned i=0;
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

// Copyright (C) 2020 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MINLPBND Bounding of Factorable Mixed-Integer Nonlinear Programs using MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 1.0
\date 2020
\bug No known bugs.

Consider a nonlinear optimization problem in the form:
\f{align*}
\mathcal{P}:\quad & \min_{x_1,\ldots,x_n}\ f(x_1,\ldots,x_n)\\
& {\rm s.t.}\ \ g_j(x_1,\ldots,x_n)\ \leq,=,\geq\ 0,\ \ j=1,\ldots,m\\
& \qquad x_i^L\leq x_i\leq x_i^U,\ \ i=1,\ldots,n\,,
\f}
where \f$f, g_1, \ldots, g_m\f$ are factorable, possibly nonlinear, real-valued functions; and \f$x_1, \ldots, x_n\f$ can be either continuous or integer decision variables. The class mc::MINLPBND computes rigorous bounds on the global solution of such (MI)NLP problems using various set arithmetics, as available in <A href="https://projects.coin-or.org/MCpp">MC++</A>.

\section sec_MINLPBND_bound How to Compute a Rigorous Bound on the Global Solution Value of my Optimization Model?

Consider the following MINLP model:
\f{align*}
  \min_{x,y}\ & -6x-y \\
  \text{s.t.} \ & 0.3(x-8)^2+0.04(y-6)^4+0.1\frac{{\rm e}^{2x}}{y^4} \leq 56 \\
                & \frac{1}{x}+\frac{1}{y}-\sqrt{x}\sqrt{y}+4 \leq 0 \\
                & 2x-5y+1 \leq 0 \\
  & 1 \leq x \leq 20\\
  & 1 \leq y \leq 20,\ y\in\mathbb{Z}
\f}

We start by instantiating an mc::MINLPBND class object, which is defined in the header file <tt>minlpbnd.hpp</tt>:

\code
  mc::MINLPBND MINLP;
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
  MINLP.setup();
  switch( MINLP.relax() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl
                <<"MINLP relaxation bound: " << MINLP.solver()->get_objective() << std::endl;
      for( unsigned i=0; i<NP; i++ ) 
        std::cout << "  " << P[i] << " = " << MINLP.solver()->get_variable( P[i] ) << std::endl;
      MINLP.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }
\endcode

The return value of mc::MINLPBND is per the enumeration mc::MIPSLV_GUROBI<I>::STATUS. The following result is displayed (with the option mc::MINLPBND::Options::MIPSLV::DISPLEVEL defaulting to 1):

\verbatim
#              |  VARIABLES      FUNCTIONS
# -------------+---------------------------
#  LINEAR      |         0              2
#  QUADRATIC   |         0              0
#  POLYNOMIAL  |         0              0
#  GENERAL     |         2              2

MINLP relaxation bound: -63.9709
  X0 = 8.82848
  X1 = 11

#  WALL-CLOCK TIMES
#  CTR PROPAG:       0.00 SEC
#  POL IMAGE:        0.00 SEC
#  MIP SETUP:        0.00 SEC
#  MIP SOLVE:        0.00 SEC, 1 PROBLEMS
\endverbatim

Other options can be modified to tailor the relaxations, tune the MIP solver, set a maximum CPU time, etc. These options can be modified through the public member mc::MINLPBND::options.
*/

//TODO: 
//- Enable RLT cuts in addition to quadratization
//- SEPARATE VARIOUS RELAXATION CLASSES AND INHERIT IN MINLPBND?

#ifndef MC__MINLPBND_HPP
#define MC__MINLPBND_HPP

#include <stdexcept>
#include <chrono>

#include "rltred.hpp"
#include "sparseexpr.hpp"
#include "polimage.hpp"
#include "squad.hpp"
#include "scmodel.hpp"
#include "ismodel.hpp"

#include "base_nlp.hpp"
#include "mipslv_gurobi.hpp"
#include "gamswriter.hpp"

//#undef MC__MINLPBND_DEBUG
//#define MC__MINLPBND_DEBUG_LIFT
//#define MC__MINLPBND_SHOW_REDUC

namespace mc
{

//! @brief C++ base class for global bounding of factorable MINLP using MC++
////////////////////////////////////////////////////////////////////////
//! mc::MINLPBND is a C++ class for global bounding of factorable MINLP
//! using MC++
////////////////////////////////////////////////////////////////////////
template < typename T,
           typename MIP=MIPSLV_GUROBI<T> >
class MINLPBND:
  public virtual BASE_NLP
{
  // Typedef's
  typedef std::map< SPolyMon, double, lt_SPolyMon > t_coefmon;
  typedef std::set< FFVar*, lt_FFVar > set_FFVar;
  typedef std::pair< SPolyMon const*, SPolyMon const* > SPolyProdMon;

protected:

  //! @brief local copy of DAG (overides BASE_NLP::_dag)
  FFGraph*                  _dag;

  //! @brief sparse expression environment for reformulations
  SparseEnv                 _SEenv;
  //! @brief sparse quadratic form environment
  SQuad                     _SQenv;
  
  //! @brief number of parameters in model
  unsigned                  _nP;
  //! @brief vector of parameters in DAG
  std::vector<FFVar>        _Pvar;

  //! @brief number of decision variables (independent and dependent) in model
  unsigned                  _nX;
  //! @brief number of original decision variables (independent and dependent) in model
  unsigned                  _nX0;
  //! @brief number of decision variables prior to lifting
  unsigned                  _nX1;
  //! @brief vector of decision variables in DAG
  std::vector<FFVar>        _Xvar;
  //! @brief vector of decision variable lower bounds
  std::vector<double>       _Xlow;
  //! @brief vector of decision variable upper bounds
  std::vector<double>       _Xupp;
  //! @brief vector of variable bounds
  std::vector<T>            _Xbnd;
  //! @brief vector of decision variable types
  std::vector<unsigned>     _Xtyp;
  //! @brief subset of variables in linear expressions
  std::set<unsigned>        _Xlin;
  //! @brief subset of variables in quadratic expressions
  std::set<unsigned>        _Xquad;
  //! @brief subset of variables in polynomial expressions
  std::set<unsigned>        _Xpol;
  //! @brief subset of variables in general expressions
  std::set<unsigned>        _Xgal;
  //! @brief vector of [-1,1] scaled variables
  std::vector<FFVar>        _Xscal;
  //! @brief Chebyshev basis map
  std::map< SPolyMon, FFVar, lt_SPolyMon > _Xmon;
  //! @brief objective variable
  FFVar                     _Xobj;
  //! @brief Map of dependent expressions in terms of original variables
  std::map< unsigned, FFVar > _Xlift;

  //! @brief number of functions (objective and constraints) in model
  unsigned                  _nF;
  //! @brief vector of functions in DAG
  std::vector<FFVar>        _Fvar;
  //! @brief vector of function lower bounds
  std::vector<double>       _Flow;
  //! @brief vector of function upper bounds
  std::vector<double>       _Fupp;
  //! @brief vector of function bounds
  std::vector<T>            _Fbnd;
  //! @brief vector of function subgraphs
  std::vector<FFSubgraph>   _Fops;
  //! @brief all function subgraphs
  FFSubgraph                _Fallops;
  //! @brief subset of linear functions
  std::set<unsigned>         _Flin;
  //! @brief subset of quadratic functions
  std::set<unsigned>         _Fquad;
  //! @brief subset of polynomial functions
  std::set<unsigned>         _Fpol;
  //! @brief subset of general functions
  std::set<unsigned>         _Fgal;
  //! @brief subset of equality-constrained functions
  std::set<unsigned>         _Fctreq;

  //! @brief Interval representation of 'unbounded' variables
  T                         _IINF;
  //! @brief Storage vector for constraint propagation in interval arithmetic
  std::vector<T>            _CPbnd;
  //! @brief Storage vector for interval arithmetic
  std::vector<T>            _Iwk;

//  //! @brief pointer to subset of dependent variables showing improvement using implicit contractors
//  std::set<unsigned> _ndxdep;
//  //! @brief pointer to subset of dependent variables showing improvement using implicit interval contractor
//  std::set<unsigned> _Indxdep;
//  //! @brief pointer to subset of dependent variables showing improvement using implicit polynomial model contractor
//  std::set<unsigned> _CMndxdep;
//  //! @brief boolean flag to ignore the dependents in the relaxations
//  bool _ignore_deps;

  //! @brief Polyhedral image environment
  PolImg<T>                 _POLenv;
  //! @brief Polyhedral image decision variables
  std::vector< PolVar<T> >  _POLXvar;
  //! @brief Polyhedral image scaled variables
  std::vector< PolVar<T> >  _POLXscal;
  //! @brief Map of monomials in polyhedral image
  std::map< SPolyMon, PolVar<T>, lt_SPolyMon > _POLXmon;
  //! @brief Map of monomial products in polyhedral image
  std::map< SPolyProdMon, PolVar<T>, lt_SQuad > _POLXprodmon;
  //! @brief Polyhedral image function variables
  std::vector< PolVar<T> >  _POLFvar;
  //! @brief Storage vector for function evaluation in polyhedral relaxation arithmetic
  std::vector< PolVar<T> >  _POLwk;

  //! @brief Chebyshev model environment
  SCModel<T>* _CMenv;
  //! @brief Chebyshev variables
  std::vector< SCVar<T> >   _CMXvar;
  //! @brief Chebyshev constraint variables
  std::vector< SCVar<T> >   _CMFvar;
  //! @brief Storage vector for function evaluation in Chebyshev arithmetic
  std::vector< SCVar<T> >   _CMwk;
  //! @brief Chebyshev basis map
  std::map< SPolyMon, FFVar, lt_SPolyMon > _CMXmon;

//  //! @brief Chebyshev reduced-space [-1,1] scaled model environment
//  SCModel<T>* _CMrenv;
//  //! @brief Chebyshev reduced-space [-1,1] scaled variables
//  std::vector< SCVar<T> > _CMrbas;
//  //! @brief Chebyshev reduced-space variables
//  std::vector< SCVar<T> > _CMrvar;
//  //! @brief Chebyshev reduced-space dependents
//  std::vector< SCVar<T> > _CMrdep;
//  //! @brief Interval reduced-space variables
//  std::vector<T> _Irvar;
//  //! @brief Interval reduced-space dependents
//  std::vector<T> _Irdep;

  //! @brief Interval superposition model environment
  ISModel<T>*               _ISMenv;
  //! @brief Interval superposition variables
  std::vector<ISVar<T>>     _ISMXvar;
  //! @brief Interval superposition constraint variables
  std::vector<ISVar<T>>     _ISMFvar;
  //! @brief Storage vector for function evaluation in Interval superposition arithmetic
  std::vector< ISVar<T> >   _ISMwk;

  //! @brief Worst dependence type in participating expressions
  FFDep::TYPE               _pbclass;
  //! @brief Flag for setup function
  bool                      _issetup;
  //! @brief Flag for MIP problem
  bool                      _ismip;
  //! @brief Direction of optimization (-1: MIN, 0: FEAS; 1: MAX)
  int                       _objsense;
  //! @brief MIP solver
  MIP*                      _MIPSLV;

public:

  //! @brief Constructor
  MINLPBND
    ()
    : _dag(0), _nX(0), _nX0(0), _nX1(0), _nF(0), _CMenv(0), _ISMenv(0), _issetup(false)
    { _MIPSLV = new MIP; }

  //! @brief Destructor
  virtual ~MINLPBND
    ()
    {
      delete _MIPSLV;
      delete _ISMenv;
      delete _CMenv;
      delete _dag;
    }

  //! @brief MINLPBND options
  struct Options
  {
    //! @brief Constructor
    Options():
      REFORMMETH({NPOL,QUAD}), RELAXMETH({DRL}), SUBSETDRL(0), SUBSETSCQ(0), BCHPRIM(0),
      OBBTMIG(1e-6), OBBTMAX(5), OBBTTHRES(5e-2), OBBTBKOFF(1e-7), OBBTLIN(2), OBBTCONT(true),
      CPMAX(10), CPTHRES(0.), ISMDIV(10), ISMMIPREL(true),
      CMODEL(), CMODPROP(2), CMODCUTS(0), CMODDMAX(BASE_OPT::INF), MONSCALE(false),
      RRLTCUTS(false), PSDQUADCUTS(0), DCQUADCUTS(false), NCOCUTS(false), NCOADIFF(ASA),
      LINCTRSEP(false), TIMELIMIT(6e2), DISPLEVEL(2),
      POLIMG(), MIPSLV(), SPARSEEXPR(), SQUAD(), RLTRED()
      { CMODEL.MIXED_IA        = true;
        CMODEL.MIN_FACTOR      = 1e-13; // compatibility with GUROBI
        POLIMG.BREAKPOINT_TYPE = PolImg<T>::Options::BIN;
        MIPSLV.DISPLEVEL       = 0;
        MIPSLV.DUALRED         = 0;
        //MIPSLV.PRESOLVE        = 1;
        MIPSLV.TIMELIMIT       = TIMELIMIT;
        SPARSEEXPR.LIFTDIV     = true;
        SPARSEEXPR.LIFTIPOW    = false;
        SQUAD.BASIS            = SQuad::Options::MONOM;
        SQUAD.ORDER            = SQuad::Options::INC;
        RLTRED.METHOD          = RLTRed::Options::ILP;
        RLTRED.LEVEL           = RLTRed::Options::PRIMSIM;
        RLTRED.TIMELIMIT       = TIMELIMIT; }
    //! @brief Assignment operator
    Options& operator= ( Options&options ){
        REFORMMETH    = options.REFORMMETH;
        RELAXMETH     = options.RELAXMETH;
        SUBSETDRL     = options.SUBSETDRL;
        SUBSETSCQ     = options.SUBSETSCQ;
        BCHPRIM       = options.BCHPRIM;
        OBBTMIG       = options.OBBTMIG;
        OBBTMAX       = options.OBBTMAX;
        OBBTTHRES     = options.OBBTTHRES;
        OBBTBKOFF     = options.OBBTBKOFF;
        OBBTLIN       = options.OBBTLIN;
        OBBTCONT      = options.OBBTCONT;
        CPMAX         = options.CPMAX;
        CPTHRES       = options.CPTHRES;
        ISMDIV        = options.ISMDIV;
        ISMMIPREL     = options.ISMMIPREL;
        CMODEL        = options.CMODEL;
        CMODPROP      = options.CMODPROP;
        CMODCUTS      = options.CMODCUTS;
        CMODDMAX      = options.CMODDMAX;
        MONSCALE      = options.MONSCALE;
        RRLTCUTS      = options.RRLTCUTS;
        PSDQUADCUTS   = options.PSDQUADCUTS;
        DCQUADCUTS    = options.DCQUADCUTS;
        NCOCUTS       = options.NCOCUTS;
        NCOADIFF      = options.NCOADIFF;
        LINCTRSEP     = options.LINCTRSEP;
        TIMELIMIT     = options.TIMELIMIT;
        DISPLEVEL     = options.DISPLEVEL;
        POLIMG        = options.POLIMG;
        MIPSLV        = options.MIPSLV;
        SPARSEEXPR    = options.SPARSEEXPR;
        SQUAD         = options.SQUAD;
        RLTRED        = options.RLTRED;
        return *this ;
      }
    //! @brief Relaxation strategy
    enum RELAX{
      DRL=0,  //!< Standard decomposition-relaxation-linearization based on convex relaxations (Tawarmalani & Sahinidis)
      SCDRL,  //!< Decomposition-relaxation-linearization based on sparse Chebyshev relaxations, controlled by parameters CMODPROP and CMODCUT
      SCQ,    //!< Quadratisation of sparse Chebyshev models, controlled by parameters CMODPROP and CMODCUT
      ISM     //!< Interval superposition model relaxations, controlled by parameters ISMDIV
    };
    //! @brief Reformulation strategy
    enum REFORM{
      NPOL=0,  //!< Reformulate non-polynomial expressions as polynomial subexpressions and transcendental terms using mc::SparseExpr
      QUAD     //!< Flatten linear/quadratic/polynomial expressions and lift polynomial into quadratic expressions
    };
    //! @brief Reduced-space strategy
    //enum REDUC{
    //  NOREDUC=0, //!< Do not use Chebyshev-reduction constraints
    //  APPEND	 //!< Append Chebyshev-reduction constraints to the other constraints
    //};
    //! @brief Sensitivity strategy
    enum SENS{
      FSA=0,      //!< Forward sensitivity analysis
      ASA         //!< Adjoint sensitivity analysis
    };
    //! @brief Reformulation methods
    std::set<REFORM> REFORMMETH;   
    //! @brief Relaxation methods
    std::set<RELAX> RELAXMETH;
    //! @brief Exclusion from decomposition-relaxation-linearization: 0: none; 1: non-polynomial functions; 2: polynomial functions
    unsigned SUBSETDRL;
    //! @brief Exclusion from quadratization: 0: none; 1: non-polynomial functions; 2: polynomial functions
    unsigned SUBSETSCQ;
    //! @brief Set higher branch priority to primary variables (e.g. over auxiliary variables in quadratization)
    unsigned BCHPRIM;
    //! @brief Minimum variable range for application of bounds tighteneting
    double OBBTMIG;
    //! @brief Maximum rounds of optimization-based bounds tighteneting
    unsigned OBBTMAX;
    //! @brief Threshold for repeating optimization-based bounds tighteneting (minimum relative reduction in any variable)
    double OBBTTHRES;
    //! @brief Backoff of tightened variable bounds to compensate for numerical errors
    double OBBTBKOFF;
    //! @brief Whether to apply optimization-based bounds tighteneting on: 0: linear constraints only; 1: linear constraints first; 2: joint linear & nonlinear constraints
    unsigned OBBTLIN;
    //! @brief Whether to relax binary/integer variables as continuous during optimization-based bounds tighteneting
    bool OBBTCONT;
    //! @brief Maximum rounds of constraint propagation
    unsigned CPMAX;
    //! @brief Threshold for repeating constraint propagation (minimum relative reduction in any variable)
    double CPTHRES;
    //! @brief Number of partition subdivisions in interval superposition model
    unsigned ISMDIV;
    //! @brief Whether to generate a MIP relaxation of ISM (true) or LP relaxation (false)
    bool ISMMIPREL;
    //! @brief CModel options
    typename SCModel<T>::Options CMODEL;
    //! @brief Chebyhev model propagation order (0: no propag.)
    unsigned CMODPROP;
    //! @brief Chebyhev model cut order (0: same as propag.)
    unsigned CMODCUTS;
    //! @brief Chebyhev model maximum diameter for cut generation
    double CMODDMAX;
    //! @brief Whether to scale monomials in sparse quadratic form
    bool MONSCALE;
    //! @brief Whether to add reduced RLT cuts
    bool RRLTCUTS;
    //! @brief Whether to add PSD cuts within quadratisation (0: none; 1: 2-by-2; >1: 3-by-3)
    unsigned PSDQUADCUTS;
    //! @brief Whether to add DC cuts within quadratisation
    bool DCQUADCUTS;
    //! @brief Whether to add NCO cuts
    bool NCOCUTS;
    //! @brief NCO method
    unsigned NCOADIFF;
    //! @brief Whether to separate linear constraints prior to relaxation
    bool LINCTRSEP;
    //! @brief Maximum run time (seconds)
    double TIMELIMIT;
    //! @brief Display level for solver
    int DISPLEVEL;
    //! @brief PolImg (polyhedral relaxation) options
    typename PolImg<T>::Options POLIMG;
    //! @brief MIPSLV_GUROBI (mixed-integer optimization) options
    typename MIP::Options MIPSLV;
    //! @brief SparseExpr (sparse reformulation/lifting) options
    typename SparseEnv::Options SPARSEEXPR;
    //! @brief SQuad (sparse quadratisation of Chebyshev models) options
    typename SQuad::Options SQUAD;
    //! @brief RLTRed (reduced RLT search) options
    typename RLTRed::Options RLTRED;
    //! @brief Display
    void display
      ( std::ostream&out=std::cout ) const;
  } options;

  //! @brief MINLPBND computational statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime_cprop = walltime_polimg = walltime_setmip = walltime_slvmip =
        std::chrono::microseconds(0); total_slvmip = 0; }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "#  WALL-CLOCK TIMES" << std::endl
           << "#  CTR PROPAG: " << std::setw(10) << to_time( walltime_cprop )  << " SEC" << std::endl
           << "#  POL IMAGE:  " << std::setw(10) << to_time( walltime_polimg ) << " SEC" << std::endl
           << "#  MIP SETUP:  " << std::setw(10) << to_time( walltime_setmip ) << " SEC" << std::endl
           << "#  MIP SOLVE:  " << std::setw(10) << to_time( walltime_slvmip ) << " SEC, "
                                << total_slvmip << " PROBLEMS" << std::endl   << std::endl; }
    //! @brief Cumulated wall-clock time for constraint propagation (in microseconds)
    std::chrono::microseconds walltime_cprop;
    //! @brief Cumulated wall-clock time for polyhedral relaxation construction (in microseconds)
    std::chrono::microseconds walltime_polimg;
    //! @brief Cumulated wall-clock time for setting-up MIP model (in microseconds)
    std::chrono::microseconds walltime_setmip;
    //! @brief Cumulated wall-clock time for solving MIP model (in microseconds)
    std::chrono::microseconds walltime_slvmip;
    //! @brief Total number of MIP model solves
    unsigned total_slvmip;
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

//  //! @brief MINLPBND computational statistics
//  struct Stats{
//    void reset()
//      { tCPROP = tPOLIMG = tMIPSOL = tMIPSET = 0.;
//        nCPROP = nMIPSOL = 0; }
//    void display
//      ( std::ostream&os=std::cout )
//      { os << std::fixed << std::setprecision(2);
//        if( nCPROP  ) os << "#  CPROP:  " << tCPROP << " CPU SEC  (" << nCPROP << ")" << std::endl;
//        os << "#  POLIMG: " << tPOLIMG << " CPU SEC" << std::endl
//           << "#  MIPSET: " << tMIPSET << " CPU SEC" << std::endl;
//        if( nMIPSOL ) os << "#  MIPSOL: " << tMIPSOL << " CPU SEC  (" << nMIPSOL << ")" << std::endl; 
//        os << std::endl; }
//    double tCPROP;
//    unsigned nCPROP;
//    double tPOLIMG;
//    double tMIPSOL;
//    unsigned nMIPSOL;
//    double tMIPSET;
//  } stats;

  //! @brief MINLPBND exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for NLGO exception handling
    enum TYPE{
      MULTOBJ=1,	//!< Optimization problem may not have more than one objective functions
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
      case MULTOBJ:
        return "MINLPBND::Exceptions  Model with multiple objectives not allowed";
      case SETUP:
        return "MINLPBND::Exceptions  Incomplete setup before a solve";
      case INTERN: default:
        return "MINLPBND::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief Setup optimization model before bounding
  void setup
    ( std::ostream& os=std::cout );
    //( unsigned const* Xtyp=nullptr, std::ostream& os=std::cout );

  //! @brief Update variable and function bounds before tightening / relaxation
  bool update_bounds
    ( T const* X=nullptr, double const* Finc=nullptr, bool const resetbnd=true );

//  //! @brief Tighten bounds using constraint propagation
//  int tighten
//    ( T const* X=nullptr, double const* Finc=nullptr )
//    { if( !update( X, Finc ) ) return -1;
//      return _propagate_bounds(); }

  //! @brief Set polyhedral relaxation
  void init_polrelax
    ();

  //! @brief Update polyhedral relaxation cuts
  void update_polrelax
    ( unsigned const addcut=2, bool const contcuts=false, bool const resetcut=true );

  //! @brief Refine polyhedral relaxation by adding breakpoints
  void refine_polrelax
    ( double const* Xinc=nullptr, bool const resetcuts=true );

  //! @brief Setup and solve polyhedral relaxation of optimization model in the variable subdomain <a>X</a>, for the incumbent value <a>Finc</a> at point <a>Xinc</a>, and applying <a>nref</a> breakpoint refinements
  int relax
    ( T const* X=nullptr, double const* Finc=nullptr, double const* Xinc=nullptr,
      unsigned const nref=0, bool const resetbnd=true, bool const reinit=true,
      std::string const gmsfile="" );

  //! @brief Setup and solve bound reduction problems using polyhedral relaxations of optimization model, starting with variable subdomain <a>X</a>, for the incumbent value <a>Finc</a>, and using the options specified in <a>MINLPBND::Options::OBBTMAX</a> and <a>MINLPBND::Options::OBBTTHRES</a> -- returns updated variable bounds <a>X</a>, and number of iterative refinements <a>nred</a>
  int reduce
    ( unsigned& nred, T* X=nullptr, double const* Finc=nullptr,
      bool const resetbnd=true, bool const reinit=true );

  //! @brief Propagate bounds, starting with variable subdomain <a>X</a>, for the incumbent value <a>Finc</a>, and using the options specified in <a>MINLPBND::Options::CPMAX</a> and <a>MINLPBND::Options::CPTHRES</a> -- returns updated variable bounds <a>X</a>
  int propagate
    ( T* X=nullptr, double const* Finc=nullptr, bool const resetbnd=true );

    //! @brief Test whether all variables of a given type are bounded
  bool bounded_domain
    ( double const& maxdiam, FFDep::TYPE const type )
    const;

  //! @brief Get const pointer to MIP solver
  MIP const* solver
    ()
    const
    { return _MIPSLV; }

  //! @brief Get non-const pointer to MIP solver
  MIP * solver
    ()
    { return _MIPSLV; }

  //! @brief Get const pointer to variable bounds
  T const* varbnd
    ()
    const
    { return _Xbnd.data(); }

  //! @brief Get problem class - i.e. worst dependence type in relaxed subproblem
  FFDep::TYPE problem_class 
    ()
    const
    { return _pbclass; }

private:

  //! @brief Time point to enable TIMELIMIT option
  std::chrono::time_point<std::chrono::system_clock> _tstart;

  //! @brief Set linear/nonlinear participating variables in functions
  void _set_variable_class
    ();

  //! @brief Set linear/polynomial/nonlinear participating functions
  void _set_function_class
    ();

  //! @brief Lift nonpolynomial functions
  void _lift_nonpolynomial
    ();

  //! @brief Lift semi-algebraic functions
  void _lift_semialgebraic
    ( bool const add2dag=true );

  //! @brief Search for reduced RLT cuts
  void _search_reduction_constraints
    ();

  //! @brief Tighten bounds using constrafint propagation
  int _propagate_bounds
    ();

  //! @brief Test if bounds are tight
  bool _tight
    ();

 //! @brief Tighten variable bounds in optimization model using polyhedral relaxation
  int _reduce
    ();
  //! @brief Solve bound reduction problem for lower/upper <a>uplo</a> bound on variable <a>ix</a>
  int _reduce
    ( unsigned const ix, bool const uplo );

  //! @brief Set model linear cuts
  void _set_cuts_LIN
   ();
  //! @brief Set model polyhedral McCormick-derived cuts
  void _set_cuts_DRL
    ();
  //! @brief Set model polyhedral Chebyshev-derived cuts
  void _set_cuts_SCDRL
    ();
  //! @brief Set model quadratic Chebyshev-derived cuts
  void _set_cuts_SCQ
    ();
  //! @brief Set model polyhedral superposition-derived cuts
  void _set_cuts_ISM
    ();

  //! @brief Create DAG variable for given quadratic form
  FFVar _var_pol
    ( SQuad::t_SQuad const& quad, std::map< SPolyMon, FFVar, lt_SPolyMon >& mapmon )
    const;

  //! @brief Create DAG variable for given Chebyshev basis function 
  FFVar _var_cheb
    ( FFVar const& x, const unsigned n )
    const;
    
  //! @brief Create DAG variable and auxiliary for given high-order monomial 
  std::pair< FFVar const*, FFVar const* > _var_mon
    ( SPolyMon const& mon )
    const;

  //! @brief Compute bound for given Chebyshev basis function 
  T _bnd_cheb
    ( T const& x, const unsigned n )
    const;

  //! @brief Compute bound of (unscaled) monomial <a>mon</a>
  T _bnd_mon
    ( SPolyMon const& mon )
    const;

  // Set monomial vector from quadratic form in polyhedral image
  void _set_mon_SCQ
    ();

  // Set monomial vector from quadratic form in polyhedral image
  void _set_mon_SQ
    ();

  // Append cuts for quadratic form in polynomial image
  void _set_cuts_SQ
    ( std::set<unsigned> ndxF, bool const chkrem );

  //! @brief Get monomial <a>mon</a> from DAG monomial map <a>_Xmon</a> or add it to the map if absent
  FFVar const& _get_mon
    ( SPolyMon const& mon );

  //! @brief Get range for quadratic terms from <a>mat</a>
  T _get_range
    ( SQuad::t_SQuad const& mat );

  //! @brief Append cuts for DC factorization in polyhedral image
  PolVar<T> _append_cuts_dcdec
    ( SQuad::t_SPolyMonCoef const& eigterm );

  //! @brief Append cuts for monomial product in polyhedral image
  PolVar<T> _append_cuts_monprod
    ( SQuad::key_SQuad const& ijmon );

  //! @brief Append quadratic terms from <a>mat</a> to cuts
  void _add_to_cuts
    ( SQuad::t_SQuad const& mat, PolCut<T>* cut1=nullptr, PolCut<T>* cut2=nullptr );

  //! @brief Append quadratic terms from <a>mat</a> to cuts with or without DC factorization
  void _add_to_cuts
    ( SQuad const& quad, SQuad::t_SQuad const& mat, PolCut<T>* cut1=nullptr,
      PolCut<T>* cut2=nullptr );

  //! @brief Function computing Hausdorff distance between intervals
  template <typename U> static double _dH
    ( const U&X, const U&Y );
  //! @brief Function computing relative reduction between interval vectors
  template <typename U> static double _reducrel
    ( const unsigned n, const U*Xred, const U*X );
  //! @brief Function computing relative reduction between interval vectors
  template <typename U> static double _reducrel
    ( const unsigned n, const U*Xred, const U*X, const U*X0 );

  //! @brief Private methods to block default compiler methods
  MINLPBND
    ( const MINLPBND& );
  MINLPBND& operator=
    ( const MINLPBND& );
};

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::setup
( std::ostream& os )
{
  _issetup = false;
  _IINF = BASE_OPT::INF * T(-1,1);
  _ismip = false;
  for( auto const& typ : _vartyp ){
    if( !typ ) continue;
    _ismip = true;
    break;
  }

  // full set of parameters
  std::vector<FFVar> Pvar = _par;

  // full set of decision variables (independent & dependent)
  std::vector<FFVar> Xvar = _var;
  Xvar.insert( Xvar.end(), _dep.begin(), _dep.end() );
  _nX0 = Xvar.size();

  // full set of variable bounds and types (independent & dependent)
  _Xlow = _varlb;
  _Xupp = _varub;
  _Xtyp = _vartyp;
  _Xlow.insert( _Xlow.end(), _deplb.begin(), _deplb.end() );
  _Xupp.insert( _Xupp.end(), _depub.begin(), _depub.end() );
  _Xtyp.insert( _Xtyp.end(), _dep.size(), 0 );
  //Xtyp? _Xtyp.assign( Xtyp, Xtyp+_nX0 ): _Xtyp.assign( _nX0, 0 );

  // full set of nonlinear functions (cost, constraints & equations)
  std::vector<FFVar> Fvar;
  _Flow.clear();
  _Fupp.clear();

  // first, cost function
  if( std::get<0>(_obj).size() > 1 ) throw Exceptions( Exceptions::MULTOBJ );
  _objsense = std::get<0>(_obj).size()? (std::get<0>(_obj)[0]==BASE_OPT::MIN? -1: 1): 0;
  std::get<0>(_obj).size()? Fvar.push_back( std::get<1>(_obj)[0] ): Fvar.push_back( 0 );
  _Flow.push_back( -BASE_OPT::INF );
  _Fupp.push_back(  BASE_OPT::INF );

  // then, regular constraints
  for( unsigned i=0; i<std::get<0>(_ctr).size(); i++ ){
    Fvar.push_back( std::get<1>(_ctr)[i] );
    switch( std::get<0>(_ctr)[i] ){
      case EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );            break;
      case LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );            break;
      case GE: _Flow.push_back( 0. );             _Fupp.push_back( BASE_OPT::INF ); break;
    }
  }

  // then, dependent equations
  Fvar.insert( _Fvar.end(), _sys.begin(), _sys.end() );
  _Flow.insert( _Flow.end(), _sys.size(), 0. );
  _Fupp.insert( _Fupp.end(), _sys.size(), 0. );
  
  // set Fritz-John cuts and corresponding multipliers
  if( options.NCOCUTS
   && set_nco( _Xtyp.data(), options.NCOADIFF==Options::ASA? true: false ) ){

    // cost multiplier
    Xvar.push_back( std::get<2>(_obj)[0] );
    _Xlow.push_back( 0. );
    _Xupp.push_back( 1. );
    _Xtyp.push_back( 0  );

    // regular constraint multipliers
    for( unsigned i=0; i<std::get<0>(_ctr).size(); ++i ){
      Xvar.push_back( std::get<1>(_ctr)[i] );
      _Xtyp.push_back( 0 ); // all constraint multipliers are continuous variables
      switch( std::get<0>(_ctr)[i] ){
        case LE:
        case GE: _Flow.push_back(  0. ); _Fupp.push_back( 1. ); break;
        case EQ: _Flow.push_back( -1. ); _Fupp.push_back( 1. ); break;
      }
    }
    //Xvar.insert( Xvar.end(), std::get<2>(_ctr).begin(), std::get<2>(_ctr).end() );
    //_Xlow.insert( _Xlow.end(), std::get<2>(_ctr).size(), 0. );
    //_Xupp.insert( _Xupp.end(), std::get<2>(_ctr).size(), 1. );
    //_Xtyp.insert( _Xtyp.end(), std::get<2>(_ctr).size(), 0  );

    // dependent equation multipliers
    Xvar.insert( Xvar.end(), _sysm.begin(), _sysm.end() );
    _Xlow.insert( _Xlow.end(), _sysm.size(), -1. ); // all dependent equations are equality constraints
    _Xupp.insert( _Xupp.end(), _sysm.size(),  1. );
    _Xtyp.insert( _Xtyp.end(), _sysm.size(),  0  ); // all constraint multipliers are continuous variables

    // variable bound multipliers
    for( unsigned i=0; i<_nX0; i++ ){
      if( _Xtyp[i] ) continue;
      Xvar.push_back( _varlm[i] );
      Xvar.push_back( _varum[i] );
      _Xlow.insert( _Xlow.end(), 2, 0. );
      _Xupp.insert( _Xupp.end(), 2, 1. );
      _Xtyp.insert( _Xtyp.end(), 2, 0  );
    }
    
    // finally, Fritz-John cuts
    for( unsigned i=0; i<std::get<0>(_nco).size(); ++i ){
      Fvar.push_back( std::get<1>(_nco)[i] );
#ifdef MC__MINLPBND_DEBUG_NCOCUTS
      BASE_NLP::_dag->output( BASE_NLP::_dag->subgraph( 1, &Fvar.back() ), " FOR NCO" );    
#endif      
      switch( std::get<0>(_nco)[i] ){
        case EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );            break;
        case LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );            break;
        case GE: _Flow.push_back( 0. );             _Fupp.push_back( BASE_OPT::INF ); break;
      }
    }
  }
  
  // local DAG copy
  if( _dag ) delete _dag;
  _dag = new FFGraph;
  _nP = Pvar.size(); _Pvar.resize( _nP );
  _dag->insert( BASE_NLP::_dag, _nP, Pvar.data(), _Pvar.data() );
  _nX = _nX1 = Xvar.size(); _Xvar.resize( _nX );
  _dag->insert( BASE_NLP::_dag, _nX, Xvar.data(), _Xvar.data() );
  _nF = Fvar.size(); _Fvar.resize( _nF );
  _dag->insert( BASE_NLP::_dag, _nF, Fvar.data(), _Fvar.data() );

  // Identify variable and function types
  _set_variable_class();
  _set_function_class();

  // reformulate nonpolynomial functions
  _Xlift.clear();
  if( options.REFORMMETH.count( Options::NPOL ) ) _lift_nonpolynomial();
  if( options.REFORMMETH.count( Options::QUAD ) ) _lift_semialgebraic( false );

  // search for reduced RLT cuts
  if( options.RRLTCUTS ) _search_reduction_constraints();

  if( options.DISPLEVEL )
    os << "#              |  VARIABLES      FUNCTIONS" << std::endl << std::right
       << "# -------------+---------------------------" << std::endl
       << "#  LINEAR      | " << std::setw(9) << _Xlin.size()  << std::setw(15) << _Flin.size()  << std::endl
       << "#  QUADRATIC   | " << std::setw(9) << _Xquad.size() << std::setw(15) << _Fquad.size() << std::endl
       << "#  POLYNOMIAL  | " << std::setw(9) << _Xpol.size()  << std::setw(15) << _Fpol.size()  << std::endl
       << "#  GENERAL     | " << std::setw(9) << _Xgal.size()  << std::setw(15) << _Fgal.size()  << std::endl;

  // setup [-1,1] scaled variables for Chebyshev model arithmetic
  // do not downsize to avoid adding more variables into DAG
  for( unsigned i=_Xscal.size(); i<_nX; i++ ){
    _Xscal.push_back( FFVar() );
    _Xscal.back().set( _dag );
  }

  // setup for objective and constraints evaluation
  _Xobj.set( _dag );
  _Fops.clear();
  for( auto && Fj : _Fvar )
    _Fops.push_back( _dag->subgraph( 1, &Fj ) );
  for( auto const& [i,Fi] : _Xlift )
    _Fops.push_back( _dag->subgraph( 1, &Fi ) );
  _Fallops = _dag->subgraph( _nF, _Fvar.data() );
#ifdef MC__MINLPBND_DEBUG  
  _dag->output( _Fallops ), " FOR ALL FUNCTIONS" );    
#endif

  stats.reset();
  _issetup = true;
  return;
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_search_reduction_constraints
()
{
  if( _Fctreq.empty() ) return;

  // search for reduced RLT cuts
  RLTRed RRLT( _dag );
  RRLT.options = options.RLTRED;
  RRLT.search( _Fctreq, _Fvar.data() );

  // append any reduced RLT cuts
  for( auto const& pFred : RRLT.constraints() ){
#ifdef MC__MINLPBND_DEBUG_RRLTCUTS
    std::ostringstream ostr; ostr << " OF REDUCTION CONSTRAINT " << *pFred;
    _dag->output( _dag->subgraph( 1, pFred ), ostr.str() );
#endif
    _Fvar.push_back( *pFred );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
  }

  // update variable and function size and type
  _nX = _Xvar.size();
  _nF = _Fvar.size();
  _set_variable_class();    
  _set_function_class();    
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_lift_semialgebraic
( bool const add2dag )
{
  auto Fsalg = _Flin;
  Fsalg.insert( _Fquad.cbegin(), _Fquad.cend() );
  Fsalg.insert( _Fpol.cbegin(),  _Fpol.cend()  ); 
  if( Fsalg.empty() ) return;

  // Create vector of all semi-algebraic expressions
  SPolyExpr::options.BASIS = SPolyExpr::Options::MONOM;
  std::vector<SPolyExpr> SPXvar, SPFvar( _nF );
  for( auto const& var: _Xvar ) SPXvar.push_back( SPolyExpr( var ) );
  _dag->eval( Fsalg, _Fvar.data(), SPFvar.data(), _nX, _Xvar.data(), SPXvar.data() );
#ifdef MC__MINLPBND_DEBUG_LIFT
  for( auto i : Fsalg ) std::cout << SPFvar[i];
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif

  // Substitute linear and quadratic expressions in DAG
  for( auto i : _Flin ){
    _Fvar[i] = SPFvar[i].insert( _dag );
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of flattened linear expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
  }
  for( auto i : _Fquad ){
    _Fvar[i] = SPFvar[i].insert( _dag );
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of flattened quadratic expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
  }

  // Transform variable indexing in quadratic and polynomial expressions
  std::map<FFVar const*, unsigned, lt_FFVar> FFmatch;
  unsigned ivar = 0;
  for( auto const& var : _Xvar ) FFmatch[&var] = ivar++;
  unsigned ifun = 0;
  auto Ftpol = _Fpol; Ftpol.insert( _Fquad.cbegin(), _Fquad.cend() );
  std::vector<SQuad::t_SPolyMonCoef> SPol( Ftpol.size() );
  for( auto i : Ftpol ){
    for( auto const& [FFmon,coef] : SPFvar[i].mapmon() ){
      SPolyMon mon( FFmon.tord, FFmon.expr, FFmatch ); 
      SPol[ifun].insert( std::make_pair( mon, coef ) );
    }
    ++ifun;
  }

  // Apply quadratisation to polynomial expressions
  _SQenv.reset();
  _SQenv.options = options.SQUAD;
#ifndef MC__MINLPBND_DEBUG_LIFT
  _SQenv.process( SPol.size(), SPol.data(), SQuad::Options::MONOM );
#else
  double viol = _SQenv.process( SPol.size(), SPol.data(), SQuad::Options::MONOM, true );
  std::cout << "violation: " << viol << _SQenv << std::endl;
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif
  if( !add2dag ) return;

  // Add higher-order monomials in basis to DAG
  std::map< SPolyMon, FFVar, lt_SPolyMon > mapmon; 
  for( auto const& mon : _SQenv.SetMon() ){
    if( mon.tord == 1 ) mapmon[mon] = _Xvar[mon.expr.cbegin()->first];
    if( mon.tord <= 1 ) continue;
    auto [pAux,pVar] = _var_mon( mon );
    _Xlift[_Xvar.size()] = *pAux; // <- stores monomial DAG expression
    _Xvar.push_back( *pVar );
    _Xlow.push_back( -BASE_OPT::INF );
    _Xupp.push_back(  BASE_OPT::INF );
    _Xtyp.push_back( 0 );
    mapmon[mon] = *pVar;
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::cout << "Lifted monomial " << *pVar << " := " << mon.display(options.SQUAD.BASIS) << std::endl;
    _dag->output( _dag->subgraph( 1, pAux ) );
#endif
  }

  // Substitute lifted quadratic expressions
  unsigned iquad = 0;
  for( auto i : Ftpol ){
    _Fvar[i] = _var_pol( _SQenv.MatFct()[iquad++], mapmon );
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of lifted quadratic expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
  }
  
  // Append reduction quadratic cuts
  for( auto const& red : _SQenv.MatRed() ){
    _Fvar.push_back( _var_pol( red, mapmon ) );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of reduction quadratic cut";
    _dag->output( _dag->subgraph( 1, &_Fvar.back() ), ostr.str() );
#endif
  }

  // Append positive semi-definite cuts
  _SQenv.tighten( options.PSDQUADCUTS>1? true: false );
  for( auto const& psd : _SQenv.MatPSD() ){
    _Fvar.push_back( _var_pol( psd, mapmon ) );
    _Flow.push_back( 0. );
    _Fupp.push_back( BASE_OPT::INF );
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of semi-definite quadratic cut >=0";
    _dag->output( _dag->subgraph( 1, &_Fvar.back() ), ostr.str() );
#endif
  }

  // update variable and function size and type
  _nX = _Xvar.size();
  _nF = _Fvar.size();
  _set_variable_class();    
  _set_function_class();
#ifdef MC__MINLPBND_DEBUG_LIFT
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }  
#endif
}

template <typename T, typename MIP>
inline FFVar
MINLPBND<T,MIP>::_var_pol
( SQuad::t_SQuad const& quad, std::map< SPolyMon, FFVar, lt_SPolyMon >& mapmon )
const
{
  FFVar varpol = 0.;
  for( auto const& [ijmon,coef] : quad ){
    if( !ijmon.first->tord && !ijmon.second->tord )
      varpol += coef;
    else if( !ijmon.first->tord ){
#ifdef MC__MINLPBND_DEBUG_LIFT
      assert( mapmon.count( *ijmon.second ) );
#endif
      varpol += coef * mapmon[*ijmon.second];
    }
    else if( ijmon.first == ijmon.second ){
#ifdef MC__MINLPBND_DEBUG_LIFT
      assert( mapmon.count( *ijmon.first ) );
#endif
      varpol += coef * sqr( mapmon[*ijmon.second] );
    }
    else{
#ifdef MC__MINLPBND_DEBUG_LIFT
      assert( mapmon.count( *ijmon.first ) && mapmon.count( *ijmon.second ) );
#endif
      varpol += coef * ( mapmon[*ijmon.first] * mapmon[*ijmon.second] );
    }
  }
  return varpol;
}

template <typename T, typename MIP>
inline FFVar
MINLPBND<T,MIP>::_var_cheb
( FFVar const& x, const unsigned n )
const
{
  switch( n ){
    case 0:  return 1.;
    case 1:  return x;
    case 2:  return 2.*sqr(x)-1.;
    default: return n%2? 2.*_var_cheb(x,n/2)*_var_cheb(x,n/2+1)-x:
                         2.*sqr(_var_cheb(x,n/2))-1.;
    //default: return 2.*x*_var_cheb(x,n-1)-_var_cheb(x,n-2);
  }
}

template <typename T, typename MIP>
inline T
MINLPBND<T,MIP>::_bnd_cheb
( T const& x, const unsigned n )
const
{
  switch( n ){
    case 0:  return 1.;
    case 1:  return x;
    case 2:  return 2.*Op<T>::sqr(x)-1.;
    default: return n%2? 2.*_bnd_cheb(x,n/2)*_bnd_cheb(x,n/2+1)-x:
                         2.*Op<T>::sqr(_bnd_cheb(x,n/2))-1.;
    //default: return 2.*x*_bnd_cheb(x,n-1)-_bnd_cheb(x,n-2);
  }
}

template <typename T, typename MIP>
inline std::pair< FFVar const*, FFVar const* >
MINLPBND<T,MIP>::_var_mon
( SPolyMon const& mon )
const
{
  // define power monomial expression
  FFVar Xlift( 1e0 );
  for( auto const& [ivar,iord] : mon.expr ){
    switch( options.SQUAD.BASIS ){
     // Monomial basis
     case SQuad::Options::MONOM:
      Xlift *= pow( _Xvar[ivar], (int)iord );
      break;
     // Chebyshev basis
     case SQuad::Options::CHEB:
      Xlift *= _var_cheb( _Xvar[ivar], iord );
      break;
    }
  }
  auto itXlift = _dag->Vars().find( &Xlift );

  // define power monomial variable
  FFVar Xmon( _dag );
  auto itXmon = _dag->Vars().find( &Xmon );

  return std::make_pair( *itXlift, *itXmon );
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_lift_nonpolynomial
()
{
  if( _Fgal.empty() ) return;

  _SEenv.set( _dag );
  _SEenv.options = options.SPARSEEXPR;
  _SEenv.process( _Fgal, _Fvar.data(), true );
#ifdef MC__MINLPBND_DEBUG_LIFT
  std::cout << std::endl << _SEenv.Var().size() << " participating variables: ";
  for( auto&& var : _SEenv.Var() ) std::cout << var << " ";
  std::cout << std::endl;
  std::cout << std::endl << _SEenv.Aux().size() << " auxiliary variables: ";
  for( auto&& aux : _SEenv.Aux() ) std::cout << *aux.first << "->" << *aux.second << " ";
  std::cout << std::endl;
  std::cout << std::endl << _SEenv.Poly().size() << " polynomial constraints: " << std::endl;
  for( auto&& expr : _SEenv.Poly() ) _dag->output( _dag->subgraph( 1, &expr ) );
  //std::cout << std::endl;
  std::cout << std::endl << _SEenv.Trans().size() << " transcendental constraints: " << std::endl;
  for( auto&& expr : _SEenv.Trans() ) _dag->output( _dag->subgraph( 1, &expr ) );
  //std::cout << _SEenv;
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif

  // append auxiliary variables
  std::set<unsigned> Fred;
  for( auto&& [pAux,pVar] : _SEenv.Aux() ){
    bool is_dep = false;
    unsigned i = 0;
    for( auto it=_Fgal.begin(); it!=_Fgal.end(); ++it ){
      i = *it;
      if( _Fvar[i] != *pAux ) continue;
      is_dep = true;
      break;
    }
    _Xlift[_Xvar.size()] = *pAux; // <- stores original DAG expression
    _Xvar.push_back( *pVar );
    _Xlow.push_back( is_dep? _Flow[i]: -BASE_OPT::INF );
    _Xupp.push_back( is_dep? _Fupp[i]:  BASE_OPT::INF );
    _Xtyp.push_back( 0 );
    
    // update/erase corresponding entries in function vectors (this is not efficient...)
    if( !is_dep ) continue;
    if( !i ){
      _Fvar[0] = *pVar;
      continue;
    }
    Fred.insert( i ); 
    //auto itFvar = _Fvar.begin(); std::advance( itFvar, i ); _Fvar.erase( itFvar );
    //auto itFlow = _Flow.begin(); std::advance( itFlow, i ); _Flow.erase( itFlow );
    //auto itFupp = _Fupp.begin(); std::advance( itFupp, i ); _Fupp.erase( itFupp );
  }
  for( auto it=Fred.rbegin(); it!=Fred.rend(); ++it ){
    unsigned i = *it;
    auto itFvar = _Fvar.begin(); std::advance( itFvar, i ); _Fvar.erase( itFvar );
    auto itFlow = _Flow.begin(); std::advance( itFlow, i ); _Flow.erase( itFlow );
    auto itFupp = _Fupp.begin(); std::advance( itFupp, i ); _Fupp.erase( itFupp );
  }
  
  // append lifted polynomial expressions
  for( auto const& poly : _SEenv.Poly() ){
    _Fvar.push_back( poly );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
  }

  // append lifted transcendental expressions
  for( auto const& trans : _SEenv.Trans() ){
    _Fvar.push_back( trans );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
  }

  // update variable and function size and type
  _nX = _Xvar.size();
  _nF = _Fvar.size();
  _set_variable_class();    
  _set_function_class();    
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_variable_class
()
{
  FFDep Fworst( 0. );
  for( auto && Fj : _Fvar )
    Fworst += Fj.dep();
#ifdef MC__MINLPBND_DEBUG
  std::cout << "DEPS <- " << Fworst << std::endl;
  //int dum; std::cin >> dum;
#endif

  _Xlin.clear();
  _Xquad.clear();
  _Xpol.clear();
  _Xgal.clear();

  for( unsigned i=0; i<_nX; i++ ){
    auto it = Fworst.dep().find( _Xvar[i].id().second );
    if( it == Fworst.dep().end() ) _Xlin.insert( i );
    else switch( it->second ){
     case FFDep::L: _Xlin.insert( i );  break;
     case FFDep::Q: _Xquad.insert( i ); break;
     case FFDep::P: _Xpol.insert( i );  break;
     case FFDep::R:
     case FFDep::N: _Xgal.insert( i ); break;
    }
  }
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_function_class
()
{
  _Flin.clear();
  _Fquad.clear();
  _Fpol.clear();
  _Fgal.clear();
  _Fctreq.clear();
 
  _pbclass = FFDep::L;
  for( unsigned j=0; j<_nF; j++ ){
    auto depworst = _Fvar[j].dep().worst();
    switch( depworst ){
     case FFDep::L: _Flin.insert( j );  break;
     case FFDep::Q: _Fquad.insert( j ); break;
     case FFDep::P: _Fpol.insert( j );  break;
     case FFDep::R:
     case FFDep::N: _Fgal.insert( j );  break;
    }
    if( _pbclass < depworst ) _pbclass = depworst;
    if( _Flow[j] == 0. && _Fupp[j] == 0. ) _Fctreq.insert( j );
  }
}

template <typename T, typename MIP>
inline bool
MINLPBND<T,MIP>::bounded_domain
( double const& maxdiam, FFDep::TYPE const type )
const
{
  switch( type ){
    case FFDep::L: for( auto i : _Xlin  ) if( Op<T>::diam( _Xbnd[i] ) >= maxdiam ) return false; break;
    case FFDep::Q: for( auto i : _Xquad ) if( Op<T>::diam( _Xbnd[i] ) >= maxdiam ) return false; break;
    case FFDep::P: for( auto i : _Xpol  ) if( Op<T>::diam( _Xbnd[i] ) >= maxdiam ) return false; break;
    case FFDep::R:
    case FFDep::N: for( auto i : _Xgal  ) if( Op<T>::diam( _Xbnd[i] ) >= maxdiam ) return false; break;
  }
  return true;
}

template <typename T, typename MIP>
inline bool
MINLPBND<T,MIP>::update_bounds
( T const* X, double const* Finc, bool const resetbnd )
{
  // Variable bounds
  if( resetbnd ){
    _Xbnd.resize( _nX );
    for( unsigned i=0; i<_nX; i++ ) _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
  }
  for( unsigned i=0; i<_nX0; i++ )
    if( X && !Op<T>::inter( _Xbnd[i], X[i], _Xbnd[i] ) ) return false;
  
  // Bound propagation for lifted variables
  unsigned j=0; 
  for( auto const& [i,Fi] : _Xlift ){
    try{
      T Xi;
#ifdef MC__MINLPBND_DEBUG_BOUNDS
      _dag->output( _dag->subgraph( 1, &Fi ), " FOR LIFTED VARIABLE" );    
#endif
      _dag->eval( _Fops.at(_nF+j), _Iwk, 1, &Fi, &Xi, _nX1, _Xvar.data(), _Xbnd.data() );
#ifdef MC__MINLPBND_DEBUG_BOUNDS
      std::cout << "Xbnd[ " << i << "] = " << _Xbnd[i] << std::endl;
      std::cout << "Xprop[ " << i << "] = " << Xi << std::endl;
#endif
      if( !Op<T>::inter( _Xbnd[i], Xi, _Xbnd[i] ) ) return false;
      j++;
    }
    catch(...){
      // No cut added for function #j in case DAG evaluation failed
      continue;
    }
  }

  // Function bounds
  if( resetbnd ){
    _Fbnd.resize( _nF );
    for( unsigned i=0; i<_nF; i++ ) _Fbnd[i] = T( _Flow[i], _Fupp[i] );
  }
  if( !Finc ) _Fbnd[0] = T( _Flow[0], _Fupp[0] );
  else if( _objsense == -1 && !Op<T>::inter( _Fbnd[0], T(-BASE_OPT::INF,*Finc), _Fbnd[0] ) ) return false;
  else if( _objsense ==  1 && !Op<T>::inter( _Fbnd[0], T( *Finc,BASE_OPT::INF), _Fbnd[0] ) ) return false;

  return true;
}

template <typename T, typename MIP>
inline int
MINLPBND<T,MIP>::_propagate_bounds
()
{
#ifdef MC__MINLPBND_DEBUG_CP
  _dag->output( _Fallops );
#endif
  
  // Apply constraint propagation
  auto tstart = stats.start();
  int flag = _dag->reval( _Fallops, _CPbnd, _nF, _Fvar.data(), _Fbnd.data(), _nX, _Xvar.data(),
                          _Xbnd.data(), _IINF, options.CPMAX, options.CPTHRES );
  stats.walltime_cprop += stats.walltime( tstart );
  
#ifdef MC__MINLPBND_DEBUG_CP
  std::cout << "\nReduced Box:\n";
  int i=0;
  for( auto && bnd : _CPbnd )
    std::cout << "WK[" << i++ << "] = " << bnd << std::endl;
#endif
  return flag;
}

template <typename T, typename MIP>
inline int
MINLPBND<T,MIP>::relax
( T const* X, double const* Finc, double const* Xinc, const unsigned nref,
  const bool resetbnd, bool const reinit, std::string const gmsfile )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _tstart = stats.start();

  // Update variable bounds
  if( !update_bounds( X, Finc, resetbnd ) ) return MIP::INFEASIBLE;
  int cpred = _propagate_bounds();
  if( cpred < 0 ) return MIP::INFEASIBLE;

  // Reset polyhedral image, LP variables and cuts
  if( reinit ) init_polrelax(); // <<== COULD PASS Xinc HERE TOO??
  update_polrelax( 2, false, reinit? false: true );
#ifdef MC__MINLPBND_DEBUG
  std::cout << _POLenv;
#endif

  // Write relaxed model to GAMS file
  if( !gmsfile.empty() ){
    GAMSWRITER<T> GMS;
    GMS.set_cuts( &_POLenv, true );
    for( unsigned i=0; i<_nX0; i++ )
      GMS.set_variable( _POLXvar[i], Xinc? &Xinc[i]: nullptr );
    GMS.set_objective( _POLFvar[0], _objsense>0? BASE_OPT::MAX: BASE_OPT::MIN );
    GMS.write( gmsfile );
    return MIP::OTHER;
  }

  // Set-up variable initial guess and branch priority
  for( unsigned i=0; i<_nX0; i++ )
    assert( _MIPSLV->set_variable( _POLXvar[i], Xinc? &Xinc[i]: nullptr, options.BCHPRIM ) );

  for( unsigned iref=0; ; iref++ ){
    // Set-up relaxed objective, options, and solve polyhedral relaxation
    auto tMIP = stats.start();
    _MIPSLV->set_objective( _POLFvar[0], _objsense>0? BASE_OPT::MAX: BASE_OPT::MIN );
    _MIPSLV->options = options.MIPSLV;
    _MIPSLV->options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime( _tstart ) );
    _MIPSLV->solve();
    stats.walltime_slvmip += stats.walltime( tMIP );
    stats.total_slvmip ++;

    // Break if relaxation unsuccessful or refinement iteration exceeded
    // Accept both optimal and suboptimal MIP solutions - could be dangerous?
    if( iref >= nref 
     || ( _MIPSLV->get_status() != MIP::OPTIMAL
       && _MIPSLV->get_status() != MIP::SUBOPTIMAL ) ) break;

    // Refine relaxation via additional breakpoints
    refine_polrelax( Xinc );
#ifdef MC__MINLPBND_DEBUG
    std::cout << _POLenv;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }

  return _MIPSLV->get_status();
}

template <typename T, typename MIP>
inline int
MINLPBND<T,MIP>::_reduce
( unsigned const ix, bool const uplo )
{
#ifdef MC__MINLPBND_DEBUG
  std::cout << "\nTIGHTENING OF VARIABLE " << ix << (uplo?"U":"L") << ":\n";
//  std::cout << _POLenv;
#endif
  // Set-up lower/upper bound objective, options, and solve polyhedral relaxation
  auto tMIP = stats.start();
  _MIPSLV->set_objective( _POLXvar[ix], (uplo? BASE_OPT::MAX: BASE_OPT::MIN) );
  _MIPSLV->options = options.MIPSLV;
  _MIPSLV->options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime( _tstart ) );
  _MIPSLV->solve();
  stats.walltime_slvmip += stats.walltime( tMIP );
  stats.total_slvmip ++;

  return _MIPSLV->get_status();
}

template <typename T, typename MIP>
inline bool
MINLPBND<T,MIP>::_tight
()
{
  // test if current bounds are tight
  for( unsigned i=0; i<_nX; i++ )
    if( Op<T>::diam( _Xbnd[i] ) > options.OBBTMIG ) return false;
  return true;
}

template <typename T, typename MIP>
inline int
MINLPBND<T,MIP>::_reduce
()
{
  // solve reduction subproblems from closest to farthest from bounds
  std::multimap<double,std::pair<unsigned,bool>> vardomred, vardomredupd;
  std::pair<unsigned,bool> varini;
  for( unsigned i=0; i<_nX; i++ ){
    // do not reduce variables whose domain is less than OBBTMIG
    if( Op<T>::diam( _Xbnd[i] ) < options.OBBTMIG ) continue;

    varini.first = i;
    varini.second = false; // lower bound
    double dist = 1.;
    vardomred.insert( std::pair<double,std::pair<unsigned,bool>>(dist,varini) );
    varini.second = true;  // upper bound
    vardomred.insert( std::pair<double,std::pair<unsigned,bool>>(dist,varini) );
  }
  
  // anything to reduce?
  if( vardomred.empty() ) return MIP::OPTIMAL;

  unsigned nred=0;
  for( ; !vardomred.empty(); nred++ ){
    // upper/lower range reduction for current subproblem
    auto itv = vardomred.begin();
    unsigned const ix = (*itv).second.first;
    bool const uplo  = (*itv).second.second;
    double xL = Op<T>::l( _Xbnd[ix] ), xU = Op<T>::u( _Xbnd[ix] );
    _reduce( ix, uplo );

    // Accept both optimal and suboptimal MIP solutions - could be dangerous?
    if( _MIPSLV->get_status() == MIP::OPTIMAL
     || _MIPSLV->get_status() == MIP::SUBOPTIMAL ){
      switch( (int)uplo ){
       case false: // lower bound
        xL = _MIPSLV->get_objective_bound();
        if( options.OBBTBKOFF > 0. )
          xL -= options.OBBTBKOFF + std::fabs(xL)*options.OBBTBKOFF;
        if( _Xtyp[ix] > 0 ) xL = std::ceil( xL );
        if( !Op<T>::inter(  _Xbnd[ix], _Xbnd[ix], T(xL,xU+1.) ) ) _Xbnd[ix] = xU;
        break;
       case true: // upper bound
        xU = _MIPSLV->get_objective_bound();
        if( options.OBBTBKOFF > 0. )
          xU += options.OBBTBKOFF + std::fabs(xU)*options.OBBTBKOFF;;
        if( _Xtyp[ix] > 0 ) xU = std::floor( xU );
        if( !Op<T>::inter(  _Xbnd[ix], _Xbnd[ix], T(xL-1.,xU) ) ) _Xbnd[ix] = xL;
        break;
      }
        
#ifdef MC__MINLPBND_DEBUG
      std::cout << "  UPDATED RANGE OF VARIABLE #" << ix << ": " << _Xbnd[ix] << std::endl;
#endif
      // update map of candidate reduction subproblems
      vardomredupd.clear();
      for( ++itv; itv!=vardomred.end(); ++itv ){
        unsigned const ix = (*itv).second.first;
        bool const uplo  = (*itv).second.second;
        double dist = ( uplo? std::fabs( _MIPSLV->get_variable( _Xvar[ix] ) - Op<T>::u( _Xbnd[ix] ) ):
                              std::fabs( _MIPSLV->get_variable( _Xvar[ix] ) - Op<T>::l( _Xbnd[ix] ) ) )
                      / Op<T>::diam( _Xbnd[ix] ); // consider relative distance to bound
        if( dist <= options.OBBTTHRES ) continue;
        if( dist > (*itv).first ) dist = (*itv).first;
        vardomredupd.insert( std::pair<double,std::pair<unsigned,bool>>(dist,std::make_pair(ix,uplo)) );
      }
      vardomred.swap( vardomredupd );
      continue;
    }

    // Some variables may be unbounded
    else if( _MIPSLV->get_status() != MIP::UNBOUNDED
          && _MIPSLV->get_status() != MIP::INFORUNBND ){
#ifdef MC__MINLPBND_PAUSE_INFEASIBLE
      int dum; std::cout << "Infeasible or interrupted problem - PAUSED"; std::cin >> dum;
#endif
      break;
    }
    
    vardomred.erase( itv );
  }
#ifdef MC__MINLPBND_DEBUG
  std::cout << "SOLVED " << nred << " RANGE REDUCTION LPs OUT OF " << 2*_nX << std::endl;
#endif

  return _MIPSLV->get_status();
}

template <typename T, typename MIP>
inline int
MINLPBND<T,MIP>::reduce
( unsigned& nred, T* X, double const* Finc, const bool resetbnd,
  bool const reinit )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _tstart = stats.start();
    
  // Update variable bounds
  if( !update_bounds( X, Finc, resetbnd ) ) return MIP::INFEASIBLE;
  int cpred = _propagate_bounds();
  if( cpred < 0 ) return MIP::INFEASIBLE;
  if( _tight() ){
    for( unsigned i=0; X && i<_nX; i++ ) X[i] = _Xbnd[i];
    return MIP::OPTIMAL;
  }

  // Reset polyhedral image, LP variables and cuts
  if( reinit ) init_polrelax();
#ifdef MC__MINLPBND_DEBUG
  std::cout << _POLenv;
#endif

  // Main loop for relaxation and domain reduction
  double vred = 0.;
  int flag = MIP::OPTIMAL;
  std::vector<T> Xbnd0 = _Xbnd, Xbnd1( _nX );
  for( nred = 0; nred < options.OBBTMAX; nred++ ){
    Xbnd1 = _Xbnd;

    // Optimization-based domain reduction considering linear functions only
    if( !_Flin.empty() && options.OBBTLIN < 2 ){
      update_polrelax( 0, options.OBBTCONT, true );
#ifdef MC__MINLPBND_DEBUG
      std::cout << _POLenv;
#endif
      flag = _reduce();
      if( flag != MIP::OPTIMAL   && flag != MIP::SUBOPTIMAL
       && flag != MIP::UNBOUNDED && flag != MIP::INFORUNBND ) break;

      // Constraint propagation-based domain reduction considering all functions
      cpred = _propagate_bounds();
      if( cpred < 0 ){
        flag = MIP::INFEASIBLE;
        //break;
      }
      if( _tight() ){
        flag = MIP::OPTIMAL;
        break;
      }
    }
    
    // Optimization-based domain reduction considering nonlinear functions as well
    if( options.OBBTLIN > 0 ){
      if( !_Flin.empty() && options.OBBTLIN < 2 )
        update_polrelax( 1, options.OBBTCONT, false );
      else
        update_polrelax( 2, options.OBBTCONT, true );
#ifdef MC__MINLPBND_DEBUG
      std::cout << _POLenv;
#endif
      flag = _reduce();
      if( flag != MIP::OPTIMAL   && flag != MIP::SUBOPTIMAL
       && flag != MIP::UNBOUNDED && flag != MIP::INFORUNBND ) break;

      // Constraint propagation-based domain reduction considering all functions
      cpred = _propagate_bounds();
      if( cpred < 0 ){
#ifdef MC__MINLPBND_SHOW_REDUC
        std::cout << "Infeasibility during constraint propagation (" << cpred << ")\n";
#endif
        flag = MIP::INFEASIBLE;
        //break;
      }
      if( _tight() ){
        flag = MIP::OPTIMAL;
        break;
      }
    }
    
    // Check reduction ratio
    vred = _reducrel( _nX, _Xbnd.data(), Xbnd1.data(), Xbnd0.data() );
    if( vred < options.OBBTTHRES ) break;
#ifdef MC__MINLPBND_SHOW_REDUC
    std::cout << "Reduction #" << nred+1 << ": "
              << std::fixed << std::setprecision(1) << vred*1e2 << "%\n";
    std::cout << "\nReduced Box:\n";
    for( unsigned i=0; i<_nX; i++ )
      std::cout << _Xvar[i] << " = " << _Xbnd[i] << std::endl;
    { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  }
  
  // Update user bounds <a>X</a>
  for( unsigned i=0; X && i<_nX0; i++ ) X[i] = _Xbnd[i];

#ifdef MC__MINLPBND_SHOW_REDUC
  std::cout << "Reduction #" << nred+1 << ": (" << flag << ") "
            << std::fixed << std::setprecision(1) << vred*1e2 << "%\n";
  std::cout << "\nReduced Box:\n";
  for( unsigned i=0; i<_nX; i++ )
    std::cout << _Xvar[i] << " = " << _Xbnd[i] << std::endl;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

  return flag;
}

template <typename T, typename MIP>
inline int
MINLPBND<T,MIP>::propagate
( T* X, double const* Finc, const bool resetbnd )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _tstart = stats.start();
    
  // Update variable bounds
  if( !update_bounds( X, Finc, resetbnd ) ) return MIP::INFEASIBLE;
  int cpred = _propagate_bounds();
  if( cpred < 0 ) return MIP::INFEASIBLE;

  // Update user bounds <a>X</a>
  for( unsigned i=0; X && i<_nX0; i++ ) X[i] = _Xbnd[i];

#ifdef MC__MINLPBND_SHOW_REDUC
  std::cout << "\nPropagated Bounds:\n";
  for( unsigned i=0; i<_nX; i++ )
    std::cout << _Xvar[i] << " = " << _Xbnd[i] << std::endl;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

  return MIP::OPTIMAL;
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::init_polrelax
()
{
  auto tstart = stats.start();

  // Reset polyhedral image
  _POLenv.reset();
  _POLenv.options = options.POLIMG;

  // Set polyhedral dependent variables
  _POLFvar.clear();
  auto itF = _Fvar.begin();
  for( unsigned i=0; itF!=_Fvar.end(); ++itF, i++ )
    _POLFvar.push_back( PolVar<T>( &_POLenv, *itF, _Fbnd[i], true ) );

  // Set polyhedral main variables
  _POLXvar.clear();
  auto itX = _Xvar.begin();
  for( unsigned i=0; itX!=_Xvar.end(); ++itX, i++ )
    _POLXvar.push_back( PolVar<T>( &_POLenv, *itX, _Xbnd[i], (_Xtyp[i]? false: true) ) );

  // Set nonlinear cuts
  for( auto const& meth : options.RELAXMETH ){
    switch( meth ){
    
      // Add McCormick-derived polyhedral cuts
      default:
      case Options::DRL:
        break;

     // Add Chebyshev-derived polyhedral cuts
     case Options::SCDRL:
       // Reset Chebyshev basis map in DAG
       _Xmon.clear();

       // Reset scaled variables in polyhedral image
       _POLXscal.clear();
       for( auto&& X : _Xscal )
         _POLXscal.push_back( PolVar<T>( &_POLenv, X, T(-1e0,1e0), true ) );
       // **no break** to continue into SCQ
       
     case Options::SCQ:
       // Chebyshev model environment reset
       if( _CMenv && (_CMenv->maxvar() != _nX || _CMenv->maxord() != options.CMODPROP) ){
         _CMXvar.clear();
         delete _CMenv; _CMenv = 0;   
       }
       if( !_CMenv ){
         // Set Chebyshev model
         _CMenv = new SCModel<T>( options.CMODPROP, _nX );
         _CMenv->options = options.CMODEL;
         _CMXvar.resize( _nX );
       }
       break;

     // Add Interval superposition-derived polyhedral cuts
     case Options::ISM:
       if( _ISMenv && (_ISMenv->nvar() != _nX || _ISMenv->ndiv() != options.ISMDIV) ){
         _ISMXvar.clear();
         delete _ISMenv; _ISMenv = 0;   
       }
       if( !_ISMenv ){
         // Set interval superposition model
         _ISMenv = new ISModel<T>( _nX, options.ISMDIV );
         //_ISMenv->options = options.ISMODEL;
         _ISMXvar.resize( _nX );
       }
       break;
    }
  }
  
  stats.walltime_polimg += stats.walltime( tstart );
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::update_polrelax
( unsigned const addcuts, bool const contcuts, bool const resetcuts )
{
  auto tstart = stats.start();

  // Reset polyhedral cuts
  if( resetcuts ) _POLenv.reset_cuts();

  // Reset monomial vectors
  _POLXmon.clear();
  _POLXprodmon.clear();

  // Update polyhedral main variables
  auto itX = _POLXvar.begin();
  for( unsigned i=0; itX!=_POLXvar.end(); ++itX, i++ )
    itX->update( _Xbnd[i] );

  // Add linear cuts
  if( addcuts != 1 ) _set_cuts_LIN();

  // Add nonlinear cuts
  if( addcuts > 0 ) for( auto && meth : options.RELAXMETH ){
    switch( meth ){

      // Add McCormick-derived polyhedral cuts
      default:
      case Options::DRL:
        // Add polyhedral cuts
        _set_cuts_DRL();
        break;

      // Add Chebyshev-derived polyhedral cuts
      case Options::SCDRL:
        // Add polyhedral cuts
        _set_cuts_SCDRL();
        break;

      // Add Chebyshev-derived polyhedral cuts
      case Options::SCQ:
        // Add quadratic cuts
        _set_cuts_SCQ();
        break;

      // Add Interval superposition-derived polyhedral cuts
      case Options::ISM:
        // Update ISM variables
        for( unsigned i=0; i<_nX; i++ )
          _ISMXvar[i].set( _ISMenv, i, _Xbnd[i] );

        // Add polyhedral cuts
        _set_cuts_ISM();
        break;
    }
  }

  // Update polyhedral dependent bounds
  for( unsigned i=0; i<_nF; i++ ){
   T Fupdi = _Fbnd[i];
    Op<T>::inter( Fupdi, _Fbnd[i], _POLFvar[i].range() );
    _POLFvar[i].update( Fupdi );
  }

  stats.walltime_polimg += stats.walltime( tstart );

  // Input cuts in MIP solver
  auto CONTRELAX = _MIPSLV->options.CONTRELAX;
  _MIPSLV->options.CONTRELAX = contcuts;
  tstart = stats.start();
  _MIPSLV->set_cuts( &_POLenv, true );
  stats.walltime_setmip += stats.walltime( tstart );
  _MIPSLV->options.CONTRELAX = CONTRELAX;
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::refine_polrelax
( double const* Xinc, bool const resetcuts )
{
  auto tstart = stats.start();
  
  // Update discretization with optimum point of relaxation
  for( auto itv=_POLenv.Vars().begin(); itv!=_POLenv.Vars().end(); ++itv ){
    double Xval = _MIPSLV->get_variable( *itv->second );
    itv->second->add_breakpt( Xval );
#ifdef MC__MINLPBND_SHOW_BREAKPTS
    std::cout << itv->second->name() << " " << Xval << " " << itv->second->range()
              << std::scientific << std::setprecision(4);
    for( auto it = itv->second->breakpts().begin(); it!=itv->second->breakpts().end(); ++it )
      std::cout << "  " << *it;
    std::cout << std::endl;
#endif
  }
  auto itX = _POLXvar.begin();
  for( unsigned i=0; itX!=_POLXvar.end(); ++itX, i++ ){
    double Xval = _MIPSLV->get_variable( *itX );
    itX->add_breakpt( Xval );
    itX->update( _Xbnd[i] );
  }

  // Update discetization with incumbent
  if( Xinc ){
    itX = _POLXvar.begin();
    for( unsigned i=0; i<_nX0 && itX!=_POLXvar.end(); ++itX, i++ ){
      itX->add_breakpt( Xinc[i] );
      auto itv = _POLenv.Vars().find( &_var[i] );
      itv->second->add_breakpt( Xinc[i] );
    }
  }

  // Reset polyhedral cuts
  if( resetcuts ) _POLenv.reset_cuts();
  stats.walltime_polimg += stats.walltime( tstart );
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_cuts_LIN
()
{
 // Add polyhedral cuts for each linear function
 //_POLFvar.resize( _nF );
 for( unsigned j=0; options.LINCTRSEP && j<_nF; j++ ){
   if( _Flin.find( j ) == _Flin.end() ) continue; // (!j && !_objsense) || 
   try{
     _dag->eval( _Fops[j], _POLwk, 1, &_Fvar[j], &_POLFvar[j], _nX, _Xvar.data(), _POLXvar.data() );
     // Update bounds of intermediate factors from constraint propagation results
     if( options.CPMAX ){
       _dag->wkextract( _Fops[j], _Iwk, _Fallops, _CPbnd );
       for( unsigned i=0; i<_Iwk.size(); i++ ) _POLwk[i].update( _Iwk[i] );
     }
     // Generate cuts
     _POLenv.generate_cuts( 1, &_POLFvar[j], false );
   }
   catch(...){
     // No cut added for function #j in case DAG evaluation failed
     continue;
   }
 }
#ifdef MC__MINLPBND_DEBUG_LIN
 std::cout << _POLenv;
 { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_cuts_DRL
()
{
  // Subset of functions to be relaxed
  std::set<unsigned> Frel, Fsalg = _Fquad; Fsalg.insert( _Fpol.cbegin(), _Fpol.cend() );
  for( unsigned j=0; j<_nF; j++ ){
    if( ( options.LINCTRSEP       && _Flin.find( j ) != _Flin.end() )   // exclude cut of linear function
     || ( options.REFORMMETH.count( Options::QUAD ) && Fsalg.find( j ) != Fsalg.end() ) // exclude cut of quadratic/polynomial function if quadratisation requested
     || ( options.SUBSETDRL == 1  && _Fgal.find( j ) != _Fgal.end() )   // exclude cut of non-polynomial function
     || ( options.SUBSETDRL == 2  && _Fgal.find( j ) == _Fgal.end() ) ) // exclude cut of polynomial function
      continue;
    Frel.insert( j );
  }
  if( Frel.empty() && Fsalg.empty() ) return;

  // Add polyhedral cuts for selected functions
  for( unsigned j : Frel ){
  //for( unsigned j=0; j<_nF; j++ ){
    try{
      _dag->eval( _Fops[j], _POLwk, 1, &_Fvar[j], &_POLFvar[j], _nX, _Xvar.data(), _POLXvar.data() );
      // Update bounds of intermediate factors from constraint propagation results
      if( options.CPMAX ){
        _dag->wkextract( _Fops[j], _Iwk, _Fallops, _CPbnd );
        for( unsigned i=0; i<_Iwk.size(); i++ ) _POLwk[i].update( _Iwk[i] );
      }
      // Generate cuts
      _POLenv.generate_cuts( 1, &_POLFvar[j], false );
    }
    catch(...){
      // No cut added for function #j in case DAG evaluation failed
      continue;
    }
  }

  // Add polyhedral cuts for quadratised polynomial functions
  if( options.REFORMMETH.count( Options::QUAD ) && !Fsalg.empty() ){
    // Set monomial vector from quadratic form into polyhedral image
    _set_mon_SQ();
    // Add cuts for quadratic form into polynomial image
    _set_cuts_SQ( Fsalg, false );
  }

#ifdef MC__MINLPBND_DEBUG_DRL
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_cuts_SCQ
()
{
  // Subset of functions to be relaxed
  std::set<unsigned> ndxF;
  for( unsigned j=0; j<_nF; j++ ){
    if( ( options.LINCTRSEP      && _Flin.find( j ) != _Flin.end() )   // exclude cut of linear function
     || ( options.SUBSETSCQ == 1 && _Fgal.find( j ) != _Fgal.end() )   // exclude cut of non-polynomial function
     || ( options.SUBSETSCQ == 2 && _Fgal.find( j ) == _Fgal.end() ) ) // exclude cut of polynomial function
      continue;
    ndxF.insert( j );
  }
  if( ndxF.empty() ) return;

  // Update sparse Chebyshev variable bounds
  for( unsigned i=0; i<_nX; i++ ){
    _CMXvar[i].set( _CMenv, i, _Xbnd[i] );
    if( _CMenv->scalvar()[i] <= options.CMODEL.MIN_FACTOR )
      _CMXvar[i] = _Xbnd[i];
  }

  // Compute sparse Chebyshev model for each nonlinear function
  const unsigned MAXORD = (!options.CMODCUTS || options.CMODCUTS>options.CMODPROP)?
                          options.CMODPROP: options.CMODCUTS;

  _CMFvar.assign( _nF, 0. );
  for( unsigned j : ndxF ){
    try{
      _dag->eval( _Fops[j], _CMwk, 1, &_Fvar[j], &_CMFvar[j], _nX, _Xvar.data(), _CMXvar.data() );
#ifdef MC__MINLPBND_DEBUG_SCQ
      std::cout << "Chebyshev model for function F[" << j << "]: " << _CMFvar[j];
#endif
      // Test for too large Chebyshev bounds or NaN
      if( !(Op<SCVar<T>>::diam(_CMFvar[j]) <= options.CMODDMAX) ) throw(0);

      // Simplify sparse Chebyshev model
      _CMFvar[j].simplify( options.CMODEL.MIN_FACTOR, MAXORD );
    }
    catch( int ecode ){
#ifdef MC__MINLPBND_DEBUG_SCQ
      std::cout << "Chebyshev model bound too weak!\n";
#endif
      T IFvarj;
      Op<T>::inter( IFvarj, _IINF, _CMFvar[j].B() );
      _CMFvar[j] = IFvarj;
    }
    catch(...){
#ifdef MC__MINLPBND_DEBUG_SCQ
      std::cout << "Chebyshev model for function F[" << j << "]: failed" << std::endl;
#endif
      // No cut added for constraint #j in case evaluation failed
      _CMFvar[j] = _IINF;
      continue;
    }
  }
 
  // Perform quadratisation of polynomial part of sparse Chebyshev models
  _SQenv.reset();
  _SQenv.options = options.SQUAD;
  SQuad::t_SPolyMonCoef coefmon;
  
  for( unsigned j : ndxF ){
    if( !(Op<SCVar<T>>::diam(_CMFvar[j]) < Op<T>::diam(_IINF)) )         // exclude unbounded constraints
      continue;
    switch( options.SQUAD.BASIS ){
     case SQuad::Options::MONOM:
     {
      auto scvmon = _CMFvar[j].to_monomial( options.MONSCALE, options.CMODEL.MIN_FACTOR, MAXORD );
      coefmon = scvmon.first;
      _CMFvar[j].R() += scvmon.second;
      break;
     }
     case SQuad::Options::CHEB:
      coefmon = _CMFvar[j].coefmon(); // already simplified earlier
      break;
    }
#ifndef MC__MINLPBND_DEBUG_SCQ
    _SQenv.process( coefmon, options.SQUAD.BASIS );
#else
    double viol = _SQenv.process( coefmon, options.SQUAD.BASIS, true );
    if( viol > 1e-15 ){
      std::cout << _CMFvar[j].display( coefmon, options.SQUAD.BASIS );
      std::cout << "violation: " << viol << std::endl;
      int dum; std::cout << "PAUSED --"; std::cin >> dum;
    }
#endif
  }
#ifdef MC__MINLPBND_DEBUG_SCQ
  std::cout << _SQenv;
 { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif

  // Set monomial vector from quadratic form into polyhedral image
  _set_mon_SCQ();

  // Add cuts for quadratic form into polynomial image
  _set_cuts_SQ( ndxF, true );

#ifdef MC__MINLPBND_DEBUG_SCQ
 std::cout << _POLenv;
 { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_mon_SQ
()
{
  // Add monomial vector of quadratic from to polyhedral image
  for( auto const& mon : _SQenv.SetMon() ){
    if( !mon.tord ) continue;
    
    // Insert monomial as auxiliary variable in polyhedral image
#ifdef MC__MINLPBND_DEBUG_SQ
    std::cout << "Current monomial: " << mon.display(options.SQUAD.BASIS) << std::endl;
#endif
    assert( _POLXmon.find( mon ) == _POLXmon.end() );

    // First-order monomials correspond to existing variables
    if( mon.tord == 1 ){
      auto ivar = mon.expr.cbegin()->first;
      _Xmon[mon]    = _Xvar[ivar];
      _POLXmon[mon] = _POLXvar[ivar];
#ifdef MC__MINLPBND_DEBUG_SQ
      std::cout << " " << _POLXvar[ivar]  << " (DAG: " << _Xvar[ivar] << "): "
                << _POLXvar[ivar].range() << ", " << _Xbnd[ivar] << std::endl;
#endif
    }
      
    // Add higher-order monomials to polyhedral image
    else{
      _POLXmon[mon].set( &_POLenv, _get_mon(mon), _bnd_mon(mon), true );
#ifdef MC__MINLPBND_DEBUG_SQ
      std::cout << " (" << mon.display(options.SQUAD.BASIS) << ") = "
                << _POLXmon[mon] << " (DAG: " << _POLXmon[mon].var() << "): "
                << _POLXmon[mon].range() << std::endl;
#endif
    }
  }

#ifdef MC__MINLPBND_DEBUG_SQ
  std::cout << "Monomial map:" << std::endl;
  for( auto const& [mon,polvar] : _POLXmon )
    std::cout << " " << mon.display(options.SQUAD.BASIS) << " == " << polvar
              << " (DAG: " << polvar.var() << ")" << std::endl;
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_mon_SCQ
()
{
  // Add monomial vector of quadratic from to polyhedral image
  for( auto const& mon : _SQenv.SetMon() ){
    if( !mon.tord ) continue;
    
    // Insert monomial as auxiliary variable in polyhedral image
#ifdef MC__MINLPBND_DEBUG_SCQ
    std::cout << "Current monomial: " << mon.display(options.SQUAD.BASIS) << std::endl;
#endif
    assert( _POLXmon.find( mon ) == _POLXmon.end() );
    switch( options.SQUAD.BASIS ){
      
     // Case of power monomials
     case SQuad::Options::MONOM:
      if( !options.MONSCALE ){
        if( mon.tord == 1 ){
          // non-scaled first-order monomials correspond to existing variables
          auto const& ivar = mon.expr.begin()->first;
          _Xmon[mon] = _Xvar[ivar];
          _POLXmon[mon] = _POLXvar[ivar];
#ifdef MC__MINLPBND_DEBUG_SCQ
          std::cout << " " << _POLXvar[ivar]  << " (DAG: " << _Xvar[ivar] << "): "
                    << _POLXvar[ivar].range() << ", " << _Xbnd[ivar] << std::endl;
#endif
        }
        else{
          // add unscaled power monomial to polyhedral image
          _POLXmon[mon].set( &_POLenv, _get_mon(mon), _bnd_mon(mon), true );
#ifdef MC__MINLPBND_DEBUG_SCQ
          std::cout << " (" << mon.display(options.SQUAD.BASIS) << ") = "
                    << _POLXmon[mon] << " (DAG: " << _POLXmon[mon].var() << "): "
                    << _POLXmon[mon].range() << std::endl;
#endif
        }
        continue; // Loop to next monomial in _SQenv.SetMon()
      }

      // add scaled power monomial to polyhedral image
      _POLXmon[mon] = PolVar<T>( &_POLenv, _get_mon(mon), (mon.gcexp()%2? T(-1e0,1e0): T(0e0,1e0)), true );
#ifdef MC__MINLPBND_DEBUG_SCQ
      std::cout << " (" << mon.display(options.SQUAD.BASIS) << ") = "
                << _POLXmon[mon] << " (DAG: " << _POLXmon[mon].var() << "): "
                << _POLXmon[mon].range() << std::endl;
#endif
      break;

     // Case of Chebyshev monomials
     case SQuad::Options::CHEB:
      // add Chebyshev monomial to polyhedral image
      _POLXmon[mon] = PolVar<T>( &_POLenv, _get_mon(mon), T(-1e0,1e0), true );
#ifdef MC__MINLPBND_DEBUG_SCQ
      std::cout << " (" << mon.display(options.SQUAD.BASIS) << ") = "
                << _POLXmon[mon] << " (DAG: " << _POLXmon[mon].var() << "): "
                << _POLXmon[mon].range() << std::endl;
#endif
      break;
    }

    // Add linear cut between degree 1 monomial and actual (unscaled) decision variable
    if( mon.tord == 1 ){
      auto const& ivar = mon.expr.begin()->first;
#ifndef MC__MINLPBND_DEBUG_SCQ
      _POLenv.add_cut( PolCut<T>::EQ, _CMenv->refvar()[ivar], _POLXvar[ivar],  1.,
                                      _POLXmon[mon], -_CMenv->scalvar()[ivar] );
#else
      auto cutX = _POLenv.add_cut( PolCut<T>::EQ, _CMenv->refvar()[ivar], _POLXvar[ivar],  1.,
                                                  _POLXmon[mon], -_CMenv->scalvar()[ivar] );
      std::cout << "Scaling cut for variable X[" << ivar << "]: " << **cutX << std::endl;
#endif
    }
  }

#ifdef MC__MINLPBND_DEBUG_SCQ
  std::cout << "Monomial map:" << std::endl;
  for( auto const& [mon,polvar] : _POLXmon )
    std::cout << " " << mon.display(options.SQUAD.BASIS) << " == " << polvar
              << " (DAG: " << polvar.var() << ")" << std::endl;
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_cuts_SQ
( std::set<unsigned> ndxF, bool const chkrem )
{
  // Add cuts for entries in MatFct
  auto itF = ndxF.begin();
  for( auto const& mat : _SQenv.MatFct() ){
    assert( itF != ndxF.end() );
    PolCut<T> *cutF1 = nullptr, *cutF2 = nullptr;
    if( !chkrem || Op<T>::diam(_CMFvar[*itF].R()) == 0. ){
      cutF1 = *_POLenv.add_cut( PolCut<T>::EQ, 0., _POLFvar[*itF], -1. );
    }
    else{
      cutF1 = *_POLenv.add_cut( PolCut<T>::LE, -Op<T>::l(_CMFvar[*itF].R()), _POLFvar[*itF], -1. );
      cutF2 = *_POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(_CMFvar[*itF].R()), _POLFvar[*itF], -1. );
    }
    // Separate quadratic term
    _add_to_cuts( _SQenv, mat, cutF1, cutF2 );
#ifdef MC__MINLPBND_DEBUG_SQ
    std::cout << "Main cuts for function F[" << *itF << "]: " << *cutF1 << std::endl;
    if( cutF2 )  std::cout << "                             " << *cutF2 << std::endl;
#endif
    ++itF;
  }

  // Check entries in MatRed
#ifdef MC__MINLPBND_DEBUG_SQ
  unsigned ired = 0;
#endif
  for( auto const& mat : _SQenv.MatRed() ){
    PolCut<T> *cutR = *_POLenv.add_cut( PolCut<T>::EQ, 0. );
    //_add_to_cuts( mat, cutR );
    _add_to_cuts( _SQenv, mat, cutR );
#ifdef MC__MINLPBND_DEBUG_SQ
    std::cout << "Reduction cuts #" << ++ired << ": " << *cutR << std::endl;
#endif
  }

  // Check entries in MatPSD
  if( options.PSDQUADCUTS ){
#ifdef MC__MINLPBND_DEBUG_SQ
    unsigned ipsd = 0;
#endif
    _SQenv.tighten( options.PSDQUADCUTS>1? true: false );
    for( auto const& mat : _SQenv.MatPSD() ){
      PolCut<T> *cutP = *_POLenv.add_cut( PolCut<T>::GE, 0. );
      _add_to_cuts( mat, cutP );
#ifdef MC__MINLPBND_DEBUG_SQ
      std::cout << "PSD cuts #" << ++ipsd << ": " << *cutP << std::endl;
#endif
    }
  }
}

template <typename T, typename MIP>
inline T
MINLPBND<T,MIP>::_bnd_mon
( SPolyMon const& mon )
const
{
  // compute (unscaled) power monomial bound
  T bndmon( 1e0 );
  for( auto const& [ivar,iord] : mon.expr ){
    switch( options.SQUAD.BASIS ){
     // Monomial basis
     case SQuad::Options::MONOM:
      bndmon *= Op<T>::pow( _POLXvar[ivar].range(), (int)iord );
      break;
     // Chebyshev basis
     case SQuad::Options::CHEB:
      bndmon *= _bnd_cheb( _POLXvar[ivar].range(), iord );
      break;
    }
  }
  return bndmon;
}

template <typename T, typename MIP>
inline FFVar const&
MINLPBND<T,MIP>::_get_mon
( SPolyMon const& mon )
{
  auto itXmapmon = _Xmon.find( mon );
  if( itXmapmon != _Xmon.end() ) return itXmapmon->second;

  FFVar Xmon( _dag );
  auto itXmon = _dag->Vars().find( &Xmon );
  _Xmon[mon] = **itXmon;
  return **itXmon;
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_add_to_cuts
( SQuad const& _SQenv, SQuad::t_SQuad const& mat, PolCut<T>* cut1, PolCut<T>* cut2 )
{
  // DC decomposition not required
    if( !options.DCQUADCUTS )
      return _add_to_cuts( mat, cut1, cut2 );

  // DC decomposition required
  for( auto const& matsep : _SQenv.separate( mat ) ){
  
    // Append monomial if single term
    if( matsep.size() == 1 ){
      _add_to_cuts( matsep, cut1, cut2 );
      continue;
    }

    // Introduce auxiliary variable in cuts and new auxiliary cut
    PolVar<T> POLsep( &_POLenv, _get_range( matsep ), true );
    if( cut1 ) cut1->append( POLsep, 1. );
    if( cut2 ) cut2->append( POLsep, 1. );
    PolCut<T> *cutR = *_POLenv.add_cut( PolCut<T>::EQ, 0., POLsep, -1. );
    _add_to_cuts( matsep, cutR );

    // Introduce auxiliary cuts for DC factorization
    PolCut<T> *cutDC = *_POLenv.add_cut( PolCut<T>::EQ, 0., POLsep, -1. );
    for( auto const& [eigval,eigterm] : _SQenv.factorize( matsep ) ){
      auto const& POLEVSQ = _append_cuts_dcdec( eigterm );
      cutDC->append( POLEVSQ, eigval );
    }
  }
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_add_to_cuts
( SQuad::t_SQuad const& mat, PolCut<T>* cut1, PolCut<T>* cut2 )
{
  for( auto const& [ijmon,coef] : mat ){
    // Constant term
    if( !ijmon.first->tord && !ijmon.second->tord ){
      if( cut1 ) cut1->rhs() -= coef;
      if( cut2 ) cut2->rhs() -= coef;
    }
    // Linear term
    else if( !ijmon.first->tord ){
      if( cut1 ) cut1->append( _POLXmon[*ijmon.second], coef );
      if( cut2 ) cut2->append( _POLXmon[*ijmon.second], coef );
    }
    else if( !ijmon.second->tord ){
      if( cut1 ) cut1->append( _POLXmon[*ijmon.first], coef );
      if( cut2 ) cut2->append( _POLXmon[*ijmon.first], coef );
    }
    // Quadratic term
    else if( options.POLIMG.AGGREG_LQ && !options.POLIMG.RELAX_QUAD ){
      if( cut1 ) cut1->append( _POLXmon[*ijmon.first], _POLXmon[*ijmon.second], coef );
      if( cut2 ) cut2->append( _POLXmon[*ijmon.first], _POLXmon[*ijmon.second], coef );
    }
    else{
      auto POLprod = _append_cuts_monprod( ijmon );
      if( cut1 ) cut1->append( POLprod, coef );
      if( cut2 ) cut2->append( POLprod, coef );
    }
  }
}

template <typename T, typename MIP>
inline T
MINLPBND<T,MIP>::_get_range
( SQuad::t_SQuad const& mat )
{
  T range = 0.;
  for( auto const& [ijmon,coef] : mat ){
    // Constant term
    if( !ijmon.first->tord && !ijmon.second->tord )
      range += coef;
    // Linear term
    else if( !ijmon.first->tord )
      range += _POLXmon[*ijmon.second].range() * coef;
    else if( !ijmon.second->tord )
      range += _POLXmon[*ijmon.first].range() * coef;
    // Quadratic term
    else
      range += _POLXmon[*ijmon.first].range() * _POLXmon[*ijmon.second].range() * coef;
  }
  return range;
}

template <typename T, typename MIP>
inline PolVar<T>
MINLPBND<T,MIP>::_append_cuts_dcdec
( SQuad::t_SPolyMonCoef const& eigterm )
{
  // New auxiliary variable and cut for linear combition of monomials
  PolCut<T> *cutEV = *_POLenv.add_cut( PolCut<T>::EQ, 0. );
  T rangeEV = 0.;
  for( auto const& [mon,coef] : eigterm ){
    if( !mon.tord ){
      rangeEV += coef;
      cutEV->rhs() -= coef;
      continue;
    }
    auto const& POLmon = _POLXmon[mon];
    rangeEV += POLmon.range() * coef;
    cutEV->append( POLmon, coef );
  }
  PolVar<T> POLEV( &_POLenv, rangeEV, true );
  cutEV->append( POLEV, -1. );

  // New auxiliary variable and cut for square term
  PolVar<T> POLEVSQ( &_POLenv, Op<T>::sqr(rangeEV), true );
  _POLenv.append_cuts_SQR( POLEVSQ, POLEV );
  return POLEVSQ;
}

template <typename T, typename MIP>
inline PolVar<T>
MINLPBND<T,MIP>::_append_cuts_monprod
( SQuad::key_SQuad const& ijmon )
{
  // Seach for pair ijmon in _POLXprodmon
  auto itijmon = _POLXprodmon.find( ijmon );

  // Append the pair if non-existent
  if( itijmon == _POLXprodmon.end() ){
    if( ijmon.first == ijmon.second ){
      auto const& Xmon1 = _POLXmon[*ijmon.first];
      PolVar<T> POLprod( &_POLenv, Op<T>::sqr( Xmon1.range() ), true );
      _POLenv.append_cuts_SQR( POLprod, Xmon1 );
      itijmon = ( _POLXprodmon.insert( std::make_pair( ijmon, POLprod ) ) ).first;
#if defined( MC__MINLPBND_DEBUG_SQ ) || defined( MC__MINLPBND_DEBUG_SCQ )
      //std::cout << POLprod << std::endl;
      std::cout << "Auxiliary variable " << itijmon->second << ": " << Xmon1 << "^2" << std::endl;
#endif
    }
    else{
      auto const& Xmon1 = _POLXmon[*ijmon.first];
      auto const& Xmon2 = _POLXmon[*ijmon.second];
      PolVar<T> POLprod( &_POLenv, Xmon1.range() * Xmon2.range(), true );
      _POLenv.append_cuts_TIMES( POLprod, Xmon1, Xmon2 );
      itijmon = ( _POLXprodmon.insert( std::make_pair( ijmon, POLprod ) ) ).first;
#if defined( MC__MINLPBND_DEBUG_SQ ) || defined( MC__MINLPBND_DEBUG_SCQ )
      //std::cout << POLprod << std::endl;
      std::cout << "Auxiliary variable " << itijmon->second << ": " << Xmon1 << "·" << Xmon2 << std::endl;
#endif
    }
  }

  return itijmon->second;
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_cuts_SCDRL
()
{
//  // reset bases maps
//  const unsigned basisord = (!options.CMODCUTS || options.CMODCUTS>options.CMODPROP)?
//                            options.CMODPROP: options.CMODCUTS;
//  _Xmon.clear();
//  _POLXmon.clear();

//  // Add cuts for variable scaling and basis polynomials (up to order options.CMODCUT)
//  // TEST FOR PARTICIPATING VARIABLES FIRST?!?
//  // if( _var_lin.find( ip ) != _var_lin.end() ) continue; // only nonlinearly participating variables
//  auto itv = _POLscalvar.begin();
//  for( unsigned i=0; itv!=_POLscalvar.end(); ++itv, i++ ){
//    if( _CMenv->scalvar()[i] > options.CMODEL.MIN_FACTOR )
//      _POLenv.add_cut( PolCut<T>::EQ, _CMenv->refvar()[i], _POLvar[i], 1., *itv, -_CMenv->scalvar()[i] );
//    //else{
//    //  _POLenv.add_cut( PolCut<T>::GE, _CMenv->refvar()[i]-_CMenv->scalvar()[i], _POLvar[i], 1. );
//    //  _POLenv.add_cut( PolCut<T>::LE, _CMenv->refvar()[i]+_CMenv->scalvar()[i], _POLvar[i], 1. );
//    //}
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "Variable #" << i << ": " << _CMvar[i];
//#endif
//  }

//  // Evaluate Chebyshev model for (nonlinear) objective and keep track of participating monomials
//  auto ito = std::get<0>(_obj).begin();
//  for( unsigned j=0; ito!=std::get<0>(_obj).end() && j<1; ++ito, j++ ){
//    if( feastest || _fct_lin.find(std::get<1>(_obj).data()) != _fct_lin.end() ) continue; // only nonlinear objectives
//    T Iobj(-SBB<T>::INF,SBB<T>::INF);
//    try{
//      // Polynomial model evaluation
//      _dag->eval( _op_f, _CMwk, 1, std::get<1>(_obj).data(), &_CMobj,
//                  _nvar, _var.data(), _CMvar.data() );
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "Objective remainder: " << _CMobj.R() << std::endl;
//      std::cout << _CMobj;
//      //if(_p_inc.data()) std::cout << _CMobj.P(_p_inc.data())+_CMobj.R() << std::endl;
//#endif
//      // Test for too large Chebyshev bounds or NaN
//      if( !(Op<SCVar<T>>::diam(_CMobj) <= options.CMODDMAX) ) throw(0);

//      // Keep track of participating Chebyshev monomials into '_Xmon'
//      auto first_CMobj = _CMobj.coefmon().lower_bound( std::make_pair( 1, std::map<unsigned,unsigned>() ) );
//      auto last_CMobj  = _CMobj.coefmon().lower_bound( std::make_pair( basisord+1, std::map<unsigned,unsigned>() ) );
//      for( auto it=first_CMobj; it!=last_CMobj; ++it )
//        _Xmon.insert( std::make_pair( it->first, FFVar() ) );
//    }
//    catch( int ecode ){
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "Objective: " << "cut too weak!\n";
//#endif
//      Op<T>::inter( Iobj, Iobj, _CMobj.B() );
//      _CMobj = Iobj;
//    }
//    catch(...){
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "Objective: " << "cut failed!\n";
//#endif
//      //Op<T>::inter( Iobj, Iobj, _CMobj.B() );
//      _CMobj = Iobj;
//      // No cut added for objective in case evaluation failed
//    }
//  }

//  // Evaluate Chebyshev model for (nonlinear) constraints and keep track of participating monomials
//  _CMctr.resize( _nctr );
//  auto itc = std::get<0>(_ctr).begin();
//  for( unsigned j=0; itc!=std::get<0>(_ctr).end(); ++itc, j++ ){
//    if( _fct_lin.find(std::get<1>(_ctr).data()+j) != _fct_lin.end() ) continue;
//    T Ictr(-SBB<T>::INF,SBB<T>::INF);
//    try{
//      // Polynomial model evaluation
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      _dag->output( _op_g[j] );
//#endif
//      _dag->eval( _op_g[j], _CMwk, 1, std::get<1>(_ctr).data()+j, _CMctr.data()+j,
//                  _nvar, _var.data(), _CMvar.data() );
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "\nConstraint #" << j << ": " << _CMctr[j];
//      //if(_p_inc.data()) std::cout << "optim: " << _CMctr[j].P(_p_inc.data())+_CMctr[j].R() << std::endl;
//      //{ int dum; std::cout << "PAUSED"; std::cin >> dum; }
//#endif
//      // Test for too large Chebyshev bounds or NaN
//      if( !(Op<SCVar<T>>::diam(_CMctr[j]) <= options.CMODDMAX) ) throw(0);

//      // Keep track of participating Chebyshev monomials into '_Xmon'
//      auto first_CMctr = _CMctr[j].coefmon().lower_bound( std::make_pair( 1, std::map<unsigned,unsigned>() ) );
//      auto last_CMctr  = _CMctr[j].coefmon().lower_bound( std::make_pair( basisord+1, std::map<unsigned,unsigned>() ) );
//      for( auto it=first_CMctr; it!=last_CMctr; ++it )
//        _Xmon.insert( std::make_pair( it->first, FFVar() ) );
//   }
//    catch( int ecode ){
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "Constraint #" << j << ": " << "cut too weak!\n";
//#endif
//      Op<T>::inter( Ictr, Ictr, _CMctr[j].B() );
//      _CMctr[j] = Ictr;
//    }
//    catch(...){
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "Constraint #" << j << ": " << "cut failed!\n";
//#endif
//      //Op<T>::inter( Ictr, Ictr, _CMctr[j].B() );
//      _CMctr[j] = Ictr;
//      // No cut added for constraint #j in case evaluation failed
//    }
//  }

//  // Keep track of participating monomials in Chebyshev model for dependents
//  for( auto it=_CMndxdep.begin(); options.CMODDEPS && options.CMODRED == OPT::APPEND && it!=_CMndxdep.end(); ++it ){
//    const unsigned irdep = _var_fperm[*it]-_nrvar;
//    T Irdep(-SBB<T>::INF,SBB<T>::INF);
//    try{
//      // Test for too large Chebyshev bounds or NaN
//      if( !(Op<SCVar<T>>::diam(_CMrdep[irdep]) <= options.CMODDMAX) ) throw(0);
//      // Keep track of participating Chebyshev monomials into '_Xmon'
//      auto first_CMrdep = _CMrdep[irdep].coefmon().lower_bound( std::make_pair( 1, std::map<unsigned,unsigned>() ) );
//      auto last_CMrdep  = _CMrdep[irdep].coefmon().lower_bound( std::make_pair( basisord+1, std::map<unsigned,unsigned>() ) );
//      for( auto jt=first_CMrdep; jt!=last_CMrdep; ++jt ){
//        t_expmon expmon_CMrdep;
//        expmon_CMrdep.first = jt->first.first;
//        // Match indices between the full and reduced Chebyshev models
//        for( auto ie = jt->first.second.begin(); ie!=jt->first.second.end(); ++ie )
//          expmon_CMrdep.second.insert( std::make_pair( _var_rperm[ie->first], ie->second ) );
//        _Xmon.insert( std::make_pair( expmon_CMrdep, FFVar() ) );
//      }
//    }
//    catch(...){
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//      std::cout << "Dependent #" << irdep << ": " << "cut failed!\n";
//#endif
//      Op<T>::inter( Irdep, Irdep, _CMrdep[irdep].B() );
//      _CMrdep[irdep] = Irdep; // <- more efficient to interesect with dependent bounds?
//      // No cut added for constraint #j in case evaluation failed
//    }
//  }

//  // Populate '_Xmon' with references to the Chebyshev monomials in the DAG
//  _CMenv->get_bndmon( _Xmon, _scalvar.data(), true );
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//  std::cout << "\nBASIS:\n";
//  for( auto it=_Xmon.begin(); it!=_Xmon.end(); ++it ){
//    std::cout << it->second << " = ";
//    for( auto ie=it->first.second.begin(); ie!=it->first.second.end(); ++ie )
//      std::cout << "T" << ie->second << "[" << ie->first << "] ";
//    std::cout << std::endl;
//  }
//  std::list<const mc::FFOp*> op_Xmon  = _dag->subgraph( _Xmon );
//  _dag->output( op_Xmon );
//  { int dum; std::cin >> dum; }
//#endif

//  // Populate '_POLXmon' with references to the Chebyshev monomials in the polyhedral relaxation
//  _dag->eval( _POLwk, _Xmon, _POLXmon, _scalvar.size(), _scalvar.data(), _POLscalvar.data() );

//  // Append cuts for the Chebyshev monomials in the polyhedral relaxation
//  _POLenv.generate_cuts( _POLXmon, false );

//  // Add Chebyshev-derived cuts for objective
//  ito = std::get<0>(_obj).begin();
//  for( unsigned j=0; ito!=std::get<0>(_obj).end() && j<1; ++ito, j++ ){
//    if( feastest
//     || _fct_lin.find(std::get<1>(_obj).data()) != _fct_lin.end()
//     || !(Op<SCVar<T>>::diam(_CMobj) < SBB<T>::INF) ) continue; // only nonlinear / finite objectives

//    // Constant, variable and bound on objective model
//    T Robj = _CMobj.bndord( basisord+1 ) + _CMobj.remainder();
//    const double a0 = (_CMobj.coefmon().empty() || _CMobj.coefmon().begin()->first.first)?
//                      0.: _CMobj.coefmon().begin()->second;
//    auto first_CMobj = _CMobj.coefmon().lower_bound( std::make_pair( 1, std::map<unsigned,unsigned>() ) );
//    auto last_CMobj  = _CMobj.coefmon().lower_bound( std::make_pair( basisord+1, std::map<unsigned,unsigned>() ) );
//    t_coefmon Cobj; Cobj.insert( first_CMobj, last_CMobj );

//    // Linear objective sparse cut
//    switch( std::get<0>(_obj)[0] ){
//      case MIN: _POLenv.add_cut( PolCut<T>::LE, -Op<T>::l(Robj)-a0, _POLXmon, Cobj, _POLobjaux, -1. ); break;
//      case MAX: _POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(Robj)-a0, _POLXmon, Cobj, _POLobjaux, -1. ); break;
//    }
//    _POLobjaux.update( _CMobj.bound(), true );

//  }
//#ifdef MC__NLGO_DEBUG
//  std::cout << _POLenv;
//#endif

//  // Add Chebyshev-derived cuts for constraints
//  itc = std::get<0>(_ctr).begin();
//  for( unsigned j=0; itc!=std::get<0>(_ctr).end(); ++itc, j++ ){
//    if( _fct_lin.find(std::get<1>(_ctr).data()+j) != _fct_lin.end()
//     || !(Op<SCVar<T>>::diam(_CMctr[j]) < SBB<T>::INF) ) continue; // only nonlinear / finite constraints

//    // Constant, variable and bound on constraint model
//    T Rctr = _CMctr[j].bndord( basisord+1 ) + _CMctr[j].remainder();
//    const double a0 = (_CMctr[j].coefmon().empty() || _CMctr[j].coefmon().begin()->first.first)?
//                      0.: _CMctr[j].coefmon().begin()->second;
//    auto first_CMctr = _CMctr[j].coefmon().lower_bound( std::make_pair( 1, std::map<unsigned,unsigned>() ) );
//    auto last_CMctr  = _CMctr[j].coefmon().lower_bound( std::make_pair( basisord+1, std::map<unsigned,unsigned>() ) );
//    t_coefmon Cctr; Cctr.insert( first_CMctr, last_CMctr );

//    // Nonlinear constraint sparse cut
//    switch( (*itc) ){
//      case EQ: if( !feastest ){ if( !Cctr.empty() ) _POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(Rctr)-a0, _POLXmon, Cctr ); }// no break
//               else { _POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(Rctr)-a0, _POLXmon, Cctr, _POLobjaux,  1. ); }// no break
//      case LE: if( !feastest ){ if( !Cctr.empty() ) _POLenv.add_cut( PolCut<T>::LE, -Op<T>::l(Rctr)-a0, _POLXmon, Cctr ); break; }
//               else { _POLenv.add_cut( PolCut<T>::LE, -Op<T>::l(Rctr)-a0, _POLXmon, Cctr, _POLobjaux, -1. ); break; }
//      case GE: if( !feastest ){ if( !Cctr.empty() ) _POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(Rctr)-a0, _POLXmon, Cctr ); break; }
//               else { _POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(Rctr)-a0, _POLXmon, Cctr, _POLobjaux,  1. ); break; }
//    }
//  }

//  // Add Chebyshev-derived cuts for dependents
//  for( auto it=_CMndxdep.begin(); options.CMODDEPS && options.CMODRED == OPT::APPEND && it!=_CMndxdep.end(); ++it ){
//    // Constant, variable and bound on constraint model
//    const unsigned irdep = _var_fperm[*it]-_nrvar;
//#ifdef MC__NLGO_CHEBCUTS_DEBUG
//    std::cout << "\nDependent #" << irdep << ": " << _CMrdep[irdep];
//#endif
//    T Rrdep = _CMrdep[irdep].bndord( basisord+1 ) + _CMrdep[irdep].remainder();
//    const double a0 = (_CMrdep[irdep].coefmon().empty() || _CMrdep[irdep].coefmon().begin()->first.first)?
//                      0.: _CMrdep[irdep].coefmon().begin()->second;
//    auto first_CMrdep = _CMrdep[irdep].coefmon().lower_bound( std::make_pair( 1, std::map<unsigned,unsigned>() ) );
//    auto last_CMrdep  = _CMrdep[irdep].coefmon().lower_bound( std::make_pair( basisord+1, std::map<unsigned,unsigned>() ) );
//    t_coefmon Crdep; //Crdep.insert( first_CMrdep, last_CMrdep );
//    for( auto jt=first_CMrdep; jt!=last_CMrdep; ++jt ){
//      t_expmon expmon_CMrdep;
//      expmon_CMrdep.first = jt->first.first;
//      // Match indices between the full and reduced Chebyshev models
//      for( auto ie = jt->first.second.begin(); ie!=jt->first.second.end(); ++ie )
//        expmon_CMrdep.second.insert( std::make_pair( _var_rperm[ie->first], ie->second ) );
//      Crdep.insert( std::make_pair( expmon_CMrdep, jt->second ) );
//    }

//    // Dependent sparse cut
//    if( !feastest ){
//      _POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(Rrdep)-a0, _POLXmon, Crdep, _POLvar[*it],  -1. );
//      _POLenv.add_cut( PolCut<T>::LE, -Op<T>::l(Rrdep)-a0, _POLXmon, Crdep, _POLvar[*it],  -1. );
//    }
//    else{
//      _POLenv.add_cut( PolCut<T>::GE, -Op<T>::u(Rrdep)-a0, _POLXmon, Crdep, _POLvar[*it],  -1., _POLobjaux,  1. );
//      _POLenv.add_cut( PolCut<T>::LE, -Op<T>::l(Rrdep)-a0, _POLXmon, Crdep, _POLvar[*it],  -1., _POLobjaux, -1. );
//    }
//  }

#ifdef MC__MINLPBND_CHEBCUTS_DEBUG
  std::cout << _POLenv;
  { int dum; std::cin >> dum; }
#endif
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::_set_cuts_ISM
()
{
//  // Auxiliary variables in polyhedral image are defined locally 
//  std::vector<std::vector<PolVar<T>>> POL_ISMaux( _nvar );
//  std::vector<double> DL_ISMaux( _ISMenv->ndiv() );
//  std::vector<double> DU_ISMaux( _ISMenv->ndiv() );

//  // Add polyhedral cuts for objective - by-pass if feasibility test or no objective function defined
//  auto ito = std::get<0>(_obj).begin();
//  for( unsigned j=0; ito!=std::get<0>(_obj).end() && j<1; ++ito, j++ ){
//    if( feastest || _fct_lin.find(std::get<1>(_obj).data()) != _fct_lin.end() ) continue;
//    try{
//#ifdef MC__NLGO_SUPCUTS_DEBUG
//      std::cout << "objective" << std::endl;
//#endif
//      // Interval superposition evaluation
//      _dag->eval( _op_f, _ISMwk, 1, std::get<1>(_obj).data(), &_ISMobj, _nvar, _var.data(), _ISMvar.data() );
//#ifdef MC__NLGO_SUPCUTS_DEBUG
//      std::cout << "IAM of objective:\n" << _ISMobj;
//      //_dag->output( _op_f );
//#endif
//      _POLobj.set( &_POLenv, std::get<1>(_obj)[0], _ISMobj.B(), true );
//      // Polyhedral cut generation
//      const double rhs = (_ISMobj.ndep()? 0.: -_ISMobj.cst());
//      auto CutObjL = _POLenv.add_cut( PolCut<T>::LE, rhs, _POLobj, -1. );
//      auto CutObjU = _POLenv.add_cut( PolCut<T>::GE, rhs, _POLobj, -1. );
//      for( unsigned ivar=0; ivar<_nvar; ivar++ ){
//        auto&& vec = _ISMobj.C()[ivar];
//        if( vec.empty() ) continue;
//        if( POL_ISMaux[ivar].empty() ){
//          POL_ISMaux[ivar].resize( _ISMenv->ndiv() );
//          for( unsigned jsub=0; jsub<_ISMenv->ndiv(); jsub++ )
//            POL_ISMaux[ivar][jsub].set( &_POLenv, Op<T>::zeroone(), !options.ISMMIPREL );
//        }
//        for( unsigned jsub=0; jsub<_ISMenv->ndiv(); jsub++ ){
//          DL_ISMaux[jsub] = Op<T>::l(vec[jsub]);
//          DU_ISMaux[jsub] = Op<T>::u(vec[jsub]);
//        }
//        (*CutObjL)->append( _ISMenv->ndiv(), POL_ISMaux[ivar].data(), DL_ISMaux.data() );
//        (*CutObjU)->append( _ISMenv->ndiv(), POL_ISMaux[ivar].data(), DU_ISMaux.data() );
//      }
//      switch( std::get<0>(_obj)[0] ){
//        case MIN: _POLenv.add_cut( PolCut<T>::GE, 0., _POLobjaux, 1., _POLobj, -1. ); break;
//        case MAX: _POLenv.add_cut( PolCut<T>::LE, 0., _POLobjaux, 1., _POLobj, -1. ); break;
//      }
//    }
//    catch(...){
//      // No cut added for objective in case evaluation failed
//#ifdef MC__NLGO_SUPCUTS_DEBUG
//      std::cout << "evaluation failed" << std::endl;
//#endif
//    }
//  }

//  // Add polyhedral cuts for constraints - add slack to inequality constraints if feasibility test
//  _POLctr.resize( _nctr );
//  _ISMctr.resize( _nctr );
//  auto itc = std::get<0>(_ctr).begin();
//  for( unsigned j=0; itc!=std::get<0>(_ctr).end(); ++itc, j++ ){
//    if( _fct_lin.find(std::get<1>(_ctr).data()+j) != _fct_lin.end() ) continue;
//    try{
//      // Interval superposition evaluation
//      _dag->eval( _op_g[j], _ISMwk, 1, std::get<1>(_ctr).data()+j, _ISMctr.data()+j, _nvar, _var.data(), _ISMvar.data() );
//#ifdef MC__NLGO_SUPCUTS_DEBUG
//      std::cout << "ISM of constraint #" << j << std::endl << _ISMctr[j];
//      //_dag->output( _op_g[j] );
//#endif
//      _POLctr[j].set( &_POLenv, std::get<1>(_ctr)[j], _ISMctr[j].B(), true );
//      // Polyhedral cut generation
//      const double rhs = (_ISMobj.ndep()? 0.: -_ISMctr[j].cst());
//      auto CutCtrL = _POLenv.add_cut( PolCut<T>::LE, rhs, _POLctr[j], -1. );
//      auto CutCtrU = _POLenv.add_cut( PolCut<T>::GE, rhs, _POLctr[j], -1. );
//      for( unsigned ivar=0; ivar<_nvar; ivar++ ){
//        auto&& vec = _ISMctr[j].C()[ivar];
//        if( vec.empty() ) continue;
//        if( POL_ISMaux[ivar].empty() ){
//          POL_ISMaux[ivar].resize( _ISMenv->ndiv() );
//          for( unsigned jsub=0; jsub<_ISMenv->ndiv(); jsub++ )
//            POL_ISMaux[ivar][jsub].set( &_POLenv, Op<T>::zeroone(), !options.ISMMIPREL );
//        }
//        for( unsigned jsub=0; jsub<_ISMenv->ndiv(); jsub++ ){
//          DL_ISMaux[jsub] = Op<T>::l(vec[jsub]);
//          DU_ISMaux[jsub] = Op<T>::u(vec[jsub]);
//        }
//        (*CutCtrL)->append( _ISMenv->ndiv(), POL_ISMaux[ivar].data(), DL_ISMaux.data() );
//        (*CutCtrU)->append( _ISMenv->ndiv(), POL_ISMaux[ivar].data(), DU_ISMaux.data() );
//      }
//      switch( (*itc) ){
//        case EQ: if( !feastest ){ _POLenv.add_cut( PolCut<T>::EQ, 0., _POLctr[j], 1. ); break; }
//                 else             _POLenv.add_cut( PolCut<T>::GE, 0., _POLctr[j], 1., _POLobjaux,  1. ); // no break
//        case LE: if( !feastest ){ _POLenv.add_cut( PolCut<T>::LE, 0., _POLctr[j], 1. ); break; }
//                 else           { _POLenv.add_cut( PolCut<T>::LE, 0., _POLctr[j], 1., _POLobjaux, -1. ); break; }
//        case GE: if( !feastest ){ _POLenv.add_cut( PolCut<T>::GE, 0., _POLctr[j], 1. ); break; }
//                 else           { _POLenv.add_cut( PolCut<T>::GE, 0., _POLctr[j], 1., _POLobjaux,  1. ); break; }
//      }
//    }
//    catch(...){
//      // No cut added for constraint #j in case evaluation failed
//#ifdef MC__NLGO_SUPCUTS_DEBUG
//      std::cout << "evaluation failed" << std::endl;
//#endif
//      continue;
//    }
//  }
//  
//  // Add polyhedral cuts for ISM-participating variables
//  for( unsigned ivar=0; ivar<_nvar; ivar++ ){
//    if( POL_ISMaux[ivar].empty() ) continue;
//    // Auxiliaries add up to 1
//    for( unsigned jsub=0; jsub<_ISMenv->ndiv(); jsub++ )
//      DL_ISMaux[jsub] = 1.;
//    _POLenv.add_cut( PolCut<T>::EQ, 1., _ISMenv->ndiv(), POL_ISMaux[ivar].data(), DL_ISMaux.data() );
//    // Relationship between variables and auxiliaries
//    PolVar<T> POLvarL( 0. ), POLvarU( 0. );
//    auto&& vec = _ISMvar[ivar].C()[ivar];
//#ifdef MC__NLGO_SUPCUTS_DEBUG
//    assert( !vec.empty() );
//#endif
//    for( unsigned jsub=0; jsub<_ISMenv->ndiv(); jsub++ ){
//      DL_ISMaux[jsub] = Op<T>::l(vec[jsub]);
//      DU_ISMaux[jsub] = Op<T>::u(vec[jsub]);
//    }
//    _POLenv.add_cut( PolCut<T>::LE, 0., _ISMenv->ndiv(), POL_ISMaux[ivar].data(), DL_ISMaux.data(), _POLvar[ivar], -1. );
//    _POLenv.add_cut( PolCut<T>::GE, 0., _ISMenv->ndiv(), POL_ISMaux[ivar].data(), DU_ISMaux.data(), _POLvar[ivar], -1. );
//  }

#ifdef MC__CSEARCH_DEBUG
  std::cout << _POLenv;
#endif
}

template <typename T, typename MIP>
template <typename U>
inline double
MINLPBND<T,MIP>::_dH
( const U&X, const U&Y )
{
  return std::max( std::fabs(Op<U>::l(X)-Op<U>::l(Y)),
                   std::fabs(Op<U>::u(X)-Op<U>::u(Y)) );
}

template <typename T, typename MIP>
template <typename U>
inline double
MINLPBND<T,MIP>::_reducrel
( const unsigned n, const U*Xred, const U*X )
{
  double drel = 0.;
  for( unsigned ip=0; ip<n; ip++ )
    drel = std::max( drel, _dH( Xred[ip], X[ip] ) / Op<T>::diam( X[ip] ) );
  return drel;
}

template <typename T, typename MIP>
template <typename U>
inline double
MINLPBND<T,MIP>::_reducrel
( const unsigned n, const U*Xred, const U*X, const U*X0 )
{
  double drel = 0.;
  for( unsigned ip=0; ip<n; ip++ )
    drel = std::max( drel, _dH( Xred[ip], X[ip] ) / Op<T>::diam( X0[ip] ) );
  return drel;
}

template <typename T, typename MIP>
inline void
MINLPBND<T,MIP>::Options::display
( std::ostream&out ) const
{
  // Display MINLPBND Options
  out << std::left;
  out << std::setw(60) << "  POLYHEDRAL RELAXATION APPROACH" << "[";
  for( auto && meth : RELAXMETH ){
   switch( meth ){
    case DRL:    out << " DRL";    break;
    case SCDRL:  out << " SCDRL";  break;
    case SCQ:    out << " SCQ";    break;
    case ISM:    out << " ISM";    break;
   }
  }
  out << " ]" << std::endl;
  out << std::setw(60) << "  ORDER OF CHEBYSHEV MODEL PROPAGATION";
  switch( CMODPROP ){
   case 0:  out << "-\n"; break;
   default: out << CMODPROP << std::endl; break;
  }
  out << std::setw(60) << "  ORDER OF CHEBYSHEV-DERIVED CUTS";
  if( !CMODCUTS)
    switch( CMODPROP ){
     case 0:  out << "-\n"; break;
     default: out << CMODPROP << std::endl; break;
    }
  else 
    switch( CMODCUTS ){
     case 0:  out << "-\n"; break;
     default: out << std::min(CMODPROP,CMODCUTS) << std::endl; break;
    }
  out << std::setw(60) << "  APPEND NCO CUTS"
      << (NCOCUTS?"Y\n":"N\n");
  if( CMODCUTS ){
    out << std::setw(60) << "  METHOD FOR NCO CUTS";
    switch( NCOADIFF ){
     case FSA: out << "FSA\n";
     case ASA: out << "ASA\n";
    }
  }
  out << std::setw(60) << "  MAXIMUM OPTIMIZATION-BASED REDUCTION LOOPS"
      << OBBTMAX << std::endl;
  out << std::setw(60) << "  THRESHOLD FOR OPTIMIZATION-BASED REDUCTION LOOP"
      << std::fixed << std::setprecision(0)
      << OBBTTHRES*1e2 << "%\n";
  out << std::setw(60) << "  BACKOFF FOR OPTIMIZATION-BASED REDUCTION"
      << std::scientific << std::setprecision(1)
      << OBBTBKOFF << std::endl;
  out << std::setw(60) << "  MAXIMUM CONSTRAINT PROPAGATION LOOPS"
      << CPMAX << std::endl;
  out << std::setw(60) << "  THRESHOLD FOR CONSTRAINT PROPAGATION LOOP"
      << std::fixed << std::setprecision(0)
      << CPTHRES*1e2 << "%\n";
  out << std::setw(60) << "  MAXIMUM CPU TIME (SEC)"
      << std::scientific << std::setprecision(1)
      << TIMELIMIT << std::endl;
  out << std::setw(60) << "  DISPLAY LEVEL"
      << DISPLEVEL << std::endl;
}

} // end namescape mc

#endif

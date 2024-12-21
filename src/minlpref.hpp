// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MINLPREF Reformulating Factorable Mixed-Integer Nonlinear Programs using MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 2.0
\date 2023
\bug No known bugs.

Consider a nonlinear optimization problem in the form:
\f{align*}
\mathcal{P}:\quad & \min_{x_1,\ldots,x_n}\ f(x_1,\ldots,x_n)\\
& {\rm s.t.}\ \ g_j(x_1,\ldots,x_n)\ \leq,=,\geq\ 0,\ \ j=1,\ldots,m\\
& \qquad x_i^L\leq x_i\leq x_i^U,\ \ i=1,\ldots,n\,,
\f}
where \f$f, g_1, \ldots, g_m\f$ are factorable, possibly nonlinear, real-valued functions; and \f$x_1, \ldots, x_n\f$ can be either continuous or integer decision variables. The class mc::MINLPREF enables reformulation of such (MI)NLP problems, both lifted and reduced-space formulations, using set arithmetics and optimization-based tools available in <A href="https://projects.coin-or.org/MCpp">MC++</A>.

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

Start by instantiating an mc::MINLPBND class object, which is defined in the header file <tt>minlpbnd.hpp</tt>:

\code
  mc::MINLPBND MINLP;
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

Finally, set up the MINLP model relaxation and solve it using:

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

Other options can be modified to tailor the relaxations, tune the MIP solver, set a maximum CPU time, etc. These options can be modified through the public member mc::MINLPREF::options.
*/

//- Enable RLT cuts in addition to quadratization

#ifndef MC__MINLPREF_HPP
#define MC__MINLPREF_HPP

#include <stdexcept>
#include <chrono>
#include <functional>

#include "squad.hpp"
#include "selim.hpp"
#include "sred.hpp"

#include "base_nlp.hpp"
#include "aebnd.hpp"
#include "gamswriter.hpp"
#include "gamsio.hpp"

//#undef MC__MINLPREF_DEBUG
//#define MC__MINLPREF_DEBUG_LIFT

namespace mc
{

//! @brief C++ base class for reformulation of factorable MINLP using MC++
////////////////////////////////////////////////////////////////////////
//! mc::MINLPREF is a C++ class for reformulation of factorable MINLP
//! using MC++
////////////////////////////////////////////////////////////////////////
template < typename T >
class MINLPREF
#if defined (MC__WITH_GAMS)
: protected virtual GAMSIO,
  public virtual BASE_NLP
#else
: public virtual BASE_NLP
#endif
{
public:

  typedef SMon< unsigned, std::less<unsigned> > t_mon;
  typedef lt_SMon< std::less<unsigned> > lt_mon;
  typedef SPoly< unsigned, std::less<unsigned> > t_poly;
  typedef std::map< t_mon, double, lt_mon > map_poly;
  typedef std::pair< t_mon const*, t_mon const* > t_prodmon;

  typedef SQuad< unsigned, std::less<unsigned> > t_quad;
  typedef lt_SQuad< std::less<unsigned> > lt_quad;

  typedef SMon< mc::FFVar const*, mc::lt_FFVar > t_ffmon;
  typedef lt_pSMon< mc::lt_FFVar > lt_ffmon;
  typedef SPoly< mc::FFVar const*, mc::lt_FFVar > t_ffpoly;
  typedef SRed< mc::FFVar const*, mc::lt_FFVar > t_red;

  typedef SLiftEnv t_lift;
  typedef SElimEnv t_elim;
  typedef AEBND<T> t_aebnd;

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

#if defined (MC__WITH_GAMS)
  using GAMSIO::read;
#endif

protected:

  // Do not use BASE_NLP::_dag since redefined locally
  using BASE_NLP::_var;
  using BASE_NLP::_vartyp;
  using BASE_NLP::_varlb;
  using BASE_NLP::_varlm;
  using BASE_NLP::_varub;
  using BASE_NLP::_varum;
  using BASE_NLP::_par;
  using BASE_NLP::_obj;
  using BASE_NLP::_ctr;
  using BASE_NLP::_nco;

#if defined (MC__WITH_GAMS)
  using GAMSIO::_varini;
#endif

protected:

  //! @brief local copy of DAG (overides BASE_AE::_dag)
  FFGraph*                  _dag;

  //! @brief environment for expression elimination
  t_elim                    _SEenv;
  //! @brief environment for expression lift
  t_lift                    _SLenv;
  //! @brief environment for sparse quadratic form
  t_quad                    _SQenv;
  //! @brief environment for redundant constraints
  t_red                     _SRenv;
  //! @brief environment for algebraic equation bounding
  t_aebnd                   _AEBND;

  //! @brief number of parameters in model
  unsigned                  _nP;
  //! @brief vector of parameters in DAG
  std::vector<FFVar>        _Pvar;

  //! @brief number of decision variables in model after reformulation
  unsigned                  _nX;
  //! @brief number of original decision variables in model (excl. multipliers)
  unsigned                  _nX0;
  //! @brief number of decision variables in model before reformulation
  unsigned                  _nX1;
  //! @brief vector of decision variables in DAG
  std::vector<FFVar>        _Xvar;
  //! @brief vector of decision variables dependencies
  std::vector<FFDep>        _Xdep;
  //! @brief vector of decision variable levels (size _nX0)
  std::vector<double>       _Xini;
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
  //! @brief objective variable
  FFVar                     _Xobj;
  //! @brief Map of lifted expressions in terms of original variables
  std::map< unsigned, FFVar > _Xlift;
  //! @brief Map of eliminated variables from original variables
  std::set< unsigned >      _Xelim;

  //! @brief number of functions (objective and constraints) in model
  unsigned                  _nF;
  //! @brief vector of functions in DAG
  std::vector<FFVar>        _Fvar;  
  //! @brief vector of functions dependencies
  std::vector<FFDep>        _Fdep;
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
  std::set<unsigned>        _Flin;
  //! @brief subset of quadratic functions
  std::set<unsigned>        _Fquad;
  //! @brief subset of polynomial functions
  std::set<unsigned>        _Fpol;
  //! @brief subset of general functions
  std::set<unsigned>        _Fgal;
  //! @brief subset of nonlinear functions
  std::set<unsigned>        _Fnlin;
  //! @brief subset of equality-constrained functions
  std::set<unsigned>        _Fctreq;
  
  //! @brief Interval representation of 'unbounded' variables
  T                         _IINF;
  //! @brief Storage vector for constraint propagation in interval arithmetic
  std::vector<T>            _CPbnd;
  //! @brief Storage vector for interval arithmetic
  std::vector<T>            _Iwk;
  //! @brief Storage vector for real arithmetic
  std::vector<double>       _dwk;

  //! @brief Worst dependence type in participating expressions
  FFDep::TYPE               _pbclass;
  //! @brief Flag for setup function
  bool                      _issetup;
  //! @brief Flag for MIP problem
  bool                      _ismip;
  //! @brief Flag indicating subgraphs needing updating
  bool                      _sgupdt;
  //! @brief Direction of optimization (-1: MIN, 0: FEAS; 1: MAX)
  int                       _objsense;

  //! @brief independent indices in elimination
  std::vector<unsigned>     _ndxIndep;
  //! @brief dependent and constraint indices in elimination
  std::vector<std::pair<unsigned,unsigned>> _ndxDep;
  //! @brief Storage vector for independent variables in elimination
  std::vector<T>            _bndIndep;
  //! @brief Storage vector for dependent variables in elimination
  std::vector<T>            _bndDep;
  
  //! @brief Storage vector for polynomial variables in flattening
  std::vector<t_ffpoly>     _SPXvar;
  //! @brief Storage vector for polynomial functions in flattening
  std::vector<t_ffpoly>     _SPFvar;

public:

  //! @brief Constructor
  MINLPREF
    ()
    : _dag(nullptr), _nX(0), _nX0(0), _nX1(0), _nF(0), _issetup(false)
    {}

  //! @brief Destructor
  virtual ~MINLPREF
    ()
    { delete _dag; }

  //! @brief MINLPREF options
  struct Options
  {
    //! @brief Constructor
    Options():
      CPMAX(20), CPTHRES(1e-10),
      DISCRELIM(false), INVBNDGS(false), INVKEEPLIN(false), REDELIM(false),
      QUADOPTIM(false), PSDQUADCUTS(0), DCQUADCUTS(false), 
      NCOCUTS(false), NCOADIFF(FSA), TIMELIMIT(6e2), DISPLEVEL(1),
      SELIM(), SLIFT(), SQUAD(), SRED(), AEBND()
      { SLIFT.LIFTDIV          = 0;
        SLIFT.LIFTIPOW         = 0;
        SLIFT.KEEPFACT         = 1;
        SELIM.MIPDISPLEVEL     = 0;
        SELIM.MIPTIMELIMIT     = TIMELIMIT;
        SQUAD.BASIS            = t_quad::Options::MONOM;
        SQUAD.ORDER            = t_quad::Options::INC;
        SQUAD.REDUC            = 0;
        SQUAD.MIPDISPLEVEL     = 0;
        SQUAD.MIPTIMELIMIT     = TIMELIMIT;
        SRED.ORDER             = 1;
        SRED.DISPLEVEL         = 0;
        SRED.MIPDISPLEVEL      = 0;
        SRED.MIPTIMELIMIT      = TIMELIMIT;
        AEBND.BOUNDER          = t_aebnd::Options::ALGORITHM::GS;
        AEBND.BLKDEC           = t_aebnd::Options::DECOMPOSITION::RECUR;
        AEBND.DISPLEVEL        = 0; }
    //! @brief Assignment operator
    Options& operator= ( Options const& opt ){
        CPMAX         = opt.CPMAX;
        CPTHRES       = opt.CPTHRES;
        DISCRELIM     = opt.DISCRELIM;
        INVBNDGS      = opt.INVBNDGS;
        INVKEEPLIN    = opt.INVKEEPLIN;
        REDELIM       = opt.REDELIM;
	QUADOPTIM     = opt.QUADOPTIM;
        PSDQUADCUTS   = opt.PSDQUADCUTS;
        DCQUADCUTS    = opt.DCQUADCUTS;
        NCOCUTS       = opt.NCOCUTS;
        NCOADIFF      = opt.NCOADIFF;
        TIMELIMIT     = opt.TIMELIMIT;
        DISPLEVEL     = opt.DISPLEVEL;
        SELIM         = opt.SELIM;
        SLIFT         = opt.SLIFT;
        SQUAD         = opt.SQUAD;
        SRED          = opt.SRED;
        AEBND         = opt.AEBND;
        return *this ;
      }
    //! @brief Sensitivity strategy
    enum SENS{
      FSA=0,      //!< Forward sensitivity analysis
      ASA         //!< Adjoint sensitivity analysis
    };
    //! @brief Maximum rounds of constraint propagation
    unsigned CPMAX;
    //! @brief Threshold for repeating constraint propagation (minimum relative reduction in any variable)
    double CPTHRES;
    //! @brief Whether to exclude discrete (binary/integer) variables from the elimination search
    bool DISCRELIM;
    //! @brief Whether to bound eliminated variables and check invertibility using interval Gauss-Siedel method
    bool INVBNDGS;
    //! @brief Whether to keep auxiliary linear variables for enforcing eliminated variable bounds
    bool INVKEEPLIN;
    //! @brief Whether to eliminate monomials through exact linearization from reduction constraints
    bool REDELIM;
    //! @brief Whether to minimize the number of auxiliary variables in quadratisation using MIP
    bool QUADOPTIM;
    //! @brief Whether to add PSD cuts within quadratisation (0: none; 1: 2-by-2; >1: 3-by-3)
    unsigned PSDQUADCUTS;
    //! @brief Whether to add DC cuts within quadratisation
    bool DCQUADCUTS;
    //! @brief Whether to add NCO cuts
    bool NCOCUTS;
    //! @brief NCO method
    unsigned NCOADIFF;
    //! @brief Maximum run time (seconds)
    double TIMELIMIT;
    //! @brief Display level for solver
    int DISPLEVEL;
    //! @brief SElimEnv options for variable elimination
    typename t_elim::Options SELIM;
    //! @brief SLiftEnv options for expression lifting
    typename t_lift::Options SLIFT;
    //! @brief SQuad options for quadratization of sparse polynomial expressions
    typename t_quad::Options SQUAD;
    //! @brief RLTRed options for reduced RLT search
    typename t_red::Options  SRED;
    //! @brief AEBND options for implicit equation bounds
    typename t_aebnd::Options AEBND;
    //! @brief Display
    void display
      ( std::ostream&out=std::cout ) const;
  } options;

  //! @brief MINLPREF exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for exception handling
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
        return "MINLPREF::Exceptions  Model with multiple objectives not allowed";
      case SETUP:
        return "MINLPREF::Exceptions  Incomplete setup before a solve";
      case INTERN: default:
        return "MINLPREF::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief Setup optimization model before bounding
  void setup
    ( std::ostream& os=std::cout );

  //! @brief Get problem class - i.e. worst dependence type in relaxed subproblem
  FFDep::TYPE problem_class 
    ()
    const
    { return _pbclass; }

  //! @brief Get pointer to DAG
  FFGraph* dag
    ()
    const
    { return _dag; }

  //! @brief Get vector of DAG variables
  std::vector<FFVar> const& variables
    ()
    const
    { return _Xvar; }

  //! @brief Get vector of DAG variables
  std::map<unsigned,FFVar> const& lifted_variables
    ()
    const
    { return _Xlift; }

  //! @brief Get vector of DAG functions
  std::vector<FFVar> const& functions
    ()
    const
    { return _Fvar; }

  //! @brief Get vector of variable bounds
  std::vector<double> const& variable_initials
    ()
    const
    { return _Xini; }

  //! @brief Update variable initial values
  //bool update_initials
  //  ( double const* Xini, std::ostream& os );

  //! @brief Get vector of variable bounds
  std::vector<T> const& variable_bounds
    ()
    const
    { return _Xbnd; }

  //! @brief Get vector of function bounds
  std::vector<T> const& function_bounds
    ()
    const
    { return _Fbnd; }

  //! @brief Update variable and function bounds
  bool update_bounds
    ( T const* X=nullptr, double const* Finc=nullptr, bool const resetbnd=true,
      std::ostream& os=std::cout );

  //! @brief Propagate bounds, starting with variable subdomain <a>X</a>, for the incumbent value <a>Finc</a>, and using the options specified in <a>MINLPREF::Options::CPMAX</a> and <a>MINLPREF::Options::CPTHRES</a> -- returns updated variable bounds <a>X</a>
  bool propagate_bounds
    ( T const* X=nullptr, double const* Finc=nullptr, const bool resetbnd=true,
      std::ostream& os=std::cout );

  //! @brief Lift polynomial subexpressions in cost and constraints
  bool lift_polynomial_subexpressions
    ( bool const add2dag, std::ostream& os=std::cout );

  //! @brief Flatten linear cost and constraint functions
  bool flatten_linear_functions
    ( bool const add2dag );

  //! @brief Flatten quadratic cost and constraint functions
  bool flatten_quadratic_functions
    ( bool const add2dag );

  //! @brief Flatten polynomial cost and constraint functions
  bool flatten_polynomial_functions
    ( bool const add2dag );

  //! @brief Quadratize polynomial subexpressions in cost and constraints
  bool quadratize_polynomial_functions
    ( bool const add2dag, std::ostream& os=std::cout );

  //! @brief Append reduction polynomial constraints
  bool append_reduction_constraints
    ( bool const add2dag, std::ostream& os=std::cout );

  //! @brief Eliminate variables from invertible equality constraints
  bool eliminate_invertible_constraints
    ( bool const add2dag, std::ostream& os=std::cout );

  //! @brief export reformulated optimization model to GAMS file
  bool export_model
    ( std::string const gmsfile, double const* Xinc=nullptr, std::ostream& os=std::cout );

protected:

  //! @brief Tighten bounds using constraint propagation
  int _propagate_bounds
    ();

  //! @brief MINLPBND computational statistics
  static struct Stats{
    //! @brief Get current time point
    static std::chrono::time_point<std::chrono::system_clock> start
      ()
      { return std::chrono::system_clock::now(); }
    //! @brief Get current time lapse with respect to start time point
    static std::chrono::microseconds walltime
      ( std::chrono::time_point<std::chrono::system_clock> const& start )
      { return std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start ); }    
    //! @brief Convert microsecond ticks to time
    static double to_time
      ( std::chrono::microseconds t )
      { return t.count() * 1e-6; }
  } stats;

  //! @brief Time point to enable TIMELIMIT option
  std::chrono::time_point<std::chrono::system_clock> _tstart;

private:

  //! @brief Update MINLPREF options
  virtual void _update_options
    ()
    {};

  //! @brief Set linear/nonlinear participating variables in functions
  void _update_model
    ();

  //! @brief Display model information
  void _display_model
    ( std::ostream& os = std::cout );

  //! @brief Set optimality cuts in original DAG
  bool _set_optimality_cuts
    ( std::vector<FFVar>& Xvar, std::vector<FFVar>& Fvar, std::ostream& os = std::cout);

  //! @brief Set dependencies in functions
  void _set_dependencies
    ();

  //! @brief Set linear/nonlinear participating variables in functions
  void _set_variable_class
    ();

  //! @brief Set linear/polynomial/nonlinear participating functions
  void _set_function_class
    ();

  //! @brief Set subgraphs for participating functions
  void _set_subgraph
    ( std::ostream& os = std::cout );

  //! @brief Flatten subset of functions in DAG
  bool _flatten_functions
    ( std::set<unsigned> const& Fndx, bool const add2dag );

  //! @brief Create DAG variable for given sparse polynomial
  FFVar _insert_ffpol
    ( t_ffpoly const& pol )
    const;
    
  //! @brief Create DAG variable and auxiliary for given high-order monomial 
  std::pair< FFVar const*, FFVar const* > _insert_ffmon
    ( t_ffmon const& mon, int const BASIS, bool const noaux=false )
    const;

  //! @brief Create DAG variable for given quadratic form
  FFVar _insert_quad
    ( t_quad::map_SQuad const& quad, std::map< t_mon, FFVar, lt_mon >& mapmon )
    const;

  //! @brief Create DAG variable for given Chebyshev basis function 
  FFVar _insert_cheb
    ( FFVar const& x, const unsigned n )
    const;
    
  //! @brief Create DAG variable and auxiliary for given high-order monomial 
  std::pair< FFVar const*, FFVar const* > _insert_mon
    ( t_mon const& mon, int const BASIS )
    const;

  //! @brief Search for invertible equality constraints and form correspond triangular constraint subsystem
  void _search_invertible_constraints
    ( std::ostream& os );

  //! @brief Bound dependent variables of invertible equality constraints using Gauss-Siedel interval methods
  bool _bound_invertible_constraints
    ();

  //! @brief Search for reduction polynomial constraints and identify linearizable monomials
  unsigned _search_reduction_constraints
    ( std::set<unsigned> Ftpol, std::ostream& os );

  //! @brief Private methods to block default compiler methods
  MINLPREF
    ( MINLPREF<T> const& );
  MINLPREF<T>& operator=
    ( MINLPREF<T> const& );
};

template <typename T>
inline void
MINLPREF<T>::setup
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

  // full set of decision variables
  std::vector<FFVar> Xvar = _var;
  _nX0 = Xvar.size();

  // full set of variable bounds and types
#if defined (MC__WITH_GAMS)
  _Xini = _varini;
#else
  _Xini.clear();
#endif
  _Xlow = _varlb;
  _Xupp = _varub;
  _Xtyp = _vartyp;

  // full set of nonlinear functions (cost, constraints & equations)
  std::vector<FFVar> Fvar;
  _Flow.clear();
  _Fupp.clear();

  // cost function
  if( std::get<0>(_obj).size() > 1 ) throw Exceptions( Exceptions::MULTOBJ );
  _objsense = std::get<0>(_obj).size()? (std::get<0>(_obj)[0]==BASE_OPT::MIN? -1: 1): 0;
  std::get<0>(_obj).size()? Fvar.push_back( std::get<1>(_obj)[0] ): Fvar.push_back( 0 );
  _Flow.push_back( -BASE_OPT::INF );
  _Fupp.push_back(  BASE_OPT::INF );

  // constraints
  for( unsigned i=0; i<std::get<0>(_ctr).size(); i++ ){
    Fvar.push_back( std::get<1>(_ctr)[i] );
    switch( std::get<0>(_ctr)[i] ){
      case BASE_OPT::EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );            break;
      case BASE_OPT::LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );            break;
      case BASE_OPT::GE: _Flow.push_back( 0. );             _Fupp.push_back( BASE_OPT::INF ); break;
    }
  }
 
  // set Fritz-John cuts and corresponding multipliers
  if( options.NCOCUTS ) _set_optimality_cuts( Xvar, Fvar, os );

  // local DAG copy
  if( _dag ) delete _dag;
  _dag = new FFGraph;
  _nP = Pvar.size(); _Pvar.resize( _nP );
  _dag->insert( BASE_NLP::_dag, _nP, Pvar.data(), _Pvar.data() );
  _nX = _nX1 = Xvar.size(); _Xvar.resize( _nX );
  _dag->insert( BASE_NLP::_dag, _nX, Xvar.data(), _Xvar.data() );
  _nF = Fvar.size(); _Fvar.resize( _nF );
  _dag->insert( BASE_NLP::_dag, _nF, Fvar.data(), _Fvar.data() );
#ifdef MC__MINLPREF_DEBUG  
  _dag->output( _dag->subgraph( 1, _Fvar.data() ), " objective" );
#endif

  // Set default variable and function bounds
  _Xbnd.resize( _nX );
  for( unsigned i=0; i<_nX; i++ ) _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
  _Fbnd.resize( _nF );
  for( unsigned i=0; i<_nF; i++ ) _Fbnd[i] = T( _Flow[i], _Fupp[i] );

  // Identify variable and function sets and create subgraphs
  _Xobj.set( _dag );
  _Xlift.clear();
  _Xelim.clear();
  _set_dependencies();
  _set_variable_class();
  _set_function_class();
  _sgupdt = true;

  //stats.reset();
  _issetup = true;
  if( options.DISPLEVEL ) _display_model( os );
}

template <typename T>
inline
bool
MINLPREF<T>::_set_optimality_cuts
( std::vector<FFVar>& Xvar, std::vector<FFVar>& Fvar, std::ostream& os )
{
  if( options.DISPLEVEL )
    os << "# APPENDING FIRST-ORDER OPTIMALITY CONDITIONS" << std::endl;

  if( !BASE_NLP::set_nco( _Xtyp.data(), options.NCOADIFF==Options::ASA ) )
    return false;

  // cost multiplier
  Xvar.push_back( std::get<2>(_obj)[0] );
  _Xlow.push_back( 0. );
  _Xupp.push_back( 1. );
  _Xtyp.push_back( 0  );

  // constraint multipliers
  for( unsigned i=0; i<std::get<0>(_ctr).size(); ++i ){
    Xvar.push_back( std::get<2>(_ctr)[i] );
    _Xtyp.push_back( 0 ); // all constraint multipliers are continuous variables
    switch( std::get<0>(_ctr)[i] ){
      case BASE_OPT::LE:
      case BASE_OPT::GE: _Xlow.push_back(  0. ); _Xupp.push_back( 1. ); break;
      case BASE_OPT::EQ: _Xlow.push_back( -1. ); _Xupp.push_back( 1. ); break;
    }
  }

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
#ifdef MC__MINLPREF_DEBUG_NCOCUTS
    BASE_NLP::_dag->output( BASE_NLP::_dag->subgraph( 1, &Fvar.back() ), " FOR NCO" );    
#endif
    switch( std::get<0>(_nco)[i] ){
      case BASE_OPT::EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );            break;
      case BASE_OPT::LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );            break;
      case BASE_OPT::GE: _Flow.push_back( 0. );             _Fupp.push_back( BASE_OPT::INF ); break;
    }
  }
  return true;
}

template <typename T>
inline
void
MINLPREF<T>::_update_model
()
{
  // update variable and function size and type
  _nX = _Xvar.size();
  _nF = _Fvar.size();
  _set_dependencies();
  _set_variable_class();
  _set_function_class();
}

template <typename T>
inline
void
MINLPREF<T>::_display_model
( std::ostream& os )
{
  os << std::endl
     << "#              |  VARIABLES      FUNCTIONS" << std::endl << std::right
     << "# -------------+---------------------------" << std::endl
     << "#  LINEAR      | " << std::setw(9) << _Xlin.size()  << std::setw(15) << _Flin.size()  << std::endl
     << "#  QUADRATIC   | " << std::setw(9) << _Xquad.size() << std::setw(15) << _Fquad.size() << std::endl
     << "#  POLYNOMIAL  | " << std::setw(9) << _Xpol.size()  << std::setw(15) << _Fpol.size()  << std::endl
     << "#  GENERAL     | " << std::setw(9) << _Xgal.size()  << std::setw(15) << _Fgal.size()  << std::endl
     << std::endl;
}

template <typename T>
inline void
MINLPREF<T>::_set_dependencies
()
{
  _Xdep.resize( _nX );
  for( unsigned i=0; i<_nX; ++i )
    _Xdep[i].indep( _Xvar[i].id().second );

  _Fdep.resize( _nF );
  for( unsigned i=0; i<_nF; i++ )
    _dag->eval( 1, &_Fvar[i], &_Fdep[i], _nX, _Xvar.data(), _Xdep.data() );
}

template <typename T>
inline void
MINLPREF<T>::_set_variable_class
()
{
  FFDep Fworst( 0. );
  for( auto const& dep : _Fdep )
    Fworst += dep;
#ifdef MC__MINLPREF_DEBUG
  std::cout << "DEPS <- " << Fworst << std::endl;
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

template <typename T>
inline void
MINLPREF<T>::_set_function_class
()
{
  _Flin.clear();
  _Fquad.clear();
  _Fpol.clear();
  _Fgal.clear();
  _Fnlin.clear();
  _Fctreq.clear();
 
  _pbclass = FFDep::L;
  for( unsigned j=0; j<_nF; j++ ){
    auto depworst = _Fdep[j].worst();
    switch( depworst ){
     case FFDep::L: _Flin.insert( j );                      break;
     case FFDep::Q: _Fquad.insert( j ); _Fnlin.insert( j ); break;
     case FFDep::P: _Fpol.insert( j );  _Fnlin.insert( j ); break;
     case FFDep::R:
     case FFDep::N: _Fgal.insert( j );  _Fnlin.insert( j ); break;
    }
    if( _pbclass < depworst ) _pbclass = depworst;
    if( j && _Flow[j] == 0. && _Fupp[j] == 0. ) _Fctreq.insert( j );
  }
}


template <typename T>
inline void
MINLPREF<T>::_set_subgraph
( std::ostream& os )
{
  if( !_sgupdt ) return;
  if( options.DISPLEVEL )
    os << "# GENERATING EXPRESSION TREES" << std::endl;

  _Fops.clear();
  for( auto const& Fj : _Fvar )
    _Fops.push_back( _dag->subgraph( 1, &Fj ) );
  for( auto const& [i,Fi] : _Xlift )
    _Fops.push_back( _dag->subgraph( 1, &Fi ) );
  _Fallops = _dag->subgraph( _nF, _Fvar.data() );
#ifdef MC__MINLPREF_DEBUG  
  _dag->output( _Fallops, " FOR ALL FUNCTIONS" );    
#endif
  _sgupdt = false;
}

template <typename T>
inline
bool
MINLPREF<T>::lift_polynomial_subexpressions
( bool const add2dag, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  //if( _Fgal.empty() ) return false;
  if( _Fnlin.empty() ) return false;
  _update_options(); // virtual function

  _SLenv.set( _dag );
  _SLenv.options = options.SLIFT;
  if( options.DISPLEVEL )
    os << "# LIFTING POLYNOMIAL SUBEXPRESSIONS" << std::endl;
#ifdef MC__MINLPREF_DEBUG_LIFT
    std::cout << std::endl << _Fnlin.size() << " NONLINEAR CONSTRAINT" << (_Fnlin.size()>1?"S:":":") << std::endl;
    auto sgExpr = _dag->subgraph( _Fnlin, _Fvar.data() );
    std::vector<double> Xval( _Xvar.size(), 0.5 ), Fval( _Fvar.size() );
    _dag->eval( sgExpr, _Fnlin, _Fvar.data(), Fval.data(), _Xvar.size(), _Xvar.data(), Xval.data() );
    auto vExpr  = FFExpr::subgraph( _dag, sgExpr );
    auto jt = _Fnlin.cbegin();
    for( auto const& expr : vExpr ){
      assert( jt != _Fnlin.cend() );
      std::cout << Fval[*jt] << " = " << expr << std::endl;
      ++jt;
    }
#endif
  _SLenv.process( _Fnlin, _Fvar.data(), true );
  //_SLenv.process( _Fgal, _Fvar.data(), true );
#ifdef MC__MINLPREF_DEBUG_LIFT
  { std::cout << _SLenv << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif
  if( !add2dag ) return true;

  // update lifted constraints
  std::map<FFVar const*, unsigned, lt_FFVar> Frem;
  assert( _Fnlin.size() == _SLenv.Dep().size() );
  auto it = _Fnlin.begin();
  //assert( _Fgal.size() == _SLenv.Dep().size() );
  //auto itgal = _Fgal.begin();
  for( auto const& expr : _SLenv.Dep() ){
    // Earmark constraint if expression matches a variable
    if( *it && expr.opdef().first->type == FFOp::VAR )
      Frem[&expr] = *it;
    else
      _Fvar[*it] = expr;
      // Do not modify lower and upper constraint range
    ++it;
  }
  
  // append auxiliary variables
#ifdef MC__MINLPREF_DEBUG_LIFT
    std::cout << std::endl << _SLenv.Aux().size() << " AUXILIARY VARIABLE" << (_SLenv.Aux().size()>1?"S:":":") << std::endl;
#endif
  for( auto const& [pAux,pVar] : _SLenv.Aux() ){
#ifdef MC__MINLPREF_DEBUG_LIFT
    double Xliftval;
    _dag->eval( 1, pAux, &Xliftval, _Xvar.size(), _Xvar.data(), Xval.data() );
    Xval.push_back( Xliftval );
    auto sgAux = _dag->subgraph( 1, pAux );
    auto vAux  = FFExpr::subgraph( _dag, sgAux );
    std::cout << Xval.back() << " = " << *pVar << " = " << vAux[0] << std::endl;
#endif
    _Xlift[_Xvar.size()] = *pAux; // <- stores original DAG expression
    _Xvar.push_back( *pVar );
    auto itVar = Frem.find( pVar );
    // Inherit function bounds in case the lifted constraint corresponds to current auxiliary variable 
    _Xlow.push_back( itVar != Frem.end()? _Flow[itVar->second]: -BASE_OPT::INF );
    _Xupp.push_back( itVar != Frem.end()? _Fupp[itVar->second]:  BASE_OPT::INF );
    //_Xbnd.push_back( T( _Xlow.back(), _Xupp.back() ) );
    _Xtyp.push_back( 0 );
  }
#ifdef MC__MINLPREF_DEBUG_LIFT
    std::cout << std::endl << _SLenv.Dep().size() << " NONLINEAR LIFTED CONSTRAINT" << (_SLenv.Dep().size()>1?"S:":":") << std::endl;
    _dag->eval( _SLenv.Dep().size(), _SLenv.Dep().data(), Fval.data(), _Xvar.size(), _Xvar.data(), Xval.data() );
    for( unsigned i=0; i<_SLenv.Dep().size(); ++i )
      std::cout << _SLenv.Dep()[i] << " = " << Fval[i] << std::endl;
#endif

  // eliminate lifted constraints that correspond to new lifted variables
  //std::cout << "#lifted constraints corresponding to new lifted variables: " << Frem.size() << std::endl;
  for( auto it=Frem.rbegin(); it!=Frem.rend(); ++it ){
    unsigned const i = it->second;
    auto itFvar = _Fvar.begin(); std::advance( itFvar, i ); _Fvar.erase( itFvar );
    auto itFlow = _Flow.begin(); std::advance( itFlow, i ); _Flow.erase( itFlow );
    auto itFupp = _Fupp.begin(); std::advance( itFupp, i ); _Fupp.erase( itFupp );
    auto itFbnd = _Fbnd.begin(); std::advance( itFbnd, i ); _Fbnd.erase( itFbnd );
  }

  // append auxiliary polynomial constraints
  for( auto const& poly : _SLenv.Poly() ){
    _Fvar.push_back( poly );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
    //_Fbnd.push_back( T( _Flow.back(), _Fupp.back() ) );
  }

  // append auxiliary non-polynomial constraints
  for( auto const& trans : _SLenv.Trans() ){
    _Fvar.push_back( trans );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
    //_Fbnd.push_back( T( _Flow.back(), _Fupp.back() ) );
  }


  // update variable and function size and type
  assert( _Fvar.size() == _Flow.size() && _Fvar.size() == _Fupp.size() ); 
  _sgupdt = true;
  _update_model();
  update_bounds( nullptr, nullptr, false, os );
  if( options.DISPLEVEL ) _display_model( os );
  
  return true;
}

template <typename T>
inline bool
MINLPREF<T>::_flatten_functions
( std::set<unsigned> const& Fndx, bool const add2dag )
{
  if( Fndx.empty() ) return false;

  // Create vector of all semi-algebraic expressions
  t_poly::options.BASIS = t_poly::Options::MONOM;
  _SPXvar.resize( _nX );
  for( unsigned ix=0; ix<_nX; ix++ ){
    auto itXvar = _dag->Vars().find( &_Xvar[ix] );
    _SPXvar[ix].var( *itXvar ); // vector _Xvar may be resized!
  }
  _SPFvar.resize( _nF );
  try{
    _dag->eval( Fndx, _Fvar.data(), _SPFvar.data(), _nX, _Xvar.data(), _SPXvar.data() );
  }
  catch(...){
    // Exception caught in case of non-polynomial term
    return false;
  }
#ifdef MC__MINLPREF_DEBUG_FLATTEN
  for( unsigned const& i : Fndx ){
    std::ostringstream ostr; ostr << " of expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
    std::cout << "Polynomial expression " << i << ":\n" << _SPFvar[i];
  }
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum;}
#endif
  if( !add2dag ) return true;
  
  // Substitute flattened expressions in DAG
  for( unsigned const& i : Fndx ){
    _Fvar[i] = _SLenv.insert_dag( _SPFvar[i] );
#ifdef MC__MINLPREF_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of flattened expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
  }
  
  return true;
}

template <typename T>
inline bool
MINLPREF<T>::flatten_linear_functions
( bool const add2dag )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _update_options(); // virtual function
  return _flatten_functions( _Flin, add2dag );
}

template <typename T>
inline bool
MINLPREF<T>::flatten_quadratic_functions
( bool const add2dag )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _update_options(); // virtual function
  return _flatten_functions( _Fquad, add2dag );
}

template <typename T>
inline bool
MINLPREF<T>::flatten_polynomial_functions
( bool const add2dag )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _update_options(); // virtual function
  return _flatten_functions( _Fpol, add2dag );
}

template <typename T>
inline bool
MINLPREF<T>::quadratize_polynomial_functions
( bool const add2dag, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _tstart = stats.start();
  _update_options(); // virtual function
    
  // Flatten quadratic and polynomial expressions
  std::set<unsigned> Ftpol = _Fpol;
  Ftpol.insert( _Fquad.cbegin(), _Fquad.cend() );
  if( !_flatten_functions( Ftpol, false ) ) return false;

  // Transform variable indexing in quadratic and polynomial expressions
  std::map<FFVar const*, unsigned, lt_FFVar> FFmatch;
  unsigned ivar = 0;
  for( auto const& var : _Xvar ) FFmatch[&var] = ivar++;
  unsigned ifun = 0;
  std::vector<t_poly> SPol( Ftpol.size() );
  for( unsigned const& i : Ftpol ){
    for( auto const& [FFmon,coef] : _SPFvar[i].mapmon() ){
      t_mon mon( FFmon.tord, FFmon.expr, FFmatch ); 
      SPol[ifun] += std::make_pair( mon, coef );
    }
    ++ifun;
  }

  // Apply quadratisation to polynomial expressions
  _SQenv.reset();
  _SQenv.options = options.SQUAD;
  if( options.DISPLEVEL )
    os << "# PERFORMING QUADRATIC DECOMPOSITION" << std::endl;
  _SQenv.process( SPol.size(), SPol.data(), &t_poly::mapmon, t_quad::Options::MONOM );
  if( options.QUADOPTIM ){
    if( options.DISPLEVEL )
      os << "# OPTIMIZING QUADRATIC DECOMPOSITION" << std::endl;
    _SQenv.options.MIPTIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime( _tstart ) );
    _SQenv.optimize( true );
  }
#ifdef MC__MINLPREF_DEBUG_LIFT
  double viol = _SQenv.check( SPol.size(), SPol.data(), &t_poly::mapmon, t_quad::Options::MONOM );
  std::cout << "violation: " << viol << std::endl << _SQenv << std::endl;
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif
  if( !add2dag ) return true;

  // Add higher-order monomials in basis to DAG
  std::map< t_mon, FFVar, lt_mon > mapmon; 
  for( auto const& mon : _SQenv.SetMon() ){
    if( mon.tord == 1 ) mapmon[mon] = _Xvar[mon.expr.cbegin()->first];
    if( mon.tord <= 1 ) continue;
    auto [pAux,pVar] = _insert_mon( mon, options.SQUAD.BASIS );
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
    _Fvar[i] = _insert_quad( _SQenv.MatFct()[iquad++], mapmon );
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of lifted quadratic expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
  }
  
  // Append reduction quadratic cuts
  for( auto const& red : _SQenv.MatRed() ){
    _Fvar.push_back( _insert_quad( red, mapmon ) );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
#ifdef MC__MINLPBND_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of reduction quadratic cut";
    _dag->output( _dag->subgraph( 1, &_Fvar.back() ), ostr.str() );
#endif
  }

  // Append positive semi-definite cuts
  if( options.PSDQUADCUTS ){
    _SQenv.tighten( options.PSDQUADCUTS>1? true: false );
    for( auto const& psd : _SQenv.MatPSD() ){
      _Fvar.push_back( _insert_quad( psd, mapmon ) );
      _Flow.push_back( 0. );
      _Fupp.push_back( BASE_OPT::INF );
#ifdef MC__MINLPBND_DEBUG_LIFT
      std::ostringstream ostr; ostr << " of semi-definite quadratic cut >=0";
      _dag->output( _dag->subgraph( 1, &_Fvar.back() ), ostr.str() );
#endif
    }
  }

  // update variable and function size and type
  _sgupdt = true;
  _update_model();
  update_bounds( nullptr, nullptr, false, os );
  if( options.DISPLEVEL ) _display_model( os );
  
  return true;
}

template <typename T>
inline
FFVar
MINLPREF<T>::_insert_ffpol
( t_ffpoly const& pol )
const
{
  FFVar varpol( 0. );
  std::cout << "inserting: " << pol.display( pol.mapmon(), pol.options.BASIS, pol.options.DISPLEN, true ) << std::endl;
  for( auto const& [mon,coef] : pol.mapmon() ){
    if( mon.tord ){
      auto [pAux,pVar] = _insert_ffmon( mon, pol.options.BASIS, true );
      varpol += coef * (*pAux);
    }
    else
      varpol += coef;
  }
  return varpol;
}

template <typename T>
inline
std::pair< FFVar const*, FFVar const* >
MINLPREF<T>::_insert_ffmon
( t_ffmon const& mon, int const BASIS, bool const noaux )
const
{
  // define power monomial expression
  FFVar Xlift( 1e0 );
  for( auto const& [pvar,iord] : mon.expr ){
    switch( BASIS ){
     // Monomial basis
     case t_ffpoly::Options::MONOM:
      Xlift *= pow( *pvar, (int)iord );
      break;
     // Chebyshev basis
     case t_ffpoly::Options::CHEB:
      Xlift *= _insert_cheb( *pvar, iord );
      break;
    }
  }
#ifdef MC__MINLPBND_DEBUG_RED
  std::ostringstream ostr; ostr << " of lifted monomial";
  _dag->output( _dag->subgraph( 1, &Xlift ), ostr.str() );
#endif
  auto itXlift = _dag->Vars().find( &Xlift );
  assert( itXlift != _dag->Vars().end() );
  if( noaux ) return std::make_pair( *itXlift, nullptr );

  // define power monomial variable
  FFVar Xmon( _dag );
  auto itXmon = _dag->Vars().find( &Xmon );
  return std::make_pair( *itXlift, *itXmon );
}

template <typename T>
inline
FFVar
MINLPREF<T>::_insert_quad
( t_quad::map_SQuad const& quad, std::map< t_mon, FFVar, lt_mon >& mapmon )
const
{
  FFVar varpol( 0. );
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

template <typename T>
inline
std::pair< FFVar const*, FFVar const* >
MINLPREF<T>::_insert_mon
( t_mon const& mon, int const BASIS )
const
{
  // define power monomial expression
  FFVar Xlift( 1e0 );
  for( auto const& [ivar,iord] : mon.expr ){
    switch( BASIS ){
     // Monomial basis
     case t_ffpoly::Options::MONOM:
      Xlift *= pow( _Xvar[ivar], (int)iord );
      break;
     // Chebyshev basis
     case t_ffpoly::Options::CHEB:
      Xlift *= _insert_cheb( _Xvar[ivar], iord );
      break;
    }
  }
  auto itXlift = _dag->Vars().find( &Xlift );

  // define power monomial variable
  FFVar Xmon( _dag );
  auto itXmon = _dag->Vars().find( &Xmon );

  return std::make_pair( *itXlift, *itXmon );
}

template <typename T>
inline
FFVar
MINLPREF<T>::_insert_cheb
( FFVar const& x, const unsigned n )
const
{
  switch( n ){
    case 0:  return 1.;
    case 1:  return x;
    case 2:  return 2.*sqr(x)-1.;
    default: return n%2? 2.*_insert_cheb(x,n/2)*_insert_cheb(x,n/2+1)-x:
                         2.*sqr(_insert_cheb(x,n/2))-1.;
    //default: return 2.*x*_insert_cheb(x,n-1)-_insert_cheb(x,n-2);
  }
}
/*
template <typename T>
inline bool
MINLPREF<T>::update_initials
( double const* Xini, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );

  // Update initial values
  _Xini.resize( _nX, 0. ); // any NCO variable initialized to zero
  for( unsigned i=0; Xini && i<_nX0; i++ )
    _Xini[i] = Xini[i];

  // Update subgraphs 
  _set_subgraph( os );
  
  // Propagate initial values for lifted variables
  unsigned j=0; 
  bool noexcp = true;
  for( auto const& [i,Fi] : _Xlift ){
    try{
      double Xi;
#ifdef MC__MINLPREF_DEBUG_BOUNDS
      _dag->output( _dag->subgraph( 1, &Fi ), " FOR LIFTED VARIABLE" );    
#endif
      _dag->eval( _Fops.at(_nF+j), _dwk, 1, &Fi, &Xi, _nX1, _Xvar.data(), _Xini.data() );
#ifdef MC__MINLPREF_DEBUG_INITIALS
      std::cout << "Xini[ " << i << "] = " << Xi << std::endl;
#endif
    }
    catch(...){
      // No cut added for function #j in case DAG evaluation failed
      continue;
      noexcp = false;
    }
    j++;
  }

  return noexcp;
}
*/
template <typename T>
inline bool
MINLPREF<T>::update_bounds
( T const* X, double const* Finc, bool const resetbnd, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _update_options(); // virtual function

  // Variable bounds
  unsigned const nX0 = _Xbnd.size();
  _Xbnd.resize( _nX );
  if( resetbnd )
    for( unsigned i=0; i<_nX; i++ )   _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
  else
    for( unsigned i=nX0; i<_nX; i++ ) _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
  for( unsigned i=0; i<_nX0; i++ )
    if( X && !_Xelim.count(i) && !Op<T>::inter( _Xbnd[i], X[i], _Xbnd[i] ) ) return false;
  
  // Update subgraphs 
  _set_subgraph( os );
  
  // Bound propagation for lifted variables
  unsigned j=0; 
  for( auto const& [i,Fi] : _Xlift ){
    try{
      T Xi;
#ifdef MC__MINLPREF_DEBUG_BOUNDS
      _dag->output( _dag->subgraph( 1, &Fi ), " FOR LIFTED VARIABLE" );    
#endif
      _dag->eval( _Fops.at(_nF+j), _Iwk, 1, &Fi, &Xi, _nX1, _Xvar.data(), _Xbnd.data() );
#ifdef MC__MINLPREF_DEBUG_BOUNDS
      std::cout << "Xbnd[ " << i << "] = " << _Xbnd[i] << std::endl;
      std::cout << "Xprop[ " << i << "] = " << Xi << std::endl;
#endif
      if( !Op<T>::inter( _Xbnd[i], Xi, _Xbnd[i] ) ) return false;
    }
    catch(...){
      // No cut added for function #j in case DAG evaluation failed
      continue;
    }
    j++;
  }

  // Function bounds
  unsigned const nF0 = _Fbnd.size();
  _Fbnd.resize( _nF );
  if( resetbnd )
    for( unsigned i=0; i<_nF; i++ )   _Fbnd[i] = T( _Flow[i], _Fupp[i] );
  else
    for( unsigned i=nF0; i<_nF; i++ ) _Fbnd[i] = T( _Flow[i], _Fupp[i] );

  if( !Finc ) _Fbnd[0] = T( _Flow[0], _Fupp[0] );
  else if( _objsense == -1 && !Op<T>::inter( _Fbnd[0], T(-BASE_OPT::INF,*Finc), _Fbnd[0] ) ) return false;
  else if( _objsense ==  1 && !Op<T>::inter( _Fbnd[0], T( *Finc,BASE_OPT::INF), _Fbnd[0] ) ) return false;

  return true;
}

template <typename T>
inline
int
MINLPREF<T>::_propagate_bounds
()
{
#ifdef MC__MINLPREF_DEBUG_CP
  _dag->output( _Fallops );
#endif
  
  // Apply constraint propagation
  //auto tstart = stats.start();
  int flag = _dag->reval( _Fallops, _CPbnd, _nF, _Fvar.data(), _Fbnd.data(), _nX, _Xvar.data(),
                          _Xbnd.data(), _IINF, options.CPMAX, options.CPTHRES );
  //stats.walltime_cprop += stats.walltime( tstart );
  
#ifdef MC__MINLPREF_DEBUG_CP
  std::cout << "\nReduced Box:\n";
  int i=0;
  for( auto const& bnd : _CPbnd )
    std::cout << "WK[" << i++ << "] = " << bnd << std::endl;
#endif

  // Round binary and integer variables accordingly
  for( unsigned ix=0; ix<_nX; ++ix )
    if( _Xtyp[ix] > 0 ) _Xbnd[ix] = T( std::ceil( Op<T>::l(_Xbnd[ix]) ), std::floor( Op<T>::u(_Xbnd[ix]) ) );

  return flag;
}

template <typename T>
inline
bool
MINLPREF<T>::propagate_bounds
( T const* X, double const* Finc, const bool resetbnd, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _update_options(); // virtual function

  // Update variable bounds
  if( !update_bounds( X, Finc, resetbnd, os ) ){
    std::cout << std::endl << "# MODEL FOUND INFEASIBLE" << std::endl;
    return false;
  }

  int cpred = _propagate_bounds();
  if( cpred < 0 ){
    std::cout << std::endl << "# MODEL FOUND INFEASIBLE (ROUND " << -cpred << ")" << std::endl;
    { int dum; std::cout << "PAUSED - ENTER 1 to CONTINUE"; std::cin >> dum; } 
    return false;
  }
  if( options.DISPLEVEL )
    os << "# BOUND PROPAGATION: " << cpred << " ROUNDS" << std::endl;
  return true;
}

template <typename T>
inline
void
MINLPREF<T>::_search_invertible_constraints
( std::ostream& os )
{
  if( _Fctreq.empty() ) return;
  
  // Set negative weight to discrete variable to prevent their elimination
  std::map<FFVar const*,double,lt_FFVar> wVar;
  if( !options.DISCRELIM )
    for( unsigned i=0; i<_nX; ++i )
      if( _Xtyp[i] > 0 ) wVar[&_Xvar[i]] = -1.;

  _SEenv.set( _dag );
  _SEenv.options = options.SELIM;
  _SEenv.options.MIPTIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime( _tstart ) );
  _SEenv.process( _Fctreq, _Fvar.data(), wVar );//, true );
#ifdef MC__MINLPREF_DEBUG_ELIM
  { std::cout << _SEenv << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif
  auto const& [vVar,vCtr,vAux] = _SEenv.VarElim();
  unsigned const nDep = vVar.size();

  // Track dependent and independent variables
  _ndxDep.resize( nDep );
  std::set<FFVar const*, lt_FFVar> setVar;
  std::vector<FFVar> vDep( nDep ), vSys( nDep );
  std::vector<FFVar>::const_reverse_iterator itvar=vVar.crbegin(), itctr=vCtr.crbegin();
  std::vector<FFVar>::reverse_iterator itdep=vDep.rbegin(), itsys=vSys.rbegin();
  std::vector<std::pair<unsigned,unsigned>>::reverse_iterator itndx=_ndxDep.rbegin(); 
  for( ; itvar!=vVar.rend(); ++itvar, ++itctr, ++itdep, ++itsys, ++itndx ){

    // track inverted constraint in _Fvar
    auto ite = _Fctreq.cbegin();
    for( ; ite != _Fctreq.cend(); ++ite )
      if( itctr->id().second == _Fvar[*ite].id().second ) break;
    assert( ite != _Fctreq.cend() );
    *itsys = _Fvar[*ite];
    itndx->second = *ite;

    // track candidate variable in _Xvar
    unsigned i = 0;
    for( ; i<_nX; ++i )
      if( itvar->id().second == _Xvar[i].id().second ) break;
    *itdep = _Xvar[i];
    setVar.erase( &_Xvar[i] );
    itndx->first = i;

    // track other participating variables in _Xvar
    auto sgsys = _dag->subgraph( 1, &_Fvar[*ite] );
    for( auto const& Op : sgsys.l_op ){
      if( Op->type != FFOp::VAR || Op->varout[0]->id().second == _Xvar[i].id().second ) continue;
      setVar.insert( Op->varout[0] );
    }
  }

  // all other participating variables in _Xvar
  _ndxIndep.clear();
  _ndxIndep.reserve( setVar.size() );
  unsigned k = 0;
  for( auto const& var : _Xvar ){
    if( setVar.find( &var ) != setVar.end() ) _ndxIndep.push_back( k );
    k++;
  }

  // Form triangular system of invertible constraints
  _AEBND.reset_par();
  _AEBND.reset_var();
  _AEBND.set_dag( _dag );
  for( unsigned const& k : _ndxIndep ) _AEBND.add_var( _Xvar[k] );
  //auto const& pVar : setVar ) _AEBND.add_var( *pVar );
  _AEBND.set_dep( vDep, vSys );
  _AEBND.options = options.AEBND;
  _AEBND.setup( nDep, nullptr, nullptr, nullptr, os );
}

template <typename T>
inline
bool
MINLPREF<T>::_bound_invertible_constraints
()
{
  std::vector<T> _bndIndep, _bndDep;
  _bndIndep.reserve( _ndxIndep.size() );
  _bndDep.reserve( _ndxDep.size() );
  for( auto const& ndx : _ndxIndep ) _bndIndep.push_back( _Xbnd[ndx] );
  for( auto const& [ndx,eqn] : _ndxDep ) _bndDep.push_back( _Xbnd[ndx] );
  return( _AEBND.solve( _bndIndep.data(), _bndDep.data(), _bndDep.data() ) == t_aebnd::NORMAL );
}

template <typename T>
inline
bool
MINLPREF<T>::eliminate_invertible_constraints
( bool const add2dag, std::ostream& os )
{
  _update_options(); // virtual function
  if( options.DISPLEVEL )
    os << "# SEARCHING INVERTIBLE CONSTRAINTS" << std::endl;
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _tstart = stats.start();
  
  _search_invertible_constraints( os );
  if( !add2dag ) return true;
  
  auto const& [vVar,vCtr,vAux] = _SEenv.VarElim();
  if( _Fctreq.empty() || vVar.empty() ) return false;

  if( options.DISPLEVEL )
    os << "# ELIMINATING INVERTIBLE CONSTRAINTS" << std::endl;
  std::set<unsigned> Fremain;
  for( unsigned j=0; j<_nF; ++j ) Fremain.insert( j );

  // Bound dependent variables of invertible equality constraints using Gauss-Siedel interval methods
  if( options.INVBNDGS && !_bound_invertible_constraints() ) return false;

  // iterate over set of eliminated variables
  std::vector<FFVar>::const_reverse_iterator itvar=vVar.crbegin(), itaux=vAux.crbegin();
  std::vector<std::pair<unsigned,unsigned>>::reverse_iterator itndx=_ndxDep.rbegin(); 
  for( unsigned iblk=0; itvar!=vVar.rend(); ++itvar, ++itaux, ++itndx, ++iblk ){

    // check uniqueness of inverted constraint for current variable ranges
    if( options.INVBNDGS && !_AEBND.uniblk( iblk ) ) continue;

    // track inverted variable/constraint in _Xvar and _Fvar
    auto const& [i,j] = *itndx;

    // compose other constraints with *itaux instead of *itvar
    Fremain.erase( j ); // Drop j temporarilly
    const mc::FFVar* Fcomp = _dag->compose( Fremain, _Fvar.data(), 1, &*itvar, &*itaux );
    for( unsigned const& j : Fremain ){
#ifdef MC__MINLPREF_DEBUG_ELIM
      std::ostringstream ostr;
      ostr << " OF FUNCTION " << _Fvar[j] << " COMPOSED WITH ELIMINATED VARIABLE " << *itvar;
      _dag->output( _dag->subgraph( 1, &Fcomp[j] ), ostr.str() );
#endif
      _Fvar[j] = Fcomp[j];
    }
    delete[] Fcomp;
    
    // substitute equality constraint with inverted expression and corresponding bounds
    if( _Xlow[i] > -0.999*BASE_OPT::INF || _Xupp[i] < 0.999*BASE_OPT::INF ){
      if( options.INVKEEPLIN ){
        _Fvar[j] = *itaux - *itvar;
        _Flow[j] = _Fupp[j] = 0;
        _Fbnd[j] = T( 0 );
      }
      else{
        _Fvar[j] = *itaux;
        _Flow[j] = _Xlow[i];
        _Fupp[j] = _Xupp[i];
        _Fbnd[j] = T( _Xlow[i], _Xupp[i] );
        _Xelim.insert( i ); // Earmark var i as eliminated
      }
      Fremain.insert( j ); // Reinsert j
#ifdef MC__MINLPREF_DEBUG_ELIM
      std::ostringstream ostr;
      ostr << " OF ELIMINATED VARIABLE " << *itvar << " IN [" << _Xlow[i] << "," << _Xupp[i] << "]";
      _dag->output( _dag->subgraph( 1, &*itaux ), ostr.str() );
#endif
    }
  }

  // erase unused constraints and corresponding bounds
  auto itFvar = _Fvar.begin();
  auto itFlow = _Flow.begin();
  auto itFupp = _Fupp.begin();
  auto itFbnd = _Fbnd.begin();
  for( unsigned j=0; j<_nF; ++j ){
    if( !Fremain.count( j ) ){
#ifdef MC__MINLPREF_DEBUG_ELIM
      std::cout << "REMOVING CONSTRAINT " << *itFvar << std::endl;
#endif
      itFvar = _Fvar.erase( itFvar ); 
      itFlow = _Flow.erase( itFlow ); 
      itFupp = _Fupp.erase( itFupp );
      itFbnd = _Fbnd.erase( itFbnd );
      continue;
    }
#ifdef MC__MINLPREF_DEBUG_ELIM
    std::cout << "KEEPING CONSTRAINT " << *itFvar << " IN [" << *itFlow << "," << *itFupp << "]" << std::endl;
#endif
    ++itFvar; ++itFlow; ++itFupp; ++itFbnd;
  }
/*
  // erase unused variables and corresponding bounds
  auto itXvar = _Xvar.begin();
  auto itXlow = _Xlow.begin();
  auto itXupp = _Xupp.begin();
  auto itXbnd = _Xbnd.begin();
  auto itXtyp = _Xtyp.begin();
  for( unsigned i=0; i<_nX; ++i ){
    if( Xelim.count( i ) ){
#ifdef MC__MINLPREF_DEBUG_ELIM
      std::cout << "REMOVING VARIABLE " << *itXvar << " IN [" << *itXlow << "," << *itXupp << "]" << std::endl;
#endif
      itXvar = _Xvar.erase( itXvar ); 
      itXlow = _Xlow.erase( itXlow ); 
      itXupp = _Xupp.erase( itXupp );
      itXbnd = _Xbnd.erase( itXbnd );
      itXtyp = _Xtyp.erase( itXtyp );
      continue;
    }
#ifdef MC__MINLPREF_DEBUG_ELIM
    std::cout << "KEEPING VARIABLE " << *itXvar << " IN [" << *itXlow << "," << *itXupp << "]" << std::endl;
#endif
    ++itXvar; ++itXlow; ++itXupp; ++itXbnd; ++itXtyp;
  }
*/
  // update variable and function size and type
  _sgupdt = true;
  _update_model();
  if( options.DISPLEVEL ) _display_model( os );

#ifdef MC__MINLPREF_DEBUG_ELIM
  for( unsigned i=0; i<_nX; ++i ){
    if( _Xelim.count( i ) ) continue;
    std::cout << "VARIABLE " << _Xvar[i] << " IN [" << _Xlow[i] << "," << _Xupp[i] << "]" << std::endl;
  }
  for( unsigned j=0; j<_nF; ++j )
    std::cout << "CONSTRAINT " << _Fvar[j] << " IN [" << _Flow[j] << "," << _Fupp[j] << "]" << std::endl;
#endif

  return true;
}

template <typename T>
inline
unsigned
MINLPREF<T>::_search_reduction_constraints
( std::set<unsigned> Ftpol, std::ostream& os )
{
  if( _Fctreq.empty() ) return 0;

  // Map linear, quadratic and polynomial expressions
  std::set<unsigned> Ftpoleq = Ftpol;
  for( auto itpol=Ftpoleq.begin(); itpol!=Ftpoleq.end(); ){
    if( _Fctreq.count(*itpol) ){
      ++itpol;
      continue;
    }
    itpol = Ftpoleq.erase( itpol );
  }

  // Populate polynomial reduction problem and search
  _SRenv.options = options.SRED;
  _SRenv.options.MIPTIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime( _tstart ) );
  _SRenv.set_monomials( Ftpol, _SPFvar.data(), &t_ffpoly::mapmon );
  unsigned nred = _SRenv.search_reductions( Ftpoleq, _SPFvar.data(), &t_ffpoly::mapmon, options.REDELIM );
//#ifdef MC__MINLPREF_DEBUG_RED
  std::cout << _SRenv;
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
//#endif
  return nred;
}

template <typename T>
inline
bool
MINLPREF<T>::append_reduction_constraints
( bool const add2dag, std::ostream& os )
{
  _update_options(); // virtual function
  if( options.DISPLEVEL )
    os << "# SEARCHING REDUCTION CONSTRAINTS" << std::endl;
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _tstart = stats.start();
  
  std::set<unsigned> Ftpol = _Fpol;
  Ftpol.insert( _Fquad.cbegin(), _Fquad.cend() );
  Ftpol.insert( _Flin.cbegin(), _Flin.cend() );
  if( !_flatten_functions( Ftpol, false ) ) return false;

  unsigned nred = _search_reduction_constraints( Ftpol, os );
  if( !nred || !add2dag ) return true;

  // Substitute linearized monomials
  if( options.REDELIM ){
    // append auxiliary variables for linearized monomials
    std::map< t_ffmon const*, FFVar const*, lt_ffmon > mapmonlin; 
    for( auto const& monlin : _SRenv.ElimMon() ){
      auto [pauxlin,pvarlin] = _insert_ffmon( monlin, t_ffpoly::Options::MONOM, false );
      _Xlift[_Xvar.size()] = *pauxlin; // <- stores monomial DAG expression
      _Xvar.push_back( *pvarlin );
      _Xlow.push_back( -BASE_OPT::INF );
      _Xupp.push_back(  BASE_OPT::INF );
      //_Xbnd.push_back( T( -BASE_OPT::INF, BASE_OPT::INF ) );
      _Xtyp.push_back( 0 );
      mapmonlin[&monlin] = pvarlin;
#ifdef MC__MINLPBND_DEBUG_RED
      std::cout << "Linearized monomial " << *pvarlin << " := " << monlin.display(t_ffpoly::Options::MONOM) << std::endl;
      _dag->output( _dag->subgraph( 1, pauxlin ) );
#endif
    }

    // substitute auxiliary variables in existing polynomial expressions
    for( auto const& i : Ftpol ){
      // start with substituting most complex monomial
      for( auto itmonlin=mapmonlin.rbegin(); itmonlin!=mapmonlin.rend(); ++itmonlin ){
        auto const& [pmonlin,pvarlin] = *itmonlin;
        for( auto itmon=_SPFvar[i].mapmon().begin(); itmon!=_SPFvar[i].mapmon().end(); ){
          auto const& [mon,coef] = *itmon;
          if( !pmonlin->subseteq( mon ) ){
            ++itmon;
            continue;
          }
          t_ffmon newmon( mon );
          newmon -= *pmonlin;
          newmon += t_ffmon( pvarlin );
          _SPFvar[i].mapmon().insert( std::make_pair( newmon, coef ) );
          itmon = _SPFvar[i].mapmon().erase( itmon );
        }
      }
      // substitute existing DAG expression
      _Fvar[i] = _insert_ffpol( _SPFvar[i] );
#ifdef MC__MINLPBND_DEBUG_RED
      std::ostringstream ostr; ostr << " of polynomial expression F[" << i << "]";
      _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
    }

    // substitute auxiliary variables in reduction polynomial expressions
    _SPFvar = _SRenv.RedCtr(); // local copy
    for( unsigned i=0; i<_SPFvar.size(); ++i ){
      // start with substituting most complex monomial
      for( auto itmonlin=mapmonlin.rbegin(); itmonlin!=mapmonlin.rend(); ++itmonlin ){
        auto const& [pmonlin,pvarlin] = *itmonlin;
        for( auto itmon=_SPFvar[i].mapmon().begin(); itmon!=_SPFvar[i].mapmon().end(); ){
          auto const& [mon,coef] = *itmon;
          if( !pmonlin->subseteq( mon ) ){
            ++itmon;
            continue;
          }
          t_ffmon newmon( mon );
          newmon -= *pmonlin;
          newmon += t_ffmon( pvarlin );
          _SPFvar[i].mapmon().insert( std::make_pair( newmon, coef ) );
          itmon = _SPFvar[i].mapmon().erase( itmon );
        }
      }
      // append new DAG expression and bounds
      _Fvar.push_back( _insert_ffpol( _SPFvar[i] ) );
      _Flow.push_back( 0. );
      _Fupp.push_back( 0. );
#ifdef MC__MINLPBND_DEBUG_RED
      std::ostringstream ostr; ostr << " of reduction polynomial cut F[" << _Fvar.size()-1 << "]";
      _dag->output( _dag->subgraph( 1, &_Fvar.back() ), ostr.str() );
#endif
    }
  }
  
  // Append reduction polynomial cuts without linearization
  else{
    // Append reduction polynomial cuts
    for( auto const& red : _SRenv.RedCtr() ){
      _Fvar.push_back( _insert_ffpol( red ) );
      _Flow.push_back( 0. );
      _Fupp.push_back( 0. );
#ifdef MC__MINLPBND_DEBUG_RED
      std::ostringstream ostr; ostr << " of reduction polynomial cut F[" << _Fvar.size()-1 << "]";
      _dag->output( _dag->subgraph( 1, &_Fvar.back() ), ostr.str() );
#endif
    }
  }

  // update variable and function size and type
  assert( _Fvar.size() == _Flow.size() && _Fvar.size() == _Fupp.size() ); 
  _sgupdt = true;
  _update_model();
  update_bounds( nullptr, nullptr, false, os );
  if( options.DISPLEVEL ) _display_model( os );
  
  return true;
}

template <typename T>
inline
bool
MINLPREF<T>::export_model
( std::string const gmsfile, double const* Xstart, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );
  _update_options(); // virtual function
  if( gmsfile.empty() ){
    if( options.DISPLEVEL > 0 )
      os << std::endl << "# GAMS FILENAME UNSPECIFIED" << std::endl;
    return false;
  }

  // Write relaxed model to GAMS file
  if( options.DISPLEVEL > 0 )
    os << std::endl << "# WRITING MODEL TO FILE: " << gmsfile << std::endl;
  GAMSWRITER<T> GMS;
  typename GAMSWRITER<T>::MODELTYPE type = (_Fgal.empty()&&_Fpol.empty()?
                                                     (_Fquad.empty()? GAMSWRITER<T>::MODELTYPE::LIN:
                                                                      GAMSWRITER<T>::MODELTYPE::QUAD):
                                                                      GAMSWRITER<T>::MODELTYPE::NLIN);

  //for( unsigned i=0; i<_nX; i++ ){
  //  if( Xinc && i<_nX0 )
  //    GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], &Xinc[i] );
  //  else if( !_Xini.empty() && i<_nX0 ) // issue is GAMS passes an initialization by default...
  //    GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], &_Xini[i] );
  //  else
  //    GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], nullptr );
  //}
  for( unsigned i=0; i<_nX0; i++ ){
    if( _Xelim.count( i ) ) continue;
#ifdef MC__MINLPREF_DEBUG_EXPORT
    std::cout << "MINLPREF::export_model ** Adding original variable " << _Xvar[i] << std::endl;
#endif
    GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], Xstart? &Xstart[i]: nullptr );
    //std::cout << _Xvar[i] << " in " << _Xbnd[i] << std::endl;
#ifdef MC__MINLPREF_DEBUG_INITIALS
    if( Xstart ) std::cout << "Xstart[ " << i << "] = " << Xstart[i] << std::endl;
#endif
  }
  unsigned j=0; 
  for( auto const& [i,Fi] : _Xlift ){
#ifdef MC__MINLPREF_DEBUG_EXPORT
    std::cout << "MINLPREF::export_model ** Adding lifted variable " << _Xvar[i] << std::endl;
#endif
    if( !Xstart )
      GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], nullptr );
    else{
      try{
        double Xi;
        _dag->eval( _Fops.at(_nF+j), _dwk, 1, &Fi, &Xi, _nX0, _Xvar.data(), Xstart );
        GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], &Xi );
#ifdef MC__MINLPREF_DEBUG_INITIALS
        std::cout << "Xstart[ " << i << "] = " << Xi << std::endl;
#endif
      }
      catch(...){
        GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], nullptr );
      }
      j++;
    }
  }

  GMS.set_functions( _dag, type, _nF, _Fvar.data(), _nX, _Xvar.data() );
  GMS.set_objective( 0, _objsense>0? BASE_OPT::MAX: BASE_OPT::MIN );
  GMS.set_constraints( 0, _nF, _Fbnd.data() );
  GMS.write( gmsfile );
  return true;
}

template <typename T>
inline
void
MINLPREF<T>::Options::display
( std::ostream& out )
const
{
  // Display MINLPREF Options
  out << std::left;
  out << std::setw(60) << "  APPEND NCO CUTS"
      << (NCOCUTS?"Y\n":"N\n");
  if( NCOCUTS ){
    out << std::setw(60) << "  METHOD FOR NCO CUTS";
    switch( NCOADIFF ){
     case FSA: out << "FSA\n";
     case ASA: out << "ASA\n";
    }
  }
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

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
#include "rltred.hpp"

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
template < typename DAG,
           typename T >
class MINLPREF
#if defined (MC__WITH_GAMS)
: protected virtual GAMSIO<DAG>,
  public virtual BASE_NLP<DAG>
#else
: public virtual BASE_NLP<DAG>
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

  typedef SLiftEnv<DAG> t_lift;
  typedef SElimEnv<DAG> t_elim;
  typedef AEBND<DAG,T> t_aebnd;

  using BASE_AE<DAG>::set;
  using BASE_AE<DAG>::dag;
  using BASE_AE<DAG>::set_dag;
  using BASE_AE<DAG>::par;
  using BASE_AE<DAG>::set_par;
  using BASE_AE<DAG>::add_par;
  using BASE_AE<DAG>::reset_par;
  using BASE_AE<DAG>::var;
  using BASE_AE<DAG>::set_var;
  using BASE_AE<DAG>::add_var;
  using BASE_AE<DAG>::reset_var;
  using BASE_AE<DAG>::update_vartyp;
  using BASE_AE<DAG>::dep;
  using BASE_AE<DAG>::set_dep;
  using BASE_AE<DAG>::add_dep;
  using BASE_AE<DAG>::reset_dep;
  using BASE_AE<DAG>::sys;
  using BASE_AE<DAG>::add_sys;
  using BASE_AE<DAG>::reset_sys;

  using BASE_NLP<DAG>::set_obj;
  using BASE_NLP<DAG>::add_ctr;

#if defined (MC__WITH_GAMS)
  using GAMSIO<DAG>::read;
#endif

protected:

  // Do not use BASE_AE<DAG>::_dag since redefined locally
  using BASE_AE<DAG>::_var;
  using BASE_AE<DAG>::_vartyp;
  using BASE_AE<DAG>::_varlb;
  using BASE_AE<DAG>::_varlm;
  using BASE_AE<DAG>::_varub;
  using BASE_AE<DAG>::_varum;
  using BASE_AE<DAG>::_dep;
  using BASE_AE<DAG>::_deplb;
  using BASE_AE<DAG>::_deplm;
  using BASE_AE<DAG>::_depub;
  using BASE_AE<DAG>::_depum;
  using BASE_AE<DAG>::_sys;
  using BASE_AE<DAG>::_sysm;
  using BASE_AE<DAG>::_par;

  using BASE_NLP<DAG>::_obj;
  using BASE_NLP<DAG>::_ctr;
  using BASE_NLP<DAG>::_nco;

#if defined (MC__WITH_GAMS)
  using GAMSIO<DAG>::_varini;
#endif

protected:

  //! @brief local copy of DAG (overides BASE_AE<DAG>::_dag)
  DAG*                      _dag;

  //! @brief environment for expression elimination
  t_elim                    _SEenv;
  //! @brief environment for expression lift
  t_lift                    _SLenv;
  //! @brief environment for sparse quadratic form
  t_quad                    _SQenv;
  //! @brief environment for sparse quadratic form
  //t_quad                    _SCQenv;
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
  std::set<unsigned>        _Flin;
  //! @brief subset of quadratic functions
  std::set<unsigned>        _Fquad;
  //! @brief subset of polynomial functions
  std::set<unsigned>        _Fpol;
  //! @brief subset of general functions
  std::set<unsigned>        _Fgal;
  //! @brief subset of equality-constrained functions
  std::set<unsigned>        _Fctreq;
  
  //! @brief Interval representation of 'unbounded' variables
  T                         _IINF;
  //! @brief Storage vector for constraint propagation in interval arithmetic
  std::vector<T>            _CPbnd;
  //! @brief Storage vector for interval arithmetic
  std::vector<T>            _Iwk;

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
  std::vector<typename t_lift::t_poly> _SPXvar;
  //! @brief Storage vector for polynomial functions in flattening
  std::vector<typename t_lift::t_poly> _SPFvar;

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
      RRLTCUTS(false), MIPQUADCUTS(false), PSDQUADCUTS(0),
      NCOCUTS(false), NCOADIFF(FSA), TIMELIMIT(6e2), DISPLEVEL(2),
      SELIM(), SLIFT(), SQUAD(), RLTRED(), AEBND()
      { SLIFT.LIFTDIV          = true;
        SLIFT.LIFTIPOW         = false;
        SELIM.MIPDISPLEVEL     = 0;
        SELIM.MIPTIMELIMIT     = TIMELIMIT;
        SQUAD.BASIS            = t_quad::Options::MONOM;
        SQUAD.ORDER            = t_quad::Options::INC;
        SQUAD.REDUC            = false;
        RLTRED.METHOD          = RLTRed::Options::ILP;
        RLTRED.LEVEL           = RLTRed::Options::PRIMSIM;
        RLTRED.TIMELIMIT       = TIMELIMIT;
        AEBND.BLKDEC           = t_aebnd::Options::DECOMPOSITION::RECUR;
        AEBND.DISPLEVEL        = 0; }
    //! @brief Assignment operator
    Options& operator= ( Options&options ){
        CPMAX         = options.CPMAX;
        CPTHRES       = options.CPTHRES;
        RRLTCUTS      = options.RRLTCUTS;
	MIPQUADCUTS   = options.MIPQUADCUTS;
        PSDQUADCUTS   = options.PSDQUADCUTS;
        NCOCUTS       = options.NCOCUTS;
        NCOADIFF      = options.NCOADIFF;
        TIMELIMIT     = options.TIMELIMIT;
        DISPLEVEL     = options.DISPLEVEL;
        SELIM         = options.SELIM;
        SLIFT         = options.SLIFT;
        SQUAD         = options.SQUAD;
        RLTRED        = options.RLTRED;
        AEBND         = options.AEBND;
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
    //! @brief Whether to add reduced RLT cuts
    bool RRLTCUTS;
    //! @brief Whether to minimize the number of auxiliary variables in quadratisation using MIP
    bool MIPQUADCUTS;
    //! @brief Whether to add PSD cuts within quadratisation (0: none; 1: 2-by-2; >1: 3-by-3)
    unsigned PSDQUADCUTS;
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
    typename RLTRed::Options RLTRED;
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
  DAG* dag
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
  std::vector<T> const& variable_bounds
    ()
    const
    { return _Xbnd; }

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
    ( bool const add2dag, std::ostream& os = std::cout );

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
    ( bool const add2dag, std::ostream& os = std::cout );

  //! @brief Eliminate variables from invertible equality constraints
  bool eliminate_invertible_constraints
    ( bool const bndinv, bool const add2dag, std::ostream& os = std::cout );

  //! @brief export reformulated optimization model to GAMS file
  bool export_model
    ( std::string const gmsfile, double const* Xinc=nullptr );

private:

  //! @brief Time point to enable TIMELIMIT option
  //std::chrono::time_point<std::chrono::system_clock> _tstart;

  //! @brief Set linear/nonlinear participating variables in functions
  void _update_model
    ();

  //! @brief Display model information
  void _display_model
    ( std::ostream& os = std::cout );

  //! @brief Set optimality cuts in original DAG
  bool _set_optimality_cuts
    ( std::vector<FFVar>& Xvar, std::vector<FFVar>& Fvar, std::ostream& os = std::cout);

  //! @brief Set linear/nonlinear participating variables in functions
  void _set_variable_class
    ();

  //! @brief Set linear/polynomial/nonlinear participating functions
  void _set_function_class
    ();

  //! @brief Set subgraphs for participating functions
  void _set_subgraph
    ( std::ostream& os = std::cout );

  //! @brief Tighten bounds using constraint propagation
  int _propagate_bounds
    ();

  //! @brief Flatten subset of functions in DAG
  bool _flatten_functions
    ( std::set<unsigned> const& Fndx, bool const add2dag );

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
    ( t_mon const& mon )
    const;

  //! @brief Search for invertible equality constraints and form correspond triangular constraint subsystem
  void _search_invertible_constraints
    ();

  //! @brief Bound dependent variables of invertible equality constraints using Gauss-Siedel interval methods
  bool _bound_invertible_constraints
    ();

  //! @brief Private methods to block default compiler methods
  MINLPREF
    ( MINLPREF<DAG,T> const& );
  MINLPREF<DAG,T>& operator=
    ( MINLPREF<DAG,T> const& );
};

template <typename DAG, typename T>
inline void
MINLPREF<DAG,T>::setup
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
#if defined (MC__WITH_GAMS)
  _Xini = _varini;
#else
  _Xini.clear();
#endif
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
      case BASE_OPT::EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );            break;
      case BASE_OPT::LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );            break;
      case BASE_OPT::GE: _Flow.push_back( 0. );             _Fupp.push_back( BASE_OPT::INF ); break;
    }
  }

  // then, dependent equations
  Fvar.insert( _Fvar.end(), _sys.begin(), _sys.end() );
  _Flow.insert( _Flow.end(), _sys.size(), 0. );
  _Fupp.insert( _Fupp.end(), _sys.size(), 0. );
 
  // set Fritz-John cuts and corresponding multipliers
  if( options.NCOCUTS ) _set_optimality_cuts( Xvar, Fvar, os );

  // local DAG copy
  if( _dag ) delete _dag;
  _dag = new DAG;
  _nP = Pvar.size(); _Pvar.resize( _nP );
  _dag->insert( BASE_NLP<DAG>::_dag, _nP, Pvar.data(), _Pvar.data() );
  _nX = _nX1 = Xvar.size(); _Xvar.resize( _nX );
  _dag->insert( BASE_NLP<DAG>::_dag, _nX, Xvar.data(), _Xvar.data() );
  _nF = Fvar.size(); _Fvar.resize( _nF );
  _dag->insert( BASE_NLP<DAG>::_dag, _nF, Fvar.data(), _Fvar.data() );
#ifdef MC__MINLPREF_DEBUG  
  _dag->output( _dag->subgraph( 1, _Fvar.data() ), " objective" );
#endif

  // Identify variable and function sets and create subgraphs
  _Xobj.set( _dag );
  _Xlift.clear();
  _set_variable_class();
  _set_function_class();
  _sgupdt = true;

  // Set default variable and function bounds
  _Xbnd.resize( _nX );
  for( unsigned i=0; i<_nX; i++ ) _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
  _Fbnd.resize( _nF );
  for( unsigned i=0; i<_nF; i++ ) _Fbnd[i] = T( _Flow[i], _Fupp[i] );

#if 0
  // search for reduced RLT cuts
  if( options.RRLTCUTS ) _search_reduction_constraints();
#endif


  //stats.reset();
  _issetup = true;
  if( options.DISPLEVEL ) _display_model( os );
}

template <typename DAG, typename T>
inline
bool
MINLPREF<DAG,T>::_set_optimality_cuts
( std::vector<FFVar>& Xvar, std::vector<FFVar>& Fvar, std::ostream& os )
{
  if( options.DISPLEVEL )
    os << "# APPENDING FIRST-ORDER OPTIMALITY CONDITIONS" << std::endl;

  if( !BASE_NLP<DAG>::set_nco( _Xtyp.data(), options.NCOADIFF==Options::ASA ) )
    return false;

  // cost multiplier
  Xvar.push_back( std::get<2>(_obj)[0] );
  _Xlow.push_back( 0. );
  _Xupp.push_back( 1. );
  _Xtyp.push_back( 0  );

  // regular constraint multipliers
  for( unsigned i=0; i<std::get<0>(_ctr).size(); ++i ){
    Xvar.push_back( std::get<2>(_ctr)[i] );
    _Xtyp.push_back( 0 ); // all constraint multipliers are continuous variables
    switch( std::get<0>(_ctr)[i] ){
      case BASE_OPT::LE:
      case BASE_OPT::GE: _Xlow.push_back(  0. ); _Xupp.push_back( 1. ); break;
      case BASE_OPT::EQ: _Xlow.push_back( -1. ); _Xupp.push_back( 1. ); break;
    }
  }

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
#ifdef MC__MINLPREF_DEBUG_NCOCUTS
    BASE_NLP<DAG>::_dag->output( BASE_NLP<DAG>::_dag->subgraph( 1, &Fvar.back() ), " FOR NCO" );    
#endif
    switch( std::get<0>(_nco)[i] ){
      case BASE_OPT::EQ: _Flow.push_back( 0. );             _Fupp.push_back( 0. );            break;
      case BASE_OPT::LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( 0. );            break;
      case BASE_OPT::GE: _Flow.push_back( 0. );             _Fupp.push_back( BASE_OPT::INF ); break;
    }
  }
  return true;
}

template <typename DAG, typename T>
inline
void
MINLPREF<DAG,T>::_update_model
()
{
  // update variable and function size and type
  _nX = _Xvar.size();
  _nF = _Fvar.size();
  _set_variable_class();    
  _set_function_class();
}

template <typename DAG, typename T>
inline
void
MINLPREF<DAG,T>::_display_model
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

template <typename DAG, typename T>
inline void
MINLPREF<DAG,T>::_set_variable_class
()
{
  FFDep Fworst( 0. );
  for( auto const& Fj : _Fvar )
    Fworst += Fj.dep();
#ifdef MC__MINLPREF_DEBUG
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

template <typename DAG, typename T>
inline void
MINLPREF<DAG,T>::_set_function_class
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
    if( j && _Flow[j] == 0. && _Fupp[j] == 0. ) _Fctreq.insert( j );
  }
}


template <typename DAG, typename T>
inline void
MINLPREF<DAG,T>::_set_subgraph
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

#if 0
template <typename DAG, typename T>
inline
void
MINLPREF<DAG,T>::_search_reduction_constraints
()
{
  if( _Fctreq.empty() ) return;

  // search for reduced RLT cuts
  RLTRed RRLT( _dag );
  RRLT.options = options.RLTRED;
  RRLT.search( _Fctreq, _Fvar.data() );

  // append any reduced RLT cuts
  for( auto const& pFred : RRLT.constraints() ){
#ifdef MC__MINLPREF_DEBUG_RRLTCUTS
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
#endif

template <typename DAG, typename T>
inline
bool
MINLPREF<DAG,T>::lift_polynomial_subexpressions
( bool const add2dag, std::ostream& os )
{
  if( _Fgal.empty() ) return false;

  _SLenv.set( _dag );
  _SLenv.options = options.SLIFT;
  if( options.DISPLEVEL )
    os << "# LIFTING POLYNOMIAL SUBEXPRESSIONS" << std::endl;
  _SLenv.process( _Fgal, _Fvar.data(), true );
#ifdef MC__MINLPREF_DEBUG_LIFT
  { std::cout << _SLenv << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif
  if( !add2dag ) return true;
  
  // append auxiliary variables
  //std::set<unsigned> Fred;
  for( auto const& [pAux,pVar] : _SLenv.Aux() ){
    //bool is_dep = false;
    //unsigned i = 0;
    //for( auto it=_Fgal.begin(); it!=_Fgal.end(); ++it ){
    //  i = *it;
    //  if( _Fvar[i] != *pAux ) continue;
    //  is_dep = true;
    //  break;
    //}
    _Xlift[_Xvar.size()] = *pAux; // <- stores original DAG expression
    _Xvar.push_back( *pVar );
    _Xlow.push_back( -BASE_OPT::INF ); //is_dep? _Flow[i]: -BASE_OPT::INF );
    _Xupp.push_back(  BASE_OPT::INF ); //is_dep? _Fupp[i]:  BASE_OPT::INF );
    _Xtyp.push_back( 0 );
    
    // update/erase corresponding entries in function vectors (this is not efficient...)
    //if( !is_dep ) continue;
    //if( !i ){
    //  _Fvar[0] = *pVar;
    //  continue;
    //}
    //Fred.insert( i ); 
  }
  //for( auto it=Fred.rbegin(); it!=Fred.rend(); ++it ){
  //  unsigned i = *it;
  //  auto itFvar = _Fvar.begin(); std::advance( itFvar, i ); _Fvar.erase( itFvar );
  //  auto itFlow = _Flow.begin(); std::advance( itFlow, i ); _Flow.erase( itFlow );
  //  auto itFupp = _Fupp.begin(); std::advance( itFupp, i ); _Fupp.erase( itFupp );
  //}

  // append auxiliary polynomial constraints
  assert( _Fgal.size() == _SLenv.Dep().size() );
  auto itgal = _Fgal.begin();
  for( auto const& expr : _SLenv.Dep() ){
    _Fvar[*itgal] = expr;
    // Do not modify lower and upper constraint range
    ++itgal;
  }

  // append auxiliary polynomial constraints
  for( auto const& poly : _SLenv.Poly() ){
    _Fvar.push_back( poly );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
  }

  // append auxiliary non-polynomial constraints
  for( auto const& trans : _SLenv.Trans() ){
    _Fvar.push_back( trans );
    _Flow.push_back( 0. );
    _Fupp.push_back( 0. );
  }

  // update variable and function size and type
  _sgupdt = true;
  _update_model();
  update_bounds( nullptr, nullptr, false, os );
  if( options.DISPLEVEL ) _display_model( os );
  
  return true;
}

template <typename DAG, typename T>
inline bool
MINLPREF<DAG,T>::_flatten_functions
( std::set<unsigned> const& Fndx, bool const add2dag )
{
  if( Fndx.empty() ) return false;

  // Create vector of all semi-algebraic expressions
  t_poly::options.BASIS = t_poly::Options::MONOM;
  _SPXvar.resize( _nX );
  for( unsigned ix=0; ix<_nX; ix++ ) _SPXvar[ix].var( &_Xvar[ix] );
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

template <typename DAG, typename T>
inline bool
MINLPREF<DAG,T>::flatten_linear_functions
( bool const add2dag )
{
  return _flatten_functions( _Flin, add2dag );
}

template <typename DAG, typename T>
inline bool
MINLPREF<DAG,T>::flatten_quadratic_functions
( bool const add2dag )
{
  return _flatten_functions( _Fquad, add2dag );
}

template <typename DAG, typename T>
inline bool
MINLPREF<DAG,T>::flatten_polynomial_functions
( bool const add2dag )
{
  return _flatten_functions( _Fpol, add2dag );
}

template <typename DAG, typename T>
inline bool
MINLPREF<DAG,T>::quadratize_polynomial_functions
( bool const add2dag, std::ostream& os )
{
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
  if( options.MIPQUADCUTS ){
    if( options.DISPLEVEL )
      os << "# OPTIMIZING QUADRATIC DECOMPOSITION" << std::endl;
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
    auto [pAux,pVar] = _insert_mon( mon );
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

  // update variable and function size and type
  _sgupdt = true;
  _update_model();
  update_bounds( nullptr, nullptr, false, os );
  if( options.DISPLEVEL ) _display_model( os );
  
  return true;
}
/*
template <typename DAG, typename T>
inline void
MINLPREF<DAG,T>::quadratize_polynomial_subexpressions
( bool const add2dag, std::ostream& os )
{
  std::set<unsigned> Fsalg = _Flin;
  Fsalg.insert( _Fquad.cbegin(), _Fquad.cend() );
  Fsalg.insert( _Fpol.cbegin(),  _Fpol.cend()  ); 
  if( Fsalg.empty() ) return;

  // Create vector of all semi-algebraic expressions
  t_poly::options.BASIS = t_poly::Options::MONOM;
  std::vector<typename t_lift::t_poly> SPXvar( _nX ), SPFvar( _nF );
  for( unsigned ix=0; ix<_nX; ix++ ) SPXvar[ix].var( &_Xvar[ix] );
  //for( auto const& var: _Xvar ) SPXvar.push_back( t_poly( var ) );
  _dag->eval( Fsalg, _Fvar.data(), SPFvar.data(), _nX, _Xvar.data(), SPXvar.data() );
#ifdef MC__MINLPREF_DEBUG_LIFT
  for( unsigned const& i : Fsalg ){
    std::ostringstream ostr; ostr << " of polynomial expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
    std::cout << "Polynomial expression " << i << ":\n" << SPFvar[i];
  }
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum;}
#endif

  // Substitute linear and quadratic expressions in DAG
  for( unsigned const& i : _Flin ){
    _Fvar[i] = _SLenv.insert_dag( SPFvar[i] );
#ifdef MC__MINLPREF_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of flattened linear expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
  }
  for( unsigned const& i : _Fquad ){
    _Fvar[i] = _SLenv.insert_dag( SPFvar[i] );
#ifdef MC__MINLPREF_DEBUG_LIFT
    std::ostringstream ostr; ostr << " of flattened quadratic expression F[" << i << "]";
    _dag->output( _dag->subgraph( 1, &_Fvar[i] ), ostr.str() );
#endif
  }

  // Transform variable indexing in quadratic and polynomial expressions
  std::map<FFVar const*, unsigned, lt_FFVar> FFmatch;
  unsigned ivar = 0;
  for( auto const& var : _Xvar ) FFmatch[&var] = ivar++;
  unsigned ifun = 0;
  std::set<unsigned> Ftpol = _Fpol; Ftpol.insert( _Fquad.cbegin(), _Fquad.cend() );
  std::vector<t_poly> SPol( Ftpol.size() );
  for( unsigned const& i : Ftpol ){
    for( auto const& [FFmon,coef] : SPFvar[i].mapmon() ){
      t_mon mon( FFmon.tord, FFmon.expr, FFmatch ); 
      SPol[ifun] += std::make_pair( mon, coef );
    }
    ++ifun;
  }

  // Apply quadratisation to polynomial expressions
  _SQenv.reset();
  _SQenv.options = options.SQUAD;
  _SQenv.process( SPol.size(), SPol.data(), &t_poly::mapmon, t_quad::Options::MONOM );
  if( options.MIPQUADCUTS ) _SQenv.optimize( true );
#ifdef MC__MINLPREF_DEBUG_LIFT
  double viol = _SQenv.check( SPol.size(), SPol.data(), &t_poly::mapmon, t_quad::Options::MONOM );
  std::cout << "violation: " << viol << std::endl << _SQenv << std::endl;
  {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif
  if( !add2dag ) return;

  // Add higher-order monomials in basis to DAG
  std::map< t_mon, FFVar, lt_mon > mapmon; 
  for( auto const& mon : _SQenv.SetMon() ){
    if( mon.tord == 1 ) mapmon[mon] = _Xvar[mon.expr.cbegin()->first];
    if( mon.tord <= 1 ) continue;
    auto [pAux,pVar] = _insert_mon( mon );
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

  // update variable and function size and type
  _sgupdt = true;
  _update_model();
  update_bounds( nullptr, nullptr, false, os );
  if( options.DISPLEVEL ) _display_model( os );
}
*/
template <typename DAG, typename T>
inline
FFVar
MINLPREF<DAG,T>::_insert_quad
( t_quad::map_SQuad const& quad, std::map< t_mon, FFVar, lt_mon >& mapmon )
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

template <typename DAG, typename T>
inline
std::pair< FFVar const*, FFVar const* >
MINLPREF<DAG,T>::_insert_mon
( t_mon const& mon )
const
{
  // define power monomial expression
  FFVar Xlift( 1e0 );
  for( auto const& [ivar,iord] : mon.expr ){
    switch( options.SQUAD.BASIS ){
     // Monomial basis
     case t_quad::Options::MONOM:
      Xlift *= pow( _Xvar[ivar], (int)iord );
      break;
     // Chebyshev basis
     case t_quad::Options::CHEB:
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

template <typename DAG, typename T>
inline
FFVar
MINLPREF<DAG,T>::_insert_cheb
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

template <typename DAG, typename T>
inline bool
MINLPREF<DAG,T>::update_bounds
( T const* X, double const* Finc, bool const resetbnd, std::ostream& os )
{
  // Variable bounds
  unsigned const nX0 = _Xbnd.size();
  _Xbnd.resize( _nX );
  if( resetbnd )
    for( unsigned i=0; i<_nX; i++ )   _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
  else
    for( unsigned i=nX0; i<_nX; i++ ) _Xbnd[i] = T( _Xlow[i], _Xupp[i] );
  for( unsigned i=0; i<_nX0; i++ )
    if( X && !Op<T>::inter( _Xbnd[i], X[i], _Xbnd[i] ) ) return false;
  
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
      j++;
    }
    catch(...){
      // No cut added for function #j in case DAG evaluation failed
      continue;
    }
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

template <typename DAG, typename T>
inline
int
MINLPREF<DAG,T>::_propagate_bounds
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

template <typename DAG, typename T>
inline
bool
MINLPREF<DAG,T>::propagate_bounds
( T const* X, double const* Finc, const bool resetbnd, std::ostream& os )
{
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );

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

template <typename DAG, typename T>
inline
void
MINLPREF<DAG,T>::_search_invertible_constraints
()
{
  if( _Fctreq.empty() ) return;

  _SEenv.set( _dag );
  _SEenv.options = options.SELIM;
  _SEenv.process( _Fctreq, _Fvar.data() );//, true );
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
  std::vector<FFVar>::iterator itdep=vDep.begin(), itsys=vSys.begin();
  std::vector<std::pair<unsigned,unsigned>>::iterator itndx=_ndxDep.begin(); 
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
    for( auto const& [Op,mov] : sgsys.l_op ){
      if( Op->type != FFOp::VAR || Op->pres->id().second == _Xvar[i].id().second ) continue;
      setVar.insert( Op->pres );
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
  _AEBND.setup();
}

template <typename DAG, typename T>
inline
bool
MINLPREF<DAG,T>::_bound_invertible_constraints
()
{
  std::vector<T> _bndIndep, _bndDep;
  _bndIndep.reserve( _ndxIndep.size() );
  _bndDep.reserve( _ndxDep.size() );
  for( auto const& ndx : _ndxIndep ) _bndIndep.push_back( _Xbnd[ndx] );
  for( auto const& [ndx,eqn] : _ndxDep ) _bndDep.push_back( _Xbnd[ndx] );
  return( _AEBND.solve( _bndIndep.data(), _bndDep.data(), _bndDep.data() ) == t_aebnd::NORMAL );
}

template <typename DAG, typename T>
inline
bool
MINLPREF<DAG,T>::eliminate_invertible_constraints
( bool const bndinv, bool const add2dag, std::ostream& os )
{
  _search_invertible_constraints();
  if( !add2dag ) return true;
  
  auto const& [vVar,vCtr,vAux] = _SEenv.VarElim();
  if( _Fctreq.empty() || vVar.empty() ) return false;

  std::set<unsigned> Fremain;
  for( unsigned j=0; j<_nF; ++j ) Fremain.insert( j );

  // Bound dependent variables of invertible equality constraints using Gauss-Siedel interval methods
  if( bndinv && !_bound_invertible_constraints() ) return false;

  // iterate over set of eliminated variables
  std::vector<FFVar>::const_reverse_iterator itvar=vVar.crbegin(), itaux=vAux.crbegin();
  std::vector<std::pair<unsigned,unsigned>>::iterator itndx=_ndxDep.begin(); 
  for( unsigned iblk=0; itvar!=vVar.rend(); ++itvar, ++itaux, ++itndx, ++iblk ){

    // check uniqueness of inverted constraint for current variable ranges
    if( bndinv && !_AEBND.uniblk( iblk ) ) continue;

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
      _Fvar[j] = *itaux - *itvar;
      _Flow[j] = _Fupp[j] = 0;
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
  for( unsigned j=0; j<_nF; ++j ){
    if( !Fremain.count( j ) ){
#ifdef MC__MINLPREF_DEBUG_ELIM
      std::cout << "REMOVING CONSTRAINT " << *itFvar << std::endl;
#endif
      itFvar = _Fvar.erase( itFvar ); 
      itFlow = _Flow.erase( itFlow ); 
      itFupp = _Fupp.erase( itFupp );
      continue;
    }
    ++itFvar; ++itFlow; ++itFupp;
  }

  // update variable and function size and type
  _sgupdt = true;
  _update_model();
  if( options.DISPLEVEL ) _display_model( os );
  return true;
}

template <typename DAG, typename T>
inline
bool
MINLPREF<DAG,T>::export_model
( std::string const gmsfile, double const* Xinc )
{
  if( gmsfile.empty() ){
    std::cout << std::endl << "# GAMS FILENAME UNSPECIFIED" << std::endl;
    return false;
  }
  if( !_issetup ) throw Exceptions( Exceptions::SETUP );

  // Write relaxed model to GAMS file
  if( options.DISPLEVEL > 0 )
    std::cout << std::endl << "# WRITING MODEL TO FILE: " << gmsfile << std::endl;
  GAMSWRITER<DAG,T> GMS;
  for( unsigned i=0; i<_nX; i++ ){
    if( Xinc && i<_nX0 )
      GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], &Xinc[i] );
    else if( !_Xini.empty() && i<_nX0 ) // issue is GAMS passes an initialization by default...
      GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], &_Xini[i] );
    else
      GMS.add_variable( _Xvar[i], _Xtyp[i], &_Xbnd[i], nullptr );
  }
  typename GAMSWRITER<DAG,T>::MODELTYPE type = (_Fgal.empty()&&_Fpol.empty()?
                                               (_Fquad.empty()? GAMSWRITER<DAG,T>::MODELTYPE::LIN:
                                                                GAMSWRITER<DAG,T>::MODELTYPE::QUAD):
                                                                GAMSWRITER<DAG,T>::MODELTYPE::NLIN);
  _Fquad.clear();
  _Fpol.clear();
  _Fgal.clear();

  GMS.set_functions( _dag, type, _nF, _Fvar.data(), _nX, _Xvar.data() );
  GMS.set_objective( 0, _objsense>0? BASE_OPT::MAX: BASE_OPT::MIN );
  GMS.set_constraints( 0, _nF, _Fbnd.data() );
  GMS.write( gmsfile );
  return true;
}

template <typename DAG, typename T>
inline
void
MINLPREF<DAG,T>::Options::display
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

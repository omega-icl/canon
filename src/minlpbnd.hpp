// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MINLPBND Bounding of Factorable Mixed-Integer Nonlinear Programs using MC++
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

Other options can be modified to tailor the relaxations, tune the MIP solver, set a maximum CPU time, etc. These options can be modified through the public member mc::MINLPBND::options.
*/

//- Enable RLT cuts in addition to quadratization
//- SEPARATE VARIOUS RELAXATION CLASSES AND INHERIT IN MINLPBND?

#ifndef MC__MINLPBND_HPP
#define MC__MINLPBND_HPP

#include <stdexcept>
#include <chrono>

#include "minlpref.hpp"
#include "mipslv_gurobi.hpp"

#include "polimage.hpp"
#include "scmodel.hpp"
#include "ismodel.hpp"

//#undef MC__MINLPBND_DEBUG
//#define MC__MINLPBND_DEBUG_LIFT
//#define MC__MINLPBND_SHOW_REDUC
//#define MC__MINLPBND_DEBUG_ISM

namespace mc
{

//! @brief C++ base class for global bounding of factorable MINLP using MC++
////////////////////////////////////////////////////////////////////////
//! mc::MINLPBND is a C++ class for global bounding of factorable MINLP
//! using MC++
////////////////////////////////////////////////////////////////////////
template < typename T,
           typename MIP=MIPSLV_GUROBI<T>,
           typename... ExtOps >
class MINLPBND
#if defined (MC__WITH_GAMS)
: protected virtual GAMSIO<ExtOps...>,
  public virtual MINLPREF<T,ExtOps...>
#else
: public virtual MINLPREF<T,ExtOps...>
#endif
{
  // Typedef's
  typedef SMon< unsigned, std::less<unsigned> > t_mon;
  typedef lt_SMon< std::less<unsigned> > lt_mon;
  
  typedef SPoly< unsigned, std::less<unsigned> > t_poly;
  typedef std::map< t_mon, double, lt_mon > map_poly;
  typedef std::pair< t_mon const*, t_mon const* > t_prodmon;

  typedef SQuad< unsigned, std::less<unsigned> > t_quad;
  typedef lt_SQuad< std::less<unsigned> > lt_quad;

  typedef SLiftEnv<ExtOps...> t_lift;
  typedef SElimEnv<ExtOps...> t_elim;

public:

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

  using MINLPREF<T,ExtOps...>::problem_class;
  using MINLPREF<T,ExtOps...>::dag;
  using MINLPREF<T,ExtOps...>::variables;
  using MINLPREF<T,ExtOps...>::lifted_variables;
  using MINLPREF<T,ExtOps...>::functions;
  using MINLPREF<T,ExtOps...>::variable_bounds;
  using MINLPREF<T,ExtOps...>::update_bounds;
  using MINLPREF<T,ExtOps...>::propagate_bounds;
  using MINLPREF<T,ExtOps...>::flatten_linear_functions;
  using MINLPREF<T,ExtOps...>::flatten_quadratic_functions;
  using MINLPREF<T,ExtOps...>::flatten_polynomial_functions;
  using MINLPREF<T,ExtOps...>::lift_polynomial_subexpressions;
  using MINLPREF<T,ExtOps...>::quadratize_polynomial_functions;
  using MINLPREF<T,ExtOps...>::eliminate_invertible_constraints;
  using MINLPREF<T,ExtOps...>::export_model;

#if defined (MC__WITH_GAMS)
  using GAMSIO<ExtOps...>::read;
#endif

protected:

  using MINLPREF<T,ExtOps...>::_dag;
  using MINLPREF<T,ExtOps...>::_issetup;
  using MINLPREF<T,ExtOps...>::_objsense;

  using MINLPREF<T,ExtOps...>::_nX;
  using MINLPREF<T,ExtOps...>::_nX0;
  using MINLPREF<T,ExtOps...>::_nX1;
  using MINLPREF<T,ExtOps...>::_Xvar;
  using MINLPREF<T,ExtOps...>::_Xbnd;
  using MINLPREF<T,ExtOps...>::_Xtyp;
  using MINLPREF<T,ExtOps...>::_Xlin;
  using MINLPREF<T,ExtOps...>::_Xquad;
  using MINLPREF<T,ExtOps...>::_Xpol;
  using MINLPREF<T,ExtOps...>::_Xgal;
  using MINLPREF<T,ExtOps...>::_Xlift;
  using MINLPREF<T,ExtOps...>::_Xobj;
  
  using MINLPREF<T,ExtOps...>::_nF;
  using MINLPREF<T,ExtOps...>::_Fvar;
  using MINLPREF<T,ExtOps...>::_Fbnd;
  using MINLPREF<T,ExtOps...>::_Flin;
  using MINLPREF<T,ExtOps...>::_Fquad;
  using MINLPREF<T,ExtOps...>::_Fpol;
  using MINLPREF<T,ExtOps...>::_Fgal;
  using MINLPREF<T,ExtOps...>::_Fops;
  using MINLPREF<T,ExtOps...>::_Fallops;

  using MINLPREF<T,ExtOps...>::_SQenv;

  using MINLPREF<T,ExtOps...>::_propagate_bounds;
  using MINLPREF<T,ExtOps...>::_dwk;
  using MINLPREF<T,ExtOps...>::_Iwk;
  using MINLPREF<T,ExtOps...>::_CPbnd;
  using MINLPREF<T,ExtOps...>::_IINF;

  using MINLPREF<T,ExtOps...>::_tstart;

protected:

  //! @brief environment for sparse quadratic form
  t_quad                    _SCQenv;
  //! @brief vector of auxiliary variables
  std::vector< FFVar >      _Xaux;
  //! @brief Map of participating monomials
  std::map< t_mon, FFVar, lt_mon > _Xmon;

  //! @brief Polyhedral image environment
  PolImg< T, ExtOps... >    _POLenv;
  //! @brief Polyhedral image decision variables
  std::vector< PolVar<T> >  _POLXvar;
  //! @brief Polyhedral image auxiliary variables
  std::vector< PolVar<T> >  _POLXaux;
  //! @brief Map of monomials in polyhedral image
  std::map< t_mon, PolVar<T>, lt_mon > _POLXmon;
  //! @brief Map of monomial products in polyhedral image
  std::map< t_prodmon, PolVar<T>, lt_quad > _POLXprodmon;
  //! @brief Polyhedral image function variables
  std::vector< PolVar<T> >  _POLFvar;
  //! @brief Storage vector for function evaluation in polyhedral relaxation arithmetic
  std::vector< PolVar<T> >  _POLwk;

  //! @brief Chebyshev model environment
  SCModel<T>* _SCMenv;
  //! @brief Chebyshev variables
  std::vector< SCVar<T> >   _SCMXvar;
  //! @brief Chebyshev constraint variables
  std::vector< SCVar<T> >   _SCMFvar;
  //! @brief Storage vector for function evaluation in Chebyshev arithmetic
  std::vector< SCVar<T> >   _SCMwk;
  //! @brief Chebyshev basis map
  std::map< t_mon, FFVar, lt_mon > _SCMXmon;

  //! @brief Interval superposition model environment
  ISModel<T>*               _ISMenv;
  //! @brief Interval superposition variables
  std::vector<ISVar<T>>     _ISMXvar;
  //! @brief Interval superposition constraint variables
  std::vector<ISVar<T>>     _ISMFvar;
  //! @brief Storage vector for function evaluation in Interval superposition arithmetic
  std::vector< ISVar<T> >   _ISMwk;

  //! @brief MIP solver
  MIP*                      _MIPSLV;

public:

  //! @brief Constructor
  MINLPBND
    ()
    : _SCMenv( nullptr ),
      _ISMenv( nullptr )
    { _MIPSLV = new MIP; }

  //! @brief Destructor
  virtual ~MINLPBND
    ()
    {
      delete _MIPSLV;
      delete _ISMenv;
      delete _SCMenv;
    }

  //! @brief MINLPBND options
  struct Options
  : public MINLPREF<T,ExtOps...>::Options
  {
    using MINLPREF<T,ExtOps...>::Options::TIMELIMIT;
    using MINLPREF<T,ExtOps...>::Options::DISPLEVEL;
    using MINLPREF<T,ExtOps...>::Options::PSDQUADCUTS;
    using MINLPREF<T,ExtOps...>::Options::DCQUADCUTS;
    
    //! @brief Constructor
    Options():
      MINLPREF<T,ExtOps...>::Options(),
      RELAXMETH({DRL}), SUBSETDRL(0), SUBSETSCM(0), SUBSETISM(0),
      OBBTMIG(1e-6), OBBTMAX(5), OBBTTHRES(5e-2), OBBTBKOFF(1e-7), OBBTLIN(2), OBBTCONT(true),
      ISMODEL(), ISMDIV(10), ISMCONT(true),
      CMODEL(), CMODPROP(2), CMODCUTS(0), CMODDMAX(BASE_OPT::INF),
      MONSCALE(true), LINCTRSEP(false), BCHPRIM(0),
      POLIMG(), SCQUAD(), MIPSLV()
      { CMODEL.MIXED_IA        = true;
        CMODEL.MIG_ATOL        = 1e-13; // compatibility with GUROBI
        POLIMG.BREAKPOINT_TYPE = PolImg<T,ExtOps...>::Options::BIN;
        MIPSLV.DISPLEVEL       = 0;
        //MIPSLV.DUALRED         = 1;
        //MIPSLV.PRESOLVE        = 1;
        MIPSLV.TIMELIMIT       = TIMELIMIT;
        SCQUAD.BASIS            = t_quad::Options::CHEB;
        SCQUAD.ORDER            = t_quad::Options::INC;
        SCQUAD.REDUC            = false; }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        MINLPREF<T,ExtOps...>::Options::operator=( options );
        RELAXMETH     = options.RELAXMETH;
        SUBSETDRL     = options.SUBSETDRL;
        SUBSETSCM     = options.SUBSETSCM;
        SUBSETISM     = options.SUBSETISM;
        OBBTMIG       = options.OBBTMIG;
        OBBTMAX       = options.OBBTMAX;
        OBBTTHRES     = options.OBBTTHRES;
        OBBTBKOFF     = options.OBBTBKOFF;
        OBBTLIN       = options.OBBTLIN;
        OBBTCONT      = options.OBBTCONT;
	ISMODEL       = options.ISMODEL;
        ISMDIV        = options.ISMDIV;
        ISMCONT       = options.ISMCONT;
        CMODEL        = options.CMODEL;
        CMODPROP      = options.CMODPROP;
        CMODCUTS      = options.CMODCUTS;
        CMODDMAX      = options.CMODDMAX;
        MONSCALE      = options.MONSCALE;
        LINCTRSEP     = options.LINCTRSEP;
        BCHPRIM       = options.BCHPRIM;
        POLIMG        = options.POLIMG;
        SCQUAD        = options.SCQUAD;
        MIPSLV        = options.MIPSLV;
        return *this ;
      }
    //! @brief Relaxation strategy
    enum RELAX{
      DRL=0,  //!< McCormick-derived polyhedral cuts
      DRLQ,   //!< McCormick-derived polyhedral and quadratic cuts
      SCDRL,  //!< Chebyshev-derived polyhedral cuts (controlled by parameters CMODPROP and CMODCUT)
      SCQ,    //!< Chebyshev-derived polyhedral and quadratic cuts (controlled by parameters CMODPROP and CMODCUT)
      ISM     //!< Interval superposition model relaxations (controlled by parameters ISMDIV and ISMCONT)
    };
    //! @brief Relaxation methods
    std::set<RELAX> RELAXMETH;
    //! @brief Exclusion from decomposition-relaxation-linearization: 0: none; 1: non-polynomial functions; 2: polynomial functions
    unsigned SUBSETDRL;
    //! @brief Exclusion from Chebyshev model: 0: none; 1: non-polynomial functions; 2: polynomial functions
    unsigned SUBSETSCM;
    //! @brief Exclusion from interval superposition: 0: none; 1: non-polynomial functions; 2: polynomial functions
    unsigned SUBSETISM;
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
    //! @brief ISModel options
    typename ISModel<T>::Options ISMODEL;
    //! @brief Number of partition subdivisions in interval superposition model
    unsigned ISMDIV;
    //! @brief Whether to generate a continuous relaxation of ISM (true) or MIP relaxation (false)
    bool ISMCONT;
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
    //! @brief Whether to separate linear constraints prior to relaxation
    bool LINCTRSEP;
    //! @brief Set higher branch priority to primary variables (e.g. over auxiliary variables in quadratization)
    unsigned BCHPRIM;
    //! @brief PolImg (polyhedral relaxation) options
    typename PolImg<T,ExtOps...>::Options POLIMG;
    //! @brief SQuad options for quadratization of sparse Chebyshev models
    typename t_quad::Options SCQUAD;
    //! @brief Display
    //! @brief MIPSLV_GUROBI (mixed-integer optimization) options
    typename MIP::Options MIPSLV;
    void display
      ( std::ostream&out=std::cout ) const;
  } options;

  //! @brief MINLPBND computational statistics
  struct Stats
  : MINLPREF<T,ExtOps...>::Stats
  {
    using MINLPREF<T,ExtOps...>::Stats::start;
    using MINLPREF<T,ExtOps...>::Stats::walltime;
    using MINLPREF<T,ExtOps...>::Stats::to_time;

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
  } stats;

  //! @brief Setup optimization model before bounding
  void setup
    ( std::ostream& os=std::cout );

  //! @brief Propagate bounds, starting with variable subdomain <a>X</a>, for the incumbent value <a>Finc</a>, and using the options specified in <a>MINLPREF::Options::CPMAX</a> and <a>MINLPREF::Options::CPTHRES</a> -- returns updated variable bounds <a>X</a>
  bool propagate_bounds
    ( T const* X=nullptr, double const* Finc=nullptr, const bool resetbnd=true,
      std::ostream& os=std::cout )
    { auto tstart = stats.start();
      bool flag = MINLPREF<T,ExtOps...>::propagate_bounds( X, Finc, resetbnd, os );
      stats.walltime_cprop += stats.walltime( tstart );
      return flag; }

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
  int relax_model
    ( T const* X=nullptr, double const* Finc=nullptr, double const* Xinc=nullptr,
      unsigned const nref=0, bool const resetbnd=true, bool const reinit=true,
      std::string const gmsfile="", std::ostream& os=std::cout );

  //! @brief Setup and solve bound reduction problems using polyhedral relaxations of optimization model, starting with variable subdomain <a>X</a>, for the incumbent value <a>Finc</a>, and using the options specified in <a>MINLPBND::Options::OBBTMAX</a> and <a>MINLPBND::Options::OBBTTHRES</a> -- returns updated variable bounds <a>X</a>, and number of iterative refinements <a>nred</a>
  int reduce_bounds
    ( unsigned& nred, T* X=nullptr, double const* Finc=nullptr,
      bool const resetbnd=true, bool const reinit=true );

    //! @brief Test whether all variables of a given type are bounded
  bool bounded_domain
    ( double const& maxdiam, FFDep::TYPE const type )
    const;

  //! @brief Get const pointer to MIP solver
  MIP const* relax_solver
    ()
    const
    { return _MIPSLV; }

  //! @brief Get non-const pointer to MIP solver
  MIP * relax_solver
    ()
    { return _MIPSLV; }

private:

  //! @brief Update MINLPREF baseclass options
  virtual void _update_options
    ();

  //! @brief Time point to enable TIMELIMIT option
  //std::chrono::time_point<std::chrono::system_clock> _tstart;

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

  //! @brief Set model McCormick-derived cuts
  void _set_cuts_DRL
    ( bool const SQuadCuts );

  //! @brief Set model Chebyshev-derived cuts
  void _set_cuts_SCM
    ( bool const SQuadCuts );

  //! @brief Set model ISM-derived cuts
  void _set_cuts_ISM
    ();

  //! @brief Append cuts for ISM in polynomial image
  void _set_cuts_ISM
    ( std::set<unsigned> const& ndxF );

  //! @brief Compute bound for given Chebyshev basis function 
  T _bnd_cheb
    ( T const& x, const unsigned n )
    const;

  //! @brief Compute bound of (unscaled) monomial <a>mon</a>
  T _bnd_mon
    ( t_mon const& mon, int const BASIS )
    const;

  //! @brief Set monomials from quadratic forms in polyhedral image
  void _set_mon_SQ
    ( t_quad const& SQenv, int const BASIS, bool const SCALED,
      bool const DAGINSERT );

  //! @brief Append cuts for quadratic form in polynomial image
  void _set_cuts_SQ
    ( t_quad& SQenv, std::set<unsigned> const& ndxF, bool const chkrem );

  //! @brief Set monomials from Chebyshev models in polyhedral image
  void _set_mon_SCDRL
    ( std::set<unsigned> const& ndxF, int const BASIS, bool const SCALED,
      bool const DAGINSERT );

  //! @brief Append cuts for Chebyshev models in polynomial image
  void _set_cuts_SCDRL
    ( std::set<unsigned> const& ndxF );

  //! @brief Set cuts for monomials in polyhedral image
  void _set_mon_DRL
    ( t_mon const& mon, int const BASIS, bool const SCALED, bool const DAGINSERT );

  //! @brief Add cuts for monomials in polyhedral image
  void _add_mon_DRL
    ();

  //! @brief Get monomial <a>mon</a> from DAG monomial map <a>_Xmon</a> or add it to the map if absent
  FFVar const& _get_mon
    ( t_mon const& mon, int const BASIS, bool const DAGINSERT );

  //! @brief Get range for quadratic terms from <a>mat</a>
  T _get_range
    ( t_quad::map_SQuad const& mat );

  //! @brief Append cuts for DC factorization in polyhedral image
  PolVar<T> _append_cuts_dcdec
    ( t_quad::map_SPoly const& eigterm );

  //! @brief Append cuts for monomial product in polyhedral image
  PolVar<T> _append_cuts_monprod
    ( t_quad::key_SQuad const& ijmon );

  //! @brief Append quadratic terms from <a>mat</a> to cuts
  void _add_to_cuts
    ( t_quad::map_SQuad const& mat, PolCut<T>* cut1=nullptr, PolCut<T>* cut2=nullptr );

  //! @brief Append quadratic terms from <a>mat</a> to cuts with or without DC factorization
  void _add_to_cuts
    ( t_quad const& SQenv, t_quad::map_SQuad const& mat, PolCut<T>* cut1=nullptr,
      PolCut<T>* cut2=nullptr );

  //! @brief Function computing Hausdorff distance between intervals
  template <typename U> static double _dH
    ( U const& X, U const& Y );
  //! @brief Function computing relative reduction between interval vectors
  template <typename U> static double _reducrel
    ( unsigned const n, U const* Xred, U const* X );
  //! @brief Function computing relative reduction between interval vectors
  template <typename U> static double _reducrel
    ( unsigned const n, U const* Xred, const U*X, U const* X0 );

  //! @brief Private methods to block default compiler methods
  MINLPBND
    ( MINLPBND<T,MIP,ExtOps...> const& );
  MINLPBND<T,MIP,ExtOps...>& operator=
    ( MINLPBND<T,MIP,ExtOps...> const& );
};

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::setup
( std::ostream& os )
{
  MINLPREF<T,ExtOps...>::options = options;
  MINLPREF<T,ExtOps...>::setup( os );
  stats.reset();
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_update_options
()
{
  MINLPREF<T,ExtOps...>::options = options;
  //stats.reset();
}

template <typename T, typename MIP, typename... ExtOps>
inline bool
MINLPBND<T,MIP,ExtOps...>::bounded_domain
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

template <typename T, typename MIP, typename... ExtOps>
inline int
MINLPBND<T,MIP,ExtOps...>::relax_model
( T const* X, double const* Finc, double const* Xinc, const unsigned nref,
  const bool resetbnd, bool const reinit, std::string const gmsfile, std::ostream& os )
{
  if( !_issetup ) throw typename MINLPREF<T,ExtOps...>::Exceptions( MINLPREF<T,ExtOps...>::Exceptions::SETUP );
  _tstart = stats.start();

  // Update variable bounds
  if( !update_bounds( X, Finc, resetbnd ) ) return MIP::INFEASIBLE;
  int cpred = _propagate_bounds();
  if( cpred < 0 ) return MIP::INFEASIBLE;

  // Reset polyhedral image, LP variables and cuts
  _MIPSLV->options = options.MIPSLV;
  if( reinit ) init_polrelax(); // <<== COULD PASS Xinc HERE TOO??
  update_polrelax( 2, false, reinit? false: true );
#ifdef MC__MINLPBND_DEBUG
  std::cout << _POLenv;
#endif

  // Write relaxed model to GAMS file
  if( !gmsfile.empty() ){
    GAMSWRITER<T,ExtOps...> GMS;
    GMS.set_cuts( &_POLenv, true );
    for( unsigned i=0; i<_nX0; i++ ){
      GMS.set_variable( _POLXvar[i], Xinc? &Xinc[i]: nullptr );
#ifdef MC__MINLPBND_DEBUG_INITIALS
      if( Xinc ) std::cout << "Xinc[ " << i << "] = " << Xinc[i] << std::endl;
#endif
    }
    if( Xinc ){
      unsigned j=0; 
      for( auto const& [i,Fi] : _Xlift ){
        try{
          double Xi;
          _dag->eval( _Fops.at(_nF+j), _dwk, 1, &Fi, &Xi, _nX0, _Xvar.data(), Xinc );
          GMS.set_variable( _POLXvar[i], &Xi );
#ifdef MC__MINLPBND_DEBUG_INITIALS
          std::cout << "Xinc[ " << i << "] = " << Xi << std::endl;
#endif
        }
        catch(...){
          continue;
        }
        j++;
      }
    }
    GMS.set_objective( _POLFvar[0], _objsense>0? BASE_OPT::MAX: BASE_OPT::MIN );
    GMS.write( gmsfile );
    if( options.MIPSLV.DISPLEVEL > 0 )
      os << std::endl << "# WRITING MIP MODEL TO FILE " << gmsfile << std::endl;
    return MIP::OTHER;
  }

  // Set-up variable initial guess and branch priority
  for( unsigned i=0; i<_nX0; i++ ){
    assert( _MIPSLV->set_variable( _POLXvar[i], Xinc? &Xinc[i]: nullptr, options.BCHPRIM ) );
#ifdef MC__MINLPBND_DEBUG_INITIALS
    if( Xinc ) std::cout << "Xinc[ " << i << "] = " << Xinc[i] << std::endl;
#endif
  }
  if( Xinc ){
    unsigned j=0; 
    for( auto const& [i,Fi] : _Xlift ){
      try{
        double Xi;
        _dag->eval( _Fops.at(_nF+j), _dwk, 1, &Fi, &Xi, _nX0, _Xvar.data(), Xinc );
        assert( _MIPSLV->set_variable( _POLXvar[i], &Xi, 0 ) );
#ifdef MC__MINLPBND_DEBUG_INITIALS
        std::cout << "Xinc[ " << i << "] = " << Xi << std::endl;
#endif
      }
      catch(...){
        continue;
      }
      j++;
    }
  }

  for( unsigned iref=0; ; iref++ ){
    // Set-up relaxed objective, options, and solve polyhedral relaxation
    auto tMIP = stats.start();
    _MIPSLV->set_objective( _POLFvar[0], _objsense>0? BASE_OPT::MAX: BASE_OPT::MIN );
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

template <typename T, typename MIP, typename... ExtOps>
inline int
MINLPBND<T,MIP,ExtOps...>::_reduce
( unsigned const ix, bool const uplo )
{
#ifdef MC__MINLPBND_DEBUG
  std::cout << "\nTIGHTENING OF VARIABLE " << ix << (uplo?"U":"L") << ":\n";
//  std::cout << _POLenv;
#endif
  // Set-up lower/upper bound objective, options, and solve polyhedral relaxation
  auto tMIP = stats.start();
  _MIPSLV->set_objective( _POLXvar[ix], (uplo? BASE_OPT::MAX: BASE_OPT::MIN) );
  _MIPSLV->options.TIMELIMIT = options.TIMELIMIT - stats.to_time( stats.walltime( _tstart ) );
  _MIPSLV->solve();
  stats.walltime_slvmip += stats.walltime( tMIP );
  stats.total_slvmip ++;

  return _MIPSLV->get_status();
}

template <typename T, typename MIP, typename... ExtOps>
inline bool
MINLPBND<T,MIP,ExtOps...>::_tight
()
{
  // test if current bounds are tight
  for( unsigned i=0; i<_nX; i++ )
    if( Op<T>::diam( _Xbnd[i] ) > options.OBBTMIG ) return false;
  return true;
}

template <typename T, typename MIP, typename... ExtOps>
inline int
MINLPBND<T,MIP,ExtOps...>::_reduce
()
{
  // solve reduction subproblems from closest to farthest from bounds
  std::multimap<double,std::pair<unsigned,bool>> vardomred, vardomredupd;
  std::pair<unsigned,bool> varini;
  for( unsigned i=0; i<_nX; i++ ){
    // do not reduce variable whose domain is less than OBBTMIG
    if( Op<T>::diam( _Xbnd[i] ) < options.OBBTMIG ) continue;
    // do not reduce variable that does not participate in relaxation
    auto itv = _POLenv.Vars().find( &_Xvar[i] );
    if( itv == _POLenv.Vars().end() || !itv->second->has_cuts() ) continue;
    
    varini.first = i;
    varini.second = false; // lower bound
    double dist = 1.;
    vardomred.insert( std::pair<double,std::pair<unsigned,bool>>(dist,varini) );
    varini.second = true;  // upper bound
    vardomred.insert( std::pair<double,std::pair<unsigned,bool>>(dist,varini) );
#ifdef MC__MINLPBND_DEBUG
    std::cout << "Reduction candidate: " << _Xvar[i] << std::endl;
#endif
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

template <typename T, typename MIP, typename... ExtOps>
inline int
MINLPBND<T,MIP,ExtOps...>::reduce_bounds
( unsigned& nred, T* X, double const* Finc, const bool resetbnd,
  bool const reinit )
{
  if( !_issetup ) throw typename MINLPREF<T,ExtOps...>::Exceptions( MINLPREF<T,ExtOps...>::Exceptions::SETUP );
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
  _MIPSLV->options = options.MIPSLV;
  if( reinit ) init_polrelax();
#ifdef MC__MINLPBND_DEBUG
  std::cout << _POLenv;
#endif
#ifdef MC__MINLPBND_SHOW_REDUC
  std::cout << "\nOriginal Box:\n";
  for( unsigned i=0; i<_nX; i++ )
    std::cout << _Xvar[i] << " = " << _Xbnd[i] << std::endl;
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
      //std::cout << *_dag;
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
      //std::cout << *_dag;
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
        break;
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

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::init_polrelax
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

  // Initialize relevant environments to construct relaxations
  for( auto const& meth : options.RELAXMETH ){
    switch( meth ){
    
      // Add McCormick-derived polyhedral cuts
      default:
      case Options::DRL:
      case Options::DRLQ:
        break;

     // Add Chebyshev-derived polyhedral cuts
     case Options::SCDRL:
       // Reset Chebyshev basis map in DAG
       _Xmon.clear();

       // Resize auxiliary variables in DAG and polyhedral image
       _Xaux.reserve( _nX );
       _POLXaux.reserve( _nX );
       // **no break** to continue into SCQ
       
     case Options::SCQ:
       // Chebyshev model environment reset
       if( _SCMenv && (_SCMenv->setvar().size() != _nX || _SCMenv->maxord() != options.CMODPROP) ){
         _SCMXvar.clear();
         delete _SCMenv; _SCMenv = nullptr;   
       }
       if( !_SCMenv ){
         // Set Chebyshev model
         _SCMenv = new SCModel<T>( options.CMODPROP );
         _SCMenv->options = options.CMODEL;
         _SCMXvar.resize( _nX );
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
         _ISMenv->options = options.ISMODEL;
         _ISMXvar.resize( _nX );
       }
       break;
    }
  }
  
  stats.walltime_polimg += stats.walltime( tstart );
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::update_polrelax
( unsigned const addcuts, bool const contcuts, bool const resetcuts )
{
  auto tstart = stats.start();

  // Reset polyhedral cuts
  if( resetcuts ) _POLenv.reset_cuts();

  // Reset monomial vectors
  _POLXmon.clear();
  _POLXprodmon.clear();

  // Update polyhedral main variables AND DEPENDENT BOUNDS???
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
        _set_cuts_DRL( false );
        break;

      // Add McCormick-derived polyhedral and quadratic cuts
      case Options::DRLQ:
        // Add polyhedral cuts
        _set_cuts_DRL( true );
        break;

      // Add Chebyshev-derived polyhedral cuts
      case Options::SCDRL:
        // Add polyhedral cuts
        _set_cuts_SCM( false );
        break;

      // Add Chebyshev-derived polyhedral and quadratic cuts
      case Options::SCQ:
        // Add quadratic cuts
        _set_cuts_SCM( true );
        break;

      // Add Interval superposition-derived polyhedral cuts
      case Options::ISM:
        // Add polyhedral cuts
        _set_cuts_ISM();
        break;
    }
  }

  // Update polyhedral dependent bounds
  for( unsigned i=0; i<_nF; i++ ){
    T Fupdi = _Fbnd[i];
    if( Op<T>::inter( Fupdi, _Fbnd[i], _POLFvar[i].range() ) )
      _POLFvar[i].update( Fupdi );
    else
      _POLFvar[i].update( _Fbnd[i] );
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

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::refine_polrelax
( double const* Xinc, bool const resetcuts )
{
  auto tstart = stats.start();
  
  // Update discretization with optimum point of relaxation
  for( auto itv=_POLenv.Vars().begin(); itv!=_POLenv.Vars().end(); ++itv ){
    // do not update variable that does not participate in relaxation
    if( !itv->second->has_cuts() ) continue;
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
  //auto itX = _POLXvar.begin();
  //for( unsigned i=0; itX!=_POLXvar.end(); ++itX, i++ ){
  for( unsigned i=0; i<_nX; i++ ){
    auto itv = _POLenv.Vars().find( &_Xvar[i] );
    if( itv == _POLenv.Vars().end() || !itv->second->has_cuts() ) continue;
    //double Xval = _MIPSLV->get_variable( *itX );
    //itX->add_breakpt( Xval );
    //itX->update( _Xbnd[i] );
    double Xval = _MIPSLV->get_variable( *itv->second );
    _POLXvar[i].add_breakpt( Xval );
    _POLXvar[i].update( _Xbnd[i] );
  }

  // Update discetization with incumbent
  if( Xinc ){
    auto itX = _POLXvar.begin();
    for( unsigned i=0; i<_nX0 && itX!=_POLXvar.end(); ++itX, i++ ){
      itX->add_breakpt( Xinc[i] );
      auto itv = _POLenv.Vars().find( &_Xvar[i] );
      itv->second->add_breakpt( Xinc[i] );
    }
  }

  // Reset polyhedral cuts
  if( resetcuts ) _POLenv.reset_cuts();
  stats.walltime_polimg += stats.walltime( tstart );
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_cuts_LIN
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
       //for( unsigned i=0; i<_Iwk.size(); i++ )
       for( unsigned i=0; i<_Fops[j].len_tap-_Fops[j].len_wrk; i++ )
          if( Op<T>::inter( _Iwk[i], _Iwk[i], _POLwk[i].range() ) ) // intersection needed for externals
           _POLwk[i].update( _Iwk[i] );
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

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_cuts_ISM
()
{
  // Subset of functions to be relaxed
  std::set<unsigned> ndxF;
  for( unsigned j=0; j<_nF; j++ ){
    if( ( options.LINCTRSEP      && _Flin.find( j ) != _Flin.end() )   // exclude cut of linear function
     || ( options.SUBSETISM == 1 && _Fgal.find( j ) != _Fgal.end() )   // exclude cut of non-polynomial function
     || ( options.SUBSETISM == 2 && _Fgal.find( j ) == _Fgal.end() ) ) // exclude cut of polynomial function
      continue;
    ndxF.insert( j );
  }
  if( ndxF.empty() ) return;

  // Update ISM variables
  for( unsigned i=0; i<_nX; i++ )
    _ISMXvar[i].set( _ISMenv, i, _Xbnd[i] );

  // Compute ISM bounds for each nonlinear function
  _ISMFvar.assign( _nF, 0. );
  for( auto itF=ndxF.begin(); itF!=ndxF.end(); ){
    unsigned const j = *itF;

    try{
      _dag->eval( _Fops[j], _ISMwk, 1, &_Fvar[j], &_ISMFvar[j], _nX, _Xvar.data(), _ISMXvar.data() );
#ifdef MC__MINLPBND_DEBUG_ISM
      std::cout << "Interval superposition model for function F[" << j << "]: " << _ISMFvar[j];
#endif
    }
    
    catch(...){
#ifdef MC__MINLPBND_DEBUG_ISM
      std::cout << "Superposition model for function F[" << j << "]: failed" << std::endl;
#endif
      // No cut added for constraint #j in case evaluation failed
      //_ISMFvar[j] = _IINF;
      itF = ndxF.erase( itF ); // Exclude polynomial from quadratization and cuts
      continue;
    }

    ++itF; // Increment only if current index wasn't erased from ndxF already
  }

  // Add cuts for ISM into polynomial image
  _set_cuts_ISM( ndxF );

#ifdef MC__MINLPBND_DEBUG_ISM
  std::cout << _POLenv;
#endif
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_cuts_ISM
( std::set<unsigned> const& ndxF )
{
  // Auxiliary variables in polyhedral image are defined locally 
  std::vector<std::vector<PolVar<T>>> POL_ISMaux( _nX );
  std::vector<double> DL_ISMaux( _ISMenv->ndiv() );
  std::vector<double> DU_ISMaux( _ISMenv->ndiv() );

  // Compute ISM bounds for each nonlinear function
  for( unsigned j : ndxF ){
    _POLFvar[j].set( &_POLenv, _Fvar[j], _ISMFvar[j].B(), true );

    // Polyhedral cut generation
    T rhs = ( !_ISMFvar[j].ndep()? _ISMFvar[j].cst(): 0. );
    auto cutF1 = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -Op<T>::l(rhs), _POLFvar[j], -1. );
    auto cutF2 = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -Op<T>::u(rhs), _POLFvar[j], -1. );
    for( unsigned i=0; i<_nX; ++i ){
      auto&& ISMFji = _ISMFvar[j].C()[i];
      if( ISMFji.empty() ) continue;
      if( POL_ISMaux[i].empty() ){
        POL_ISMaux[i].resize( _ISMenv->ndiv() );
        for( unsigned k=0; k<_ISMenv->ndiv(); ++k )
          POL_ISMaux[i][k].set( &_POLenv, Op<T>::zeroone(), options.ISMCONT );
      }
      for( unsigned k=0; k<_ISMenv->ndiv(); ++k ){
        DL_ISMaux[k] = Op<T>::l( ISMFji[k] );
        DU_ISMaux[k] = Op<T>::u( ISMFji[k] );
      }
      cutF1->append( _ISMenv->ndiv(), POL_ISMaux[i].data(), DL_ISMaux.data() );
      cutF2->append( _ISMenv->ndiv(), POL_ISMaux[i].data(), DU_ISMaux.data() );
    }
  }
 
  // Add polyhedral cuts for ISM-participating variables
  for( unsigned i=0; i<_nX; i++ ){
    if( POL_ISMaux[i].empty() ) continue;
    // Auxiliaries add up to 1
    for( unsigned jsub=0; jsub<_ISMenv->ndiv(); jsub++ )
      DL_ISMaux[jsub] = 1.;
    _POLenv.add_cut( nullptr, PolCut<T>::EQ, 1., _ISMenv->ndiv(), POL_ISMaux[i].data(), DL_ISMaux.data() );
    // Relationship between variables and auxiliaries
    PolVar<T> POLvarL( 0. ), POLvarU( 0. );
    auto&& ISMXi = _ISMXvar[i].C()[i];
#ifdef MC__MINLPBND_DEBUG_ISM
    assert( !ISMXi.empty() );
#endif
    for( unsigned k=0; k<_ISMenv->ndiv(); k++ ){
      DL_ISMaux[k] = Op<T>::l(ISMXi[k]);
      DU_ISMaux[k] = Op<T>::u(ISMXi[k]);
    }
    _POLenv.add_cut( nullptr, PolCut<T>::LE, 0., _ISMenv->ndiv(), POL_ISMaux[i].data(), DL_ISMaux.data(), _POLXvar[i], -1. );
    _POLenv.add_cut( nullptr, PolCut<T>::GE, 0., _ISMenv->ndiv(), POL_ISMaux[i].data(), DU_ISMaux.data(), _POLXvar[i], -1. );
  }
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_cuts_DRL
( bool const SQuadCuts )
{
  // Subset of functions to be relaxed
  std::set<unsigned> Frel, Fsalg = _Fquad; Fsalg.insert( _Fpol.cbegin(), _Fpol.cend() );
  for( unsigned j=0; j<_nF; j++ ){
    if( ( options.LINCTRSEP       && _Flin.find( j ) != _Flin.end() )   // exclude cut of linear function
     || ( options.SUBSETDRL == 1  && _Fgal.find( j ) != _Fgal.end() )   // exclude cut of non-polynomial function
     || ((options.SUBSETDRL == 2 || SQuadCuts) && _Fgal.find( j ) == _Fgal.end() ) ) // exclude cut of polynomial function
      continue;
    Frel.insert( j );
  }
  if( Frel.empty() && Fsalg.empty() ) return;

  // Add polyhedral cuts for selected functions
  bool const USEMOVE_SAVE = _dag->options.USEMOVE;
  if( options.CPMAX ) _dag->options.USEMOVE = false; // Disable move for constraint propagation
  for( unsigned j : Frel ){
    try{
#ifdef MC__MINLPBND_DEBUG_DRL
      _dag->output( _Fops[j] );
#endif
      _dag->eval( _Fops[j], _POLwk, 1, &_Fvar[j], &_POLFvar[j], _nX, _Xvar.data(), _POLXvar.data() );
  
      // Update bounds of intermediate factors from constraint propagation results
      if( options.CPMAX ){
        _dag->wkextract( _Fops[j], _Iwk, _Fallops, _CPbnd );
        //for( unsigned i=0; i<_Iwk.size(); i++ ){
        for( unsigned i=0; i<_Fops[j].len_tap-_Fops[j].len_wrk; i++ ){
          if( Op<T>::inter( _Iwk[i], _Iwk[i], _POLwk[i].range() ) ) // intersection needed for externals
            _POLwk[i].update( _Iwk[i] );
#ifdef MC__MINLPBND_DEBUG_DRL
          std::cout << _POLwk[i] << _POLwk[i].range() << _Iwk[i] << std::endl;
#endif
        }
      }

      // Generate cuts
      _POLenv.generate_cuts( 1, &_POLFvar[j], false );
    }
    
    catch(...){
      // No cut added for function #j in case DAG evaluation failed
      continue;
    }
  }
  if( options.CPMAX ) _dag->options.USEMOVE = USEMOVE_SAVE;

  // Add polyhedral cuts for quadratised polynomial functions
  if( SQuadCuts && !Fsalg.empty() ){

    // Set monomial vector from quadratic form into polyhedral image
    _set_mon_SQ( _SQenv, t_quad::Options::MONOM, false, false );

    // Add cuts for quadratic form into polynomial image
    _set_cuts_SQ( _SQenv, Fsalg, false );
  }

#ifdef MC__MINLPBND_DEBUG_DRL
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_cuts_SCM
( bool const SQuadCuts )
{
  // Subset of functions to be relaxed
  std::set<unsigned> ndxF;
  for( unsigned j=0; j<_nF; j++ ){
    if( ( options.LINCTRSEP      && _Flin.find( j ) != _Flin.end() )   // exclude cut of linear function
     || ( options.SUBSETSCM == 1 && _Fgal.find( j ) != _Fgal.end() )   // exclude cut of non-polynomial function
     || ( options.SUBSETSCM == 2 && _Fgal.find( j ) == _Fgal.end() ) ) // exclude cut of polynomial function
      continue;
    ndxF.insert( j );
  }
  if( ndxF.empty() ) return;

  // Update sparse Chebyshev variable bounds
  for( unsigned i=0; i<_nX; i++ ){
    _SCMXvar[i].set( _SCMenv, i, _Xbnd[i] );
    if( _SCMenv->scalvar().at(i) <= options.CMODEL.MIG_ATOL )
      _SCMXvar[i] = _Xbnd[i];
  }

  // Compute sparse Chebyshev model for each nonlinear function
  const unsigned MAXORD = (!options.CMODCUTS || options.CMODCUTS>options.CMODPROP)?
                          options.CMODPROP: options.CMODCUTS;

  _SCMFvar.assign( _nF, 0. );
  for( auto itF=ndxF.begin(); itF!=ndxF.end(); ){
    unsigned const j = *itF;
    try{
      _dag->eval( _Fops[j], _SCMwk, 1, &_Fvar[j], &_SCMFvar[j], _nX, _Xvar.data(), _SCMXvar.data() );
#ifdef MC__MINLPBND_DEBUG_SCM
      std::cout << "Chebyshev model for function F[" << j << "]: " << _SCMFvar[j];
#endif
      // Test for too large Chebyshev bounds or NaN
      if( !(Op<SCVar<T>>::diam(_SCMFvar[j]) <= options.CMODDMAX) ) throw(0);

      // Convert and simplify sparse Chebyshev model
      switch( options.SCQUAD.BASIS ){
       case t_quad::Options::MONOM:
       {
        auto&& [coefmon,bndrem] = _SCMFvar[j].to_monomial( options.MONSCALE, options.CMODEL.MIG_ATOL, options.CMODEL.MIG_RTOL, MAXORD );
        _SCMFvar[j].set( coefmon );
        _SCMFvar[j].R() += bndrem;
        break;
       }
       case t_quad::Options::CHEB:
        _SCMFvar[j].simplify( options.CMODEL.MIG_ATOL, options.CMODEL.MIG_RTOL, MAXORD );
        break;
      }
    }

    catch( int ecode ){
#ifdef MC__MINLPBND_DEBUG_SCM
      std::cout << "Chebyshev model bound too weak!\n";
#endif
      T IFvarj;
      Op<T>::inter( IFvarj, _IINF, _SCMFvar[j].B() );
      _SCMFvar[j] = IFvarj;
      itF = ndxF.erase( itF ); // Exclude polynomial from quadratization and cuts
      continue;
    }

    catch(...){
#ifdef MC__MINLPBND_DEBUG_SCM
      std::cout << "Chebyshev model for function F[" << j << "]: failed" << std::endl;
#endif
      // No cut added for constraint #j in case evaluation failed
      _SCMFvar[j] = _IINF;
      itF = ndxF.erase( itF ); // Exclude polynomial from quadratization and cuts
      continue;
    }

    ++itF; // Increment only if current index wasn't erased from ndxF already
  }
 
  // Apply decomposition-linearization-relaxation to Chebyshev-derived cuts
  if( !SQuadCuts ){
    // Set monomial vector from Chebyshev model into polyhedral image
    _set_mon_SCDRL( ndxF, options.SCQUAD.BASIS, options.MONSCALE, true );

    // Add Chebyshev-derived cuts to polynomial image
    _set_cuts_SCDRL( ndxF );
  }
 
  // Apply quadratisation to Chebyshev-derived cuts
  else{
    // Perform quadratisation of polynomial part of sparse Chebyshev models
    _SCQenv.reset();
    _SCQenv.options = options.SCQUAD;
    _SCQenv.process( ndxF, _SCMFvar.data(), &SCVar<T>::coefmon, options.SQUAD.BASIS );
#ifdef MC__MINLPBND_DEBUG_SCM
    double viol = _SCQenv.check( ndxF, _SCMFvar.data(), &SCVar<T>::coefmon, options.SQUAD.BASIS );
    std::cout << "violation: " << viol << std::endl << _SCQenv << std::endl;
    {std::cout << "PAUSED, ENTER <1> TO CONTINUE "; int dum; std::cin >> dum; }
#endif

    // Set monomial vector from quadratic forms into polyhedral image
    _set_mon_SQ( _SCQenv, options.SCQUAD.BASIS, options.MONSCALE, false );

    // Add quadratic cuts into polynomial image
    _set_cuts_SQ( _SCQenv, ndxF, true );
  }

#ifdef MC__MINLPBND_DEBUG_SCM
 std::cout << _POLenv;
 { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_cuts_SCDRL
( std::set<unsigned> const& ndxF )
{
  // Add Chebyshev-derived cuts for selected expressions
  for( unsigned j : ndxF ){
    T Rj = _SCMFvar[j].R();
    double aj0 = _SCMFvar[j].constant( true ); // get constant coefficient and remove it from model
    _POLenv.add_cut( nullptr, PolCut<T>::GE, -Op<T>::u(Rj)-aj0, _POLXmon, _SCMFvar[j].coefmon(), _POLFvar[j],  -1. );
    _POLenv.add_cut( nullptr, PolCut<T>::LE, -Op<T>::l(Rj)-aj0, _POLXmon, _SCMFvar[j].coefmon(), _POLFvar[j],  -1. );
  }

#ifdef MC__MINLPBND_DEBUG_SCDRL
  std::cout << _POLenv;
  { int dum; std::cin >> dum; }
#endif
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_mon_SQ
( t_quad const& SQenv, int const BASIS, bool const SCALED,
  bool const DAGINSERT )
{
  // Add all monomials in quadratic forms to polyhedral image
  for( auto const& mon : SQenv.SetMon() ){
    if( !mon.tord ) continue;
#ifdef MC__MINLPBND_DEBUG_MONSQ
    std::cout << "Current monomial: " << mon.display(BASIS) << std::endl;
#endif
    assert( _POLXmon.find( mon ) == _POLXmon.end() );
    _set_mon_DRL( mon, BASIS, SCALED, DAGINSERT );
  }

#ifdef MC__MINLPBND_DEBUG_MONSQ
  std::cout << "Monomial map:" << std::endl;
  for( auto const& [mon,polvar] : _POLXmon )
    std::cout << " " << mon.display(BASIS) << " == " << polvar
              << " (DAG: " << polvar.var() << ")" << std::endl;
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  if( !DAGINSERT ) return;

  // Add DRL cuts for participating monomials
  _add_mon_DRL();
#ifdef MC__MINLPBND_DEBUG_MONSCDRL
  std::cout << "Monomial map:" << std::endl;
  for( auto const& [mon,polvar] : _POLXmon )
    std::cout << " " << mon.display(BASIS) << " == " << polvar
              << " (DAG: " << polvar.var() << ")" << std::endl;
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_mon_SCDRL
( std::set<unsigned> const& ndxF, int const BASIS, bool const SCALED,
  bool const DAGINSERT )
{
  // Add all monomials in sparse Chebyshev models to polyhedral image
  for( unsigned j : ndxF ){
    for( auto const& [mon,coef] : _SCMFvar[j].coefmon() ){
#ifdef MC__MINLPBND_DEBUG_MONSCDRL
      std::cout << "Current monomial: " << mon.display(BASIS) << std::endl;
#endif
      _set_mon_DRL( mon, BASIS, SCALED, DAGINSERT );
    }
  }

#ifdef MC__MINLPBND_DEBUG_MONSCDRL
  std::cout << "Monomial map:" << std::endl;
  for( auto const& [mon,polvar] : _POLXmon )
    std::cout << " " << mon.display(BASIS) << " == " << polvar
              << " (DAG: " << polvar.var() << ")" << std::endl;
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
  if( !DAGINSERT ) return;

  // Add DRL cuts for participating monomials
  _add_mon_DRL();
#ifdef MC__MINLPBND_DEBUG_MONSCDRL
  std::cout << "Monomial map:" << std::endl;
  for( auto const& [mon,polvar] : _POLXmon )
    std::cout << " " << mon.display(BASIS) << " == " << polvar
              << " (DAG: " << polvar.var() << ")" << std::endl;
  std::cout << _POLenv;
  { int dum; std::cout << "PAUSED --"; std::cin >> dum; } 
#endif
}


template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_add_mon_DRL
()
{
  // Gather DAG variables participating in monomials
  auto first_Xaux = _Xmon.lower_bound( t_mon( 1, t_mon::t_expr() ) );
  auto last_Xaux  = _Xmon.lower_bound( t_mon( 2, t_mon::t_expr() ) );
  _Xaux.clear();
  for( auto it = first_Xaux; it != last_Xaux; ++it )
    _Xaux.push_back( it->second );

  // Gather POL variables participating in monomials
  auto first_POLXaux = _POLXmon.lower_bound( t_mon( 1, t_mon::t_expr() ) );
  auto last_POLXaux  = _POLXmon.lower_bound( t_mon( 2, t_mon::t_expr() ) );
  _POLXaux.clear();
  for( auto it = first_POLXaux; it != last_POLXaux; ++it )
    _POLXaux.push_back( it->second );

  // Add DRL cuts for participating monomials
  assert( _Xaux.size() == _POLXaux.size() );
  std::map< t_mon, PolVar<T>, lt_mon > POLXauxmon;
  _dag->eval( _POLwk, _Xmon, _POLXmon, _Xaux.size(), _Xaux.data(), _POLXaux.data() );
  _POLenv.generate_cuts( _POLXmon, false );

#ifdef MC__MINLPBND_DEBUG_MONDRL
  _dag->output( _dag->subgraph( _Xmon ) );
  { int dum; std::cin >> dum; }
#endif
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_mon_DRL
( t_mon const& mon, int const BASIS, bool const SCALED, bool const DAGINSERT )
{
  if( !mon.tord || _POLXmon.find( mon ) != _POLXmon.end() ) return;

  switch( BASIS ){
      
   // Case of power monomials
   case t_poly::Options::MONOM:
    if( !SCALED ){
      if( mon.tord == 1 ){
        // non-scaled first-order monomials correspond to existing variables
        auto const& ivar = mon.expr.begin()->first;
        _Xmon[mon] = _Xvar[ivar];
        _POLXmon[mon] = _POLXvar[ivar];
#ifdef MC__MINLPBND_DEBUG_MONDRL
        std::cout << " " << _POLXvar[ivar]  << " (DAG: " << _Xvar[ivar] << "): "
                  << _POLXvar[ivar].range() << ", " << _Xbnd[ivar] << std::endl;
#endif
      }
      else{
        // add unscaled power monomial to polyhedral image
	FFVar const& Xmon = _get_mon( mon, BASIS, DAGINSERT );
        _POLXmon[mon].set( &_POLenv, Xmon, _bnd_mon( mon, BASIS ), true );
#ifdef MC__MINLPBND_DEBUG_MONDRL
        std::cout << " (" << mon.display(BASIS) << ") = "
                  << _POLXmon[mon] << " (DAG: " << _POLXmon[mon].var() << "): "
                  << _POLXmon[mon].range() << std::endl;
#endif
      }
      return;
    }

    // add scaled power monomial to polyhedral image
    _POLXmon[mon].set( &_POLenv, _get_mon( mon, BASIS, DAGINSERT ), (mon.gcexp()%2? T(-1e0,1e0): T(0e0,1e0)), true );
#ifdef MC__MINLPBND_DEBUG_MONDRL
    std::cout << " (" << mon.display(BASIS) << ") = "
              << _POLXmon[mon] << " (DAG: " << _POLXmon[mon].var() << "): "
              << _POLXmon[mon].range() << std::endl;
#endif
    break;

    // Case of Chebyshev monomials
    case t_quad::Options::CHEB:
      // add Chebyshev monomial to polyhedral image
      _POLXmon[mon].set( &_POLenv, _get_mon( mon, BASIS, DAGINSERT ), T(-1e0,1e0), true );
#ifdef MC__MINLPBND_DEBUG_MONDRL
      std::cout << " (" << mon.display(BASIS) << ") = "
                << _POLXmon[mon] << " (DAG: " << _POLXmon[mon].var() << "): "
                << _POLXmon[mon].range() << std::endl;
#endif
      break;
  }

  // Add linear cut between degree 1 monomial and actual (unscaled) decision variable
  if( mon.tord == 1 ){
    auto const& ivar = mon.expr.begin()->first;
#ifndef MC__MINLPBND_DEBUG_MONDRL
    _POLenv.add_cut( nullptr, PolCut<T>::EQ, _SCMenv->refvar().at(ivar), _POLXvar[ivar],  1.,
                     _POLXmon[mon], -_SCMenv->scalvar().at(ivar) );
#else
    auto cutX = _POLenv.add_cut( nullptr, PolCut<T>::EQ, _SCMenv->refvar().at(ivar), _POLXvar[ivar],  1.,
                                 _POLXmon[mon], -_SCMenv->scalvar().at(ivar) );
    std::cout << "Scaling cut for variable X[" << ivar << "]: " << **cutX << std::endl;
#endif
  }
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_set_cuts_SQ
( t_quad& SQenv, std::set<unsigned> const& ndxF, bool const chkrem )
{
  // Add cuts for entries in MatFct
  auto itF = ndxF.cbegin();
  for( auto const& mat : SQenv.MatFct() ){
    assert( itF != ndxF.cend() );
    PolCut<T> *cutF1 = nullptr, *cutF2 = nullptr;
    if( !chkrem || Op<T>::diam(_SCMFvar[*itF].R()) == 0. ){
      cutF1 = *_POLenv.add_cut( nullptr, PolCut<T>::EQ, 0., _POLFvar[*itF], -1. );
    }
    else{
      cutF1 = *_POLenv.add_cut( nullptr, PolCut<T>::LE, -Op<T>::l(_SCMFvar[*itF].R()), _POLFvar[*itF], -1. );
      cutF2 = *_POLenv.add_cut( nullptr, PolCut<T>::GE, -Op<T>::u(_SCMFvar[*itF].R()), _POLFvar[*itF], -1. );
    }
    // Separate quadratic term
    _add_to_cuts( SQenv, mat, cutF1, cutF2 );
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
  for( auto const& mat : SQenv.MatRed() ){
    PolCut<T> *cutR = *_POLenv.add_cut( nullptr, PolCut<T>::EQ, 0. );
    //_add_to_cuts( mat, cutR );
    _add_to_cuts( SQenv, mat, cutR );
#ifdef MC__MINLPBND_DEBUG_SQ
    std::cout << "Reduction cuts #" << ++ired << ": " << *cutR << std::endl;
#endif
  }

  // Check entries in MatPSD
  if( options.PSDQUADCUTS ){
#ifdef MC__MINLPBND_DEBUG_SQ
    unsigned ipsd = 0;
#endif
    SQenv.tighten( options.PSDQUADCUTS>1? true: false );
    for( auto const& mat : SQenv.MatPSD() ){
      PolCut<T> *cutP = *_POLenv.add_cut( nullptr, PolCut<T>::GE, 0. );
      _add_to_cuts( mat, cutP );
#ifdef MC__MINLPBND_DEBUG_SQ
      std::cout << "PSD cuts #" << ++ipsd << ": " << *cutP << std::endl;
#endif
    }
  }
}

template <typename T, typename MIP, typename... ExtOps>
inline T
MINLPBND<T,MIP,ExtOps...>::_bnd_cheb
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

template <typename T, typename MIP, typename... ExtOps>
inline T
MINLPBND<T,MIP,ExtOps...>::_bnd_mon
( t_mon const& mon, int const BASIS )
const
{
  // compute (unscaled) power monomial bound
  T bndmon( 1e0 );
  for( auto const& [ivar,iord] : mon.expr ){
    switch( BASIS ){
     // Monomial basis
     case t_poly::Options::MONOM:
      bndmon *= Op<T>::pow( _POLXvar[ivar].range(), (int)iord );
      break;
     // Chebyshev basis
     case t_poly::Options::CHEB:
      bndmon *= _bnd_cheb( _POLXvar[ivar].range(), iord );
      break;
    }
  }
  return bndmon;
}

template <typename T, typename MIP, typename... ExtOps>
inline FFVar const&
MINLPBND<T,MIP,ExtOps...>::_get_mon
( t_mon const& mon, int const BASIS, bool const DAGINSERT )
{
  assert( mon.tord );
  auto itXmapmon = _Xmon.find( mon );
  if( itXmapmon != _Xmon.end() ) return itXmapmon->second;

  if( mon.tord == 1 || !DAGINSERT ){
    FFVar Xmon( _dag );
    auto itXmon = _dag->Vars().find( &Xmon );
    _Xmon[mon] = **itXmon;
    return **itXmon;
  }

  _Xaux.resize( mon.expr.size() );
  unsigned nvar = 0;
  for( auto const& [ivar,ord] : mon.expr ){
    // This assumes that variables are already available in _Xmon
    FFVar const& var = _Xmon[t_mon(ivar)];
    switch( BASIS ){
      case t_poly::Options::MONOM: _Xaux[nvar++] = pow( var, (int)ord ); break;
      case t_poly::Options::CHEB:  _Xaux[nvar++] = cheb(var, ord );      break;
    }
  }
  FFVar Xmon;
  switch( BASIS ){
    case t_poly::Options::MONOM: Xmon = FFBase::prod( nvar, _Xaux.data() );    break;
    case t_poly::Options::CHEB:  Xmon = Op<FFVar>::prod( nvar, _Xaux.data() ); break;
  }
#ifdef MC__MINLPBND_DEBUG_MONDRL
  std::cout << "Subgraph of monomial " << mon.display(BASIS) << std::endl;
  _dag->output( _dag->subgraph( 1, &Xmon ) );
  { int dum; std::cin >> dum; }
#endif
  auto itXmon = _dag->Vars().find( &Xmon );
  _Xmon[mon] = **itXmon;
  return **itXmon;
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_add_to_cuts
( t_quad const& SQenv, t_quad::map_SQuad const& mat, PolCut<T>* cut1, PolCut<T>* cut2 )
{
  // DC decomposition not required
    if( !options.DCQUADCUTS )
      return _add_to_cuts( mat, cut1, cut2 );

  // DC decomposition required
  for( auto const& matsep : SQenv.separate( mat ) ){
  
    // Append monomial if single term
    if( matsep.size() == 1 ){
      _add_to_cuts( matsep, cut1, cut2 );
      continue;
    }

    // Introduce auxiliary variable in cuts and new auxiliary cut
    PolVar<T> POLsep( &_POLenv, _get_range( matsep ), true );
    if( cut1 ) cut1->append( POLsep, 1. );
    if( cut2 ) cut2->append( POLsep, 1. );
    PolCut<T> *cutR = *_POLenv.add_cut( nullptr, PolCut<T>::EQ, 0., POLsep, -1. );
    _add_to_cuts( matsep, cutR );

    // Introduce auxiliary cuts for DC factorization
    PolCut<T> *cutDC = *_POLenv.add_cut( nullptr, PolCut<T>::EQ, 0., POLsep, -1. );
    for( auto const& [eigval,eigterm] : SQenv.factorize( matsep ) ){
      auto const& POLEVSQ = _append_cuts_dcdec( eigterm );
      cutDC->append( POLEVSQ, eigval );
    }
  }
}

template <typename T, typename MIP, typename... ExtOps>
inline void
MINLPBND<T,MIP,ExtOps...>::_add_to_cuts
( t_quad::map_SQuad const& mat, PolCut<T>* cut1, PolCut<T>* cut2 )
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
    else if( options.POLIMG.AGGREG_LQ && options.POLIMG.ALLOW_QUAD ){
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

template <typename T, typename MIP, typename... ExtOps>
inline T
MINLPBND<T,MIP,ExtOps...>::_get_range
( t_quad::map_SQuad const& mat )
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

template <typename T, typename MIP, typename... ExtOps>
inline PolVar<T>
MINLPBND<T,MIP,ExtOps...>::_append_cuts_dcdec
( t_quad::map_SPoly const& eigterm )
{
  // New auxiliary variable and cut for linear combition of monomials
  PolCut<T> *cutEV = *_POLenv.add_cut( nullptr, PolCut<T>::EQ, 0. );
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

template <typename T, typename MIP, typename... ExtOps>
inline PolVar<T>
MINLPBND<T,MIP,ExtOps...>::_append_cuts_monprod
( t_quad::key_SQuad const& ijmon )
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

template <typename T, typename MIP, typename... ExtOps>
template <typename U>
inline double
MINLPBND<T,MIP,ExtOps...>::_dH
( U const& X, U const& Y )
{
  return std::max( std::fabs(Op<U>::l(X)-Op<U>::l(Y)),
                   std::fabs(Op<U>::u(X)-Op<U>::u(Y)) );
}

template <typename T, typename MIP, typename... ExtOps>
template <typename U>
inline double
MINLPBND<T,MIP,ExtOps...>::_reducrel
( unsigned const n, U const* Xred, U const* X )
{
  double drel = 0.;
  for( unsigned ip=0; ip<n; ip++ )
    drel = std::max( drel, _dH( Xred[ip], X[ip] ) / Op<T>::diam( X[ip] ) );
  return drel;
}

template <typename T, typename MIP, typename... ExtOps>
template <typename U>
inline double
MINLPBND<T,MIP,ExtOps...>::_reducrel
( unsigned const n, U const* Xred, U const* X, U const* X0 )
{
  double drel = 0.;
  for( unsigned ip=0; ip<n; ip++ )
    drel = std::max( drel, _dH( Xred[ip], X[ip] ) / Op<T>::diam( X0[ip] ) );
  return drel;
}

template <typename T, typename MIP, typename... ExtOps>
inline
void
MINLPBND<T,MIP,ExtOps...>::Options::display
( std::ostream& out )
const
{
  // Display MINLPREF Options
  MINLPREF<T,ExtOps...>::display( out );
  
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
  out << std::setw(60) << "  MAXIMUM OPTIMIZATION-BASED REDUCTION LOOPS"
      << OBBTMAX << std::endl;
  out << std::setw(60) << "  THRESHOLD FOR OPTIMIZATION-BASED REDUCTION LOOP"
      << std::fixed << std::setprecision(0)
      << OBBTTHRES*1e2 << "%\n";
  out << std::setw(60) << "  BACKOFF FOR OPTIMIZATION-BASED REDUCTION"
      << std::scientific << std::setprecision(1)
      << OBBTBKOFF << std::endl;
}

} // end namescape mc

#endif

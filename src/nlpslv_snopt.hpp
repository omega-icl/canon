// Copyright (C) 2020 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_NLPSLV_SNOPT Local (Continuous) Nonlinear Optimization interfacing SNOPT with MC++
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
where \f$f, g_1, \ldots, g_m\f$ are factorable, potentially nonlinear, real-valued functions; and \f$x_1, \ldots, x_n\f$ are continuous decision variables. The class mc::NLPSLV_SNOPT solves such NLP problems using the software package <A href="https://web.stanford.edu/group/SOL/snopt.htm">SNOPT</A>, which implements a a sparse sequential quadratic programming (SQP) algorithm with limited-memory quasi-Newton approximations to the Hessian of the Lagrangian. SNOPT requires the first-order derivatives and exploits the sparsity pattern of the objective and constraint functions in the NLP model. This information is generated using direct acyclic graphs (DAG) in <A href="https://projects.coin-or.org/MCpp">MC++</A>.

\section sec_NLPSLV_SNOPT_solve How to Solve an NLP Model using mc::NLPSLV_SNOPT?

Consider the following NLP:
\f{align*}
  \max_{\bf p}\ & p_1+p_2 \\
  \text{s.t.} \ & p_1\,p_2 \leq 4 \\
  & 0 \leq p_1 \leq 6\\
  & 0 \leq p_2 \leq 4\,.
\f}

Start by instantiating an mc::NLPSLV_SNOPT class object, which is defined in the header file <tt>nlpslv_snopt.hpp</tt>:

\code
  mc::NLPSLV_SNOPT NLP;
\endcode

Next, set the variables and objective/constraint functions after creating a DAG of the problem: 

\code
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  NLP.set_dag( &DAG );                           // DAG
  NLP.add_var( P[0], 0, 6 );                     // decision variables and bounds
  NLP.add_var( P[1], 0, 4 );
  NLP.set_obj( mc::BASE_NLP::MAX, P[0]+P[1] );   // objective function
  NLP.add_ctr( mc::BASE_NLP::LE, P[0]*P[1]-4. ); // constraints
\endcode

Possibly set options using the class member NLPSLV_SNOPT::options:

\code
  NLP.options.DISPLEVEL = 0;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
\endcode

Finally, set up the NLP problem and solve it from a specified initial point:

\code
  NLP.setup();
  double p0[NP] = { 5., 1. };
  NLP.solve( p0 );
\endcode

The optimization results can be retrieved or displayed as an instance of mc::SOLUTION_OPT using the method <a>NLPSLV_SNOPT::solution</a>:

\code
  std::cout << NLP.solution();
\endcode

producing the following display:

\verbatim
  STATUS: 1
  X[0]:  LEVEL =  6.000000e+00  MARGINAL =  8.888889e-01
  X[1]:  LEVEL =  6.666667e-01  MARGINAL =  0.000000e+00
  F[0]:  LEVEL =  6.666667e+00  MARGINAL =  0.000000e+00
  F[1]:  LEVEL =  8.881784e-16  MARGINAL =  1.666667e-01
\endverbatim
*/

#ifndef MC__NLPSLV_SNOPT_HPP
#define MC__NLPSLV_SNOPT_HPP

#include <stdexcept>
#include <cassert>
#include <thread>
#include "snoptProblem.hpp"

#include "mctime.hpp"
#include "base_nlp.hpp"

#ifdef MC__USE_SOBOL
  #include <boost/random/sobol.hpp>
  #include <boost/random/uniform_01.hpp>
  #include <boost/random/variate_generator.hpp>
#endif

#define MC__NLPSLV_SNOPT_CHECK
//#define MC__NLPSLV_SNOPT_DEBUG
//#define MC__NLPSLV_SNOPT_TRACE

namespace mc
{

class WORKER_SNOPT;

void REG_CBACK_SNOPT
( WORKER_SNOPT*th, void (WORKER_SNOPT::*usr)( int*, int*, double*,
  int*, int*, double*, int*, int*, double*, int* ) );

void UNREG_CBACK_SNOPT
();

extern "C"
void WRP_CBACK_SNOPT
  ( int *Status, int *n, double *x, int *needF, int *neF, double *F, int *needG, int *neG,
    double *G, char *cu, int *lencu, int *iu, int *leniu, double *ru, int *lenru );

//! @brief C++ class for calling SNOPT on local threads
////////////////////////////////////////////////////////////////////////
//! mc::WORKER_SNOPT is a C++ class for calling SNOPT on local threads
////////////////////////////////////////////////////////////////////////
struct WORKER_SNOPT
{
  //! @brief Constructor
  WORKER_SNOPT
    ()
    {
      snOptA.initialize( "", 0, "", 0 );
      is_registered = false;
    }

  //! @brief Function registering thread
  void registration
    ()
    {
      if( is_registered ) return;
      REG_CBACK_SNOPT( this, &WORKER_SNOPT::callback );
      is_registered = true;
    }

  //! @brief Function unregistering thread
  void unregistration
    ()
    {
      if( !is_registered ) return;
      UNREG_CBACK_SNOPT();
      is_registered = false;
    }

  //! @brief flag indicating whether the thread is registered
  bool                is_registered;
  
  //! @brief SNOPTA class object
  snoptProblemA       snOptA;

  //! @brief vector of parameters in DAG
  std::vector<FFVar>  Pvar;
  //! @brief vector of decision variable values
  std::vector<double> Xval;
  //! @brief vector of decision variable multipliers
  std::vector<double> Xmul;
  //! @brief vector of decision variable states
  std::vector<int>    Xstate;
  //! @brief vector of decision variable lower bounds
  std::vector<double> Xlow;
  //! @brief vector of decision variable upper bounds
  std::vector<double> Xupp;

  //! @brief vector of function values
  std::vector<double> Fval;
  //! @brief vector of function offsets
  std::vector<double> Foff;
  //! @brief vector of function multipliers
  std::vector<double> Fmul;
  //! @brief vector of function states
  std::vector<int>    Fstate;
  //! @brief vector of constraint lower bounds
  std::vector<double> Flow;
  //! @brief vector of constraint upper bounds
  std::vector<double> Fupp;

  //!@brief row coordinates of nonzero elements in the linear part of each function
  std::vector<int>    iAfun;
  //!@brief column coordinates of nonzero elements in the linear part of each function
  std::vector<int>    jAvar;
  //!@brief values of nonzero elements in the linear part of each function
  std::vector<double> Aval;

  //!@brief row coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>    iGfun;
  //!@brief column coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>    jGvar;
  //!@brief set of indices of nonlinear constraints 
  std::set<unsigned>  Gndx;
  //!@brief values of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<double> Gval;

  //!@brief number of superbasic variables
  int                 nS;
  //!@brief sum of constraints that lie outside of their bounds by more than the 'minor feasibility tolerance' before the solution is unscaled
  int                 nInf;
  //!@brief sum of infeasibilities for the constraints that lie outside of their bounds by more than the 'minor feasibility tolerance' before the solution is unscaled
  double              sInf;
  
  //! @brief local copy of DAG
  FFGraph DAG;
  //! @brief vector of decision variables in DAG
  std::vector<FFVar>  Xvar;
  //! @brief vector of functions in DAG
  std::vector<FFVar>  Fvar;
  //! @brief vector of derivatives for the nonlinear part of each function
  std::vector<FFVar>  Gvar;

  //! @brief list of operations in functions
  FFSubgraph          op_F;
  //! @brief list of operations in function derivatives
  FFSubgraph          op_G;
  //! @brief Storage vector for DAG evaluation in double arithmetic
  std::vector<double> dwk;

  //! @brief Time limit
  double              tMAX;
  //! @brief Structure holding solution information
  SOLUTION_OPT        solution;

  //! @brief Function updating NLP bounds and parameters
  void update
    ( int const nX, double const* Xl, double const* Xu, int const nP,
      double const* Pval );

  //! @brief Function initializing NLP solution
  void initialize
    ( int const nF, int const nX, double const* Xini, bool const warm );
    
  //! @brief Function calling NLP solver
  void solve
    ( int& stat, int iStart, int nF, int nX, double ObjAdd, int ObjRow,
      int nA, int nG );

  //! @brief Function finalizing NLP solution
  void finalize
    ( int const stat, int const ObjRow, double const ObjMul );

 //! @brief User function evaluating nonlinear functions and gradients
 void callback
   ( int *Status, int *neX, double *X, int *needF, int *neF, double *F,
     int *needG, int *neG, double *G, int *fdG );

  //! @brief Function testing NLP solution feasibility
  bool feasible
    ( double const CTRTOL, int const nF, int const nX, int const ObjRow,
      int const nA );

  //! @brief Function testing NLP solution stationarity
  bool stationary
    ( double const GRADTOL, int const nX, int const nA, int const nG );

  //! @brief Function computing NLP cost correction to compensate for infeasibility
  double correction
    ( int const nF, int const nX, int const ObjRow, int const nA );
};

inline
void
WORKER_SNOPT::update
( int const nX, double const* Xl, double const* Xu, int const nP, double const* Pval )
{
#ifdef MC__NLPSLV_SNOPT_TRACE
    std::cout << "  WORKER_SNOPT::initialize  " << warm << std::endl;
#endif

  // variable bounds
  if( Xl ) Xlow.assign( Xl, Xl+nX );
  if( Xu ) Xupp.assign( Xu, Xu+nX );
#ifdef MC__NLPSLV_SNOPT_DEBUG
  for( int i=0; i<nX; i++ )
    std::cout << "  Xlow[" << i << "] = " << Xlow[i]
              << "  Xupp[" << i << "] = " << Xupp[i] << std::endl;
#endif

  // parameter values
  for( int i=0; !Pvar.empty() && Pval && i<nP; i++ ){
    Pvar[i].set( Pval[i] );
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << "  Pvar[" << i << "] = " << Pvar[i] << std::endl;
#endif
  }
}

inline
void
WORKER_SNOPT::initialize
( int const nF, int const nX, double const* Xini, bool const warm )
{
#ifdef MC__NLPSLV_SNOPT_TRACE
    std::cout << "  WORKER_SNOPT::initialize  " << warm << std::endl;
#endif

  // initial starting point
  if( Xini )       Xval.assign( Xini, Xini+nX );
  else if( !warm ) Xval.assign( nX, 0. );
#ifdef MC__NLPSLV_SNOPT_DEBUG
  for( int i=0; i<nX; i++ )
    std::cout << "  Xval[" << i << "] = " << Xval[i] << std::endl;
#endif

  // other basis arrays and initialization
  if( warm ) return;

  Xstate.assign( nX, 0 );
  Xmul.assign( nX, 0. );

  Fstate.assign( nF, 0 );
  Fval.assign( nF, 0. );
  Fmul.assign( nF, 0. );
}
    
inline
void
WORKER_SNOPT::solve
( int& stat, int iStart, int nF, int nX, double ObjAdd, int ObjRow,
  int nA, int nG )
{
#ifdef MC__NLPSLV_SNOPT_TRACE
  std::cout << "  WORKER_SNOPT::solve\n";
#endif
  // Run NLP solver
  stat = snOptA.solve( iStart, nF, nX, ObjAdd, ObjRow, WRP_CBACK_SNOPT,
    iAfun.data(), jAvar.data(), Aval.data(), nA, iGfun.data(), jGvar.data(),
    nG, Xlow.data(), Xupp.data(), Flow.data(), Fupp.data(), Xval.data(),
    Xstate.data(), Xmul.data(), Fval.data(), Fstate.data(), Fmul.data(),
    nS, nInf, sInf );
}

inline
void
WORKER_SNOPT::finalize
( int const stat, int const ObjRow, double const ObjMul )
{
#ifdef MC__NLPSLV_SNOPT_TRACE
  std::cout << "  WORKER_SNOPT::finalize\n";
#endif
  solution.stat       = stat;
  solution.x          = Xval;
  solution.ux         = Xmul;
  solution.f          = Fval;
  assert( Fval.size() == Foff.size() );
  for( unsigned i=0; i<Foff.size(); i++ ){
    //std::cout << "Fval[" << i << "] = " << Fval[i] << std::endl;
    //std::cout << "Foff[" << i << "] = " << Foff[i] << std::endl;
    solution.f[i] += Foff[i];
    //std::cout << "solution.f[" << i << "] = " << solution.f[i] << std::endl;
  }
  solution.uf         = Fmul;
  if( ObjRow >= 0 ) solution.uf[ObjRow] = ObjMul;
}

inline
void
WORKER_SNOPT::callback
( int *Status, int *neX, double *X, int *needF, int *neF, double *F,
  int *needG, int *neG, double *G, int *fdG )
{
#ifdef MC__NLPSLV_SNOPT_TRACE
    std::cout << "  WORKER_SNOPT::callback\n";
#endif
  if( *Status > 1 ) return; // <- Could be set as on option?

  if( userclock() > tMAX ){
    *Status = -2;
    return;
  }
  
#ifdef MC__NLPSLV_SNOPT_DEBUG
  for( int i=0; i<*neX; i++ )
    std::cout << "  X[" << i << "] = " << X[i] << std::endl;
#endif

  try{
    // Needs nonlinear function values
    if( *needF > 0 ){
      DAG.eval( op_F, dwk, Gndx, Fvar.data(), F, *neX, Xvar.data(), X );
#ifdef MC__NLPSLV_SNOPT_DEBUG
      for( auto && i : Gndx )
        std::cout << "  F[" << i << "] = " << F[i] << std::endl;
#endif
    }
    
    // Needs nonlinear function derivatives
    if( *needG > 0 && !*fdG ){
      DAG.eval( op_G, dwk, *neG, Gvar.data(), G, *neX, Xvar.data(), X );
#ifdef MC__NLPSLV_SNOPT_DEBUG
      for( int ie=0; ie<*neG; ie++ )
        std::cout << "  G[" << iGfun[ie] << "," << jGvar[ie] << "] = " << G[ie] << std::endl;
#endif
    }
  }
  catch(...){
    *Status = -1;
  }
}

inline
bool
WORKER_SNOPT::feasible
( double const CTRTOL, int const nF, int const nX, int const ObjRow,
  int const nA )
{
  double maxinfeas = 0.;
  for( int i=0; i<nX; i++ ){
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << "X[" << i << "]: " << Xlow[i] << " <= " << solution.x[i] << " <= " << Xupp[i] << std::endl;
#endif
    maxinfeas = Xlow[i] - solution.x[i];
    if( maxinfeas > CTRTOL ) return false;
    maxinfeas = solution.x[i] - Xupp[i];
    if( maxinfeas > CTRTOL ) return false;
  }

  try{
    solution.f.assign( nF, 0. );
    DAG.eval( op_F, dwk, Gndx, Fvar.data(), solution.f.data(), nX, Xvar.data(), solution.x.data() );
    for( int iA=0; iA<nA; iA++ )
      solution.f[iAfun[iA]] += Aval[iA] * solution.x[jAvar[iA]];
  }
  catch(...){
    return false;
  }
  for( int i=0; i<nF; i++ ){
    if( i == ObjRow ) continue;      
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << "F[" << i << "]: " << Flow[i] << " <= " << solution.f[i] << " <= " << Fupp[i] << std::endl;
#endif
    maxinfeas = Flow[i] - solution.f[i];
    if( maxinfeas > CTRTOL ) return false;
    maxinfeas = solution.f[i] - Fupp[i];
    if( maxinfeas > CTRTOL ) return false;
  }

  return true;
}

inline
bool
WORKER_SNOPT::stationary
( double const GRADTOL, int const nX, int const nA, int const nG )
{
  try{
    Gval.resize( nG );
    DAG.eval( op_G, dwk, nG, Gvar.data(), Gval.data(), nX, Xvar.data(), solution.x.data() );
  }
  catch(...){
    return false;
  }
  std::vector<double> gradL = solution.ux;
  for( int ie=0; ie<nG; ie++ )
    gradL[jGvar[ie]] += Gval[ie] * solution.uf[iGfun[ie]];
  for( int ie=0; ie<nA; ie++ )
    gradL[jAvar[ie]] += Aval[ie] * solution.uf[iAfun[ie]];
  for( int i=0; i<nX; i++ ){
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << "  gradL[" << i << "] : " << gradL[i] << " = 0" << std::endl;
#endif
    if( std::fabs( gradL[i] ) > GRADTOL ) return false;
  }
  return true;
}

inline
double
WORKER_SNOPT::correction
( int const nF, int const nX, int const ObjRow, int const nA )
{
  double costcorr = 0.;
  for( int i=0; i<nX; i++ ){
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << "X[" << i << "]: " << Xlow[i] << " <= " << solution.x[i] << " <= " << Xupp[i] << std::endl;
#endif
    costcorr += std::max( Xlow[i] - solution.x[i], 0. ) * solution.ux[i];
    costcorr -= std::max( solution.x[i] - Xupp[i], 0. ) * solution.ux[i];
  }

  try{
    solution.f.assign( nF, 0. );
    DAG.eval( op_F, dwk, Gndx, Fvar.data(), solution.f.data(), nX, Xvar.data(), solution.x.data() );
    for( int iA=0; iA<nA; iA++ )
      solution.f[iAfun[iA]] += Aval[iA] * solution.x[jAvar[iA]];
  }
  catch(...){
    return false;
  }
  for( int i=0; i<nF; i++ ){
    if( i == ObjRow ) continue;      
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << "F[" << i << "]: " << Flow[i] << " <= " << solution.f[i] << " <= " << Fupp[i] << std::endl;
#endif
    costcorr += std::max( Flow[i] - solution.f[i], 0. ) * solution.uf[i];
    costcorr -= std::max( solution.f[i] - Fupp[i], 0. ) * solution.uf[i];
  }

  return costcorr;
}

//! @brief C++ class for NLP solution using SNOPT and MC++
////////////////////////////////////////////////////////////////////////
//! mc::NLPSLV_SNOPT is a C++ class for solving NLP problems
//! using SNOPT and MC++
////////////////////////////////////////////////////////////////////////
class NLPSLV_SNOPT:
  public virtual BASE_NLP
{
  // Overloading stdout operator
  //friend std::ostream& operator<<
  //  ( std::ostream &, NLPSLV_SNOPT const& );

public:

  //! @brief NLP solution status
  enum STATUS{
     SUCCESSFUL=0,		//!< Optimal solution found (possibly not within required accuracy)
     INFEASIBLE,	    //!< The problem appears to be infeasible
     UNBOUNDED,	        //!< The problem appears to be unbounded
     INTERRUPTED,       //!< Resource limit reached
     FAILURE,           //!< Terminated after numerical difficulties
     ABORTED            //!< Critical error encountered
  };

  //! @brief NLP starting point
  enum START{
     COLD=0,            //!< Crash procedure used
     WARM=2             //!< Warm start with _nS, _Xstate, _Fstate defining a valid starting point from previous call
  };

private:

  //! @brief vector of SNOPTA workers
  std::vector<WORKER_SNOPT*> _worker;

  //! @brief number of parameters in problem
  int                 _nP;
  //! @brief vector of parameters in DAG
  std::vector<FFVar>  _Pvar;

  //! @brief number of decision variables (independent and dependent) in problem
  int                 _nX;
  //! @brief vector of decision variables in DAG
  std::vector<FFVar>  _Xvar;
  //! @brief vector of decision variable lower bounds
  std::vector<double> _Xlow;
  //! @brief vector of decision variable upper bounds
  std::vector<double> _Xupp;
  //! @brief vector of zeros for constant term calculation in functions
  std::vector<double> _X0;

  //! @brief number of functions (objective and constraints) in problem
  int                 _nF;
  //! @brief vector of functions in DAG
  std::vector<FFVar>  _Fvar;
  //! @brief vector of function offsets
  std::vector<double> _Foff;
  //! @brief vector of function lower bounds
  std::vector<double> _Flow;
  //! @brief vector of function upper bounds
  std::vector<double> _Fupp;

  //! @brief specifies how a starting point is to be obtained
  START               _iStart;
  //! @brief direction of objective function (-1:minmize; 0:feasibility; 1:maximize)
  int                 _ObjDir;
  //! @brief row to act as the objective function (0 for feasibility problem)
  int                 _ObjRow;
  //! @brief constant to be added to the objective rowF (_ObjRow) for printing purposes
  double              _ObjAdd;
  //! @brief multiplier associated to the objective
  double              _ObjMul;

  //!@brief number of nonzero elements in the linear part of each function
  int                 _nA;
  //!@brief row coordinates of nonzero elements in the linear part of each function
  std::vector<int>    _iAfun;
  //!@brief column coordinates of nonzero elements in the linear part of each function
  std::vector<int>    _jAvar;
  //!@brief values of nonzero elements in the linear part of each function
  std::vector<double> _Aval;

  //!@brief number of nonzero elements in the derivative of the nonlinear part of each function
  int                 _nG;
  //!@brief row coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>    _iGfun;
  //!@brief column coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>    _jGvar;
  //! @brief vector of derivatives for the nonlinear part of each function
  std::vector<FFVar>  _Gvar;
  //!@brief set of indices of nonlinear constraints 
  std::set<unsigned>  _Gndx;

  //! @brief function sparse derivatives
  std::tuple< unsigned, unsigned const*, unsigned const*, FFVar const* > _fct_grad;
  //!@brief pointer to user array passed to snOptA
  int*  _iusr;

  //! @brief whether a record of the original model is available
  bool                _recModel;
  //! @brief direction of objective function (-1:minmize; 0:feasibility; 1:maximize)
  int                 _recObjDir;
  //! @brief row to act as the objective function (0 for feasibility problem)
  int                 _recObjRow;
  //! @brief constant to be added to the objective rowF (_ObjRow) for printing purposes
  double              _recObjAdd;
  //! @brief multiplier associated to the objective
  double              _recObjMul;

  //! @brief number of functions (objective and constraints) in problem
  int                 _recnF;
  //! @brief vector of functions in DAG
  std::vector<FFVar>  _recFvar;
  //! @brief vector of function offsets
  std::vector<double> _recFoff;
  //! @brief vector of function lower bounds
  std::vector<double> _recFlow;
  //! @brief vector of function upper bounds
  std::vector<double> _recFupp;

  //!@brief number of nonzero elements in the linear part of each function
  int                 _recnA;
  //!@brief row coordinates of nonzero elements in the linear part of each function
  std::vector<int>    _reciAfun;
  //!@brief column coordinates of nonzero elements in the linear part of each function
  std::vector<int>    _recjAvar;
  //!@brief values of nonzero elements in the linear part of each function
  std::vector<double> _recAval;

  //!@brief number of nonzero elements in the derivative of the nonlinear part of each function
  int                 _recnG;
  //!@brief row coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>    _reciGfun;
  //!@brief column coordinates of nonzero elements in the derivative of the nonlinear part of each function
  std::vector<int>    _recjGvar;
  //! @brief vector of derivatives for the nonlinear part of each function
  std::vector<FFVar>  _recGvar;
  //!@brief set of indices of nonlinear constraints 
  std::set<unsigned>  _recGndx;

  //! @brief Structure holding solution information
  SOLUTION_OPT        _solution;

public:

  //! @brief Constructor
  NLPSLV_SNOPT()
    : _nP(0), _nX(0), _nF(0), _iStart(COLD),
      _ObjDir(0), _ObjRow(-1), _ObjAdd(0.), _ObjMul(0.),
      _nA(0), _nG(0), _recModel(false), _solution(FAILURE)
    {
      _iusr = new int[1];
    }

  //! @brief Destructor
  virtual ~NLPSLV_SNOPT()
    {
      _cleanup_grad();
      delete[] _iusr;
      for( auto && th : _worker ) delete th;
    }

  //! @brief NLP solver options
  struct Options
  {
    //! @brief Constructor
    Options():
      FEASTOL(1e-7), OPTIMTOL(1e-5), MAXITER(200), GRADMETH(FAD), GRADCHECK(false),
      QPFEASTOL(1e-7), QPMAXITER(500), QPMETH(CHOL), DISPLEVEL(0), LOGFILE(),
      FEASPB(false), TIMELIMIT(72e2), MAXTHREAD(0)
      {}
    //! @brief Assignment operator
    Options& operator= ( Options&options ){
        FEASTOL      = options.FEASTOL;
        OPTIMTOL     = options.OPTIMTOL;
        MAXITER      = options.MAXITER;
        GRADMETH     = options.GRADMETH;
        GRADCHECK    = options.GRADCHECK;
        QPFEASTOL    = options.QPFEASTOL;
        QPMAXITER    = options.QPMAXITER;
        QPMETH       = options.QPMETH;
        DISPLEVEL    = options.DISPLEVEL;
        LOGFILE      = options.LOGFILE;
        TIMELIMIT    = options.TIMELIMIT;
        FEASPB       = options.FEASPB;
        MAXTHREAD    = options.MAXTHREAD;
        return *this;
      }
    //! @brief Enumeration type for Hessian strategy
    enum QP_STRATEGY{
      CHOL=0,  //!< Cholesky QP solver
      CG,      //!< Conjugate-gradient QP solver 
      QN       //!< Quasi-Newton QP solver       
    };
    //! @brief Enumeration type for gradient strategy
    enum GRADIENT_STRATEGY{
      FAD=0,	//!< Forward AD
      BAD,		//!< Backward AD
      FD        //!< Finite differences
    };
    //! @brief Corresponds to "Major feasibility tolerance" in snOptA, which specifies how accurately the nonlinear constraints should be satisfied.
    double FEASTOL;
    //! @brief Corresponds to "Major optimality tolerance" in snOptA, which specifies the final accuracy of the dual variables.
    double OPTIMTOL;
   //! @brief Corresponds to "Major iterations limit" in snOptA, which is the maximum number of major iterations allowed. It is intended to guard againstan excessive number of linearizations of the constraints. If non-positive value given, both feasibility and optimality are checked.
    int MAXITER;
    //! @brief Corresponds to "Derivative option" in snOptA, which specifies whether nonlinear function gradients are known analytically (FAD, BAD) or estimated using finite differences (FD).
    GRADIENT_STRATEGY GRADMETH;
    //! @brief Corresponds to "Verify level" in snOptA, which enables finite-difference checks on the derivatives computed by the user-provided routines at the first point that satisfies all bounds and linear constraints.
    bool GRADCHECK;
    //! @brief Corresponds to "Minor feasibility tolerance" in snOptA, which ensures that all linear constraints eventually satisfy their upper and lower bounds to within this tolerance.
    double QPFEASTOL;
   //! @brief Corresponds to "Minor iterations limit" in snOptA. If the number of minor iterations for the optimality phase of the QP subproblem exceeds this value, then all nonbasic QP variables that have not yet moved are frozen at their current values and the reduced QP is solved to optimality.
    int QPMAXITER;
    //! @brief Corresponds to "QPSolver" in snOptA, which specifies the method used to solve the QP subproblems: CHOL - Cholesky QP solver; CG - conjugate-gradient QP solver; QN - quasi-Newton QP solver.
    QP_STRATEGY QPMETH;
    //! @brief Corresponds to "Summary file" in snOptA, which specifies whether (>0) or not (<=0) to generate the summary file.
    int DISPLEVEL;
    //! @brief Corresponds to "Print file" in snOptA, which specifies the file name for the "Summary file". Displays to screen if an empty string is passed (default).
    std::string LOGFILE;
    //! @brief Corresponds to "Feasible point" in snOptA, which specifies to “Ignore the objective function” while finding afeasible point for the linear and nonlinear constraints.
    bool FEASPB;
    //! @brief Maximum run-time (in seconds) - this is checked externally to snOptA based on the wall clock.
    double TIMELIMIT;
    //! @brief Maximum number of threads for multistart solve.
    unsigned MAXTHREAD;
  } options;

  //! @brief Setup NLP model before solution
  bool setup
    ();

  //! @brief Change objective function of NLP model
  bool set_obj_lazy
    ( t_OBJ const& type, FFVar const& obj );

  //! @brief Append general constraint to NLP model
  bool add_ctr_lazy
    ( t_CTR const type, FFVar const& ctr );

  //! @brief Restore original NLP model
  bool restore_model
    ();

  //! @brief Solve NLP model -- return value is SNOPT status
  template <typename T>
  int solve
    ( double const* Xini, T const* Xbnd, double const* Pval=0,
      bool const warm=false );

  //! @brief Solve NLP model -- return value is SNOPT status
  int solve
    ( double const* Xini=0, double const* Xlow=0, double const* Xupp=0,
      double const* Pval=0, bool const warm=false );

#ifdef MC__USE_SOBOL
  //! @brief Solve NLP model using multistart search -- return value is SNOPT status
  template <typename T>
  int solve
    ( unsigned const NSAM, T const* Xbnd, double const* Pval=0,
      bool const* logscal=0, bool const disp=false );

  //! @brief Solve NLP model using multistart search -- return value is SNOPT status
  int solve
    ( unsigned const NSAM, double const* Xlow=0, double const* Xupp=0,
      double const* Pval=0, bool const* logscal=0, bool const disp=false );
#endif

  //! @brief Test primal feasibility
  bool is_feasible
    ( double const* x, double const CTRTOL );

  //! @brief Test primal feasibility of current solution point
  bool is_feasible
    ( const double CTRTOL );

  //! @brief Test dual feasibility
  bool is_stationary
    ( double const* x, double const* ux, double const* uf, double const GRADTOL );

  //! @brief Test dual feasibility of current solution point
  bool is_stationary
    ( double const GRADTOL );//, const double NUMTOL=1e-8 );

  //! @brief Compute cost correction to compensate for infeasibility of current solution
  double cost_correction
    ();

  //! @brief Compute cost correction to compensate for infeasibility
  double cost_correction
    ( double const* x, double const* ux, double const* uf );

  //! @brief Get solution info
  SOLUTION_OPT const& solution() const
    {
      return _solution;
    }

  //! @brief Status after last NLP call
  STATUS get_status
    ()
    const
    {
      if     ( _solution.stat < 10 ) return SUCCESSFUL;
      else if( _solution.stat < 20 ) return INFEASIBLE;
      else if( _solution.stat < 30 ) return UNBOUNDED;
      else if( _solution.stat < 40 ) return INTERRUPTED;
      else if( _solution.stat < 50 ) return FAILURE;
      else if( _solution.stat < 70 ) return ABORTED;
      else if( _solution.stat < 80 ) return INTERRUPTED;
      else                           return ABORTED;
    }

protected:

  //! @brief Make a local copy of the original NLP model
  bool _record_model
    ();

  //! @brief set the solver options
  int _set_options
    ( WORKER_SNOPT * th );

  //! @brief set the worker internal variables
  void _set_worker
    ( WORKER_SNOPT * th );

  //! @brief resize the number of workers
  void _resize_workers
    ( int const noth );

#ifdef MC__USE_SOBOL
  //! @brief call multistart NLP solver on worker th
  void _mssolve
    ( int const th, unsigned const NOTHREADS, unsigned const NSAM, bool const* logscal,
      bool const DISP, int& feasible, SOLUTION_OPT& stat );
#endif

  //! @brief Cleanup gradient storage
  void _cleanup_grad
    ()
    {
      delete[] std::get<1>(_fct_grad);  std::get<1>(_fct_grad) = 0;
      delete[] std::get<2>(_fct_grad);  std::get<2>(_fct_grad) = 0;
      delete[] std::get<3>(_fct_grad);  std::get<3>(_fct_grad) = 0;
    }

private:

  //! @brief Private methods to block default compiler methods
  NLPSLV_SNOPT( NLPSLV_SNOPT const& );
  NLPSLV_SNOPT& operator=( NLPSLV_SNOPT const& );
};

thread_local WORKER_SNOPT* PTR_WORKER_SNOPT = nullptr;
thread_local void (WORKER_SNOPT::*PTR_CBACK_SNOPT)( int*, int*, double*,
  int*, int*, double*, int*, int*, double*, int* ) = nullptr;

void REG_CBACK_SNOPT
( WORKER_SNOPT*th, void (WORKER_SNOPT::*usr)( int*, int*, double*,
  int*, int*, double*, int*, int*, double*, int* ) )
{
#ifdef MC__NLPSLV_SNOPT_CHECK
  assert( PTR_WORKER_SNOPT == nullptr && PTR_CBACK_SNOPT == nullptr );
#endif
  PTR_WORKER_SNOPT = th;
  PTR_CBACK_SNOPT  = usr;
}

void UNREG_CBACK_SNOPT
()
{
#ifdef MC__NLPSLV_SNOPT_CHECK
  assert( PTR_WORKER_SNOPT != nullptr && PTR_CBACK_SNOPT != nullptr );
#endif
  PTR_WORKER_SNOPT = nullptr;
  PTR_CBACK_SNOPT  = nullptr;
}

extern "C"
void WRP_CBACK_SNOPT
( int *Status, int *n, double *x, int *needF, int *neF, double *F, int *needG, int *neG,
  double *G, char *cu, int *lencu, int *iu, int *leniu, double *ru, int *lenru )
{
#ifdef MC__NLPSLV_SNOPT_CHECK
  //std::cout << "PTR_WORKER_SNOPT: " << PTR_WORKER_SNOPT << "  PTR_CBACK_SNOPT: " << PTR_CBACK_SNOPT << std::endl;
  assert( PTR_WORKER_SNOPT != nullptr && PTR_CBACK_SNOPT != nullptr);
#endif
  return (PTR_WORKER_SNOPT->*PTR_CBACK_SNOPT)( Status, n, x, needF, neF, F, needG, neG, G, iu );
}

inline
bool
NLPSLV_SNOPT::setup
()
{
  // full set of parameters
  _Pvar = _par;
  _nP = _Pvar.size();
  
  // full set of decision variables (independent & dependent)
  _Xvar = _var;
  _Xvar.insert( _Xvar.end(), _dep.begin(), _dep.end() );
  _nX = _Xvar.size();
  _X0.resize( _nX, 0. );

  // full set of variable bounds (independent & dependent)
  _Xlow = _varlb;
  _Xupp = _varub;
  _Xlow.insert( _Xlow.end(), _deplb.begin(), _deplb.end() );
  _Xupp.insert( _Xupp.end(), _depub.begin(), _depub.end() );

  // full set of nonlinear functions (cost, constraints & equations)
  _Fvar.clear();
  _Flow.clear();
  _Fupp.clear();
  _Foff.clear();
  
  int ndxF = 0;
  if( std::get<0>(_obj).size() ){   // First, cost function
    _Fvar.push_back( std::get<1>(_obj)[0] );
    _ObjDir = (std::get<0>(_obj)[0]==BASE_OPT::MIN? -1: 1 );
    _ObjRow = ndxF;
    _ObjAdd =  0.;
    _ObjMul = -1.; 
    if( _Fvar[ndxF].dep().worst() > FFDep::L )
      _Gndx.insert( ndxF );
    else
      _dag->eval( 1, &_Fvar[ndxF], &_ObjAdd, _nX, _Xvar.data(), _X0.data() ); 
    _Flow.push_back( -BASE_OPT::INF );
    _Fupp.push_back(  BASE_OPT::INF );
    _Foff.push_back( _ObjAdd );
  }
  else{
    _ObjDir = 0;
    _ObjRow = ndxF = -1;
    _ObjAdd = 0.;
    _ObjMul = 0.;
  }

  for( unsigned i=0; i<std::get<0>(_ctr).size(); i++ ){ // Then, regular constraints
    if( std::get<3>(_ctr)[i] ) continue; // ignore if redundant
    ndxF++;
    _Fvar.push_back( std::get<1>(_ctr)[i] );
    double CtrCst = 0.;
    if( _Fvar[ndxF].dep().worst() > FFDep::L )
      _Gndx.insert( ndxF );
    else
      _dag->eval( 1, &_Fvar[ndxF], &CtrCst, _nX, _Xvar.data(), _X0.data() ); 
    switch( std::get<0>(_ctr)[i] ){
      case EQ: _Flow.push_back( -CtrCst );        _Fupp.push_back( -CtrCst );        break;
      case LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( -CtrCst );        break;
      case GE: _Flow.push_back( -CtrCst );        _Fupp.push_back(  BASE_OPT::INF ); break;
    }
    _Foff.push_back( CtrCst );
  }

  for( auto its=_sys.begin(); its!=_sys.end(); ++its ){ // Last, dependent equations
    ndxF++;
    _Fvar.push_back( *its );
    double CtrCst = 0.;
    if( _Fvar[ndxF].dep().worst() > FFDep::L )
      _Gndx.insert( ndxF );
    else
      _dag->eval( 1, &_Fvar[ndxF], &CtrCst, _nX, _Xvar.data(), _X0.data() ); 
    _Flow.push_back( -CtrCst );
    _Fupp.push_back( -CtrCst );
    _Foff.push_back(  CtrCst );
  }
  _nF = _Fvar.size();
#ifdef MC__NLPSLV_SNOPT_DEBUG
  assert( _nF == (int)ndxF+1 );
#endif

  // setup function derivatives
  _cleanup_grad();
  switch( options.GRADMETH ){
    default:
    case Options::FAD: _fct_grad = _dag->SFAD( _nF, _Fvar.data(), _nX, _Xvar.data() ); break;
    case Options::BAD: _fct_grad = _dag->SBAD( _nF, _Fvar.data(), _nX, _Xvar.data() ); break;
  }

  _iAfun.clear(); _jAvar.clear(); _Aval.clear(); 
  _iGfun.clear(); _jGvar.clear(); _Gvar.clear(); 
  for( unsigned k=0; k<std::get<0>(_fct_grad); ++k ){
    // derivative term in nonlinear constraint
    if( _Gndx.find( std::get<1>(_fct_grad)[k] ) != _Gndx.end() ){
      _iGfun.push_back( std::get<1>(_fct_grad)[k] );
      _jGvar.push_back( std::get<2>(_fct_grad)[k] );
      _Gvar.push_back( std::get<3>(_fct_grad)[k] );
#ifdef MC__NLPSLV_SNOPT_DEBUG
     std::cout << "  _Gvar[" << std::get<1>(_fct_grad)[k] << "," << std::get<2>(_fct_grad)[k]
               << "] = " << std::get<3>(_fct_grad)[k] << std::endl;
#endif
    }
    // derivative term in linear constraint
    else{
      _iAfun.push_back( std::get<1>(_fct_grad)[k] );
      _jAvar.push_back( std::get<2>(_fct_grad)[k] );
      assert( std::get<3>(_fct_grad)[k].cst() );
      _Aval.push_back( std::get<3>(_fct_grad)[k].num().val() );    
#ifdef MC__NLPSLV_SNOPT_DEBUG
     std::cout << "  _Aval[" << std::get<1>(_fct_grad)[k] << "," << std::get<2>(_fct_grad)[k]
               << "] = " << std::get<3>(_fct_grad)[k].num().val() << std::endl;
#endif
    }
  }
  _nG = _Gvar.size();
  _nA = _Aval.size();

  _recModel = false;
  return true;
}

inline
bool
NLPSLV_SNOPT::_record_model
()
{
  if( _recModel ) return false;

  _recObjDir = _ObjDir;
  _recObjRow = _ObjRow;
  _recObjAdd = _ObjAdd;
  _recObjMul = _ObjMul;
  _recFvar   = _Fvar;
  _recFlow   = _Flow;
  _recFupp   = _Fupp;
  _recFoff   = _Foff;
  _recnF     = _nF;
  _recGndx   = _Gndx;
  _reciGfun  = _iGfun;
  _recjGvar  = _jGvar;
  _recGvar   = _Gvar;
  _recnG     = _nG;
  _reciAfun  = _iAfun;
  _recjAvar  = _jAvar;
  _recAval   = _Aval; 
  _recnA     = _nA;

  _recModel = true;
  return true;
}

inline
bool
NLPSLV_SNOPT::restore_model
()
{
  if( !_recModel ) return false;

  _ObjDir = _recObjDir;
  _ObjRow = _recObjRow;
  _ObjAdd = _recObjAdd;
  _ObjMul = _recObjMul;
  _Fvar.swap( _recFvar );
  _Flow.swap( _recFlow );
  _Fupp.swap( _recFupp );
  _Foff.swap( _recFoff );
  _nF = _recnF;
  _Gndx.swap( _recGndx );
  _iGfun.swap( _reciGfun );
  _jGvar.swap( _recjGvar );
  _Gvar.swap( _recGvar );
  _nG = _recnG;
  _iAfun.swap( _reciAfun );
  _jAvar.swap( _recjAvar );
  _Aval.swap( _recAval ); 
  _nA = _recnA;

  _recModel = false;
  return true;
}

inline
bool
NLPSLV_SNOPT::set_obj_lazy
( t_OBJ const& type, FFVar const& obj )
{
  // Keep track of original model 
  if( _recModel && _Fvar[_ObjRow] == obj
   && (type == BASE_OPT::MIN? _ObjDir == -1: _ObjDir == 1) ) return false;
  _record_model();
  
  // Change to new objective
  _ObjDir = (type==BASE_OPT::MIN? -1: 1 );
  if( _ObjRow < 0 ){
    _ObjRow = 0;
    _Fvar.insert( _Fvar.begin(), obj );
    _Flow.insert( _Flow.begin(), -BASE_OPT::INF );
    _Fupp.insert( _Fupp.begin(),  BASE_OPT::INF );
    _Foff.insert( _Foff.begin(), 0. );
  }
  else{
    _Fvar[_ObjRow] = obj;
    // Previous objective function was nonlinear
    if( _Gndx.erase( _ObjRow ) ){
      auto it_iGfun = _iGfun.begin();
      auto it_jGvar = _jGvar.begin();
      auto it_Gvar  = _Gvar.begin();
      for( ; it_iGfun != _iGfun.end(); ){
        if( *it_iGfun != _ObjRow ){
          ++it_iGfun;
          ++it_jGvar;
          ++it_Gvar;
          continue;
        }
        it_iGfun = _iGfun.erase( it_iGfun );
        it_jGvar = _jGvar.erase( it_jGvar );
        it_Gvar  = _Gvar.erase( it_Gvar );
      }
    }
    // Previous objective function was linear
    else{
      auto it_iAfun = _iAfun.begin();
      auto it_jAvar = _jAvar.begin();
      auto it_Aval  = _Aval.begin();
      for( ; it_iAfun != _iAfun.end(); ){
        if( *it_iAfun != _ObjRow ){
          ++it_iAfun;
          ++it_jAvar;
          ++it_Aval;
          continue;
        }
        it_iAfun = _iAfun.erase( it_iAfun );
        it_jAvar = _jAvar.erase( it_jAvar );
        it_Aval  = _Aval.erase( it_Aval );
      }    
    }
  }
  _ObjAdd =  0.;
  _ObjMul = -1.; 
  if( _Fvar[_ObjRow].dep().worst() > FFDep::L )
    _Gndx.insert( _ObjRow );
  else
    _dag->eval( 1, &_Fvar[_ObjRow], &_ObjAdd, _nX, _Xvar.data(), _X0.data() ); 
  _Foff[_ObjRow] = _ObjAdd;
  _nF = _Fvar.size();
  
  // Update objective derivatives
  _cleanup_grad();
  switch( options.GRADMETH ){
    default:
    case Options::FAD: _fct_grad = _dag->SFAD( 1, &_Fvar[_ObjRow], _nX, _Xvar.data() ); break;
    case Options::BAD: _fct_grad = _dag->SBAD( 1, &_Fvar[_ObjRow], _nX, _Xvar.data() ); break;
  }
  for( unsigned k=0; k<std::get<0>(_fct_grad); ++k ){
    // derivative term in nonlinear constraint
    if( _Gndx.find( _ObjRow ) != _Gndx.end() ){
      _iGfun.push_back( _ObjRow );
      _jGvar.push_back( std::get<2>(_fct_grad)[k] );
      _Gvar.push_back( std::get<3>(_fct_grad)[k] );
#ifdef MC__NLPSLV_SNOPT_DEBUG
     std::cout << "  _Gvar[" << _ObjRow << "," << std::get<2>(_fct_grad)[k]
               << "] = " << std::get<3>(_fct_grad)[k] << std::endl;
#endif
    }
    // derivative term in linear constraint
    else{
      _iAfun.push_back( _ObjRow );
      _jAvar.push_back( std::get<2>(_fct_grad)[k] );
      assert( std::get<3>(_fct_grad)[k].cst() );
      _Aval.push_back( std::get<3>(_fct_grad)[k].num().val() );    
#ifdef MC__NLPSLV_SNOPT_DEBUG
     std::cout << "  _Aval[" << _ObjRow << "," << std::get<2>(_fct_grad)[k]
               << "] = " << std::get<3>(_fct_grad)[k].num().val() << std::endl;
#endif
    }
  }
  _nG = _Gvar.size();
  _nA = _Aval.size();

  return true;
}

inline
bool
NLPSLV_SNOPT::add_ctr_lazy
( t_CTR const type, FFVar const& ctr )
{
  // Keep track of original model 
  _record_model();
  
  // Append new constraint
  unsigned CtrPos = _Fvar.size();
  double CtrCst = 0.;
  _Fvar.push_back( ctr );
  if( _Fvar.back().dep().worst() > FFDep::L )
    _Gndx.insert( CtrPos );
  else
    _dag->eval( 1, &_Fvar.back(), &CtrCst, _nX, _Xvar.data(), _X0.data() ); 
  switch( type ){
    case EQ: _Flow.push_back( -CtrCst );        _Fupp.push_back( -CtrCst );        break;
    case LE: _Flow.push_back( -BASE_OPT::INF ); _Fupp.push_back( -CtrCst );        break;
    case GE: _Flow.push_back( -CtrCst );        _Fupp.push_back(  BASE_OPT::INF ); break;
  }
  _Foff.push_back( CtrCst );
  _nF = _Fvar.size();
  
  // Append new constraint derivatives
  _cleanup_grad();
  switch( options.GRADMETH ){
    default:
    case Options::FAD: _fct_grad = _dag->SFAD( 1, &_Fvar.back(), _nX, _Xvar.data() ); break;
    case Options::BAD: _fct_grad = _dag->SBAD( 1, &_Fvar.back(), _nX, _Xvar.data() ); break;
  }
  for( unsigned k=0; k<std::get<0>(_fct_grad); ++k ){
    // derivative term in nonlinear constraint
    if( _Gndx.find( CtrPos ) != _Gndx.end() ){
      _iGfun.push_back( CtrPos );
      _jGvar.push_back( std::get<2>(_fct_grad)[k] );
      _Gvar.push_back( std::get<3>(_fct_grad)[k] );
#ifdef MC__NLPSLV_SNOPT_DEBUG
     std::cout << "  _Gvar[" << CtrPos << "," << std::get<2>(_fct_grad)[k]
               << "] = " << std::get<3>(_fct_grad)[k] << std::endl;
#endif
    }
    // derivative term in linear constraint
    else{
      _iAfun.push_back( CtrPos );
      _jAvar.push_back( std::get<2>(_fct_grad)[k] );
      assert( std::get<3>(_fct_grad)[k].cst() );
      _Aval.push_back( std::get<3>(_fct_grad)[k].num().val() );    
#ifdef MC__NLPSLV_SNOPT_DEBUG
     std::cout << "  _Aval[" << CtrPos << "," << std::get<2>(_fct_grad)[k]
               << "] = " << std::get<3>(_fct_grad)[k].num().val() << std::endl;
#endif
    }
  }
  _nG = _Gvar.size();
  _nA = _Aval.size();

  return true;
}

inline
int
NLPSLV_SNOPT::_set_options
( WORKER_SNOPT * th )
{
  _iusr[0] = ( options.GRADMETH == Options::FD ? 1 : 0 );
  th->snOptA.setUserI( _iusr, 1 );
  th->snOptA.setPrintFile( options.LOGFILE.c_str(), 21 );

  int error = 0;
  //if( th->snOptA.setIntParameter ( "Print file ",                   0 )                                         ) error++;
  if( th->snOptA.setIntParameter ( "Summary file ",                 options.DISPLEVEL>0? 6: 0 )                   ) error++;
  if( th->snOptA.setIntParameter ( "Major Iterations limit ",       options.MAXITER>0? options.MAXITER:0 )        ) error++;
  if( th->snOptA.setIntParameter ( "Minor Iterations limit ",       options.QPMAXITER>0? options.QPMAXITER: 500)  ) error++;
  if( th->snOptA.setIntParameter ( "Verify level ",                 options.GRADCHECK? 3: -1 )                    ) error++;
  if( th->snOptA.setRealParameter( "Major feasibility tolerance ",  options.FEASTOL<0.?   0.: options.FEASTOL )   ) error++;
  if( th->snOptA.setRealParameter( "Major optimality tolerance ",   options.OPTIMTOL<0.?  0.: options.OPTIMTOL )  ) error++;
  if( th->snOptA.setRealParameter( "Minor feasibility tolerance ",  options.QPFEASTOL<0.? 0.: options.QPFEASTOL ) ) error++;
  if( th->snOptA.setRealParameter( "Infinite bound ",               BASE_OPT::INF )                               ) error++;
  if( th->snOptA.setIntParameter ( "Iterations limit ",             options.MAXITER*options.QPMAXITER>0?
                                                                    options.MAXITER*options.QPMAXITER: 10000 )    ) error++;
  switch( options.QPMETH ){
   case Options::CHOL: if( th->snOptA.setParameter( "QPSolver Cholesky" )         ) error++; break;
   case Options::CG:   if( th->snOptA.setParameter( "QPSolver CG" )               ) error++; break;
   case Options::QN:   if( th->snOptA.setParameter( "QPSolver QN" )               ) error++; break;
  }
 
  switch( options.GRADMETH ){
   case Options::FD:   if( th->snOptA.setIntParameter( "Derivative option", 0 )   ) error++; break;
   case Options::FAD:
   case Options::BAD:  if( th->snOptA.setIntParameter( "Derivative option", 1 )   ) error++; break;
  }
 
  if( options.FEASPB || !_ObjDir ){
    if( th->snOptA.setParameter( "Feasible point" ) ) error++;
  }
  else if( _ObjDir == -1 ){
    if( th->snOptA.setParameter( "Minimize" ) )       error++;
  }
  else{
    if( th->snOptA.setParameter( "Maximize" ) )       error++;  
  }

#ifdef MC__NLPSLV_SNOPT_CHECK
  assert( !error );
#endif
  return error;
}

inline
void
NLPSLV_SNOPT::_set_worker
( WORKER_SNOPT * th )
{
  th->Pvar.resize( _nP );
  th->Xvar.resize( _nX );
  th->Fvar.resize( _nF );
  th->Gvar.resize( _nG );
  th->DAG.insert( _dag, _nP, _Pvar.data(), th->Pvar.data() );
  th->DAG.insert( _dag, _nX, _Xvar.data(), th->Xvar.data() );
  th->DAG.insert( _dag, _nF, _Fvar.data(), th->Fvar.data() );
  th->DAG.insert( _dag, _nG, _Gvar.data(), th->Gvar.data() );
  th->op_F  = th->DAG.subgraph( _Gndx, th->Fvar.data() );
  th->op_G  = th->DAG.subgraph( _nG, th->Gvar.data() );
  th->iAfun = _iAfun;
  th->jAvar = _jAvar;
  th->Aval  = _Aval;
  th->iGfun = _iGfun;
  th->jGvar = _jGvar;
  th->Gndx  = _Gndx;
  th->Xlow  = _Xlow;
  th->Xupp  = _Xupp;
  th->Flow  = _Flow;
  th->Fupp  = _Fupp;
  th->Foff  = _Foff;
  th->tMAX  = userclock() + options.TIMELIMIT;
}

inline
void
NLPSLV_SNOPT::_resize_workers
( int const noth )
{
  while( (int)_worker.size() < noth )
    _worker.push_back( new WORKER_SNOPT );
}

template <typename T>
inline
int
NLPSLV_SNOPT::solve
( double const* Xini, T const* Xbnd, double const* Pval, bool const warm )
{
  std::vector<double> Xlow, Xupp;
  if( Xbnd ){
    Xlow.resize(_nX);
    Xupp.resize(_nX);
    for( int i=0; i<_nX; i++ ){
      Xlow[i] = Op<T>::l( Xbnd[i] );
      Xupp[i] = Op<T>::u( Xbnd[i] );
    }
  }
  return solve( Xini, Xlow.data(), Xupp.data(), Pval, warm );
}

inline
int
NLPSLV_SNOPT::solve
( double const* Xini, double const* Xlow, double const* Xupp, double const* Pval,
  bool const warm )
{
  // Set worker
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _set_worker( _worker[th] );
  _set_options( _worker[th] );
  _worker[th]->update( _nX, Xlow, Xupp, _nP, Pval );
  
  // Run NLP solver
  _iStart = ( warm? WARM: COLD );
  int stat;
  _worker[th]->registration();
  _worker[th]->initialize( _nF, _nX, Xini, false );
  _worker[th]->solve( stat, _iStart, _nF, _nX, _ObjAdd, _ObjRow, _nA, _nG );
  _worker[th]->finalize( stat, _ObjRow, _ObjMul );
  _solution = _worker[th]->solution;
  _worker[th]->unregistration();
 
  return stat;
}

#ifdef MC__USE_SOBOL
template <typename T>
inline
int
NLPSLV_SNOPT::solve
( unsigned const NSAM, T const* Xbnd, double const* Pval, bool const* logscal,
  bool const DISP )
{
  std::vector<double> Xlow(_nX), Xupp(_nX);
  for( int i=0; i<_nX; i++ ){
    Xlow[i] = Op<T>::l( Xbnd[i] );
    Xupp[i] = Op<T>::u( Xbnd[i] );
  }
  return solve( NSAM, Xlow.data(), Xupp.data(), Pval, logscal, DISP );
}

inline
int
NLPSLV_SNOPT::solve
( unsigned const NSAM, double const* Xlow, double const* Xupp, double const* Pval,
  bool const* logscal, bool const DISP )
{
  // Set workers
  const unsigned NOTHREADS = ( options.MAXTHREAD>0? options.MAXTHREAD: std::thread::hardware_concurrency() );
  _resize_workers( NOTHREADS );
  for( unsigned th=0; th<NOTHREADS; th++ ){
    _set_options( _worker[th] );
    _set_worker( _worker[th] );
    _worker[th]->update( _nX, Xlow, Xupp, _nP, Pval );
  }
  std::vector<std::thread> vth( NOTHREADS-1 ); // Main threads also solves some NLPs

  // Initialize multistart
  if( DISP ) std::cout << "\nMultistart: ";
  std::vector<int> feasible( NOTHREADS, false );
  std::vector<SOLUTION_OPT> solution( NOTHREADS );
  _iStart = COLD;
  _solution.reset();

  // Run NLP solver on auxiliary threads
  for( unsigned th=1; th<NOTHREADS; th++ ){
    //_worker[th]->snOptA.setIntParameter ( "Summary file ", 6 );
    vth[th-1] = std::thread( &NLPSLV_SNOPT::_mssolve, this, th, NOTHREADS, NSAM,
                             logscal, DISP, std::ref(feasible[th]), std::ref(solution[th]) );
  }

  // Run NLP solver on main thread
  //_worker[0]->snOptA.setIntParameter ( "Summary file ", 6 );
  _mssolve( 0, NOTHREADS, NSAM, logscal, DISP, feasible[0], solution[0] ); 

  bool found = false;
  if( feasible[0] ){
    _solution = solution[0];
    if( options.FEASPB || !_ObjDir )
      return _solution.stat;
    found = true;
  }

  // Join all the threads to the main one
  for( unsigned th=1; th<NOTHREADS; th++ ){
    vth[th-1].join();
    if( feasible[th] ){
      if( !found ){
        _solution = solution[th];
        found = true;
        continue;
      }
      if( options.FEASPB || !_ObjDir ){
        _solution = solution[th];
        return _solution.stat;
      }
      else if( _ObjDir == -1 && solution[th].f[_ObjRow] < _solution.f[_ObjRow] ){
        _solution = solution[th];
      }
      else if( _ObjDir == 1  && solution[th].f[_ObjRow] > _solution.f[_ObjRow] ){
        _solution = solution[th];
      }
    }
  }

  if( DISP ) std::cout << std::endl;

  return _solution.stat;
}

inline
void
NLPSLV_SNOPT::_mssolve
( int const th, unsigned const NOTHREADS, unsigned const NSAM, bool const* logscal,
  bool const DISP, int& feasible, SOLUTION_OPT& solution )
{
  // Initialize multistart and seed at position th
  std::vector<double> vSAM(_nX), Xini(_nX);
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( _nX );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( th );
  
  // Run multistart
#ifdef MC__NLPSLV_SNOPT_DEBUG
  std::cout << "Thread #" << th << std::endl;
#endif
  int stat;
  _worker[th]->registration();
  for( unsigned k=th; k<NSAM && userclock()<_worker[th]->tMAX; k+=NOTHREADS ){
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << "**** k = " << k << std::endl;
#endif

    // Sample variable domain
    for( int i=0; i<_nX; i++ ){
      vSAM[i] = gen();
      if( !logscal || !logscal[i] || _worker[th]->Xlow[i] <= 0. )
        Xini[i] = _worker[th]->Xlow[i] + ( _worker[th]->Xupp[i] - _worker[th]->Xlow[i] ) * vSAM[i];
      else
        Xini[i] = std::exp( std::log(_worker[th]->Xlow[i]) + ( std::log(_worker[th]->Xupp[i]) - std::log(_worker[th]->Xlow[i]) ) * vSAM[i] );
        //Xini[i] = std::exp( Op<T>::l(Op<T>::log(Xbnd[i])) + Op<T>::diam(Op<T>::log(Xbnd[i])) * vSAM[i] );
#ifdef MC__NLPSLV_SNOPT_DEBUG
      std::cout << "  " << Xini[i];
#endif
    }
#ifdef MC__NLPSLV_SNOPT_DEBUG
    std::cout << std::endl;
#endif

    // Set initial point and run NLP solver
    _worker[th]->initialize( _nF, _nX, Xini.data(), false );
    _worker[th]->solve( stat, _iStart, _nF, _nX, _ObjAdd, _ObjRow, _nA, _nG );
    _worker[th]->finalize( stat, _ObjRow, _ObjMul );    

    // Test for feasibility and improvement
    if( !_worker[th]->feasible( options.FEASTOL, _nF, _nX, _ObjRow, _nA ) ){
      if( DISP ) std::cout << "·";
    }
    // Solution point is feasible
    else{
      if( DISP ) std::cout << "*";
      if( options.FEASPB || !_ObjDir ){
        solution = _worker[th]->solution;
        break;
      }
      else if( !feasible ){
        solution = _worker[th]->solution;
        feasible = true;
      }
      else if( _ObjDir == -1 && _worker[th]->solution.f[_ObjRow] < solution.f[_ObjRow] ){
        solution = _worker[th]->solution;
      }
      else if( _ObjDir == 1  && _worker[th]->solution.f[_ObjRow] > solution.f[_ObjRow] ){
        solution = _worker[th]->solution;
      }
    }

    // Advance quasi-random counter
    gen.engine().discard( (NOTHREADS-1) * _nX );
  }

  _worker[th]->unregistration();
}
#endif

inline
bool
NLPSLV_SNOPT::is_feasible
( const double*x, const double CTRTOL )
{
  if( !x ) return false;

  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _set_worker( _worker[th] );
  _worker[th]->solution.x.assign( x, x+_nX );
  bool feas = _worker[th]->feasible( CTRTOL, _nF, _nX, _ObjRow, _nA );
  _solution = _worker[th]->solution;
  return feas;
}

inline
bool
NLPSLV_SNOPT::is_feasible
( const double CTRTOL )
{
  if( _solution.x.empty() ) return false;
  
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _worker[th]->solution = _solution;
  bool feas = _worker[th]->feasible( CTRTOL, _nF, _nX, _ObjRow, _nA );
  _solution = _worker[th]->solution;
  return feas;
}

inline
bool
NLPSLV_SNOPT::is_stationary
( const double*x, const double*ux, const double*uf, const double GRADTOL )
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _set_worker( _worker[th] );
  _worker[th]->solution.x.assign( x, x+_nX );
  _worker[th]->solution.ux.assign( ux, ux+_nX );
  _worker[th]->solution.uf.assign( uf, uf+_nF );
  return _worker[th]->stationary( GRADTOL, _nX, _nA, _nG );
}

inline
bool
NLPSLV_SNOPT::is_stationary
( const double GRADTOL )
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _worker[th]->solution = _solution;
  return _worker[th]->stationary( GRADTOL, _nX, _nA, _nG );
}

inline
double
NLPSLV_SNOPT::cost_correction
( const double*x, const double*ux, const double*uf )
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _set_worker( _worker[th] );
  _worker[th]->solution.x.assign( x, x+_nX );
  _worker[th]->solution.ux.assign( ux, ux+_nX );
  _worker[th]->solution.uf.assign( uf, uf+_nF );
  return _worker[th]->correction( _nF, _nX, _ObjRow, _nA );
}

inline
double
NLPSLV_SNOPT::cost_correction
()
{
  // Initialize main thread
  const int th = 0, noth = 1;
  _resize_workers( noth );
  _worker[th]->solution = _solution;
  return _worker[th]->correction( _nF, _nX, _ObjRow, _nA );
}

} // end namescape mc

#endif

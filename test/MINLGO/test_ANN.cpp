#define TEST4       // <-- select test function here
#define USE_MCISM      // <-- select relaxation approach here: USE_ISM / USE_MC / USE_MCISM

#undef MC__MINLPBND_DEBUG_DRL

#ifdef MC__USE_PROFIL
 #include "mcprofil.hpp"
 typedef INTERVAL I;
#else
 #ifdef MC__USE_BOOST
  #include "mcboost.hpp"
   typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
   typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
   typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
   typedef boost::numeric::interval<double,T_boost_policy> I;
 #else
  #ifdef MC__USE_FILIB
   #include "mcfilib.hpp"
   typedef filib::interval<double> I;
  #else
   #include "interval.hpp"
   typedef mc::Interval I;
  #endif
 #endif
#endif

#include "ismodel.hpp"
#include "mccormick.hpp"
#include "minlgo.hpp"

unsigned const NP = 2;
unsigned const ISMDIV = 64;
bool const     ISMCONT = true;//false;

namespace mc
{

#if defined( TEST0 )
template <typename T>
T myANN
( unsigned const nx, T const* x )
{
  assert( nx == 2 );
  return Op<T>::tanh( FFBase::sum( nVar, vVar ) );
}
double const xL = -10., xU = 10.;

#elif defined( TEST1 )
template <typename T>
T myANN
( unsigned const nx, T const* x )
{
  assert( nx == 2 );
  return Op<T>::tanh(0.2*x[0]+0.3*x[1]) - Op<T>::tanh(0.5*x[0]-0.2*x[1]) + Op<T>::tanh(-0.1*x[1]);
}
double const xL = -10., xU = 10.;

#elif defined( TEST2 )
template <typename T>
T myANN
( unsigned const nx, T const* x )
{
  assert( nx == 2 );

  T y[nx] = { 0.33393267662969284 * x[0] - 0.0004457779206294976,
              0.3345887696749378  * x[1] - 0.0002029955280284934 };
  return 0.7601908525553558 + 6.969979437085875 * ( - 0.060465315174905
         - 0.065704 * Op<T>::tanh(   9.10703116886844 * y[0] + 2.94337038212874 * y[1] - 9.63015944232877 )
         + 0.033719 * Op<T>::tanh( - 6.93102790414726 * y[0] + 3.75196310820969 * y[1] + 7.31017275430626 )
         - 0.76133  * Op<T>::tanh( - 3.36432203670221 * y[0] - 0.82525521613015 * y[1] + 2.98200536306973 )
         + 0.48805  * Op<T>::tanh( - 1.64194568809983 * y[0] - 4.48523835606202 * y[1] + 2.77237105961281 )
         - 0.10716  * Op<T>::tanh(  0.00226437988375905 * y[0] - 4.867765277275 * y[1] - 4.42497349082323 )
         + 0.12256  * Op<T>::tanh( - 7.19863262227284 * y[0] - 1.87895940079179 * y[1] + 6.25777205138423 )
         + 0.47773  * Op<T>::tanh(   0.74811043850941 * y[0] - 4.20191684807602 * y[1] - 2.62626886981211 )
         + 1.4593   * Op<T>::tanh( - 2.65321880831968 * y[0] - 2.19715149631687 * y[1] + 1.60241051102647 )
         + 0.12829  * Op<T>::tanh( - 1.95895703624561 * y[0] - 6.61501085095858 * y[1] - 3.21372438941966 )
         + 0.17066  * Op<T>::tanh( - 0.976462786697995 * y[0] - 5.4722226686442 * y[1] - 3.43511278570442)
         - 0.031784 * Op<T>::tanh( - 9.05631954225151 * y[0] + 0.860748840616473 * y[1] + 6.80778429540444 )
         + 1.0343   * Op<T>::tanh( - 2.51698650443046 * y[0] + 1.12325188704783 * y[1] + 1.46066391523409 )
         - 0.7515   * Op<T>::tanh( - 2.53263349216247 * y[0] + 2.83124829526224 * y[1] + 2.00914786562159 )
         + 0.17572  * Op<T>::tanh( - 1.64022926216511 * y[0] + 6.71323991116741 * y[1] + 2.30586986684063 )
         + 0.89455  * Op<T>::tanh( - 1.7873770187918 * y[0] + 3.68278580116741 * y[1] + 1.39928381929576 )
         - 1.0327   * Op<T>::tanh( - 4.22195373550896 * y[0] + 0.396477134417836 * y[1] + 0.964209592889774 )
         + 0.053874 * Op<T>::tanh( - 2.41239470358791 * y[0] + 8.80924154690345 * y[1] + 4.23134871494298 )
         + 0.32397  * Op<T>::tanh(   0.442923139035927 * y[0] + 5.28020902797231 * y[1] + 2.04241734851624 )
         + 1.8663   * Op<T>::tanh( - 3.51805314742391 * y[0] + 1.30278510763279 * y[1] + 0.0651561391008471 )
         + 0.5493   * Op<T>::tanh(   0.113212023605603 * y[0] - 4.72324347930784 * y[1] - 0.853205151783354 )
         - 0.60609  * Op<T>::tanh( - 3.95665396270945 * y[0] - 1.27626711582506 * y[1] + 0.814440074164862 )
         + 0.40068  * Op<T>::tanh( - 4.7695955444504 * y[0] - 0.773487861729487 * y[1] - 0.11124760272183 )
         - 0.92391  * Op<T>::tanh( - 2.05027781679787 * y[0] + 2.50975207869453 * y[1] + 0.0820486511860433 )
         - 0.93709  * Op<T>::tanh( - 3.66654493886071 * y[0] - 2.38892762517696 * y[1] - 0.891390460987188 )
         - 0.24686  * Op<T>::tanh( - 3.27998735486886 * y[0] + 2.59050296919374 * y[1] - 0.739390789012633 )
         - 1.7127   * Op<T>::tanh( - 1.6153208627896 * y[0] - 2.33417975474504 * y[1] + 0.871854475217263 )
         - 0.30011  * Op<T>::tanh(   1.11358422125162 * y[0] + 5.30599002390186 * y[1] - 1.00704834952813 )
         - 0.28025  * Op<T>::tanh(   2.92402686265607 * y[0] + 2.89613202071523 * y[1] - 0.232468557606477 )
         - 0.23534  * Op<T>::tanh(   1.81867167443697 * y[0] - 7.33812407332642 * y[1] + 2.54135224708381 )
         + 1.205    * Op<T>::tanh( - 3.10542197925959 * y[0] - 2.12619668948654 * y[1] - 0.68717003462119 )
         - 0.50945  * Op<T>::tanh( - 1.74472150970003 * y[0] - 4.95282077029394 * y[1] - 2.17551955892824 )
         - 0.20317  * Op<T>::tanh(   2.65152859120729 * y[0] - 4.38271460759243 * y[1] + 1.53042126339426 )
         + 0.26555  * Op<T>::tanh( - 4.1487314966423 * y[0] - 1.36838462771498 * y[1] - 2.02423652126007 )
         - 1.6129   * Op<T>::tanh( - 3.71026588633397 * y[0] + 0.240322230905286 * y[1] - 0.902116571066426 )
         - 0.28066  * Op<T>::tanh( - 0.136858569888465 * y[0] - 6.22839150715604 * y[1] + 2.22245776134561 )
         - 0.080875 * Op<T>::tanh(   2.29273626384379 * y[0] - 9.49277618264744 * y[1] + 4.61447478864626 )
         - 0.26354  * Op<T>::tanh(   3.29404103074185 * y[0] + 4.22870341832834 * y[1] + 2.65015883530254 )
         + 0.17364  * Op<T>::tanh( - 0.236622607550426 * y[0] - 6.32282202964636 * y[1] + 4.10022407859802 )
         - 0.58227  * Op<T>::tanh(   3.50739896377168 * y[0] - 1.52421925481854 * y[1] + 1.81299728715722 )
         - 0.15634  * Op<T>::tanh(   3.82419999460229 * y[0] - 3.12341880053042 * y[1] + 3.64656884309529 )
         + 0.20363  * Op<T>::tanh(   3.66913363409212 * y[0] - 0.319990400846123 * y[1] + 2.89624962044085 )
         + 0.041905 * Op<T>::tanh(   8.1823967031854 * y[0] + 4.01934818658146 * y[1] + 7.19040411962699 )
         - 0.43558  * Op<T>::tanh( - 2.45448375388046 * y[0] + 4.00420657536838 * y[1] - 2.8116643770919 )
         + 0.031562 * Op<T>::tanh( - 6.4400062979969 * y[0] - 8.07408273005427 * y[1] - 6.51730187495992 )
         - 1.0758   * Op<T>::tanh(   4.28496179289514 * y[0] - 8.05160576591809 * y[1] + 9.22764002248408 )
         - 0.96224  * Op<T>::tanh( - 4.52908595891183 * y[0] + 8.45807037184382 * y[1] - 9.71433325733342 )
         + 0.042862 * Op<T>::tanh( - 9.03003295779029 * y[0] - 5.87007164848379 * y[1] - 8.54369465648149 ) );
}
double const xL = -3., xU = 3.;

#elif defined( TEST3 )
template <typename T>
T ReLU
( T const& x )
{
  return Op<T>::max( x, T(0.) );
}

template <typename T>
ISVar<T> ReLU
( ISVar<T> const& x )
{
  return relu( x );
}

template <typename T>
fadbad::F<T> ReLU
( fadbad::F<T> const& x )
{
  fadbad::F<T> z = ReLU( x.val() );
  z.setDepend( x );
  for( unsigned j=0; j<z.size(); ++j )
    z[j] = Op<T>::fstep( x.val() ) * x[j];
  return z;
}

template <typename T>
T myANN
( unsigned const nx, T const* x )
{
  assert( nx == 2 );
  return ReLU(0.2*x[0]+0.3*x[1]) - ReLU(0.5*x[0]-0.2*x[1]-3.) + ReLU(0.2*x[0]-0.4*x[1]) + ReLU(-0.5*x[0]);
}
double const xL = -10., xU = 10.;


#elif defined( TEST4 )
template <typename T>
T ReLU
( T const& x )
{
  return Op<T>::max( x, T(0.) );
}

template <typename T>
ISVar<T> ReLU
( ISVar<T> const& x )
{
  return relu( x );
}

template <typename T>
fadbad::F<T> ReLU
( fadbad::F<T> const& x )
{
  fadbad::F<T> z = ReLU( x.val() );
  z.setDepend( x );
  for( unsigned j=0; j<z.size(); ++j )
    z[j] = Op<T>::fstep( x.val() ) * x[j];
  return z;
}

template <typename T>
T tanhapp
( T const& x )
{
  return ReLU( -ReLU( -x + 1. ) + 2. ) - 1.;
}

template <typename T>
T myANN
( unsigned const nx, T const* x )
{
  assert( nx == 2 );

  T y[nx] = { 0.33393267662969284 * x[0] - 0.0004457779206294976,
              0.3345887696749378  * x[1] - 0.0002029955280284934 };
  return 0.7601908525553558 + 6.969979437085875 * ( - 0.060465315174905
         - 0.065704 * tanhapp(   9.10703116886844 * y[0] + 2.94337038212874 * y[1] - 9.63015944232877 )
         + 0.033719 * tanhapp( - 6.93102790414726 * y[0] + 3.75196310820969 * y[1] + 7.31017275430626 )
         - 0.76133  * tanhapp( - 3.36432203670221 * y[0] - 0.82525521613015 * y[1] + 2.98200536306973 )
         + 0.48805  * tanhapp( - 1.64194568809983 * y[0] - 4.48523835606202 * y[1] + 2.77237105961281 )
         - 0.10716  * tanhapp(  0.00226437988375905 * y[0] - 4.867765277275 * y[1] - 4.42497349082323 )
         + 0.12256  * tanhapp( - 7.19863262227284 * y[0] - 1.87895940079179 * y[1] + 6.25777205138423 )
         + 0.47773  * tanhapp(   0.74811043850941 * y[0] - 4.20191684807602 * y[1] - 2.62626886981211 )
         + 1.4593   * tanhapp( - 2.65321880831968 * y[0] - 2.19715149631687 * y[1] + 1.60241051102647 )
         + 0.12829  * tanhapp( - 1.95895703624561 * y[0] - 6.61501085095858 * y[1] - 3.21372438941966 )
         + 0.17066  * tanhapp( - 0.976462786697995 * y[0] - 5.4722226686442 * y[1] - 3.43511278570442)
         - 0.031784 * tanhapp( - 9.05631954225151 * y[0] + 0.860748840616473 * y[1] + 6.80778429540444 )
         + 1.0343   * tanhapp( - 2.51698650443046 * y[0] + 1.12325188704783 * y[1] + 1.46066391523409 )
         - 0.7515   * tanhapp( - 2.53263349216247 * y[0] + 2.83124829526224 * y[1] + 2.00914786562159 )
         + 0.17572  * tanhapp( - 1.64022926216511 * y[0] + 6.71323991116741 * y[1] + 2.30586986684063 )
         + 0.89455  * tanhapp( - 1.7873770187918 * y[0] + 3.68278580116741 * y[1] + 1.39928381929576 )
         - 1.0327   * tanhapp( - 4.22195373550896 * y[0] + 0.396477134417836 * y[1] + 0.964209592889774 )
         + 0.053874 * tanhapp( - 2.41239470358791 * y[0] + 8.80924154690345 * y[1] + 4.23134871494298 )
         + 0.32397  * tanhapp(   0.442923139035927 * y[0] + 5.28020902797231 * y[1] + 2.04241734851624 )
         + 1.8663   * tanhapp( - 3.51805314742391 * y[0] + 1.30278510763279 * y[1] + 0.0651561391008471 )
         + 0.5493   * tanhapp(   0.113212023605603 * y[0] - 4.72324347930784 * y[1] - 0.853205151783354 )
         - 0.60609  * tanhapp( - 3.95665396270945 * y[0] - 1.27626711582506 * y[1] + 0.814440074164862 )
         + 0.40068  * tanhapp( - 4.7695955444504 * y[0] - 0.773487861729487 * y[1] - 0.11124760272183 )
         - 0.92391  * tanhapp( - 2.05027781679787 * y[0] + 2.50975207869453 * y[1] + 0.0820486511860433 )
         - 0.93709  * tanhapp( - 3.66654493886071 * y[0] - 2.38892762517696 * y[1] - 0.891390460987188 )
         - 0.24686  * tanhapp( - 3.27998735486886 * y[0] + 2.59050296919374 * y[1] - 0.739390789012633 )
         - 1.7127   * tanhapp( - 1.6153208627896 * y[0] - 2.33417975474504 * y[1] + 0.871854475217263 )
         - 0.30011  * tanhapp(   1.11358422125162 * y[0] + 5.30599002390186 * y[1] - 1.00704834952813 )
         - 0.28025  * tanhapp(   2.92402686265607 * y[0] + 2.89613202071523 * y[1] - 0.232468557606477 )
         - 0.23534  * tanhapp(   1.81867167443697 * y[0] - 7.33812407332642 * y[1] + 2.54135224708381 )
         + 1.205    * tanhapp( - 3.10542197925959 * y[0] - 2.12619668948654 * y[1] - 0.68717003462119 )
         - 0.50945  * tanhapp( - 1.74472150970003 * y[0] - 4.95282077029394 * y[1] - 2.17551955892824 )
         - 0.20317  * tanhapp(   2.65152859120729 * y[0] - 4.38271460759243 * y[1] + 1.53042126339426 )
         + 0.26555  * tanhapp( - 4.1487314966423 * y[0] - 1.36838462771498 * y[1] - 2.02423652126007 )
         - 1.6129   * tanhapp( - 3.71026588633397 * y[0] + 0.240322230905286 * y[1] - 0.902116571066426 )
         - 0.28066  * tanhapp( - 0.136858569888465 * y[0] - 6.22839150715604 * y[1] + 2.22245776134561 )
         - 0.080875 * tanhapp(   2.29273626384379 * y[0] - 9.49277618264744 * y[1] + 4.61447478864626 )
         - 0.26354  * tanhapp(   3.29404103074185 * y[0] + 4.22870341832834 * y[1] + 2.65015883530254 )
         + 0.17364  * tanhapp( - 0.236622607550426 * y[0] - 6.32282202964636 * y[1] + 4.10022407859802 )
         - 0.58227  * tanhapp(   3.50739896377168 * y[0] - 1.52421925481854 * y[1] + 1.81299728715722 )
         - 0.15634  * tanhapp(   3.82419999460229 * y[0] - 3.12341880053042 * y[1] + 3.64656884309529 )
         + 0.20363  * tanhapp(   3.66913363409212 * y[0] - 0.319990400846123 * y[1] + 2.89624962044085 )
         + 0.041905 * tanhapp(   8.1823967031854 * y[0] + 4.01934818658146 * y[1] + 7.19040411962699 )
         - 0.43558  * tanhapp( - 2.45448375388046 * y[0] + 4.00420657536838 * y[1] - 2.8116643770919 )
         + 0.031562 * tanhapp( - 6.4400062979969 * y[0] - 8.07408273005427 * y[1] - 6.51730187495992 )
         - 1.0758   * tanhapp(   4.28496179289514 * y[0] - 8.05160576591809 * y[1] + 9.22764002248408 )
         - 0.96224  * tanhapp( - 4.52908595891183 * y[0] + 8.45807037184382 * y[1] - 9.71433325733342 )
         + 0.042862 * tanhapp( - 9.03003295779029 * y[0] - 5.87007164848379 * y[1] - 8.54369465648149 ) );
}
double const xL = -3., xU = 3.;
#endif

class FFExt
: public FFOp
{
public:

#if defined( USE_ISM )
  static ISModel<I> ISMEnv;
  static std::vector<ISVar<I>> ISMVar;
  static std::vector<ISVar<I>> ISMRes;
  static std::vector<std::vector<PolVar<I>>> POLISMAux;
  static std::vector<double> DLISMAux;
  static std::vector<double> DUISMAux;

#elif defined( USE_MC )
  static std::vector<McCormick<I>> MCVar;
  static std::vector<McCormick<I>> MCRes;

#elif defined( USE_MCISM )
  static ISModel<I> ISMEnv;
  static std::vector<ISVar<I>> ISMVar;
  static std::vector<ISVar<I>> ISMRes;
  static std::vector<std::vector<PolVar<I>>> POLISMAux;
  static std::vector<double> DLISMAux;
  static std::vector<double> DUISMAux;
  static std::vector<McCormick<ISVar<I>>> MCISMVar;
  static std::vector<McCormick<ISVar<I>>> MCISMRes;
#endif

  // Constructors
  FFExt
    ()
    : FFOp( (int)EXTERN )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      auto dep = FFDep();
      for( unsigned i=0; i<nVar; ++i ) dep += pVar[i].dep();
      dep.update( FFDep::TYPE::N );
      return *(insert_external_operation( *this, 1, dep, nVar, pVar )[0]);
    }

  // Evaluation overloads
  template <typename T>
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFEXT_TRACE
      std::cout << "FFExt::eval generic instantiation\n"; 
      std::cout << typeid( vRes[0] ).name() << std::endl;
#endif
      assert( nVar == NP && nRes == 1 );
      vRes[0] = myANN( nVar, vVar );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFEXT_TRACE
      std::cout << "FFExt::eval FFVar instantiation\n"; 
#endif
      assert( nVar == NP && nRes == 1 );
      vRes[0] = operator()( nVar, vVar );
    }
    
  void eval
    ( unsigned const nRes, SLiftVar* vRes, unsigned const nVar, SLiftVar const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFEXT_TRACE
      std::cout << "FFExt::eval SLiftVar instantiation\n"; 
#endif
      assert( nVar == NP && nRes == 1 );
      vVar->env()->lift( nRes, vRes, nVar, vVar );
    }

  void eval
    ( unsigned const nRes, PolVar<I>* vRes, unsigned const nVar, PolVar<I> const* vVar,
      unsigned const* mVar )
    const
    {
#ifdef MC__FFEXT_TRACE
      std::cout << "FFExt::eval PolVar<I> instantiation\n"; 
#endif
      assert( nVar == NP && nRes == 1 );
      PolBase<I>* img = vVar[0].image();
      FFBase* dag = vVar[0].var().dag();
      assert( img && dag );
      FFVar* pRes = dag->curOp()->varout[0];

#if defined( USE_ISM )
      // evaluate interval superposition
      for( unsigned i=0; i<nVar; ++i )
        ISMVar[i].set( &ISMEnv, i, vVar[i].range() );
      ISMRes[0] = myANN( nVar, ISMVar.data() );
      //std::cout << "MCRes[0] in " << ISMRes[0];
      vRes[0].set( img, *pRes, ISMRes[0].B() );
      //std::cout << "vRes[0] in " << vRes[0].range() << std::endl;

#elif defined( USE_MC )
      //for( unsigned i=0; i<nVar; ++i )
      //  IVar[i] = vVar[i].range();
      //IRes[0] = myANN( nVar, IVar.data() );
      //vRes[0].set( img, *pRes, IRes[0] );
      // compute McCormick relaxation at mid-point with subgradient in each direction
      for( unsigned i=0; i<nVar; ++i )
        MCVar[i] = McCormick<I>( vVar[i].range(), Op<I>::mid( vVar[i].range() ) ).sub( nVar, i );
      MCRes[0] = myANN( nVar, MCVar.data() );
      //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
      vRes[0].set( img, *pRes, MCRes[0].I() );
      //std::cout << "vRes[0] in " << vRes[0].range() << std::endl;

#elif defined( USE_MCISM )
      // compute McCormick relaxation with ISM bounds at mid-point with subgradient in each direction
      for( unsigned i=0; i<nVar; ++i )
        MCISMVar[i] = McCormick<ISVar<I>>( ISVar<I>( &ISMEnv, i, vVar[i].range() ), Op<I>::mid( vVar[i].range() ) ).sub( nVar, i );
      MCISMRes[0] = myANN( nVar, MCISMVar.data() );
      //std::cout << "MCISMRes[0] in " << MCISMRes[0] << std::endl;
      vRes[0].set( img, *pRes, MCISMRes[0].I().B() );
      //std::cout << "vRes[0] in " << vRes[0].range() << std::endl;
#endif
    }

  template< typename T >
  bool reval
    ( unsigned const nRes, T const* vRes, unsigned const nVar, T* vVar )
    const
    {
      throw std::runtime_error("Error: FFExt::eval no generic implementation\n");
    }

  bool reval
    ( unsigned const nRes, PolVar<I> const* vRes, unsigned const nVar, PolVar<I>* vVar )
    const
    {
#ifdef MC__FFEXT_TRACE
      std::cout << "FFExt::reval PolVar<T> instantiation\n"; 
#endif
      assert( nVar == NP && nRes == 1 );
      PolBase<I>* img = vVar[0].image();
      FFOp* pop = vVar[0].var().opdef().first;
      assert( img && pop );

#if defined( USE_ISM )
      assert( ISMEnv.ndiv() == ISMDIV );
      // define auxiliary variables 
      for( unsigned i=0; i<nVar; ++i ){
        POLISMAux[i].resize( ISMEnv.ndiv() );
        for( unsigned k=0; k<ISMEnv.ndiv(); ++k )
          POLISMAux[i][k].set( img, Op<I>::zeroone(), ISMCONT );
      }

      // polyhedral cut generation
      //std::cout << ISMRes[0];
      auto cutF1 = *img->add_cut( pop, PolCut<I>::LE, 0., vRes[0], -1. );
      auto cutF2 = *img->add_cut( pop, PolCut<I>::GE, 0., vRes[0], -1. );
      for( unsigned i=0; i<nVar; ++i ){
        auto&& ISMi = ISMRes[0].C()[i];
        if( ISMi.empty() ) continue;
        for( unsigned k=0; k<ISMEnv.ndiv(); ++k ){
          DLISMAux[k] = Op<I>::l( ISMi[k] );
          DUISMAux[k] = Op<I>::u( ISMi[k] );
        }
        cutF1->append( ISMEnv.ndiv(), POLISMAux[i].data(), DLISMAux.data() );
        cutF2->append( ISMEnv.ndiv(), POLISMAux[i].data(), DUISMAux.data() );
      }

      // add polyhedral cuts for ISM-participating variables
      for( unsigned i=0; i<nVar; i++ ){
        if( POLISMAux[i].empty() ) continue;
        // auxiliaries add up to 1
        for( unsigned jsub=0; jsub<ISMEnv.ndiv(); jsub++ )
          DLISMAux[jsub] = 1.;
        img->add_cut( pop, PolCut<I>::EQ, 1., ISMEnv.ndiv(), POLISMAux[i].data(), DLISMAux.data() );

        // link auxiliaries to model variables
        PolVar<I> POLvarL( 0. ), POLvarU( 0. );
        auto&& ISMi = ISMVar[i].C()[i];
        assert( !ISMi.empty() );
        for( unsigned k=0; k<ISMEnv.ndiv(); k++ ){
          DLISMAux[k] = Op<I>::l(ISMi[k]);
          DUISMAux[k] = Op<I>::u(ISMi[k]);
        }
        img->add_cut( pop, PolCut<I>::LE, 0., ISMEnv.ndiv(), POLISMAux[i].data(), DLISMAux.data(), vVar[i], -1. );
        img->add_cut( pop, PolCut<I>::GE, 0., ISMEnv.ndiv(), POLISMAux[i].data(), DUISMAux.data(), vVar[i], -1. );
      }

#elif defined( USE_MC )
      // polyhedral cut generation
      //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
      double rhs1 = -MCRes[0].cv(),
             rhs2 = -MCRes[0].cc();
      for( unsigned i=0; i<nVar; ++i ){
        rhs1 += MCRes[0].cvsub(i)*MCVar[i].cv();
        rhs2 += MCRes[0].ccsub(i)*MCVar[i].cc();
      }
      img->add_cut( pop, PolCut<I>::LE, rhs1, nVar, vVar, MCRes[0].cvsub(), vRes[0], -1. );
      img->add_cut( pop, PolCut<I>::GE, rhs2, nVar, vVar, MCRes[0].ccsub(), vRes[0], -1. );

#elif defined( USE_MCISM )
      assert( ISMEnv.ndiv() == ISMDIV );
      // define ISM auxiliary variables 
      for( unsigned i=0; i<nVar; ++i ){
        POLISMAux[i].resize( ISMEnv.ndiv() );
        for( unsigned k=0; k<ISMEnv.ndiv(); ++k )
          POLISMAux[i][k].set( img, Op<I>::zeroone(), ISMCONT );
      }

      // polyhedral cut generation for ISM
      //std::cout << MCISMRes[0].I();
      auto cutF1 = *img->add_cut( pop, PolCut<I>::LE, 0., vRes[0], -1. );
      auto cutF2 = *img->add_cut( pop, PolCut<I>::GE, 0., vRes[0], -1. );
      for( unsigned i=0; i<nVar; ++i ){
        auto&& ISMi = MCISMRes[0].I().C()[i];
        if( ISMi.empty() ) continue;
        for( unsigned k=0; k<ISMEnv.ndiv(); ++k ){
          DLISMAux[k] = Op<I>::l( ISMi[k] );
          DUISMAux[k] = Op<I>::u( ISMi[k] );
        }
        cutF1->append( ISMEnv.ndiv(), POLISMAux[i].data(), DLISMAux.data() );
        cutF2->append( ISMEnv.ndiv(), POLISMAux[i].data(), DUISMAux.data() );
      }

      // add polyhedral cuts for ISM-participating variables
      for( unsigned i=0; i<nVar; i++ ){
        if( POLISMAux[i].empty() ) continue;
        // auxiliaries add up to 1
        for( unsigned jsub=0; jsub<ISMEnv.ndiv(); jsub++ )
          DLISMAux[jsub] = 1.;
        img->add_cut( pop, PolCut<I>::EQ, 1., ISMEnv.ndiv(), POLISMAux[i].data(), DLISMAux.data() );

        // link ISM auxiliaries to model variables
        PolVar<I> POLvarL( 0. ), POLvarU( 0. );
        auto&& ISMi = MCISMVar[i].I().C()[i];
        assert( !ISMi.empty() );
        for( unsigned k=0; k<ISMEnv.ndiv(); k++ ){
          DLISMAux[k] = Op<I>::l(ISMi[k]);
          DUISMAux[k] = Op<I>::u(ISMi[k]);
        }
        img->add_cut( pop, PolCut<I>::LE, 0., ISMEnv.ndiv(), POLISMAux[i].data(), DLISMAux.data(), vVar[i], -1. );
        img->add_cut( pop, PolCut<I>::GE, 0., ISMEnv.ndiv(), POLISMAux[i].data(), DUISMAux.data(), vVar[i], -1. );
      }

      // polyhedral cut generation for MC
      //std::cout << "MCRes[0] in " << MCRes[0] << std::endl;
      double rhs1 = -MCISMRes[0].cv(),
             rhs2 = -MCISMRes[0].cc();
      for( unsigned i=0; i<nVar; ++i ){
        rhs1 += MCISMRes[0].cvsub(i) * MCISMVar[i].cv();
        rhs2 += MCISMRes[0].ccsub(i) * MCISMVar[i].cc();
      }
      img->add_cut( pop, PolCut<I>::LE, rhs1, nVar, vVar, MCISMRes[0].cvsub(), vRes[0], -1. );
      img->add_cut( pop, PolCut<I>::GE, rhs2, nVar, vVar, MCISMRes[0].ccsub(), vRes[0], -1. );
#endif
      return true;
    }

  // Properties
  std::string name
    ()
    const
    { return "EXT"; }
};

#if defined( USE_ISM )
  inline ISModel<I> FFExt::ISMEnv = ISModel<I>( NP, ISMDIV );
  inline std::vector<ISVar<I>> FFExt::ISMVar = std::vector<ISVar<I>>( NP );
  inline std::vector<ISVar<I>> FFExt::ISMRes = std::vector<ISVar<I>>( 1 );
  inline std::vector<std::vector<PolVar<I>>> FFExt::POLISMAux = std::vector<std::vector<PolVar<I>>>( NP );
  inline std::vector<double> FFExt::DLISMAux = std::vector<double>( ISMDIV );
  inline std::vector<double> FFExt::DUISMAux = std::vector<double>( ISMDIV );

#elif defined( USE_MC )
  inline std::vector<McCormick<I>> FFExt::MCVar = std::vector<McCormick<I>>( NP );
  inline std::vector<McCormick<I>> FFExt::MCRes = std::vector<McCormick<I>>( 1 );

#elif defined( USE_MCISM )
  inline ISModel<I> FFExt::ISMEnv = ISModel<I>( NP, ISMDIV );
  inline std::vector<std::vector<PolVar<I>>> FFExt::POLISMAux = std::vector<std::vector<PolVar<I>>>( NP );
  inline std::vector<double> FFExt::DLISMAux = std::vector<double>( ISMDIV );
  inline std::vector<double> FFExt::DUISMAux = std::vector<double>( ISMDIV );
  inline std::vector<McCormick<ISVar<I>>> FFExt::MCISMVar = std::vector<McCormick<ISVar<I>>>( NP );
  inline std::vector<McCormick<ISVar<I>>> FFExt::MCISMRes = std::vector<McCormick<ISVar<I>>>( 1 );
#endif

} // end namespace mc


#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIP;
#elif  MC__USE_IPOPT
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIP;
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
 typedef mc::NLPSLV_SNOPT<mc::FFExt> NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT<mc::FFExt> NLP;
#endif


int main()
{
  mc::FFGraph< mc::FFExt > DAG;
  mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );
  mc::FFExt ANN;

  mc::MINLGO<I,NLP,MIP,mc::FFExt> MINLP;
  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, mc::xL, mc::xU, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_OPT::MIN, ANN( NP, P ) ); // objective

  //MINLP.options.GAMSEXPORT                  = "test_ANN.gms";
  //MINLP.options.PRESOLVE                    = 0;
  MINLP.options.STRATEGY                    = mc::MINLGO<I,NLP,MIP,mc::FFExt>::Options::SBB;
  MINLP.options.DISPLEVEL                   = 1;
  MINLP.options.CVATOL                      = 1e-4;
  MINLP.options.CVRTOL                      = 1e-4;
  MINLP.options.MAXITER                     = 0;
  MINLP.options.MINLPBND.OBBTMAX            = 5;
  MINLP.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  //MINLP.options.MINLPBND.MIPSLV.OUTPUTFILE  = "test_ANN.lp";
  MINLP.setup();
  MINLP.presolve();
  //MINLP.GAMSexport();
  MINLP.optimize();
  MINLP.stats.display();

  return 0;
}

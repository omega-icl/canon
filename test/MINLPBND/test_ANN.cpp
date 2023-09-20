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

#include "minlpbnd.hpp"

unsigned const NP = 2;
unsigned const ISMDIV = 8;
bool const     ISMCONT = false;//true;

namespace mc
{

//template <typename T>
//T myANN
//( unsigned const nVar, T const* vVar )
//{ return Op<T>::tanh( FFBase::sum( nVar, vVar ) ); }

template <typename T>
T myANN
( unsigned const nx, T const* x )
{
  assert( nx == 2 );
  return Op<T>::tanh(0.2*x[0]+0.3*x[1]) - Op<T>::tanh(0.5*x[0]-0.2*x[1]) + Op<T>::tanh(-0.1*x[1]);
}

class FFExt
: public FFOp
{
public:

  static ISModel<I> ISMEnv;
  static std::vector<ISVar<I>> ISMVar;
  static std::vector<ISVar<I>> ISMRes;
  static std::vector<std::vector<PolVar<I>>> POLISMAux;
  static std::vector<double> DLISMAux;
  static std::vector<double> DUISMAux;

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
      assert( nVar == NP && nRes == 1 );
      std::cout << "FFExt::eval generic instantiation\n"; 
      vRes[0] = myANN( nVar, vVar );
      std::cout << typeid( vRes[0] ).name() << std::endl;
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      //throw std::runtime_error("Error: No implementation for FFExt::eval with mc::FFVar\n");
      std::cout << "FFExt::eval FFVar instantiation\n"; 
      assert( nVar == NP && nRes == 1 );
      vRes[0] = operator()( nVar, vVar );
    }

  void eval
    ( unsigned const nRes, PolVar<I>* vRes, unsigned const nVar, PolVar<I> const* vVar,
      unsigned const* mVar )
    const
    {
      assert( nVar == NP && nRes == 1 );
      std::cout << "FFExt::eval PolVar<I> instantiation\n"; 
      PolBase<I>* img = vVar[0].image();
      FFBase* dag = vVar[0].var().dag();
      assert( img && dag );
      FFVar* pRes = dag->curOp()->varout[0];

      //static std::vector<I> IVar( nVar );
      //for( unsigned i=0; i<nVar; ++i )
      //  IVar[i] = vVar[i].range();
      //I IRes = myANN( nVar, IVar.data() );
      //vRes[0].set( img, *pRes, IRes );

      // evaluate interval superposition
      for( unsigned i=0; i<nVar; ++i )
        ISMVar[i].set( &ISMEnv, i, vVar[i].range() );
      ISMRes[0] = myANN( nVar, ISMVar.data() );
      std::cout << ISMRes[0];
      vRes[0].set( img, *pRes, ISMRes[0].B() );
      std::cout << "vRes[0] in " << vRes[0].range() << std::endl;
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
      assert( nVar == NP && nRes == 1 && ISMEnv.ndiv() == ISMDIV );
      std::cout << "FFExt::reval PolVar<T> instantiation\n"; 
      PolBase<I>* img = vVar[0].image();
      FFBase* dag = vVar[0].var().dag();
      FFOp* pop = vVar[0].var().opdef().first;
      assert( img && dag && pop );

      // define auxiliary variables 
      for( unsigned i=0; i<nVar; ++i ){
        POLISMAux[i].resize( ISMEnv.ndiv() );
        for( unsigned k=0; k<ISMEnv.ndiv(); ++k )
          POLISMAux[i][k].set( img, Op<I>::zeroone(), ISMCONT );
      }

      // polyhedral cut generation
      std::cout << ISMRes[0];
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
        // relate auxiliaries to model variables
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

      return true;
    }

  // Properties
  std::string name
    ()
    const
    { return "EXT"; }
};

inline ISModel<I> FFExt::ISMEnv = ISModel<I>( NP, ISMDIV );
inline std::vector<ISVar<I>> FFExt::ISMVar = std::vector<ISVar<I>>( NP );
inline std::vector<ISVar<I>> FFExt::ISMRes = std::vector<ISVar<I>>( 1 );
inline std::vector<std::vector<PolVar<I>>> FFExt::POLISMAux = std::vector<std::vector<PolVar<I>>>( NP );
inline std::vector<double> FFExt::DLISMAux = std::vector<double>( ISMDIV );
inline std::vector<double> FFExt::DUISMAux = std::vector<double>( ISMDIV );

} // end namespace mc

int main()
{
  mc::FFGraph< mc::FFExt > DAG;
  mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );
  mc::FFExt ANN;

  mc::MINLPBND<I,mc::MIPSLV_GUROBI<I>,mc::FFExt> MINLP;
  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, -10, 10, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_OPT::MIN, ANN( NP, P ) ); // objective

  // Solving for a MIP relaxation using polyhedral relaxations
  MINLP.options.RELAXMETH           = { MINLP.options.DRL };
  MINLP.options.POLIMG.AGGREG_LQ    = 1;
  MINLP.options.MIPSLV.DISPLEVEL    = 1;
  MINLP.options.MIPSLV.OUTPUTFILE   = "test_ANN.lp";

  MINLP.setup();
/*
  unsigned nred;
  MINLP.reduce_bounds( nred );
  std::cout << std::endl
            <<"MINLP reduced bounds:" << std::endl;
  for( unsigned i=0; i<NP; i++ ) 
    std::cout << "  " << P[i] << " = " << MINLP.variable_bounds()[i] << std::endl;
*/
  switch( MINLP.relax_model() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl
                <<"MINLP relaxation bound: " << MINLP.relax_solver()->get_objective() << std::endl;
      for( unsigned i=0; i<NP; i++ ) 
        std::cout << "  " << P[i] << " = " << MINLP.relax_solver()->get_variable( P[i] ) << std::endl;
      MINLP.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }

  return 0;
}

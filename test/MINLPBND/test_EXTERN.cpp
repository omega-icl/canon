#define TEST_SHEKEL	// <-- select test function here
////////////////////////////////////////////////////////////////////////

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

#include "ffdagext.hpp"
#include "minlpbnd.hpp"

////////////////////////////////////////////////////////////////////////

#if defined( TEST_EXP )
unsigned const NX = 4;
const double XL[4] = { -2, -1, -2, -1 };	// <-- X range lower bound
const double XU[4] = {  1,  2,  1,  2 };	// <-- X range upper bound
const double XREF[4] = { -0.5, 0.5, -0.5, 0.5 };		// <-- X reference point

mc::FFVar extFunc
( mc::FFGraph* DAG, std::vector<mc::FFVar> const& x )
{
  mc::FFVar w = x[0],
            z = x[0]*(exp(x[0])-exp(-x[0]));
  for( unsigned i=1; i<NX; ++i ){
    w *= x[i];
    if( i%2 ) z -= x[i]*(exp(x[i])-exp(-x[i]));
    else      z += x[i]*(exp(x[i])-exp(-x[i]));
  }
  return w*z;
}

#elif defined( TEST_SHEKEL )
unsigned const NX = 2;
unsigned const M  = 10;

const double B[10]    = {0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5};
const double C[6][10] = {{4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0},
                         {4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6},
                         {4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0},
                         {4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6},
                         {4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0},
                         {4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6}};

const double XL[6] = {  0,  0,  0,  0,  0,  0 };	// <-- X range lower bound
const double XU[6] = { 10, 10, 10, 10, 10, 10 };	// <-- X range upper bound
const double XREF[6] = { 4, 4, 4, 4, 4, 4 };		// <-- X reference point

mc::FFVar extFunc
( mc::FFGraph* DAG, std::vector<mc::FFVar> const& x )
{
  mc::FFVar z = 0.;
  for( unsigned i=0; i<M; ++i ){
    mc::FFVar w = 0.;
    for( unsigned j=0; j<NX; ++j )
      w += pow(x[j]-C[j][i],2);
    z += 1./(w+B[i]);
  }
  return z;
}
#endif

////////////////////////////////////////////////////////////////////////

int main()
{
  // Create DAG
  mc::FFGraph DAG;
  std::vector<mc::FFVar> X = DAG.add_vars( NX );
  std::vector<mc::FFVar> F{ extFunc( &DAG, X ) };
  mc::DAGEXT<I> DAGF( &DAG, X, F );

  mc::FFDAGEXT<I> OpF;
  OpF.options.RELAX  = { OpF.options.PWLS };//AUX };//PWCS };//MC };//INT };
  OpF.options.PWCDIV = 32;
  OpF.options.PWCREL = 0;
  OpF.options.PWCSUP.USE_SHADOW = 1;
  OpF.options.PWCSHADOW = 1;
  OpF.options.PWLINI = 8;
  OpF.options.PWLMAX = 0;
  OpF.options.PWLREL = 0;
  OpF.options.PWLSUP.MAX_SUBDIV = 32;
  OpF.options.PWLSUP.USE_SHADOW = 1;
  OpF.options.PWLSHADOW = 1;

  auto SgF = DAG.subgraph( { OpF( 0, X, &DAGF ) } );
  DAG.output( SgF, " OF F" );
  auto StrF = mc::FFExpr::subgraph( &DAG, SgF );
  std::cout << "F: " << StrF[0] << std::endl;

  mc::MINLPBND<I> Model;
  Model.set_dag( &DAG );
  Model.add_var( NX, X.data(), XL, XU );
  Model.set_obj( mc::BASE_OPT::MAX, OpF( 0, X, &DAGF ) );

  // Solving for a MIP relaxation using polyhedral relaxations
  Model.options.RELAXMETH           = { Model.options.DRL };
  Model.options.LINCTRSEP           = 1;
  Model.options.POLIMG.AGGREG_LQ    = 1;
  Model.options.MIPSLV.DISPLEVEL    = 1;
  Model.options.MIPSLV.OUTPUTFILE   = "test_EXTERN.lp";

  Model.setup();
  //unsigned nred;
  //Model.reduce_bounds( nred );
  switch( Model.relax_model() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl
                <<"MINLP relaxation bound: " << Model.relax_solver()->get_objective() << std::endl;
      for( unsigned i=0; i<NX; i++ ) 
        std::cout << "  " << X[i] << " = " << Model.relax_solver()->get_variable( X[i] ) << std::endl;
      Model.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }

  return 0;
}

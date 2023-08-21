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

#undef MC__SELIM_DEBUG_PROCESS
#undef MC__SQUAD_DEBUG_MIP
#include "minlpref.hpp"

int main()
{
  mc::MINLPREF<I> MINLP;

/*
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );
  MINLP.add_var( P[0], 1, 20, 0 );
  MINLP.add_var( P[1], 1, 20, 1 );
  MINLP.set_obj( mc::BASE_OPT::MIN, -6*P[0]-P[1] );
  MINLP.add_ctr( mc::BASE_OPT::LE, 0.3*pow(P[0]-8,2)+0.04*pow(P[1]-6,4)+0.1*exp(2*P[0])/pow(P[1],4)-56 );
  MINLP.add_ctr( mc::BASE_OPT::LE, 1/P[0]+1/P[1]-sqrt(P[0])*sqrt(P[1])+4 );
  MINLP.add_ctr( mc::BASE_OPT::LE, 2*P[0]-5*P[1]+1 );

  mc::FFGraph DAG;
  const unsigned NP = 4; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, 1, 5, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_OPT::MIN, (P[0]*P[3])*(P[0]+P[1]+P[2])+P[2] ); // objective
  MINLP.add_ctr( mc::BASE_OPT::GE,  P[0]*P[1]*P[2]*P[3]-25 );          // constraints
  MINLP.add_ctr( mc::BASE_OPT::EQ,  sqr(P[0])+sqr(P[1])+sqr(P[2])+sqr(P[3])-40 );
  //MINLP.add_ctr( mc::BASE_OPT::EQ,  sqr(P[0])-P[0]/P[3] );
*/
  //std::string gamsfile( "tuncphd_29.gms");
  //std::string gamsfile( "tuncphd_30.gms");
  //std::string gamsfile( "transswitch0009r.gms" );
  std::string gamsfile( "ex6_1_4.gms" );
  if( !MINLP.read( gamsfile ) ){//, true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return -1;
  }

  // Export original model
  MINLP.options.NCOCUTS             = 0;
  MINLP.setup();
 
  MINLP.propagate_bounds();
  MINLP.export_model( "test_MINLPREF_original.gms" );
/*
  // Export original model with NCO cuts
  MINLP.options.NCOCUTS             = 1;
  MINLP.setup();
 
  MINLP.propagate_bounds();
  MINLP.export_model( "test_MINLPREF_NCOcuts.gms" );
*/
  // Formulate reduced-space model and export to GAMS
  MINLP.options.NCOCUTS             = 0;
  MINLP.options.INVBNDGS            = 1;
  MINLP.options.INVKEEPLIN          = 1;
  MINLP.options.SELIM.MIPDISPLEVEL  = 1;
  MINLP.options.SELIM.MIPTIMELIMIT  = 10;
  MINLP.options.SELIM.ELIMLIN       = 1;
  MINLP.options.SELIM.ELIMMLIN      = 1;
  MINLP.options.SELIM.ELIMNLIN      = {};//{mc::FFInv::Options::IPOW,mc::FFInv::Options::LOG,mc::FFInv::Options::INV};
  MINLP.options.AEBND.DISPLEVEL     = 0;
  MINLP.setup();

  MINLP.propagate_bounds();
  MINLP.eliminate_invertible_constraints( true );
  MINLP.export_model( "test_MINLPREF_elim.gms" );
/*
  MINLP.options.INVKEEPLIN          = 0;
  MINLP.setup();

  MINLP.propagate_bounds();
  MINLP.eliminate_invertible_constraints( true );
  MINLP.export_model( "test_MINLPREF_elimfull.gms" );
*/
  // Formulate full-space model after lifting of non-polynomial terms and quadratization of polynomials
  MINLP.options.NCOCUTS             = 0;
  MINLP.options.QUADOPTIM           = 0;
  MINLP.options.SQUAD.MIPDISPLEVEL  = 1;
  MINLP.options.REDELIM             = 0;
  MINLP.options.SRED.MIPDISPLEVEL   = 1;
  MINLP.options.SRED.ORDER          = 1;
  MINLP.options.SRED.NODIV          = 0;
  MINLP.setup();

  MINLP.propagate_bounds();
  MINLP.lift_polynomial_subexpressions( true );
  MINLP.append_reduction_constraints( true );
  //MINLP.flatten_linear_functions( true );
  //MINLP.flatten_quadratic_functions( true );
  //MINLP.flatten_polynomial_functions( true );
  //MINLP.quadratize_polynomial_functions( true );
  //MINLP.append_reduction_constraints( true );
  MINLP.propagate_bounds();
  MINLP.export_model( "test_MINLPREF_lift.gms" );
/*
  MINLP.quadratize_polynomial_functions( true );
  MINLP.propagate_bounds();
  MINLP.export_model( "test_MINLPREF_liftquad.gms" );
*/
  return 0;
}

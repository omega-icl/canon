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
   
int main()
{
  mc::MINLPBND<I> MINLP;

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
*/

  mc::FFGraph DAG;
  const unsigned NP = 4; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, 1, 5, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_OPT::MIN, (P[0]*P[3])*(P[0]+P[1]+P[2])+P[2] ); // objective
  MINLP.add_ctr( mc::BASE_OPT::GE,  (P[0]*P[3])*P[1]*P[2]-25 );          // constraints
  MINLP.add_ctr( mc::BASE_OPT::EQ,  sqr(P[0])+sqr(P[1])+sqr(P[2])+sqr(P[3])-40 );

/*
  std::string gamsfile( "tuncphd_30.gms"); 
  if( !MINLP.read( gamsfile, true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return -1;
  }
*/

  // Solving for a MIP relaxation using ISM arithmetic
  MINLP.options.RELAXMETH           = { MINLP.options.ISM };
  MINLP.options.ISMDIV              = 50;
  MINLP.options.ISMCONT             = 0;
  MINLP.options.MIPSLV.DISPLEVEL    = 1;
  MINLP.options.MIPSLV.OUTPUTFILE   = "test_MINLPBND1.lp";

  MINLP.setup();
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

  // Solving for a MIP relaxation using polyhedral relaxations
  MINLP.options.RELAXMETH           = { MINLP.options.DRL };
  MINLP.options.LINCTRSEP           = 1;
  MINLP.options.POLIMG.AGGREG_LQ    = 1;
  MINLP.options.MIPSLV.DISPLEVEL    = 1;
  MINLP.options.MIPSLV.OUTPUTFILE   = "test_MINLPBND2.lp";

  MINLP.setup();
  //unsigned nred;
  //MINLP.reduce_bounds( nred );
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
/*
  // Solving for a MIP relaxation using Chebyshev models and polyhedral relaxations
  MINLP.options.RELAXMETH           = { MINLP.options.SCDRL };
  MINLP.options.LINCTRSEP           = 1;
  MINLP.options.CMODPROP            = 4;
  MINLP.options.MIPSLV.OUTPUTFILE   = "test_MINLPBND3.lp";

  MINLP.setup();
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
*/
  // Solving for a nonconvex MIQCP relaxation in Gurobi after lifting of non-polynomial terms and quadratization of polynomials
  // in the objective and constraints
  MINLP.options.RELAXMETH           = { MINLP.options.DRLQ };
  MINLP.options.LINCTRSEP           = 1;
  MINLP.options.POLIMG.ALLOW_QUAD   = 1;
  MINLP.options.POLIMG.ALLOW_NLIN   = {};
  MINLP.options.POLIMG.ALLOW_DISJ   = {};
  MINLP.options.MIPSLV.PWLRELGAP    = 1e-6;
  MINLP.options.MIPSLV.FUNCMAXVAL   = 1e12;
  MINLP.options.MIPSLV.OUTPUTFILE   = "test_MINLPBND4.lp";
  MINLP.options.QUADOPTIM           = 1;
  MINLP.options.SQUAD.MIPFIXEDBASIS = 0;

  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.lift_polynomial_subexpressions( true );
  MINLP.quadratize_polynomial_functions( false );
  MINLP.propagate_bounds();
  MINLP.export_model( "test_MINLPBND4.gms" );

  switch( MINLP.relax_model() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl << std::scientific << std::setprecision(5)
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

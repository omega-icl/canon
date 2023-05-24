#undef  CHECK_REFORMULATION

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

#include "minlpref.hpp"
#include "nlpslv_snopt.hpp"

int main()
{
  mc::MINLPREF<mc::FFGraph<>,I> MINLP;

  // Read original GAMS model from file
  //std::string gamsfile( "CSI_Wnet.gms"); 
  std::string gamsfile( "CSI_LCOE.gms"); 
  //std::string gamsfile( "CSII_Wnet.gms"); 
  //std::string gamsfile( "CSII_LCOE.gms"); 
  //std::string gamsfile( "CSIII_Wnet.gms"); 
  //std::string gamsfile( "CSIII_LCOE.gms"); 
  if( !MINLP.read( gamsfile ) ){//, true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return -1;
  }

  // Rewriting GAMS model to file - no transformation
  MINLP.options.CPMAX  = 100;
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.export_model( "CS_DB_original.gms" );

#ifdef CHECK_REFORMULATION
  mc::NLPSLV_SNOPT<mc::FFGraph<>> NLP;
  NLP.options.DISPLEVEL = 0;
  NLP.options.MAXITER   = 100;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_SNOPT<mc::FFGraph<>>::Options::FAD;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 0;
  NLP.read( "CS_DB_original.gms" );//, true );
  NLP.setup();
  NLP.solve( 100 );//p0 ); //, Ip );
  //std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  //std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  //std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  unsigned int NP = MINLP.variables().size();
  double Dp[NP];
  std::cout << "Variables: " << NP << std::endl;
  for( unsigned i=0; i<NP; ++i ){
    Dp[i] = NLP.solution().x[i];
    std::cout << MINLP.variables()[i] << " = " << Dp[i] << std::endl;
  }
  
  unsigned const NF0 = MINLP.functions().size();
  double Df0[NF0];
  std::cout << "Functions:" << NF0 << std::endl;
  MINLP.dag()->eval( NF0, MINLP.functions().data(), Df0, NP, MINLP.variables().data(), Dp );
  for( unsigned i=0; i<NF0; ++i ) std::cout << MINLP.functions()[i] << " = " << Df0[i] << std::endl;
#endif

  // Eliminating variables using invertible equality constraints
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.options.AEBND.DISPLEVEL     = 1;
  MINLP.options.SELIM.MIPDISPLEVEL  = 0;
  MINLP.options.SELIM.ELIMMLIN      = 1;
  MINLP.options.SELIM.ELIMNLIN      = {};
  //MINLP.options.SELIM.MULTMAX       = 3;
  MINLP.eliminate_invertible_constraints( true, true );  
  MINLP.export_model( "CS_DB_reduced.gms" );

  // Lifting non-polynomial terms and quadratizing polynomials
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.lift_polynomial_subexpressions( true );
  MINLP.flatten_linear_functions( true );
  //MINLP.flatten_quadratic_functions( true );
  //MINLP.flatten_polynomial_functions( true );
  MINLP.quadratize_polynomial_functions( true );
  MINLP.export_model( "CS_DB_liftedquad.gms" );

#ifdef CHECK_REFORMULATION
  unsigned const NL = MINLP.lifted_variables().size();
  double Dl[NL];
  mc::FFVar Fl[NL];
  unsigned i = 0;
  for( auto& [j,Fj] : MINLP.lifted_variables() ) Fl[i++] = Fj;
  std::cout << "Lifted Variables: " << NL << std::endl;
  MINLP.dag()->eval( NL, Fl, Dl, NP, MINLP.variables().data(), Dp );
  for( unsigned i=0; i<NL; ++i ) std::cout << Fl[i] << " = " << Dl[i] << std::endl;

  unsigned const NF = MINLP.functions().size();
  double Df[NF+1];
  std::cout << "Functions:" << NF << std::endl;
  MINLP.dag()->eval( NF, MINLP.functions().data(), Df, NP, MINLP.variables().data(), Dp, NL, MINLP.variables().data()+NP, Dl );
  for( unsigned i=0; i<NF; ++i ) std::cout << MINLP.functions()[i] << " = " << Df[i] << std::endl;
#endif

  return 0;
}

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
  mc::MINLPREF<I> MINLP;

  // Read original GAMS model from file
  std::string gamsfile( "ANN_FS.gms"); 
  //std::string gamsfile( "CSI_Wnet.gms"); 
  //std::string gamsfile( "CSI_LCOE.gms"); 
  //std::string gamsfile( "CSII_Wnet.gms"); 
  //std::string gamsfile( "CSII_LCOE.gms"); 
  //std::string gamsfile( "CSIII_Wnet.gms"); 
  //std::string gamsfile( "CSIII_LCOE.gms"); 
  if( !MINLP.read( gamsfile, true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return -1;
  }

  // Rewriting GAMS model to file - no transformation
  MINLP.options.CPMAX  = 100;
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.export_model( "CS_DB_original.gms" );

#ifdef CHECK_REFORMULATION
  mc::NLPSLV_SNOPT NLP;
  NLP.options.DISPLEVEL = 0;
  NLP.options.MAXITER   = 100;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_SNOPT<>::Options::FAD;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 0;
  NLP.read( "CS_DB_original.gms" );//, true );
  NLP.setup();
  NLP.solve( 100 );//p0 ); //, Ip );
  //std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  //std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  //std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  unsigned int NP = MINLP.variables().size();
//  double Dp[NP];
//  std::cout << "Variables: " << NP << std::endl;
//  for( unsigned i=0; i<NP; ++i ){
//    Dp[i] = NLP.solution().x[i];
//    std::cout << MINLP.variables()[i] << " = " << Dp[i] << std::endl;
//  }

  double Dp[NP] = { 49.1902, 349.2087, 349.4950, 530.8828, 652.5192, 349.2087, 0.2000, 48.0725, 48.0725, 48.0725, 48.0725, 0.2000, 0.2000, 0.8500, 0.8042, 147.8951, 153.8791, 912.0801, 2952.2664, 3184.4701, 2192.7309, 2082.5376, 5.5081, 5.5081, 349.2087, 540.8828, 540.8828, 540.8828, 540.8828, 349.2087, 349.2087, 147.8951, 147.8951, 147.8951, 2952.2664, 2952.2664, 2553.5843, 2553.5843, 0.4465, 6.7405, 29.8292, 865.3678, 561.0825, 90400.0000, 5.9841, 178.4995, 991.7393, 29582.7542, 29404.2547, 60995.7453, 22616.5022, 60857.0545, 6926.4433, 51.0587, 98.5050, 20.1997, 324.4850, 56.0587, 30.1997, 324.4850, 247.4808, 53.5197, 57.6336, 267.6451, 284.2445, 3256.2491, 6540.3160, 3789.6610, 812.2635, 189046.8313, 352760.9232, 215251.7429, 67059.6321, 1.0000, 1.2042, 1.2042, 1.2042, 1381951.2418, 2966777.4438, 1810302.5977, 563982.5472, 140496.5090, 9149502.3205, 1.6013013E+7, 99080.2547, 0.5433, 25.7673, 19.4229 }; 

  unsigned const NF0 = MINLP.functions().size();
  double Df0[NF0];
  std::cout << "Functions:" << NF0 << std::endl;
  MINLP.dag()->eval( NF0, MINLP.functions().data(), Df0, NP, MINLP.variables().data(), Dp );
  for( unsigned i=0; i<NF0; ++i ) std::cout << MINLP.functions()[i] << " = " << Df0[i] << std::endl;
#endif

  // Eliminating variables using invertible equality constraints
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.options.INVBNDGS            = 1;
  MINLP.options.INVKEEPLIN          = 1;
  MINLP.options.AEBND.DISPLEVEL     = 0;
  MINLP.options.SELIM.MIPDISPLEVEL  = 0;
  MINLP.options.SELIM.ELIMMLIN      = 0;
  MINLP.options.SELIM.ELIMNLIN      = {};
  //MINLP.options.SELIM.ELIMNLIN      = {mc::FFInv::Options::INV,mc::FFInv::Options::SQRT,mc::FFInv::Options::EXP,
  //                                     mc::FFInv::Options::LOG,mc::FFInv::Options::RPOW};
  //MINLP.options.SELIM.MULTMAX       = 3;
  MINLP.eliminate_invertible_constraints( true );  
  MINLP.export_model( "CS_DB_reduced.gms" );

#ifdef CHECK_REFORMULATION
  double Dp0[NP] = { 49.1902, 349.2087, 349.4950, 530.8828, 652.5192, 349.2087, 0.2000, 48.0725, 48.0725, 48.0725, 48.0725, 0.2000, 0.2000, 0.8500, 0.8042, 147.8951, 153.8791, 912.0801, 2952.2664, 3184.4701, 2192.7309, 2082.5376, 5.5081, 5.5081, 349.2087, 540.8828, 540.8828, 540.8828, 540.8828, 349.2087, 349.2087, 147.8951, 147.8951, 147.8951, 2952.2664, 2952.2664, 2553.5843, 2553.5843, 0.4465, 6.7405, 29.8292, 865.3678, 561.0825, 90400.0000, 5.9841, 178.4995, 991.7393, 29582.7542, 29404.2547, 60995.7453, 22616.5022, 60857.0545, 6926.4433, 51.0587, 98.5050, 20.1997, 324.4850, 56.0587, 30.1997, 324.4850, 247.4808, 53.5197, 57.6336, 267.6451, 284.2445, 3256.2491, 6540.3160, 3789.6610, 812.2635, 189046.8313, 352760.9232, 215251.7429, 67059.6321, 1.0000, 1.2042, 1.2042, 1.2042, 1381951.2418, 2966777.4438, 1810302.5977, 563982.5472, 140496.5090, 9149502.3205, 1.6013013E+7, 99080.2547, 0.5433, 25.7673, 19.4229 }; 

  std::cout << "Functions:" << NF0 << std::endl;
  MINLP.dag()->eval( NF0, MINLP.functions().data(), Df0, NP, MINLP.variables().data(), Dp0 );
  for( unsigned i=0; i<NF0; ++i ) std::cout << MINLP.functions()[i] << " = " << Df0[i] << std::endl;

  mc::MINLPREF<mc::FFGraph<>,I> MINLP2;
  std::string gamsfile2( "CS_DB_reduced.gms"); 
  !MINLP2.read( gamsfile2, true );
  MINLP2.setup();
  unsigned const NF00 = MINLP2.functions().size();
  double Df00[NF00];
  std::cout << "Functions:" << NF00 << std::endl;
  MINLP2.dag()->eval( NF00, MINLP2.functions().data(), Df00, NP, MINLP2.variables().data(), Dp );
  for( unsigned i=0; i<NF00; ++i ) std::cout << MINLP2.functions()[i] << " = " << Df00[i] << std::endl;
#endif

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

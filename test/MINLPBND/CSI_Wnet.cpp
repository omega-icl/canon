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
  mc::MINLPBND<mc::FFGraph<>,I> MINLP;

  // Read original GAMS model from file
  std::string gamsfile( "CSI_Wnet.gms"); 
  if( !MINLP.read( gamsfile, true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return -1;
  }


  // Lifting non-polynomial terms and quadratizing polynomials
  MINLP.options.REFORMMETH         = {};
  MINLP.setup();
  MINLP.write( "CSI_Wnet_same.gms" );


  // Lifting non-polynomial terms and quadratizing polynomials
  MINLP.options.REFORMMETH         = { MINLP.options.NPOL, MINLP.options.QUAD };
  MINLP.setup();
  MINLP.write( "CSI_Wnet_lift.gms" );


  // Eliminating variables using invertible equality constraints
  MINLP.options.REFORMMETH          = { MINLP.options.ELIM };
  MINLP.options.SELIM.MIPDISPLEVEL  = 1;
  MINLP.options.SELIM.MIPOUTPUTFILE = "CSI_Wnet_elim.lp";
  MINLP.options.SELIM.ELIMNLIN      = {};
  MINLP.options.SELIM.ELIMMLIN      = 1;
  MINLP.setup();
  MINLP.write( "CSI_Wnet_elim.gms" );

  return 0;
}

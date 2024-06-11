#undef  MC__MINLPREF_DEBUG_LIFT
#undef  MC__SELIM_DEBUG_PROCESS
#undef  MC__SLIFT_DEBUG_PROCESS

#include <fstream>
#include <iomanip>

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

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIP;
#elif  MC__USE_IPOPT
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIP;
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
 typedef mc::NLPSLV_SNOPT<> NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT<> NLP;
#endif

#include "minlpslv.hpp"
#include "minlpbnd.hpp"

int main()
{
  mc::MINLPBND<I> MINLP;

  // Read original GAMS model from file
  std::string gamsfile( "UNIFAC_relu_model_20.gms"); 
  if( !MINLP.read( gamsfile, true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return -1;
  }

  // Rewriting GAMS model to file - no transformation
  MINLP.options.CPMAX  = 0;
  MINLP.setup();
  //MINLP.propagate_bounds();
  //unsigned nred;
  //MINLP.reduce_bounds( nred );

  MINLP.export_model( "UNIFAC_relu_model_20_original.gms" );

  // Eliminating variables using invertible equality constraints
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.options.INVBNDGS            = 1;
  MINLP.options.INVKEEPLIN          = 1;
  MINLP.options.AEBND.DISPLEVEL     = 0;
  MINLP.options.SELIM.MIPDISPLEVEL  = 1;
  MINLP.options.SELIM.ELIMMLIN      = 0;
  MINLP.options.SELIM.ELIMNLIN      = {};
  //MINLP.options.SELIM.ELIMNLIN      = {mc::FFInv::Options::INV,mc::FFInv::Options::SQRT,mc::FFInv::Options::EXP,
  //                                     mc::FFInv::Options::LOG,mc::FFInv::Options::RPOW};
  //MINLP.options.SELIM.MULTMAX       = 3;
  try{
    MINLP.eliminate_invertible_constraints( false );//true );  
  }
  catch( mc::SElimEnv<>::Exceptions &eObj ){
    std::cerr << "Error " << eObj.ierr()
              << " in variable elimination manipulation:" << std::endl
              << eObj.what() << std::endl
              << "Aborts." << std::endl;
    return eObj.ierr();
  }
  MINLP.export_model( "UNIFAC_relu_model_20_elim.gms" );
/*
  // Lifting non-polynomial terms and quadratizing polynomials
  MINLP.setup();
  MINLP.options.SLIFT.KEEPFACT = 1;
  MINLP.options.SLIFT.LIFTDIV  = 0;
  MINLP.options.SLIFT.LIFTIPOW = 0;
  MINLP.propagate_bounds();
  MINLP.lift_polynomial_subexpressions( true );
  MINLP.flatten_linear_functions( true );
  //MINLP.flatten_quadratic_functions( true );
  //MINLP.flatten_polynomial_functions( true );
  //MINLP.quadratize_polynomial_functions( true );
  MINLP.export_model( "UNIFAC_relu_model_20_lift.gms" );
  //MINLP.export_model( "UNIFAC_relu_model_20_quadlift.gms" );
*/
  return 0;
}

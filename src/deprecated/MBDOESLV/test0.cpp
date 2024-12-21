#define CANON__MBDOE_SETUP_DEBUG
#define CANON__MBDOE_SHOW_APPORTION
#define MC__FFDOECRIT_CHECK
#define MC__FFGRADDOECRIT_CHECK
#define MC__FFDOEEFF_CHECK
#define MC__FFGRADDOEEFF_CHECK
#define MC__FFBREFF_CHECK
#define MC__FFGRADBREFF_CHECK

#include "mbdoeslv.hpp"

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define model

  mc::FFGraph DAG;  // DAG describing the model

  const unsigned NX = 1;       // Number of experimental controls
  std::vector<mc::FFVar> X(NX);  // Controls
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );

  const unsigned NP = 2;       // Number of estimated parameters
  std::vector<mc::FFVar> P(NP);  // Parameters
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );

  const unsigned NY = 1;       // Number of measured outputs
  std::vector<mc::FFVar> Y(NY);  // Outputs
  Y[0] = P[0] * exp( P[1] * X[0] );

  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  // Sampled parameters - uniform Sobol' sampling
  unsigned const NSAM = 21;
  std::vector<double> PLB( { 1e0, -1e1 } );
  std::vector<double> PUB( { 1e1,  0e0 } );
  //std::vector<double> PSCA( { 1e0, 1e0 } );
  std::vector<double> PSCA( { std::fabs( PUB[0]-PLB[0]), std::fabs( PUB[1]-PLB[1]) } );

  // Experimental control space
  std::vector<double> XLB( { 0e0  } );
  std::vector<double> XUB( { 5e-1 } );

  // Output variance
  std::vector<double> YVAR( { 1e-1 } );

  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::MBDOESLV::BROPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV::Options::AVERSE;//NEUTRAL;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 1;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-8;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 1;
  //DOE.options.NLPSLV.GRADMETH = DOE.options.NLPSLV.FSYM;//FAD;

  DOE.set_dag( DAG );
  DOE.set_model( Y );//, YVAR );
  DOE.set_controls( X, XLB, XUB );
  DOE.set_parameters( P, DOE.uniform_sample( NSAM, PLB, PUB ), PSCA );

  std::list<std::pair<double,std::vector<double>>> prior_campaign
  {
    { 1, { 1e-1 } }
  };
  DOE.add_prior_campaign( prior_campaign );


  DOE.setup();
  //DOE.sample_supports( 50 );
  //DOE.combined_solve( 5, false ); // continuous design
  //auto CNTEFF = DOE.efforts();
  //DOE.combined_solve( 5, true, CNTEFF ); // exact design
  DOE.combined_solve( 5 );
  //DOE.effort_solve( 5, false );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 5, DOE.efforts() );
  //DOE.file_export( "test0" );
  auto campaign = DOE.campaign();
/*
  std::list<std::pair<double,std::vector<double>>> campaign // ** EFFORT-BASED EXACT DESIGN: 2.58167e+01
  {
    //SUPPORT #19: 2 x [ 2.34375e-01 ]
    { 2, { 2.34375e-01 } },
    //SUPPORT #30: 3 x [ 1.56250e-02 ]
    { 3, { 1.56250e-02 } }
  };
*/
  DOE.options.CRITERION = mc::MBDOESLV::DOPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-NEUTRAL" );

  DOE.options.CRITERION = mc::MBDOESLV::DOPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::AVERSE;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-AVERSE" );

  DOE.options.CRITERION = mc::MBDOESLV::BROPT;
  DOE.setup();
  DOE.evaluate_design( campaign, "BROPT" );
  
  return 0;
}

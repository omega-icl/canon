#define SAVE_RESULTS		// <- Whether to save bounds to file
#define MC__MBDOE_SHOW_APPORTION
//#define MC__MBDOE_SETUP_DEBUG
//#define MC__MBDOE_SAMPLE_DEBUG

#include "mbdoeslv.hpp"

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define IVP-ODE

  mc::FFGraph DAG;  // DAG describing the IVP-ODE

  const unsigned NS = 5;  // Time stages
  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( unsigned k=0; k<NS; k++ ) tk[k+1] = tk[k] + 2e0; // [hour]

  const unsigned NP = 4;  // Number of estimated parameters
  const unsigned NC = 2;  // Number of experimental controls
  const unsigned NX = 2;  // Number of states
  const unsigned NY = 2;  // Number of outputs

  std::vector<mc::FFVar> C( NC );  // Controls
  for( unsigned int i=0; i<NC; i++ ) C[i].set( &DAG );
  C[0].set("u1");
  C[1].set("u2");
  mc::FFVar& u1 = C[0];
  mc::FFVar& u2 = C[1];

  std::vector<mc::FFVar> P( NP );  // Parameters
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );
  P[0].set("p1");
  P[1].set("p2");
  P[2].set("p3");
  P[3].set("p4");
  mc::FFVar& p1 = P[0]; // max growth rate
  mc::FFVar& p2 = P[1]; // half-saturation constant
  mc::FFVar& p3 = P[2]; // proudct yield
  mc::FFVar& p4 = P[3]; // respiration rate

  std::vector<mc::FFVar> X( NX );  // States & state sensitivities
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );
  X[0].set("y1");
  X[1].set("y2");
  mc::FFVar& y1  = X[0]; // biomass
  mc::FFVar& y2  = X[1]; // substrate

  std::vector<mc::FFVar> RHS( NX*NS );  // Right-hand side function
  for( unsigned i=0; i<NS; i++ ){
    mc::FFVar r = p1 * y2 / ( p2 + y2 );
    RHS[NX*i]   = ( r - u1 - p4 ) * y1;
    RHS[NX*i+1] = -r * y1 / p3 + u1 * ( u2 - y2 );
  }
  
  std::vector<mc::FFVar> IC( NX );   // Initial value function
  IC[0] = 7e0;
  IC[1] = 1e-1;

  std::vector<std::vector<mc::FFVar>> FCT( NS, std::vector<mc::FFVar>( NY*NS, 0. ) );  // State functions
  for( unsigned i=0; i<NS; i++ ){
    FCT[i][NY*i]   = y1;
    FCT[i][NY*i+1] = y2;
  }

  mc::ODESLVS_CVODES IVP;
  IVP.options.INTMETH   = mc::BASE_CVODES::Options::MSBDF;//MSADAMS;//
  IVP.options.NLINSOL   = mc::BASE_CVODES::Options::NEWTON;//FIXEDPOINT;//
  IVP.options.LINSOL    = mc::BASE_CVODES::Options::DIAG;//DENSE;//
  IVP.options.FSACORR   = mc::BASE_CVODES::Options::STAGGERED;//STAGGERED1;//SIMULTANEOUS;
  IVP.options.NMAX      = 2000;
  IVP.options.DISPLAY   = 0;
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-10;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-8;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
  IVP.options.ASACHKPT  = 2000;
#if defined( SAVE_RESULTS )
  IVP.options.RESRECORD = 100;
#endif

  IVP.set_dag( &DAG );
  IVP.set_time( tk );
  IVP.set_state( X );
  IVP.set_parameter( C );
  IVP.set_differential( RHS );
  IVP.set_initial( IC );
  IVP.set_function( FCT );
  IVP.setup();
/*
  /////////////////////////////////////////////////////////////////////////
  // Simulate IVP-ODE

  // Nominal model parameters
  p1.set( 0.1 );
  p2.set( 0.1 );
  p3.set( 0.1 );
  p4.set( 0.1 );
  IVP.setup();

  double dC[NC] = { 5.000000e-02, 8.915310e+00 };
  IVP.states( dC );
#if defined( SAVE_RESULTS )
  std::ofstream direcSTA;
  direcSTA.open( "test2_STA.dat", std::ios_base::out );
  IVP.record( direcSTA );
#endif

  return 0;
*/
  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  // Sampled parameters - uniform Sobol' sampling
  unsigned const NSAM = 1;
  std::vector<double> PLB( NP ), PUB( NP );
  PLB[0] = PUB[0] = 1e-1;
  PLB[1] = PUB[1] = 1e-1;
  PLB[2] = PUB[2] = 1e-1;
  PLB[3] = PUB[3] = 1e-1;

  // Experimental control space
  std::vector<double> CLB( NC ), CUB( NC );
  CLB[0] = 5e-2;   CUB[0] = 2e-1;
  CLB[1] = 5e0;    CUB[1] = 35e0;

  // Output variance
  std::vector<double> YVAR( NY*NS, 4e-2 );

  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::DOEBase::DOPT;
  DOE.options.RISK      = mc::MBDOESLV<>::Options::NEUTRAL;//AVERSE;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 0;
  DOE.set_model( IVP, YVAR );
  DOE.set_controls( C, CLB, CUB );
  DOE.set_parameters( P, DOE.uniform_sample( NSAM, PLB, PUB ) );
  DOE.setup();
  DOE.sample_supports( 100 );
  DOE.combined_solve( 4 );
  //DOE.effort_solve( 4 );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 4, DOE.efforts() );
  //DOE.file_export( "test2" );

  return 0;
}

#undef SAVE_RESULTS		// <- Whether to save bounds to file
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

  const unsigned NS = 8;  // Time stages
  double tk[NS+1]; tk[0] = 0.;
  for( unsigned k=0; k<NS; k++ ) tk[k+1] = tk[k] + 2.5e1; // [min]

  const unsigned NK = 4;       // Number of estimated parameters
  const unsigned NC = NS/2+1;//NS+1;  // Number of experimental controls
  const unsigned NX = 3;       // Number of states
  const unsigned NF = 2;       // Number of outputs

  mc::FFVar C[NC];  // Controls
  for( unsigned int i=0; i<NC; i++ ) C[i].set( &DAG );
  mc::FFVar& T     = C[NS/2];//C[NS];

  mc::FFVar K[NK];  // Parameters
  for( unsigned int i=0; i<NK; i++ ) K[i].set( &DAG );
  mc::FFVar& nu    = K[0];
  mc::FFVar& alpha = K[1];
  mc::FFVar& K0    = K[2];
  mc::FFVar& K1    = K[3];
  double CAin = 10;     // [mol/L]
  double CA0  = 5;      // [mol/L]
  double Tref = 273.15; // [K]
  double eps  = 1e-5;      // [mol/L]

  mc::FFVar X[NX];  // States & state sensitivities
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );
  mc::FFVar& CA  = X[0];
  mc::FFVar& CB  = X[1];
  mc::FFVar& V   = X[2];

  mc::FFVar RHS[NX*NS];  // Right-hand side function
  for( unsigned i=0; i<NS; i++ ){
    mc::FFVar& Qin = C[i/2];//C[i]; // [L/min]
    mc::FFVar R = exp( K0 + K1 * ( 1 - T / Tref ) ) * ( pow( CA + eps, alpha ) - pow( eps, alpha ) );
    RHS[NX*i]   = Qin / V * ( CAin - CA ) - R;
    RHS[NX*i+1] = - Qin / V * CB + nu * R;
    RHS[NX*i+2] = Qin;
    //DAG.output( DAG.subgraph( NX, RHS+NX*i ) );
  }
  
  mc::FFVar IC[NX];   // Initial value function
  IC[0] = CA0;
  IC[1] = 0e0;
  IC[2] = 1e0; // [L]

  std::vector<mc::FFVar> FCT( NF*NS*NS, 0 );  // State functions
  for( unsigned i=0; i<NS; i++ ){
    FCT[NF*NS*i+NF*i]   = X[0];
    FCT[NF*NS*i+NF*i+1] = X[1];
  }
//  std::vector<mc::FFVar> FCT( NF*NS, 0 );  // State functions
//  for( unsigned i=0; i<NS; i++ ){
//    FCT[NF*i]   = X[0];
//    FCT[NF*i+1] = X[1];
//  }

  mc::ODESLVS_CVODES IVP;
  IVP.options.INTMETH   = mc::BASE_CVODES::Options::MSBDF;//MSADAMS;//
  IVP.options.NLINSOL   = mc::BASE_CVODES::Options::NEWTON;//FIXEDPOINT;//
  IVP.options.LINSOL    = mc::BASE_CVODES::Options::DIAG;//DENSE;//
  IVP.options.FSACORR   = mc::BASE_CVODES::Options::STAGGERED;//STAGGERED1;//SIMULTANEOUS;
  IVP.options.NMAX      = 2000;
  IVP.options.DISPLAY   = 0;
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-8;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-8;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
  IVP.options.ASACHKPT  = 2000;
#if defined( SAVE_RESULTS )
  IVP.options.RESRECORD = 100;
#endif

  IVP.set_dag( &DAG );
  IVP.set_time( NS, tk );
  IVP.set_state( NX, X );
  IVP.set_parameter( NC, C );
  IVP.set_differential( NS, NX, RHS );
  IVP.set_initial( NX, IC );
//  IVP.set_function( NS, NF, FCT.data() );
  IVP.set_function( NS, NF*NS, FCT.data() );
  IVP.setup();

  /////////////////////////////////////////////////////////////////////////
  // Simulate IVP-ODE
/*
  // Nominal model parameters
  K0.set( -3.1 );
  K1.set(  2.4 );
  nu.set(  0.5 );
  alpha.set( 1.0 );
  IVP.setup();

  double dC[NC] = { 0.1, 0.0, 0.0, 0.0, 0.0, 323.15 };//, -3.1, 2.4, 0.5, 1.0 };  // Parameter values
  IVP.states( dC ); //, xk, f );
#if defined( SAVE_RESULTS )
  std::ofstream direcSTA;
  direcSTA.open( "test1_STA.dat", std::ios_base::out );
  IVP.record( direcSTA );
#endif
*/
  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  // Sampled parameters - uniform Sobol' sampling
  unsigned const NSAM = 50;
  double KLB[NK], KUB[NK];//, KSAM[NK];
  KLB[0] =         KUB[0] = 5e-1;    // nu
  KLB[1] =         KUB[1] = 1e0;     // alpha
  KLB[2] = -5.866; KUB[2] = -0.543;  // K0
  KLB[3] = 0.454;  KUB[3] = 4.388;   // K1 

  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 engK( NK );
  qrgen genK( engK, boost::uniform_01<double>() );
  genK.engine().seed( 0 );

  std::vector< std::vector< double > > v_KSAM( NSAM );
  std::list< double const* > l_KSAM;
  for( unsigned s=0; s<NSAM; ++s ){
    v_KSAM[s].resize( NK );
    for( unsigned k=0; k<NK; k++ )
      v_KSAM[s][k] = KLB[k] + ( KUB[k] - KLB[k] ) * genK();
    l_KSAM.push_back( v_KSAM[s].data() );
  }

  // Experimental control space
  double CLB[NC], CUB[NC];
  for( unsigned i=0; i<NC-1; ++i ){
    CLB[i] = 0e0;
    CUB[i] = 1e-1;    // [L/h]
  }
  CLB[NC-1] = 273.15; // [K]
  CUB[NC-1] = 323.15; // [K]

  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::DOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::AVERSE;//NEUTRAL;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-6;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 0;
  DOE.set_model( &IVP );
  DOE.set_controls( NC, C, CLB, CUB );
  DOE.set_parameters( NK, K, l_KSAM );
  DOE.setup();
  DOE.sample_supports( 100 );
  DOE.combined_solve( 5 );
  //DOE.effort_solve( 5 );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 5, DOE.efforts() );
  //DOE.file_export( "test1" );

  auto campaign = DOE.campaign();
  DOE.options.CRITERION = mc::DOEBase::BROPT;//DOPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::NEUTRAL;//AVERSE;//
  DOE.setup();
  DOE.evaluate_design( campaign );
 
  DOE.options.CRITERION = mc::DOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::NEUTRAL;//AVERSE;//
  DOE.setup();
  DOE.evaluate_design( campaign );

  return 0;
}

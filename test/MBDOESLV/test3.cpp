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

  const size_t NS = 5;  // Time stages
  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( size_t k=0; k<NS; k++ ) tk[k+1] = tk[k] + 2e0; // [hour]

  const size_t NC = 3;  // Number of experimental controls
  std::vector<mc::FFVar> C( NC );  // Controls
  for( size_t i=0; i<NC; i++ ) C[i].set( &DAG );
  C[0].set("u1");
  C[1].set("u2");
  C[2].set("y10");
  mc::FFVar& u1  = C[0];
  mc::FFVar& u2  = C[1];
  mc::FFVar& y10 = C[2];

  const size_t NP = 4;  // Number of estimated parameters
  std::vector<mc::FFVar> P( NP );  // Parameters
  for( size_t i=0; i<NP; i++ ) P[i].set( &DAG );
  P[0].set("p1");
  P[1].set("p2");
  P[2].set("p3");
  P[3].set("p4");
  mc::FFVar& p1 = P[0]; // max growth rate
  mc::FFVar& p2 = P[1]; // half-saturation constant
  mc::FFVar& p3 = P[2]; // proudct yield
  mc::FFVar& p4 = P[3]; // respiration rate

  const size_t NX = 2;  // Number of states
  std::vector<mc::FFVar> X( NX );  // States & state sensitivities
  for( size_t i=0; i<NX; i++ ) X[i].set( &DAG );
  X[0].set("y1");
  X[1].set("y2");
  mc::FFVar& y1  = X[0]; // biomass
  mc::FFVar& y2  = X[1]; // substrate

  std::vector<mc::FFVar> RHS( NX*NS );  // Right-hand side function
  for( size_t i=0; i<NS; i++ ){
    mc::FFVar r = p1 * y2 / ( p2 + y2 );
    RHS[NX*i]   = ( r - u1 - p4 ) * y1;
    RHS[NX*i+1] = -r * y1 / p3 + u1 * ( u2 - y2 );
  }
  
  std::vector<mc::FFVar> IC( NX );   // Initial value function
  IC[0] = y10; //7e0;
  IC[1] = 1e-1;

  const size_t NY = 2;  // Number of outputs
  std::vector<std::vector<mc::FFVar>> FCT( NS, std::vector<mc::FFVar>( NY*NS, 0. ) );  // State functions
  for( size_t i=0; i<NS; i++ ){
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
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-9;
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
  p1.set( 0.31 );
  p2.set( 0.18 );
  p3.set( 0.55 );
  p4.set( 0.05 );
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

  size_t const NEXP = 5;

  // Sampled parameters - uniform Sobol' sampling
  size_t const NPSAM = 500;
  std::vector<double> PLB( NP ), PUB( NP );
//  PLB[0] =  PUB[0] = 0.31;
//  PLB[1] =  PUB[1] = 0.18;
//  PLB[2] =  PUB[2] = 0.55;
//  PLB[3] =  PUB[3] = 0.05;
  PLB[0] = 1e-1;  PUB[0] = 1e0;
  PLB[1] = 5e-2;  PUB[1] = 1e0;
  PLB[2] = 1e-1;  PUB[2] = 2e0;
  PLB[3] = 1e-2;  PUB[3] = 2e-1;

  // Experimental control space
  size_t const NCSAM = 100;
  std::vector<double> CLB( NC ), CUB( NC );
  CLB[0] = 5e-2;   CUB[0] = 2e-1;
  CLB[1] = 5e0;    CUB[1] = 35e0;
  CLB[2] = 1e0;    CUB[2] = 1e1;

  // Output variance
  std::vector<double> YVAR( NY*NS, 4e-2 );

  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::DOEBase::BROPT;
  DOE.options.RISK      = mc::MBDOESLV<>::Options::NEUTRAL;//AVERSE;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.MAXITER = 100;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.OPTIMTOL  = 2e-5;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 0;
  DOE.set_model( IVP, YVAR );
  DOE.set_controls( C, CLB, CUB );
  DOE.set_parameters( P, DOE.uniform_sample( NPSAM, PLB, PUB ) );

  // Solve MBDOE
  DOE.setup();
  DOE.sample_supports( NCSAM );
  DOE.combined_solve( NEXP );
  //DOE.effort_solve( NEXP );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( NEXP, DOE.efforts() );
  //DOE.file_export( "test2" );
  auto campaign = DOE.campaign();

/*
  // Sobol samples campaign
  std::multimap<double,std::vector<double>> campaign;
  for( auto const& c : DOE.uniform_sample( NEXP, CLB, CUB ) )
    campaign.insert( std::make_pair( 1, c ) );
*/
/*
  PLB[0] = 1e-1;  PUB[0] = 1e0;
  PLB[1] = 5e-2;  PUB[1] = 1e0;
  PLB[2] = 1e-1;  PUB[2] = 2e0;
  PLB[3] = 1e-2;  PUB[3] = 2e-1;
  DOE.set_parameters( P, DOE.uniform_sample( 100, PLB, PUB ) );
*/
  DOE.options.CRITERION = mc::DOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::NEUTRAL;//AVERSE;//
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-NEUTRAL" );

  DOE.options.CRITERION = mc::DOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::AVERSE;//NEUTRAL;//
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-AVERSE" );

  DOE.options.CRITERION = mc::DOEBase::BROPT;//DOPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::NEUTRAL;//AVERSE;//
  DOE.setup();
  DOE.evaluate_design( campaign, "BROPT" );


  /////////////////////////////////////////////////////////////////////////
  // Simulate experimental campaign

  // Nominal model parameters
  p1.set( 0.31 );
  p2.set( 0.18 );
  p3.set( 0.55 );
  p4.set( 0.05 );
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-10;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-10;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
  IVP.setup();

  // Nomimal model predictions
  std::vector<std::vector<double>> simulated_campaign; 
  for( auto const& c : campaign ){
    IVP.solve_state( c.second );
    simulated_campaign.push_back( IVP.val_function() );
  }


  /////////////////////////////////////////////////////////////////////////
  // Repeated model calibration
  size_t const NREP = 200;
  std::list<std::vector<double>> MLEREP;
  for( size_t irep=0; irep<NREP; ++irep ){

    // Define and solve MLE problem
    mc::FFGraph< mc::FFODE<0>, mc::FFGRADODE<0> > DAGMLE;
    std::vector<mc::FFVar> PMLE( NP );  // Parameters
    for( size_t i=0; i<NP; i++ ){
       PMLE[i].set( &DAGMLE );
       P[i].unset();
    }
    mc::FFODE<0> OpODE;
    mc::FFVar FMLE( 0. );

    arma::vec YM( NY*NS, arma::fill::zeros );
    arma::mat YC( NY*NS, NY*NS, arma::fill::zeros ); YC.diag() = arma::vec( YVAR );
    size_t iexp = 0;
    for( auto const& c : campaign ){
      std::vector<std::vector<mc::FFVar>> MLE( NS, std::vector<mc::FFVar>( 1, 0. ) );
  
      for( size_t ieff=0; ieff< std::round(c.first); ++ieff ){
        // Add measurement noise
        arma::mat dY = arma::mvnrnd( YM, YC );
        //std::cout << dY;
        //std::cout << "Simulated experiment " << iexp << "." << ieff << ": ";
        //size_t k = 0;
        //for( auto const& Yk : simulated_campaign[iexp] )
        //  std::cout << "  " << Yk + dY(k++);
        //std::cout << std::endl;
      
        // Append terms to ML estimator
        for( size_t i=0, k=0; i<NS; ++i )
          for( size_t j=0; j<NY; ++j, ++k )
            MLE[i][0] += mc::sqr( FCT[i][k] - simulated_campaign[iexp][k] - dY(k) ) / YVAR[k];
      }

      for( size_t i=0; i<NC; i++ ){
        C[i].set( c.second[i] );
        //std::cout << "C[" << i << "] = " << c.second[i] << std::endl;
      }
      IVP.set_parameter( P );
      IVP.set_function( MLE );
      IVP.setup();
      mc::ODESLVS_CVODES<>* pIVP = &IVP;
      FMLE += OpODE( 0, NP, PMLE.data(), pIVP );

      iexp++;
    }
    //std::cout << DAGMLE;

    // Local optimization
#ifdef MC__USE_SNOPT
    mc::NLPSLV_SNOPT< mc::FFODE<0>, mc::FFGRADODE<0> > NLP;
    NLP.options.DISPLEVEL = 0;
    NLP.options.MAXITER   = 50;
    NLP.options.FEASTOL   = 1e-5;
    NLP.options.OPTIMTOL  = 1e-5;
    NLP.options.GRADMETH  = mc::NLPSLV_SNOPT< mc::FFODE<0>, mc::FFGRADODE<0> >::Options::FSYM;
    NLP.options.GRADCHECK = false;
    NLP.options.MAXTHREAD = 6;
#else
    mc::NLPSLV_IPOPT< mc::FFODE<0>, mc::FFGRADODE<0> > NLP;
    NLP.options.DISPLEVEL = 5;
    NLP.options.MAXITER   = 50;
    NLP.options.FEASTOL   = 1e-6;
    NLP.options.OPTIMTOL  = 1e-6;
    NLP.options.GRADMETH  = mc::NLPSLV_IPOPT< mc::FFODE<0>, mc::FFGRADODE<0> >::Options::FAD;
    NLP.options.GRADCHECK = false;
    NLP.options.MAXTHREAD = 8;
#endif

    NLP.set_dag( &DAGMLE );
    NLP.add_var( NP, PMLE.data(), 1e-2, 1e1 );
    NLP.set_obj( mc::BASE_OPT::MIN, FMLE );
    NLP.setup();

    std::vector<double> PMLE0( { 0.31, 0.18, 0.55, 0.05 } );
    NLP.solve( PMLE0.data() );
    //NLP.solve( 100 );
    //std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
    //std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-6 )   << std::endl;
    //std::cout << "STATIONARY: " << NLP.is_stationary( 1e-6 ) << std::endl;

    // store MLE results
    MLEREP.push_back( NLP.solution().x );
    std::cout << irep << ":";
    for( auto const& xk : NLP.solution().x )
      std::cout << "  " << xk;
    std::cout << std::endl;
  }

  return 0;
}

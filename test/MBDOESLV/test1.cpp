#undef SAVE_RESULTS		// <- Whether to save bounds to file
//#define MC__MBDOE_SETUP_DEBUG
//#define MC__MBDOE_SAMPLE_DEBUG

#include "mbdoeslv.hpp"

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define model

  mc::FFGraph DAG;  // DAG describing the IVP-ODE
  DAG.options.MAXTHREAD = 0;

  size_t const NS = 12;       // Time stages
  size_t const NK = 4;       // Number of estimated parameters
  size_t const NC = NS+1;//NS/2+1;  // Number of experimental controls
  size_t const NX = 3;       // Number of states
  size_t const NY = 2;       // Number of outputs

  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( size_t k=0; k<NS; k++ ) tk[k+1] = tk[k] + 2.5e1; // [min]

  std::vector<mc::FFVar> C( NC );  // Controls
  for( size_t i=0; i<NC; i++ ) C[i].set( &DAG );
  mc::FFVar& T     = C[NS];//C[NS/2];

  std::vector<mc::FFVar> K( NK );  // Parameters
  for( size_t i=0; i<NK; i++ ) K[i].set( &DAG );
  mc::FFVar& nu    = K[0];
  mc::FFVar& alpha = K[1];
  mc::FFVar& K0    = K[2];
  mc::FFVar& K1    = K[3];
  double CAin = 10;     // [mol/L]
  double CA0  = 5;      // [mol/L]
  double Tref = 273.15; // [K]
  double eps  = 1e-5;      // [mol/L]

  std::vector<mc::FFVar> X( NX );  // States & state sensitivities
  for( size_t i=0; i<NX; i++ ) X[i].set( &DAG );
  mc::FFVar& CA  = X[0];
  mc::FFVar& CB  = X[1];
  mc::FFVar& V   = X[2];

  std::vector<std::vector<mc::FFVar>> RHS( NS, std::vector<mc::FFVar>(NX) );  // Right-hand side function
  for( size_t i=0; i<NS; i++ ){
    mc::FFVar& Qin = C[i];//C[i/2]; // [L/min]
    mc::FFVar R = exp( K0 + K1 * ( 1 - T / Tref ) ) * ( pow( CA + eps, alpha ) - pow( eps, alpha ) );
    RHS[i][0]   = Qin / V * ( CAin - CA ) - R;
    RHS[i][1] = - Qin / V * CB + nu * R;
    RHS[i][2] = Qin;
  }
  
  std::vector<mc::FFVar> IC( NX );   // Initial value function
  IC[0] = CA0;
  IC[1] = 0e0;
  IC[2] = 1e0; // [L]

  std::vector<std::vector<mc::FFVar>> FCT( NS, std::vector<mc::FFVar>( NY*NS, 0. ) );  // State functions
  for( unsigned i=0; i<NS; i++ ){
    FCT[i][NY*i]   = CA;
    FCT[i][NY*i+1] = CB;
  }

  mc::ODESLVS_CVODES IVP;
  IVP.options.INTMETH   = mc::BASE_CVODES::Options::MSBDF;//MSADAMS;//
  IVP.options.NLINSOL   = mc::BASE_CVODES::Options::NEWTON;//FIXEDPOINT;//
  IVP.options.LINSOL    = mc::BASE_CVODES::Options::DIAG;//DENSE;//
  IVP.options.FSACORR   = mc::BASE_CVODES::Options::STAGGERED;//STAGGERED1;//SIMULTANEOUS;
  IVP.options.NMAX      = 2000;
  IVP.options.DISPLAY   = 0;
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-9;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-9;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
  IVP.options.ASACHKPT  = 2000;
#if defined( SAVE_RESULTS )
  IVP.options.RESRECORD = 100;
#endif

  IVP.set_dag( &DAG );
  IVP.set_time( tk );
  IVP.set_state( X );
  IVP.set_constant( K );
  IVP.set_parameter( C );
  IVP.set_differential( RHS );
  IVP.set_initial( IC );
  IVP.set_function( FCT );
  IVP.setup();

  mc::FFODE OpODE;
  std::vector<mc::FFVar> Y(NY*NS);
  for( unsigned int j=0; j<NY*NS; j++ ) Y[j] = OpODE( j, NC, C.data(), NK, K.data(), &IVP );//, mc::FFODE::SHALLOW );
  //std::cout << DAG;
/*
  /////////////////////////////////////////////////////////////////////////
  // Simulate model

  // Nominal control and model parameters
  std::vector<double> dC{ 0.1, 0.0, 0.0, 0.0, 0.0, 323.15 };
  std::vector<double> dK{ 0.5, 1.0, -3.1, 2.4 };
  std::vector<double> dY( NY*NS );
  DAG.eval( NY*NS, Y.data(), dY.data(), NC, C.data(), dC.data(), NK, K.data(), dK.data() );
  for( unsigned i=0, k=0; i<NS; i++ )
    for( unsigned j=0; j<NY; j++, k++ )
      std::cout << "Y[" << i << "][" << j << "] = " << dY[k] << std::endl;
*/
  /////////////////////////////////////////////////////////////////////////
  // Perform MBDOE

  size_t const NEXP = 5;

  // Sampled parameters - uniform Sobol' sampling
  size_t const NKSAM = 100;
  std::vector<double> KLB(NK), KUB(NK);
//  KLB[0] =         KUB[0] = 5e-1;    // nu
//  KLB[1] =         KUB[1] = 1e0;     // alpha
//  KLB[2] =         KUB[2] = -3.1;  // K0
//  KLB[3] =         KUB[3] = 2.4;   // K1 
  KLB[0] =         KUB[0] = 5e-1;    // nu
  KLB[1] =         KUB[1] = 1e0;     // alpha
  KLB[2] = -5.866; KUB[2] = -0.543;  // K0
  KLB[3] = 0.454;  KUB[3] = 4.388;   // K1 

  // Experimental control space
  size_t const NCSAM = 200;
  std::vector<double> CLB(NC), CUB(NC);
  for( size_t i=0; i<NC-1; ++i ){
    CLB[i] = 0e0;
    CUB[i] = 1e-1;    // [L/h]
  }
  CLB[NC-1] = 273.15; // [K]
  CUB[NC-1] = 323.15; // [K]

  // Output variance
  std::vector<double> YVAR( NY*NS, 4e-2 );

  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::FFDOEBase::DOPT;//BROPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;//AVERSE;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-6;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 0;
  DOE.set_dag( DAG );
  DOE.set_model( Y, YVAR );
  DOE.set_controls( C, CLB, CUB );
  DOE.set_parameters( K, DOE.uniform_sample( NKSAM, KLB, KUB ) );

  // Solve MBDOE
  DOE.setup();
  DOE.sample_supports( NCSAM );
  DOE.combined_solve( NEXP );
  //DOE.effort_solve( NEXP );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( NEXP, DOE.efforts() );
  //DOE.file_export( "test1" );
  auto campaign = DOE.campaign();

/*
  // Sobol samples campaign
  std::multimap<double,std::vector<double>> campaign;
  for( auto const& c : DOE.uniform_sample( NEXP, CLB, CUB ) )
    campaign.insert( std::make_pair( 1, c ) );
*/
/*
  KLB[0] =         KUB[0] = 5e-1;    // nu
  KLB[1] =         KUB[1] = 1e0;     // alpha
  KLB[2] = -5.866; KUB[2] = -0.543;  // K0
  KLB[3] = 0.454;  KUB[3] = 4.388;   // K1 
  DOE.set_parameters( K, DOE.uniform_sample( 200, KLB, KUB ) );
*/
/*
  std::multimap<double,std::vector<double>> campaign // ** EFFORT-BASED EXACT DESIGN: 2.58167e+01
  {
    //SUPPORT #100: 2 x [ 1.00000e-01 7.31311e-02 0.00000e+00 0.00000e+00 3.23150e+02 ]
    { 2, { 1.00000e-01, 7.31311e-02, 0.00000e+00, 0.00000e+00, 3.23150e+02 } },
    //SUPPORT #101: 3 x [ 1.00000e-01 3.68750e-02 0.00000e+00 0.00000e+00 2.73150e+02 ]
    { 3, { 1.00000e-01, 3.68750e-02, 0.00000e+00, 0.00000e+00, 2.73150e+02 } }
    // SUPPORT #73: 2 x [ 9.60938e-02 4.60938e-02 3.82813e-02 8.98438e-02 3.21197e+02 ]
    //{ 2, { 9.60938e-02, 4.60938e-02, 3.82813e-02, 8.98438e-02, 3.21197e+02 } },
    // SUPPORT #84: 3 x [ 9.92188e-02 9.92188e-02 5.39063e-02 1.17188e-02 2.75884e+02 ]
    //{ 3, { 9.92188e-02, 9.92188e-02, 5.39063e-02, 1.17188e-02, 2.75884e+02 } }
  };
*/
  DOE.options.CRITERION = mc::FFDOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;//AVERSE;//
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-NEUTRAL" );
 
  DOE.options.CRITERION = mc::FFDOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV::Options::AVERSE;//NEUTRAL;//
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-AVERSE" );

  DOE.options.CRITERION = mc::FFDOEBase::BROPT;//DOPT;//
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;//AVERSE;//
  DOE.setup();
  DOE.evaluate_design( campaign, "BROPT" );


  /////////////////////////////////////////////////////////////////////////
  // Simulate experimental campaign

  // Nominal model parameters
  std::vector<double> dK{ 0.5, 1.0, -3.1, 2.4 };

  // Nomimal model predictions
  std::vector<std::vector<double>> simulated_campaign; 
  //std::vector<double> dY( NY*NS );
  for( auto const& c : campaign ){
    //DAG.eval( NY*NS, Y.data(), dY.data(), NC, C.data(), c.second.data(), NK, K.data(), dK.data() );
    //simulated_campaign.push_back( dY );
    IVP.solve_state( c.second, dK );
    simulated_campaign.push_back( IVP.val_function() );
  }

  /////////////////////////////////////////////////////////////////////////
  // Confidence analysis

  mc::FFVar FMLE( 0. );
  std::vector<mc::FFVar> CMLE, YMMLE;
  std::vector<double> dCMLE, dYMMLE;

  // Swap model parameters and controls in ODE model
  IVP.set_constant( C );
  IVP.set_parameter( K );
  IVP.setup();

  // Define MLE criterion
  size_t iexp = 0;
  for( auto const& c : campaign ){
    // Append controls for current experiment
    std::vector<mc::FFVar> CMLEc;
    for( unsigned i=0; i<NC; ++i ){
      mc::FFVar CMLEci( &DAG );// CMLEci.set( c.second[i] );
      CMLEc.push_back( CMLEci );
    }
    CMLE.insert( CMLE.end(), CMLEc.cbegin(), CMLEc.cend() );
    dCMLE.insert( dCMLE.end(), c.second.cbegin(), c.second.cend() );

    // Append prediction error to MLE
    for( size_t ieff=0; ieff< std::round(c.first); ++ieff ){
      for( unsigned i=0, k=0; i<NS; ++i ){
        for( unsigned j=0; j<NY; ++j, ++k ){
          mc::FFVar YMk( &DAG );// YMk.set( simulated_campaign[iexp][k] );
          YMMLE.push_back( YMk );
          dYMMLE.push_back( simulated_campaign[iexp][k] );
          mc::FFVar& Yk = OpODE( k, NK, K.data(), NC, CMLEc.data(), &IVP );
          FMLE += mc::sqr( Yk - YMk ) / YVAR[k];
        }
      }
    }
    iexp++;
  }

  // Linearised confidence region
  size_t const NYM = YMMLE.size();
  mc::FFODE::options.DIFF = mc::FFODE::Options::SYM_P;
  auto DFMLE = DAG.FAD( 1, &FMLE, NK, K.data(), NYM, YMMLE.data() );
  mc::FFODE::options.DIFF = mc::FFODE::Options::NUM_P;
  auto D2FMLE = DAG.SFAD( NK+NYM, DFMLE, NK, K.data() );
  size_t const NELE = std::get<0>(D2FMLE);
  std::vector<double> dD2FMLE( NELE );
  DAG.eval( NELE, std::get<3>(D2FMLE), dD2FMLE.data(),
            NK, K.data(), dK.data(),
            CMLE.size(), CMLE.data(), dCMLE.data(),
            NYM, YMMLE.data(), dYMMLE.data() );  
  arma::mat dD2FMLEDYDK( NYM, NK );//, arma::fill::zeros );
  arma::mat dD2FMLEDK2( NK, NK, arma::fill::zeros );
  for( unsigned k=0; k<std::get<0>(D2FMLE); ++k ){
    unsigned i = std::get<1>(D2FMLE)[k];
    unsigned j = std::get<2>(D2FMLE)[k];
    if( i < NK ) dD2FMLEDK2(i,j)     = dD2FMLE[k];
    else         dD2FMLEDYDK(i-NK,j) = dD2FMLE[k];
  }
  //std::cout << "d2FMLE/dK2 =\n" << dD2FMLEDK2;
  //std::cout << "d2FMLE/dYdK =\n " << dD2FMLEDYDK;
  
  delete[] DFMLE;
  delete[] std::get<1>(D2FMLE);
  delete[] std::get<2>(D2FMLE);
  delete[] std::get<3>(D2FMLE);
  
  arma::mat COVY = arma::kron( arma::eye(NEXP,NEXP), arma::diagmat( arma::vec( YVAR ) ) );
  //std::cout << "Measurement covariance\n " << COVY;
  arma::mat A = arma::inv( dD2FMLEDK2 ) * arma::trans( dD2FMLEDYDK );
  arma::mat COVK = A * COVY * arma::trans(A);
  std::cout << "Parameter covariance\n " << COVK;

  //return 0;
  
  // Bootstrapped MLE calculations
  size_t const NREP = 200;
  std::list<std::vector<double>> MLEREP;
  for( size_t irep=0; irep<NREP; ++irep ){

    // Update measurement values
    arma::vec YM( NY*NS, arma::fill::zeros );
    arma::mat YC( NY*NS, NY*NS, arma::fill::zeros ); YC.diag() = arma::vec( YVAR );
    size_t iexp = 0, imeas = 0;
    for( auto const& c : campaign ){
      for( size_t ieff=0; ieff< std::round(c.first); ++ieff ){
        // Add measurement noise
        arma::mat dY = arma::mvnrnd( YM, YC );
        //std::cout << dY;
        //std::cout << "Simulated experiment " << iexp << "." << ieff << ": ";
        //size_t k = 0;
        //for( auto const& Yk : simulated_campaign[iexp] )
        //  std::cout << "  " << Yk + dY(k++);
        //std::cout << std::endl;

        for( size_t i=0, k=0; i<NS; ++i )
          for( size_t j=0; j<NY; ++j, ++k, ++imeas )
            YMMLE[imeas].set( simulated_campaign[iexp][k] + dY(k) );
      }
      iexp++;
    }

    //DAG.output( DAG.subgraph( 1, &FMLE ), " OF MLE" );
    //double dFMLE;
    //DAG.eval( 1, &FMLE, &dFMLE, CMLE.size(), CMLE.data(), dCMLE.data(), NK, K.data(), dK.data() );
    //std::cout << "FMLE = " << dFMLE << std::endl;

    // Local optimization
#ifdef MC__USE_SNOPT
    mc::NLPSLV_SNOPT NLP;
    NLP.options.DISPLEVEL = 0;
    NLP.options.MAXITER   = 40;
    NLP.options.FEASTOL   = 1e-5;
    NLP.options.OPTIMTOL  = 1e-5;
    NLP.options.GRADMETH  = mc::NLPSLV_SNOPT::Options::FSYM;
    NLP.options.GRADCHECK = false;
    NLP.options.MAXTHREAD = 6;
#else
    mc::NLPSLV_IPOPT NLP;
    NLP.options.DISPLEVEL = 5;
    NLP.options.MAXITER   = 40;
    NLP.options.FEASTOL   = 1e-5;
    NLP.options.OPTIMTOL  = 1e-5;
    NLP.options.GRADMETH  = mc::NLPSLV_IPOPT::Options::FAD;
    NLP.options.GRADCHECK = false;
    NLP.options.MAXTHREAD = 8;
#endif

    NLP.set_dag( &DAG );
    NLP.add_par( CMLE );
    NLP.add_var( K, -1e1, 1e1 );
    NLP.set_obj( mc::BASE_OPT::MIN, FMLE );
    NLP.setup();
    std::vector<double> dK0{ 0.5, 1.0, -3.1, 2.4 };
    NLP.solve( dK0.data(), nullptr, nullptr, dCMLE.data() );
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

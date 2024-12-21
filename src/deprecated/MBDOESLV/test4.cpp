#undef SAVE_RESULTS		// <- Whether to save bounds to file
#define MC__MBDOE_SHOW_APPORTION
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

  const size_t NS = 10;  // Time stages
  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( size_t k=0; k<NS; k++ ) tk[k+1] = tk[k] + 1e0; // [hour]

  const size_t NC = NS+2;  // Number of experimental controls
  std::vector<mc::FFVar> C( NC );  // Controls
  for( size_t i=0; i<NC; i++ ) C[i].set( &DAG );
  C[NS].set("y2in");
  C[NS+1].set("y10");
  mc::FFVar& y2in = C[NS];
  mc::FFVar& y10  = C[NS+1];

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

  std::vector<std::vector<mc::FFVar>> RHS( NS, std::vector<mc::FFVar>(NX) );  // Right-hand side function
  for( size_t i=0; i<NS; i++ ){
    mc::FFVar& d = C[i]; C[i].set("d");
    mc::FFVar r = p1 * y2 / ( p2 + y2 );
    RHS[i][0]   = ( r - d - p4 ) * y1;
    RHS[i][1] = -r * y1 / p3 + d * ( y2in - y2 );
  }
  
  std::vector<mc::FFVar> IC( NX );   // Initial value function
  IC[0] = y10; //7e0;
  IC[1] = 1e-1;

  const size_t NY = 2;  // Number of outputs
  std::vector<std::vector<mc::FFVar>> FCT( NS, std::vector<mc::FFVar>( NY*NS, 0. ) );  // State functions
  for( size_t i=0; i<NS; i++ ){
    if( !(i%2) ) continue; // measurement at the end of every other time stage
    std::cout << "Measurement at t=" << tk[i+1] << std::endl;
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
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-10;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
  IVP.options.ASACHKPT  = 2000;
#if defined( SAVE_RESULTS )
  IVP.options.RESRECORD = 100;
#endif

  IVP.set_dag( &DAG );
  IVP.set_time( tk );
  IVP.set_state( X );
  IVP.set_constant( P );
  IVP.set_parameter( C );
  IVP.set_differential( RHS );
  IVP.set_initial( IC );
  IVP.set_function( FCT );
  IVP.setup();

  mc::FFODE OpODE;
  std::vector<mc::FFVar> Y(NY*NS);
  for( unsigned int j=0; j<NY*NS; j++ ) Y[j] = OpODE( j, NC, C.data(), NP, P.data(), &IVP );//, mc::FFODE::SHALLOW );
  //std::cout << DAG;

  std::vector<double> dP, dC;
/*
  /////////////////////////////////////////////////////////////////////////
  // Simulate model

  // Nominal control and model parameters
  dC.assign( { 1e-1, 1e0, 5e0 } );
  dP.assign( { 0.31, 0.18, 0.55, 0.05 } );
  std::vector<double> dY( NY*NS );
  DAG.eval( NY*NS, Y.data(), dY.data(), NC, C.data(), dC.data(), NP, P.data(), dP.data() );
  for( unsigned i=0, k=0; i<NS; i++ )
    for( unsigned j=0; j<NY; j++, k++ )
      std::cout << "Y[" << i << "][" << j << "] = " << dY[k] << std::endl;
*/
  /////////////////////////////////////////////////////////////////////////
  // Perform MBDOE

  size_t const NEXP = 5;

  // Sampled parameters - uniform Sobol' sampling
  size_t const NPSAM = 512; // 512
  std::vector<double> PLB( NP ), PUB( NP );
//  PLB[0] =  PUB[0] = 0.31;
//  PLB[1] =  PUB[1] = 0.18;
//  PLB[2] =  PUB[2] = 0.55;
//  PLB[3] =  PUB[3] = 0.05;
//  PLB[0] = 1e-1;  PUB[0] = 1e0;
//  PLB[1] = 5e-2;  PUB[1] = 1e0;
//  PLB[2] = 1e-1;  PUB[2] = 1e0;
//  PLB[3] = 1e-2;  PUB[3] = 2e-1;
  dP.assign( { 0.31, 0.18, 0.55, 0.05 } );
  PLB[0] = dP[0]*6e-1;  PUB[0] = dP[0]*14e-1;
  PLB[1] = dP[1]*6e-1;  PUB[1] = dP[1]*14e-1;
  PLB[2] = dP[2]*6e-1;  PUB[2] = dP[2]*14e-1;
  PLB[3] = dP[3]*6e-1;  PUB[3] = dP[3]*14e-1;

  // Experimental control space
  size_t const NCSAM = 256; // 512
  std::vector<double> CLB( NC, 5e-2 ), CUB( NC, 2e-1 );
  CLB[NS] = 5e0;      CUB[NS] = 35e0;
  CLB[NS+1] = 1e0;    CUB[NS+1] = 1e1;

  // Output variance
  std::vector<double> YVAR( NY*NS, 4e-2 );

  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::MBDOESLV::BROPT;//BROPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;//AVERSE;//
  DOE.options.CVARTHRES = 0.25;
  DOE.options.UNCREDUC  = 1e-3;
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.MAXITER   = 100;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.NLPSLV.GRADMETH  = DOE.options.MINLPSLV.NLPSLV.FAD;
  DOE.options.NLPSLV.OPTIMTOL    = 1e-5;
  DOE.options.NLPSLV.MAXITER     = 250;
  DOE.options.NLPSLV.DISPLEVEL   = 1;
  DOE.options.NLPSLV.GRADCHECK   = 0;
  DOE.options.NLPSLV.GRADMETH    = DOE.options.NLPSLV.FSYM;//FAD;
  DOE.options.NLPSLV.GRADLSEARCH = 0;
  DOE.options.NLPSLV.FCTPREC     = 1e-7;

  DOE.set_dag( DAG );
  DOE.set_model( Y, YVAR );
  DOE.set_controls( C, CLB, CUB );
  DOE.set_parameters( P, DOE.uniform_sample( NPSAM, PLB, PUB ) ); // dP );

  // Solve MBDOE
  DOE.setup();
  DOE.sample_supports( NCSAM );
  DOE.combined_solve( NEXP );
  //DOE.effort_solve( NEXP );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( NEXP );//, DOE.efforts() );
  //DOE.gradient_solve( DOE.efforts(), false );//true );
  //DOE.file_export( "test2" );
  auto campaign = DOE.campaign();

  DOE.set_parameters( P, DOE.uniform_sample( NPSAM, PLB, PUB ) ); // dP );
  DOE.options.CRITERION = mc::MBDOESLV::DOPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-NEUTRAL" );

  DOE.options.CRITERION = mc::MBDOESLV::DOPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::AVERSE;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-AVERSE" );

  DOE.options.CRITERION = mc::MBDOESLV::BROPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "BROPT" );

  return 0;

  /////////////////////////////////////////////////////////////////////////
  // Confidence analysis

  mc::FFVar FMLE( 0. );
  std::vector<mc::FFVar> CMLE, YMMLE;
  std::vector<double> dCMLE, dYMMLE;

  // Swap model parameters and controls in ODE model
  IVP.set_constant( C );
  IVP.set_parameter( P );
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
          mc::FFVar& Yk = OpODE( k, NP, P.data(), NC, CMLEc.data(), &IVP );
          FMLE += mc::sqr( Yk - YMk ) / YVAR[k];
        }
      }
    }
    iexp++;
  }

  // Linearised confidence region
  size_t const NYM = YMMLE.size();
  mc::FFODE::options.DIFF = mc::FFODE::Options::SYM_P;
  auto DFMLE = DAG.FAD( 1, &FMLE, NP, P.data(), NYM, YMMLE.data() );
  mc::FFODE::options.DIFF = mc::FFODE::Options::NUM_P;
  auto D2FMLE = DAG.SFAD( NP+NYM, DFMLE, NP, P.data() );
  size_t const NELE = std::get<0>(D2FMLE);
  std::vector<double> dD2FMLE( NELE );
  
  auto&& vPSAM = DOE.uniform_sample( NPSAM+1000, PLB, PUB );
  auto itPSAMlast = vPSAM.begin();
  std::advance(itPSAMlast, NPSAM);
  vPSAM.erase( vPSAM.begin(), itPSAMlast ); // consider only new samples

  unsigned isam=0;
  for( auto const& dP : vPSAM ){

    dYMMLE.clear();
    for( auto const& c : campaign ){
      IVP.solve_state( dP, c.second );
      for( size_t ieff=0; ieff< std::round(c.first); ++ieff )
        dYMMLE.insert( dYMMLE.end(), IVP.val_function().cbegin(), IVP.val_function().cend() );
    }

    DAG.eval( NELE, std::get<3>(D2FMLE), dD2FMLE.data(),
              NP, P.data(), dP.data(),
              CMLE.size(), CMLE.data(), dCMLE.data(),
              NYM, YMMLE.data(), dYMMLE.data() );  
    arma::mat dD2FMLEDYDP( NYM, NP );//, arma::fill::zeros );
    arma::mat dD2FMLEDP2( NP, NP, arma::fill::zeros );
    for( unsigned k=0; k<std::get<0>(D2FMLE); ++k ){
      unsigned i = std::get<1>(D2FMLE)[k];
      unsigned j = std::get<2>(D2FMLE)[k];
      if( i < NP ) dD2FMLEDP2(i,j)     = dD2FMLE[k];
      else         dD2FMLEDYDP(i-NP,j) = dD2FMLE[k];
    }
    //std::cout << "d2FMLE/dP2 =\n" << dD2FMLEDP2;
    //std::cout << "d2FMLE/dYdP =\n " << dD2FMLEDYDP;
  
    arma::mat COVY = arma::kron( arma::eye(NEXP,NEXP), arma::diagmat( arma::vec( YVAR ) ) );
    //std::cout << "Measurement covariance\n " << COVY;
    arma::mat A = arma::inv( dD2FMLEDP2 ) * arma::trans( dD2FMLEDYDP );
    arma::mat COVP = A * COVY * arma::trans(A);
    
    //std::cout << "Parameter covariance\n " << COVP;
    //std::cout << "Eivenvalues\n " << arma::trans( arma::eig_sym( COVP ) );
    //std::cout << "Rank: " << arma::rank( COVP ) << std::endl;
    std::cout << isam++ << ":" << std::scientific << std::setprecision(6);
    for( auto const& dPi : dP ) std::cout << "  " << dPi;
    std::cout << arma::trans( arma::sqrt( COVP.diag() ) );
  }
  
  delete[] DFMLE;
  delete[] std::get<1>(D2FMLE);
  delete[] std::get<2>(D2FMLE);
  delete[] std::get<3>(D2FMLE);

  return 0;

  // Bootstrapped MLE calculations
  dP.assign( { 0.31, 0.18, 0.55, 0.05 } );
  dYMMLE.clear();
  for( auto const& c : campaign ){
    IVP.solve_state( dP, c.second );
    dYMMLE.insert( dYMMLE.end(), IVP.val_function().cbegin(), IVP.val_function().cend() );
  }

  size_t const NREP = 200;
  std::list<std::vector<double>> MLEREP;
  for( size_t irep=0; irep<NREP; ++irep ){

    // Update measurement values
    arma::vec YM( NY*NS, arma::fill::zeros );
    arma::mat YC( NY*NS, NY*NS, arma::fill::zeros ); YC.diag() = arma::vec( YVAR );
    size_t imeas = 0;
    for( auto const& c : campaign ){
      for( size_t ieff=0; ieff< std::round(c.first); ++ieff ){
        // Add measurement noise
        arma::mat dY = arma::mvnrnd( YM, YC );
        for( size_t i=0, k=0; i<NS; ++i )
          for( size_t j=0; j<NY; ++j, ++k, ++imeas )
            YMMLE[imeas].set( dYMMLE[imeas] + dY(k) );
      }
    }

    //DAG.output( DAG.subgraph( 1, &FMLE ), " OF MLE" );
    //double dFMLE;
    //DAG.eval( 1, &FMLE, &dFMLE, CMLE.size(), CMLE.data(), dCMLE.data(), NP, P.data(), dP.data() );
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
    NLP.add_var( P, -1e1, 1e1 );
    NLP.set_obj( mc::BASE_OPT::MIN, FMLE );
    NLP.setup();
    NLP.solve( dP.data(), nullptr, nullptr, dCMLE.data() );
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

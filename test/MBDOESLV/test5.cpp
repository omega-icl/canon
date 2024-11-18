#undef SAVE_RESULTS		// <- Whether to save bounds to file
#define CANON__MBDOE_SHOW_APPORTION
//#define MC__MBDOE_SETUP_DEBUG
//#define MC__MBDOE_SAMPLE_DEBUG
//#define MC__NLPSLV_SNOPT_DEBUG_CALLBACK

#include "mbdoeslv.hpp"

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define model

  mc::FFGraph DAG;  // DAG describing the IVP-ODE
  DAG.options.MAXTHREAD = 0;

  const size_t NS = 8;  // Time stages
  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( size_t k=0; k<NS; k++ ) tk[k+1] = tk[k] + 7.5e-1; // [hour]

  const size_t NC = 3;  // Number of experimental controls
  std::vector<mc::FFVar> C( NC );  // Controls
  for( size_t i=0; i<NC; i++ ) C[i].set( &DAG );
  mc::FFVar& CSM1_0 = C[0];  C[0].set("CSM1_0");
  mc::FFVar& CD_0   = C[1];  C[1].set("CD_0");
  mc::FFVar& CSM2_0 = C[2];  C[2].set("CSM2_0");

  const size_t NP = 7;  // Number of estimated parameters
  std::vector<mc::FFVar> P( NP );  // Parameters
  for( size_t i=0; i<NP; i++ ) P[i].set( &DAG );

  mc::FFVar& k1f = P[0];  P[0].set("k1f");
  mc::FFVar& Keq = P[1];  P[1].set("Keq");
  mc::FFVar& k2  = P[2];  P[2].set("k2");
  mc::FFVar& k3  = P[3];  P[3].set("k3");
  mc::FFVar& k4  = P[4];  P[4].set("k4");
  mc::FFVar& k5  = P[5];  P[5].set("k5");
  mc::FFVar& k6  = P[6];  P[6].set("k6");

  const size_t NX = 10;  // Number of states
  std::vector<mc::FFVar> X( NX );  // States & state sensitivities
  for( size_t i=0; i<NX; i++ ) X[i].set( &DAG );

  mc::FFVar& CSM1  = X[0];  X[0].set("CSM1");
  mc::FFVar& CD    = X[1];  X[1].set("CD");
  mc::FFVar& CSM1D = X[2];  X[2].set("CSM1D");
  mc::FFVar& CSM2  = X[3];  X[3].set("CSM2");
  mc::FFVar& CP    = X[4];  X[4].set("CP");
  mc::FFVar& CH2O  = X[5];  X[5].set("CH2O");
  //mc::FFVar& CI1   = X[6];  X[6].set("CI1");
  //mc::FFVar& CI2   = X[7];  X[7].set("CI2");
  //mc::FFVar& CI3   = X[8];  X[8].set("CI3");
  //mc::FFVar& CI4   = X[9];  X[9].set("CI4");

  std::vector<mc::FFVar> RHS( NX );  // Right-hand side function
  mc::FFVar r1 = k1f * CSM1 * CD - k1f / Keq * CSM1D;
  mc::FFVar r2 = k2 * CSM2 * CSM1D;
  mc::FFVar r3 = k3 * CSM2 * CP;
  mc::FFVar r4 = k4 * CSM1 * CH2O;
  mc::FFVar r5 = k5 * CD * CH2O;
  mc::FFVar r6 = k6 * CP;
  RHS[0] = - r1 - r4;
  RHS[1] = - r1 + r2 - r5;
  RHS[2] =   r1 - r2;
  RHS[3] = - r2 - r3;
  RHS[4] =   r2 - r3 - r6;
  RHS[5] = - r4 - r5;
  RHS[6] =   r3;
  RHS[7] =   r4;
  RHS[8] =   r5;
  RHS[9] =   r6;

  std::vector<mc::FFVar> IC( NX );   // Initial value function
  IC[0] = CSM1_0;
  IC[1] = CD_0;
  IC[3] = CSM2_0;
  IC[5] = 1e-1;
  IC[2] = IC[4] = IC[6] = IC[7] = IC[8] = IC[9] = 0e0;

  const size_t NY = 4;  // Number of outputs
  std::vector<std::vector<mc::FFVar>> FCT( NS, std::vector<mc::FFVar>( NY*NS, 0. ) );  // State functions
  for( size_t i=0; i<NS; i++ ){
    FCT[i][NY*i]   = CSM1;
    FCT[i][NY*i+1] = CD;
    FCT[i][NY*i+2] = CSM2;
    FCT[i][NY*i+3] = CP;
  }

  mc::ODESLVS_CVODES IVP;
  IVP.options.INTMETH   = mc::BASE_CVODES::Options::MSBDF;//MSADAMS;//
  IVP.options.NLINSOL   = mc::BASE_CVODES::Options::NEWTON;//FIXEDPOINT;//
  IVP.options.LINSOL    = mc::BASE_CVODES::Options::DIAG;//DENSE;//
  IVP.options.FSACORR   = mc::BASE_CVODES::Options::STAGGERED;//STAGGERED1;//SIMULTANEOUS;
  IVP.options.NMAX      = 5000;
  IVP.options.DISPLAY   = 0;//1;
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-12;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-12;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
  IVP.options.ASACHKPT  = 5000;
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
  for( unsigned int j=0; j<NY*NS; j++ ) Y[j] = OpODE( j, NC, C.data(), NP, P.data(), &IVP );

  std::vector<double> dP, dC;
  dP.assign( { 1.09e-1*36e2, 9.33e0, 3.39e-3*36e2, 1.09e-6*36e2, 1.17e-5*36e2,
               1.93e-6*36e2, 1.87e-8*36e2 } );
/*
  /////////////////////////////////////////////////////////////////////////
  // Simulate model

  // Nominal control and model parameters
  dC.assign( { 1e0, 5e-2, 1.15e0 } );
  std::vector<double> dY( NY*NS );
  DAG.eval( NY*NS, Y.data(), dY.data(), NC, C.data(), dC.data(), NP, P.data(), dP.data() );
  for( unsigned i=0, k=0; i<NS; i++ )
    for( unsigned j=0; j<NY; j++, k++ )
      std::cout << "Y[" << i << "][" << j << "] = " << dY[k] << std::endl;
  return 0;
*/
  /////////////////////////////////////////////////////////////////////////
  // Perform MBDOE

  size_t const NEXP = 8;

  // Sampled parameters - uniform Sobol' sampling
  size_t const NPSAM = 128;//1024;//128;//512;
  std::vector<double> PLB( NP ), PUB( NP );
  for( size_t i=0; i<NP; ++i ){
    //PLB[i] = PUB[i] = dP[i];
    PLB[i] = dP[i]*8e-1;  PUB[i] = dP[i]*12e-1;
  }

  // Experimental control space
  size_t const NCSAM = 128;//256;
  std::vector<double> CLB( NC ), CUB( NC );
  CLB[0] = 1.0e-1;   CUB[0] = 1.5e0;
  CLB[1] = 1.0e-2;   CUB[1] = 5.0e-1;
  CLB[2] = 1.0e-1;   CUB[2] = 1.5e0;

  // Output variance
  std::vector<double> YVAR( NY*NS );
  for( unsigned i=0, k=0; i<NS; i++, k+=NY ){
    YVAR[k]   = 4.8e-2;
    YVAR[k+1] = 2.8e-4;
    YVAR[k+2] = 5.2e-2;
    YVAR[k+3] = 5.0e-2;
  }
  
  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::MBDOESLV::DOPT;//BROPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;//AVERSE;//
  DOE.options.CVARTHRES = 0.25;
  DOE.options.UNCREDUC  = 1e-3;//-20;
  DOE.options.FIMSTOL   = 1e-5;
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.MAXITER = 100;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.NLPSLV.GRADMETH = DOE.options.MINLPSLV.NLPSLV.FAD;
  DOE.options.NLPSLV.OPTIMTOL    = 1e-6;
  DOE.options.NLPSLV.MAXITER     = 100;
  DOE.options.NLPSLV.DISPLEVEL   = 1;
  DOE.options.NLPSLV.GRADCHECK   = 0;
  DOE.options.NLPSLV.GRADMETH    = DOE.options.NLPSLV.FSYM;//FAD;//FD;
  DOE.options.NLPSLV.GRADLSEARCH = 0;
  DOE.options.NLPSLV.FCTPREC     = 1e-7;
  DOE.set_dag( DAG );
  DOE.set_model( Y, YVAR );
  DOE.set_parameters( P, dP );//, dP ); // scaling with nominal values
  //DOE.set_parameters( P, DOE.uniform_sample( NPSAM, PLB, PUB ) ); // dP );
  DOE.set_controls( C, CLB, CUB );
/*
  std::list<std::pair<double,std::vector<double>>> prior_campaign
  {
    //{ 1, { CLB[0], CLB[1], CLB[2] } },
    //{ 1, { CLB[0], CUB[1], CUB[2] } },
    //{ 1, { CUB[0], CLB[1], CUB[2] } },
    //{ 1, { CUB[0], CUB[1], CLB[2] } },
    { 1, { (CLB[0]+CUB[0])/2., (CLB[1]+CUB[1])/2., (CLB[2]+CUB[2])/2. } }
  };
  DOE.add_prior_campaign( prior_campaign );
*/
  // Solve MBDOE
  DOE.setup();
  DOE.sample_supports( NCSAM );
  DOE.combined_solve( NEXP );
  //DOE.effort_solve( NEXP );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( NEXP );//, DOE.efforts() );
  //DOE.gradient_solve( DOE.efforts(), false );//true );
  //DOE.file_export( "test2" );
  auto&& campaign = DOE.campaign();
  //arma::mat&& PSCA = DOE.parameter_scaling();
  //std::cout << "FIM SCALING:\n" << PSCA;
  //return 0;

  DOE.set_parameters( P, DOE.uniform_sample( NPSAM, PLB, PUB ) );//, PSCA );
  DOE.options.CRITERION = mc::MBDOESLV::BROPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "BROPT" );
  
  DOE.options.CRITERION = mc::MBDOESLV::DOPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-NEUTRAL" );

  DOE.options.CRITERION = mc::MBDOESLV::DOPT;
  DOE.options.RISK      = mc::MBDOESLV::Options::AVERSE;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-AVERSE" );

  //return 0;
  
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

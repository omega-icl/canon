#define MC__MBDOE_SETUP_DEBUG
#define MC__MBDOE_SHOW_APPORTION
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
  // Define IVP-ODE

  mc::FFGraph DAG;  // DAG describing the IVP-ODE

  const unsigned NP = 2;       // Number of estimated parameters
  const unsigned NX = 1;       // Number of experimental controls
  const unsigned NY = 1;       // Number of outputs

  mc::FFVar X[NX];  // Controls
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );

  mc::FFVar P[NP];  // Parameters
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );

  mc::FFVar Y[NY];  // Outputs
  Y[0] = P[0] * exp( P[1] * X[0] );

  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  // Sampled parameters - uniform Sobol' sampling
  unsigned const NSAM = 21;
  double PLB[NP] = { 1e0, -1e1 };
  double PUB[NP] = { 1e1,  0e0 };
  //double PSCA[NP] = { 1e0, 1e0 };
  double PSCA[NP] = { std::fabs( PUB[0]-PLB[0]), std::fabs( PUB[1]-PLB[1]) };

  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 engP( NP );
  qrgen genP( engP, boost::uniform_01<double>() );
  genP.engine().seed( 0 );

  std::vector< std::vector< double > > v_PSAM( NSAM );
  std::list< double const* > l_PSAM;
  for( unsigned s=0; s<NSAM; ++s ){
    v_PSAM[s].resize( NP );
    for( unsigned k=0; k<NP; k++ )
      v_PSAM[s][k] = PLB[k] + ( PUB[k] - PLB[k] ) * genP();
    l_PSAM.push_back( v_PSAM[s].data() );
  }

  // Experimental control space
  double XLB[NX] = { 0e0  };
  double XUB[NX] = { 5e-1 };

  // Output variance
  double YVAR[NY] = { 1e0 };

  mc::MBDOESLV DOE;
  DOE.options.CRITERION = mc::DOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::AVERSE;//NEUTRAL;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 1;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-7;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 1;
  DOE.set_dag( &DAG );
  DOE.set_model( NY, Y, YVAR );
  DOE.set_controls( NX, X, XLB, XUB );
  DOE.set_parameters( NP, P, l_PSAM, PSCA );
  DOE.setup();
  DOE.sample_supports( 50 );
  DOE.combined_solve( 5 );
  //DOE.effort_solve( 5 );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 5, DOE.efforts() );
  DOE.file_export( "test0" );

  auto campaign = DOE.campaign();
  DOE.options.CRITERION = mc::DOEBase::DOPT;//BROPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::NEUTRAL;//AVERSE;//
  DOE.setup();
  DOE.evaluate_design( campaign );
 
  DOE.options.CRITERION = mc::DOEBase::BROPT;//DOPT;//
  DOE.options.RISK      = mc::MBDOESLV<>::Options::AVERSE;//NEUTRAL;//
  DOE.setup();
  DOE.evaluate_design( campaign );
 
  return 0;
}

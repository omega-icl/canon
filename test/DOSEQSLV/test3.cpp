#define USE_PROFIL
#include <fstream>
#include <iomanip>

#ifdef MC__USE_SNOPT
  #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
  #include "nlpslv_ipopt.hpp"
#endif

#include "ffode.hpp"

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{

  double const T   = 2.0;
  double const KX  = 1.0;
  double const KY  = 0.5;
  double const DX  = 0.2;
  double const DY  = 0.0;
  double const X0  = 1.0;
  double const Y0  = 0.0;
  double const UM  = 1.0;

  /////////////////////////////////////////////////////////////////////////
  // Define IVP-ODE

  mc::FFGraph DAG;  // DAG describing the problem

  const unsigned NS = 20;  // Time stages
  std::vector<double> TS( NS+1 );  // Time stages
  for( unsigned int i=0; i<=NS; i++ ) TS[i] = i * T / NS; 

  const unsigned NU = NS; // Number of parameters
  std::vector<mc::FFVar> U(NU);  // Parameters
  for( unsigned int i=0; i<NU; i++ ) U[i].set( &DAG );

  const unsigned NX = 2;  // Number of states
  std::vector<mc::FFVar> X(NX);  // States
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );

  std::vector<mc::FFVar> IC(NX);   // Initial value function
  IC[0] = X0;
  IC[1] = Y0;

  std::vector<std::vector<mc::FFVar>> RHS(NS); // Right-hand side function
  for( unsigned k=0; k<NS; k++ )
    RHS[k].assign( { KX * U[k] * X[0] - DX * X[0],
                     KY * ( 1. - U[k] ) * X[0] - DY * X[1] } );

  const unsigned NF = 1;  // Number of state functions
  std::vector<mc::FFVar> FCT{ X[1] };

  mc::ODESLVS_CVODES IVP;

  IVP.options.INTMETH   = mc::BASE_CVODES::Options::MSBDF;//MSADAMS;//
  IVP.options.NLINSOL   = mc::BASE_CVODES::Options::FIXEDPOINT;//NEWTON;//
  IVP.options.LINSOL    = mc::BASE_CVODES::Options::DIAG;//DENSE;//
  IVP.options.FSACORR   = mc::BASE_CVODES::Options::STAGGERED;//STAGGERED1;//SIMULTANEOUS;
  IVP.options.NMAX      = 2000;
  IVP.options.DISPLAY   = 0;
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-9;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-9;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
  IVP.options.ASACHKPT  = 2000;

  IVP.set_dag( &DAG );
  IVP.set_time( TS );
  IVP.set_state( X );
  IVP.set_parameter( U );
  IVP.set_differential( RHS );
  IVP.set_initial( IC );
  IVP.set_function( FCT );
  IVP.setup();

  /////////////////////////////////////////////////////////////////////////
  // Define DOSEQ

  mc::FFODE OpODE;
  std::vector<mc::FFVar> F(NF);
  for( unsigned int j=0; j<NF; j++ ) F[j] = OpODE( j, NU, U.data(), &IVP );//, mc::FFODE::SHALLOW );
//  std::cout << DAG;
  auto F_op  = DAG.subgraph( NF, F.data() );
  DAG.output( F_op );

  // Local optimization
#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT NLP;
  NLP.options.DISPLEVEL   = 0;
  NLP.options.MAXITER     = 200;
  NLP.options.FEASTOL     = 1e-7;
  NLP.options.OPTIMTOL    = 1e-7;
  NLP.options.GRADMETH    = mc::NLPSLV_SNOPT::Options::FSYM;//FAD;
  NLP.options.GRADCHECK   = false;
  NLP.options.GRADLSEARCH = false;
  NLP.options.FCTPREC     = 1e-7;
  NLP.options.MAXTHREAD   = 0;
#else
  mc::NLPSLV_IPOPT NLP;
  NLP.options.DISPLEVEL = 5;
  NLP.options.MAXITER   = 200;
  NLP.options.FEASTOL   = 1e-6;
  NLP.options.OPTIMTOL  = 1e-6;
  NLP.options.GRADMETH  = mc::NLPSLV_IPOPT::Options::FAD;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 8;
#endif
  NLP.set_dag( &DAG );                     // DAG
  NLP.add_var( U, 0., UM );                // decision variables
  NLP.set_obj( mc::BASE_OPT::MAX, F[0] );  // objective
  NLP.setup();

  std::vector<double> U0( NU, 1e-1 );

  NLP.solve( U0.data() );
  NLP.solve( 100 );
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-6 )   << std::endl;
  std::cout << "STATIONARY: " << NLP.is_stationary( 1e-6 ) << std::endl;

  OpODE.pODESLV()->options.RESRECORD = 50;
  std::vector<double> dF(NF);
  DAG.eval( NF, F.data(), dF.data(), NU, U.data(), NLP.solution().x.data() );
  std::ofstream xfile, ufile;
  xfile.open( "test3_XOPT.log", std::ios_base::out );
  OpODE.pODESLV()->record( xfile );
  ufile.open( "test3_UOPT.log", std::ios_base::out );
  for( unsigned int k=0; k<NS; k++ ){
    ufile << std::right << std::setw(15) << TS[k] << std::right << std::setw(15) << NLP.solution().x[k] << std::endl;
    ufile << std::right << std::setw(15) << TS[k+1] << std::right << std::setw(15) << NLP.solution().x[k] << std::endl;
  }

  return 0;
}

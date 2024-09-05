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

  double const T  = 1.;
  double const A  = 2.;
  double const B  = 3.;
  double const X0 = 2.;//1.;
  double const UM = 5.;

  /////////////////////////////////////////////////////////////////////////
  // Define IVP-ODE

  mc::FFGraph DAG;  // DAG describing the problem

  const unsigned NS = 40;  // Time stages
  std::vector<double> TS( NS+1 );  // Time stages
  for( unsigned int i=0; i<=NS; i++ ) TS[i] = i * T / NS; 

  const unsigned NU = NS; // Number of parameters
  std::vector<mc::FFVar> U(NU);  // Parameters
  for( unsigned int i=0; i<NU; i++ ) U[i].set( &DAG );

  const unsigned NX = 1;  // Number of states
  std::vector<mc::FFVar> X(NX);  // States
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );

  std::vector<mc::FFVar> IC(NX);   // Initial value function
  IC[0] = X0;

  std::vector<std::vector<mc::FFVar>> RHS(NS); // Right-hand side function
  for( unsigned k=0; k<NS; k++ )
    RHS[k].assign( { A * ( X[0] - sqr(X[0]) / B ) - U[k] } );

  const unsigned NQ = 1;  // Number of state quadratures
  std::vector<mc::FFVar> Q(NQ);  // State quadratures
  for( unsigned i=0; i<NQ; i++ ) Q[i].set( &DAG );

  std::vector<std::vector<mc::FFVar>> QUAD(NS); // Quadrature function
  for( unsigned k=0; k<NS; k++ )
    QUAD[k].assign( { U[k] } );

  const unsigned NF = 2;  // Number of state functions
  std::vector<std::vector<mc::FFVar>> FCT(NS);  // State functions
  for( unsigned k=0; k<NS-1; k++ )
    FCT[k].assign( { Q[0], 0. } );
  FCT[NS-1].assign( { Q[0], X[0] - X0 } );

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
  IVP.set_quadrature( QUAD, Q );
  IVP.set_function( FCT );
  IVP.setup();

  /////////////////////////////////////////////////////////////////////////
  // Define DOSEQ

  mc::FFODE OpODE;
  std::vector<mc::FFVar> F(NF);
  for( unsigned int j=0; j<NF; j++ ) F[j] = OpODE( j, NU, U.data(), &IVP );
  std::cout << DAG;

  // Local optimization
#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT NLP;
  NLP.options.DISPLEVEL = 0;
  NLP.options.MAXITER   = 200;
  NLP.options.FEASTOL   = 1e-7;
  NLP.options.OPTIMTOL  = 1e-7;
  NLP.options.GRADMETH  = mc::NLPSLV_SNOPT::Options::FSYM;//FAD;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 4;
#else
  mc::NLPSLV_IPOPT NLP;
  NLP.options.DISPLEVEL = 0;
  NLP.options.MAXITER   = 100;
  NLP.options.FEASTOL   = 1e-6;
  NLP.options.OPTIMTOL  = 1e-6;
  NLP.options.GRADMETH  = mc::NLPSLV_IPOPT::Options::FAD;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 8;
#endif
  NLP.set_dag( &DAG );                     // DAG
  NLP.add_var( U, 0., UM );                // decision variables
  NLP.set_obj( mc::BASE_OPT::MAX, F[0] );  // objective
  NLP.add_ctr( mc::BASE_OPT::EQ,  F[1] );  // constraints
  NLP.setup();

  //std::vector<double> U0( NU, 1e-1 );
  //NLP.solve( U0 );
  NLP.solve( 100 );
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-6 )   << std::endl;
  std::cout << "STATIONARY: " << NLP.is_stationary( 1e-6 ) << std::endl;

  OpODE.pODESLV()->options.RESRECORD = 50;
  std::vector<double> dF(NF);
  DAG.eval( NF, F.data(), dF.data(), NU, U.data(), NLP.solution().x.data() );

  std::ofstream xfile, ufile;
  xfile.open( "test2_XOPT.log", std::ios_base::out );
  OpODE.pODESLV()->record( xfile );
  ufile.open( "test2_UOPT.log", std::ios_base::out );
  for( unsigned int k=0; k<NS; k++ ){
    ufile << std::right << std::setw(15) << TS[k] << std::right << std::setw(15) << NLP.solution().x[k] << std::endl;
    ufile << std::right << std::setw(15) << TS[k+1] << std::right << std::setw(15) << NLP.solution().x[k] << std::endl;
  }

  return 0;
}

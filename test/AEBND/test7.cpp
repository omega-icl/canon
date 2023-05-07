const unsigned int NPM   = 1;	// <- Order of Taylor/Chebyshev model
#define USE_CMODEL		// <- Use Chebyshev models?
#undef  MC__AEBND_SHOW_PRECONDITIONING

#include "aebnd.hpp"
typedef mc::FFGraph<> DAG;

#include "interval.hpp"
typedef mc::Interval I;

#ifdef USE_CMODEL
  #include "cmodel.hpp"
  typedef mc::CModel<I> PM;
  typedef mc::CVar<I> PV;
#else
  #include "tmodel.hpp"
  typedef mc::TModel<I> PM;
  typedef mc::TVar<I> PV;
#endif

int main()
{
  mc::FFGraph NLE;  // DAG describing the problem

  const unsigned NP = 2;  // Parameter dimension
  mc::FFVar P[NP];  // Parameters p
  for( unsigned i=0; i<NP; i++ ) P[i].set( &NLE );

//  const unsigned NZ = 2;  // State dimension
//  mc::FFVar Z[NZ];  // Dependents z
//  for( unsigned i=0; i<NZ; i++ ) Z[i].set( &NLE );

//  // f1(x1,x2) = sqrt(x1+x2)
//  // f2(x1,x2) = (x1-x2)^2+3*x2
//  mc::FFVar F[NZ];  // Equations f(x,p)=0
//  F[0] = Z[0] - ( P[0] + P[1] );
//  F[1] = Z[1] - sqrt( Z[0] );

//  const unsigned NZ = 4;  // State dimension
//  mc::FFVar Z[NZ];  // Dependents z
//  for( unsigned i=0; i<NZ; i++ ) Z[i].set( &NLE );

//  // f1(x1,x2) = sqrt(x1+x2)
//  // f2(x1,x2) = (x1-x2)^2+3*x2
//  mc::FFVar F[NZ];  // Equations f(x,p)=0
//  F[0] = Z[0] - P[0];
//  F[1] = Z[1] - P[1];
//  F[2] = Z[2] - ( Z[0] + Z[1] );
//  F[3] = Z[3] - sqrt( Z[2] );

  const unsigned NZ = 8;  // State dimension
  mc::FFVar Z[NZ];  // Dependents z
  for( unsigned i=0; i<NZ; i++ ) Z[i].set( &NLE );

//  // f1(x1,x2) = sqrt(x1+x2)+x1*x2
//  // f2(x1,x2) = (x1-x2)^2+3*x2
//  mc::FFVar F[NZ];  // Equations f(x,p)=0
//  F[0] = Z[0] - ( P[0] + P[1] );
//  F[1] = Z[1] - sqrt( Z[0] );
//  F[2] = Z[2] - P[0] * P[1];
//  F[3] = Z[3] - ( Z[1] + Z[2] );
//  F[4] = Z[4] - ( P[0] - P[1] );
//  F[5] = Z[5] - sqr ( Z[4] );
//  F[6] = Z[6] - 3 * P[1];
//  F[7] = Z[7] - ( Z[5] + Z[6] );

  // f1(x1,x2) = sqrt(x1+x2)+x1*x2
  // f2(x1,x2) = (x1-x2)^2+3*x2
  mc::FFVar F[NZ];  // Equations f(x,p)=0
  F[0] = Z[0] - ( P[0] + P[1] );
  F[1] = Z[1] - P[0] * P[1];
  F[2] = Z[2] - sqrt( Z[0] );
  F[3] = Z[3] - ( Z[1] + Z[2] );
  F[4] = Z[4] - ( P[0] - P[1] );
  F[5] = Z[5] - sqr ( Z[4] );
  F[6] = Z[6] - 3 * P[1];
  F[7] = Z[7] - ( Z[5] + Z[6] );

//  const unsigned NZ = 10;  // State dimension
//  mc::FFVar Z[NZ];  // Dependents z
//  for( unsigned i=0; i<NZ; i++ ) Z[i].set( &NLE );

//  // f1(x1,x2) = sqrt(x1+x2)+x1*x2
//  // f2(x1,x2) = (x1-x2)^2+3*x2
//  mc::FFVar F[NZ];  // Equations f(x,p)=0
//  F[0] = Z[0] - P[0];
//  F[1] = Z[1] - P[1];
//  F[2] = Z[2] - ( Z[0] + Z[1] );
//  F[3] = Z[3] - Z[0] * Z[1];
//  F[4] = Z[4] - sqrt( Z[2] );
//  F[5] = Z[5] - ( Z[3] + Z[4] );
//  F[6] = Z[6] - ( Z[0] - Z[1] );
//  F[7] = Z[7] - sqr ( Z[6] );
//  F[8] = Z[8] - 3 * Z[1];
//  F[9] = Z[9] - ( Z[7] + Z[8] );

//  I Ip[NP]  = { 3+sqrt(2)*I(-1e-2,1e-2), 4+sqrt(2)*I(-1e-2,1e-2) },
  I Ip[NP]  = { 3+sqrt(2)*I(-1e0,1e0), 4+sqrt(2)*I(-1e0,1e0) },
    Iz0[NZ]  = { Ip[0] + Ip[1], I(-2e0,2e0)*sqrt(Ip[0] + Ip[1]) },
    Iz[NZ];
  PM PMEnv( NP, NPM );
  PV PMp[NP], PMz[NZ];
  for( unsigned i=0; i<NP; i++ ) PMp[i].set( &PMEnv, i, Ip[i] );

  /////////////////////////////////////////////////////////////////////////
  // Bound AE solution set
  mc::AEBND<DAG,I,PM,PV> BND;

  BND.set_dag( &NLE );
  BND.set_var( NP, P );
  BND.set_dep( NZ, Z, F );

  BND.options.DISPLAY  = 1;
  BND.options.INTERBND = true; //false;
  BND.options.MAXIT    = 20;
  BND.options.RTOL     =
  BND.options.ATOL     = 1e-8;
  BND.options.BOUNDER  = mc::AEBND<DAG,I,PM,PV>::Options::ALGORITHM::AUTO;//GS;//KRAW;//GE;
  BND.options.PRECOND  = mc::AEBND<DAG,I,PM,PV>::Options::PRECONDITIONING::INVMD;//QRMD;//NONE;

  BND.options.BLKDEC = mc::AEBND<DAG,I,PM,PV>::Options::DECOMPOSITION::RECUR;
  BND.setup();
  std::cout << "\nSuccessful? " << (BND.solve( Ip, Iz )==mc::AEBND<DAG,I,PM,PV>::NORMAL?"Y\n":"N\n");
  //std::cout << "\nSuccessful? " << (BND.solve( PMp, PMz )==mc::AEBND<DAG,I,PM,PV>::NORMAL?"Y\n":"N\n");

  BND.options.BLKDEC = mc::AEBND<DAG,I,PM,PV>::Options::DECOMPOSITION::NONE;
  //std::cout << "\nSuccessful? " << (BND.solve( Ip, Iz, Iz0 )==mc::AEBND<DAG,I,PM,PV>::NORMAL?"Y\n":"N\n");
  //std::cout << "\nSuccessful? " << (BND.solve( PMp, PMz )==mc::AEBND<DAG,I,PM,PV>::NORMAL?"Y\n":"N\n");

  BND.options.BLKDEC = mc::AEBND<DAG,I,PM,PV>::Options::DECOMPOSITION::DIAG;
  BND.setup();
  std::cout << "\nSuccessful? " << (BND.solve( Ip, Iz )==mc::AEBND<DAG,I,PM,PV>::NORMAL?"Y\n":"N\n");
  //std::cout << "\nSuccessful? " << (BND.solve( PMp, PMz )==mc::AEBND<DAG,I,PM,PV>::NORMAL?"Y\n":"N\n");

  BND.options.BLKDEC = mc::AEBND<DAG,I,PM,PV>::Options::DECOMPOSITION::RECUR;
  BND.setup();
  std::cout << "\nSuccessful? " << (BND.solve( Ip, Iz, Iz )==mc::AEBND<DAG,I,PM,PV>::NORMAL?"Y\n":"N\n");

//  std::cout << "\nF[0] = " << sqrt( Ip[0] + Ip[1] );
  std::cout << "\nF[0] = " << sqrt( Ip[0] + Ip[1] ) + Ip[0] * Ip[1];
  std::cout << "\nF[1] = " << sqr( Ip[0] - Ip[1] ) + 3 * Ip[1];
  //std::cout << "\nF[0] = " << sqrt( PMp[0] + PMp[1] ) + PMp[0] * PMp[1];
  //std::cout << "\nF[1] = " << sqr( PMp[0] - PMp[1] ) + 3 * PMp[1];
  return 0;
}



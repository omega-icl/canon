#undef  CHECK_REFORMULATION

#ifdef MC__USE_PROFIL
 #include "mcprofil.hpp"
 typedef INTERVAL I;
#else
 #ifdef MC__USE_BOOST
  #include "mcboost.hpp"
   typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
   typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
   typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
   typedef boost::numeric::interval<double,T_boost_policy> I;
 #else
  #ifdef MC__USE_FILIB
   #include "mcfilib.hpp"
   typedef filib::interval<double> I;
  #else
   #include "interval.hpp"
   typedef mc::Interval I;
  #endif
 #endif
#endif

#include "minlpref.hpp"
#include "nlpslv_snopt.hpp"

int main()
{
  //No. stages, feed location, No. components 
  //const unsigned nc = 3, ns = 3, pf = 1;
  const unsigned nc = 3, ns = 5, pf = 3;
  //const unsigned nc = 3, ns = 7, pf = 4;
  //const unsigned nc = 3, ns = 10, pf = 5;

  // Model Parameters
  double Patm = 1e5;				//Atmospheric pressure
  double V    = 1.38;				//Vapour flow rate

  /*
  	let a[1] := 23.4832; 	let a[2] := 20.5110; 	let a[3] := 20.9064;
	let b[1] := -3634.01;	let b[2] := -2664.30;	let b[3] := -3096.52;
	let c[1] := -33.768;	let c[2] := -79.483;	let c[3] := -53.668;
	
	let r[1,1] := 0.0;	let r[1,2] :=  0.7411;	let r[1,3] :=  0.9645;	
	let r[2,1] := -1.0250;	let r[2,2] := 0.0;	let r[2,3] := -1.4350;
	let r[3,1] := -0.9645;	let r[3,2] :=  2.7470;	let r[3,3] := 0.0;

	let s[1,1] := 0.0;	let s[1,2] := -477.00;	let s[1,3] := -903.1024;
	let s[2,1] :=  72.78;	let s[2,2] := 0.0;	let s[2,3] :=  768.20;
	let s[3,1] := -140.9995;let s[3,2] := -1419.0;	let s[3,3] := 0.0;
  */
  double a[nc] = { 23.4832,  20.5110, 20.9064 }; 
  double b[nc] = { -3634.01, -2664.30, -3096.52}; 
  double c[nc] = { -33.768, -79.483, -53.668}; 
  double r[nc*nc] = { 0.0,  0.7411, 0.9645, -1.0250, 0.0, -1.4350, -0.9645, 2.7470, 0.0};
  double s[nc*nc] = { 0.0, -477.00, -903.1024, 72.78, 0.0, 768.20, -140.9995, -1419.0, 0.0};
  /*
    for {i in 1..C, j in 0..N}
	let f[i,j] := 0.0;

	let f[1, N_F] := 0.4098370;
	let f[2, N_F] := 0.01229769;
	let f[3, N_F] := 0.06090665;
  */
  double f[nc*(ns+1)];   
  for( unsigned i = 0; i < nc; ++i ){
  	for( unsigned j = 0; j < ns+1; ++j ) f[i*ns+j] = 0.0;
  }
  f[ 0*ns+pf ] = 0.4098370;
  f[ 1*ns+pf ] = 0.01229769;
  f[ 2*ns+pf ] = 0.06090665;

  double FF[ns+1], sumd;
  //for {j in 0..N} let F[j] := sum{i in 1..C} f[i,j];
  for( unsigned j = 0; j < ns+1; ++j ){
  	sumd = 0.0;
  	for( unsigned i = 0; i < nc; ++i ) sumd += f[ i*ns + j ];
  	FF[j] = sumd;
  } 

  // Equilibrium Model
  mc::FFGraph DAG;
  const unsigned NF = ns*(nc*3+1), NP = NF+1;
  mc::FFVar P[NP], F[NF];
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );

  // Equilibrium Parameters
  mc::FFVar  D  = *P;
  mc::FFVar *X  = P+1;
  mc::FFVar *K  = X+nc*ns;
  mc::FFVar *G  = K+nc*ns;
  mc::FFVar *Te = G+nc*ns;
  //for( unsigned i = 0; i <    NP ; ++i  ) std::cout << "P[" << i <<"]=" << P[i] << "\n";
  //std::cout << "D=" << D << "\n";
  //for( unsigned i = 0; i < nc*ns ; ++i  ) std::cout << "X[" << i <<"]=" << X[i] << "\n";
  //for( unsigned i = 0; i < nc*ns ; ++i  ) std::cout << "K[" << i <<"]=" << K[i] << "\n";
  //for( unsigned i = 0; i <    ns ; ++i  ) std::cout << "Te[" << i <<"]=" << Te[i] << "\n";

  // Auxiliary Equations 
  mc::FFVar B, L[ns+1], p[nc*ns], Lambda[nc*nc*ns], sum_xLambda[nc*ns], sum;
  // param B := F[N_F] - D;
  B = FF[pf] - D;
  /*
  	param L{j in 0..N} = V - D + sum{k in 0..j} F[k];
  */
  for( unsigned j = 0; j < ns+1 ; ++j ){
    sumd = 0.0;
    for( unsigned k = 0; k <= j; ++k )
      sumd += FF[k];
    L[j] = V - D + sumd ;	
  }
  /*
  	var p{i in 1..C, j in 1..N} = exp(a[i]+b[i]/(T[j]+c[i]));
  */
  for( unsigned i = 0; i < nc; ++i  )
    for( unsigned j = 0; j < ns; ++j )
      p[i*ns+j] = exp( a[i] + b[i] / (Te[j]+c[i]) );  	
  /*
  	var Lambda{i1 in 1..C, i2 in 1..C, j in 1..N} = exp(r[i1,i2]+s[i1,i2]*rcp_T[j]);
  */
  for( unsigned i1 = 0; i1 < nc; ++i1 )
    for( unsigned i2 = 0; i2 < nc; ++i2 )
      for( unsigned j = 0; j < ns; ++j )
        Lambda[nc*ns*i1+ns*i2+j] = exp(r[nc*i1+i2] + s[nc*i1+i2]/Te[j]);
  /*
  	var sum_xLambda{i in 1..C, j in 1..N} = sum{i1 in 1..C} (x[i1,j]*Lambda[i,i1,j]);
  */
  for( unsigned i = 0; i < nc; ++i  )
    for( unsigned j = 0; j < ns; ++j ){
      sum = 0.;
      for( unsigned i1 = 0; i1 < nc; ++i1 )
        sum += X[i1*ns+j] * Lambda[i*nc*ns+i1*ns+j];	
      sum_xLambda[i*ns+j] = sum;
    }

  // Equilibrium Conditions
  unsigned ieq=0;
  /*
  	ACTIVITY EQUATIONS
  	E_aux_G{j in 1..N, i in 1..C}: 	G[i,j] - exp( -log(sum_xLambda[i,j]) + 1.0 - (sum{i2 in 1..C} (x[i2,j]*Lambda[i2,i,j]*rcp_sum_xLambda[i2,j])) );
  */
  for( unsigned i = 0; i < nc; ++i  )
    for( unsigned j = 0; j < ns; ++j ){
      sum = 0.;
      for( unsigned i2 = 0; i2 < nc; ++i2 )
        sum += X[i2*ns+j] * Lambda[nc*ns*i2+ns*i+j] / sum_xLambda[i2*ns+j];
      //F[ ieq++ ] = G[i*ns+j] - exp( 1.0 - sum - log( sum_xLambda[i*ns+j] ) );
      F[ ieq++ ] = G[i*ns+j] * sum_xLambda[i*ns+j] - exp( 1.0 - sum );
      //F[ ieq++ ] = 1. - log( G[i*ns+j] * sum_xLambda[i*ns+j] ) + sum;
    }
  /*
  	AUXILIARY EQUATIONS
  	E_aux_K{j in 1..N, i in 1..C}: 	K[i,j] - gamma[i,j]*(p[i,j]/P) = 0.0;  
  */
  for( unsigned j = 0; j < ns; ++j  ){
    for( unsigned i = 0; i < nc; ++i ){
      F[ ieq++ ] = K[i*ns+j] - G[i*ns+j] * p[i*ns+j] / Patm;
      //std::cout << "E_aux_K->F[" << j*nc+i <<"]=" << F[j*nc+i] << std::endl;
    }
  }
  /*
  	MATERIAL BALANCES
  	M_tot{i in 1..C}: D*(K[i,1]*x[i,1]) + B*x[i,N] - f[i,N_F] = 0.0;
  */
  for( unsigned i = 0; i < nc; ++i  ){
    F[ ieq++ ] = D*(K[i*ns+0]*X[i*ns+0]) + B*X[i*ns+(ns-1)] - f[i*ns+pf];
    //std::cout << "M_tot->F[" << i+(ns*nc) <<"]=" << F[i+(ns*nc)] << std::endl;
  } 
  /* 
  	NOTE THE UNUSUAL FORMULATION
	M_eq{j in 1..N-1, i in 1..C}:
	L[j]*x[i,j] + sum{i1 in j+1..N} f[i,i1] - B*x[i,N] - V*(K[i,j+1]*x[i,j+1]) = 0.0;
  */
  for( unsigned j = 0; j < ns-1; ++j  ){
    for( unsigned i = 0; i < nc; ++i ){
      sum = 0.; 
      for( unsigned i2 = j+1; i2 < ns; ++i2 )
        sum += f[ i*ns+i2 ];
      F[ ieq++ ] = L[j]*X[i*ns+j] + sum - B*X[i*ns+(ns-1)] - V*( K[i*ns+j+1]*X[i*ns+j+1] );
      //std::cout << "M_eq->F[" << j*nc+i+(ns*nc)+nc <<"]=" << F[j*nc+i+(ns*nc)+nc] << std::endl;
    } 
  } 
  /*
  	SUMMATION EQUATIONS
	S_x_eq{j in 1..N}: 	sum{i in 1..C} x[i,j] - 1.0 = 0.0;
  */
  for( unsigned j = 0; j < ns; ++j  ){
    sum = -1.;
    for( unsigned i = 0; i < nc; ++i )
      sum += X[i*ns+j];
    F[ ieq++ ] = sum;
    //std::cout << "S_x_eq->F[" << j+(ns*nc)+nc+(ns-1)*nc <<"]=" << F[j+(ns*nc)+nc+(ns-1)*nc] << std::endl;
  }
 
  // Variable bounds
  I Ip[NP], &ID = *Ip, *IX = Ip+1, *IK = IX+nc*ns, *IG = IK+nc*ns, *ITe = IG+nc*ns;
  //ID = I( 0.455, 0.455 );
  //ID = I( 0.45, 0.46 );
  ID = I( 0.44, 0.47 );
  for( unsigned j=0; j<ns; ++j ){
    IX[j]  = I( 0.0001, 0.9999 ); IX[ns+j] = I( 0.0001, 0.9999 ); IX[2*ns+j] = I( 0.0001, 0.9999 );
    IK[j]  = I( 0.97 , 40.52  ); IK[ns+j] = I( 0.2445, 1.317  ); IK[2*ns+j] = I( 0.2745, 1.975  );
    IG[j]  = I( 0.01 , 100.   ); IG[ns+j] = I( 0.01  , 100.   ); IG[2*ns+j] = I( 0.01  , 100.   );
    ITe[j] = I( 336.3, 383.4  );
  }
  //std::cout << DAG;

  mc::MINLPREF<I> MINLP;
  MINLP.set_dag( &DAG );
  for( unsigned i=0; i<NP; i++ )
    MINLP.add_var( P[i], mc::Op<I>::l(Ip[i]), mc::Op<I>::u(Ip[i]), 0 );
  for( unsigned i=0; i<NF; i++ )
    MINLP.add_ctr( mc::BASE_OPT::EQ, F[i] );
  MINLP.set_obj( mc::BASE_OPT::MIN, X[0] );//D );

  // Export original model to GAMS
  MINLP.options.CPMAX  = 100;
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.export_model( "CS_Xmin_original.gms" );

#ifdef CHECK_REFORMULATION
  mc::NLPSLV_SNOPT NLP;
  NLP.options.DISPLEVEL = 0;
  NLP.options.MAXITER   = 100;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_SNOPT<>::Options::FAD;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 0;
  NLP.read( "CS_Xmin_original.gms" );//, true );
  NLP.setup();
  NLP.solve( 100 );//p0 ); //, Ip );
  //std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  //std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  //std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  assert( NP == MINLP.variables().size() );
  double Dp[NP];
  std::cout << "Variables: " << NP << std::endl;
  for( unsigned i=0; i<NP; ++i ){
    Dp[i] = NLP.solution().x[i];
    std::cout << MINLP.variables()[i] << " = " << Dp[i] << std::endl;
  }
  
  assert( NF+1 == MINLP.variables().size() );
  unsigned NF0 = MINLP.functions().size();
  double Df0[NF0];
  std::cout << "Functions:" << NF0 << std::endl;
  MINLP.dag()->eval( NF0, MINLP.functions().data(), Df0, NP, MINLP.variables().data(), Dp );
  for( unsigned i=0; i<NF0; ++i ) std::cout << MINLP.functions()[i] << " = " << Df0[i] << std::endl;
#endif

  // Formulate reduced-space model and export to GAMS
  MINLP.setup();
  MINLP.propagate_bounds();

  MINLP.options.INVBNDGS            = 1;
  MINLP.options.INVKEEPLIN          = 1;
  MINLP.options.AEBND.DISPLEVEL     = 1;
  MINLP.options.SELIM.MIPDISPLEVEL  = 0;
  MINLP.options.SELIM.ELIMMLIN      = 0;
  MINLP.options.SELIM.ELIMNLIN      = {};
  MINLP.options.SELIM.MULTMAX       = 3;
  MINLP.eliminate_invertible_constraints( true );  
  MINLP.options.SELIM.ELIMMLIN      = 1;
  MINLP.options.SELIM.ELIMNLIN      = {mc::FFInv::Options::INV,mc::FFInv::Options::SQRT,mc::FFInv::Options::EXP,
                                       mc::FFInv::Options::LOG,mc::FFInv::Options::RPOW};
  MINLP.eliminate_invertible_constraints( true );
  MINLP.export_model( "CS_Xmin_reduced.gms" );
  
#ifdef CHECK_REFORMULATION
  NF0 = MINLP.functions().size();
  double Df1[NF0];
  std::cout << "Functions (after elimination):" << NF0 << std::endl;
  MINLP.dag()->eval( NF0, MINLP.functions().data(), Df1, NP, MINLP.variables().data(), Dp );
  for( unsigned i=0; i<NF0; ++i ) std::cout << MINLP.functions()[i] << " = " << Df1[i] << std::endl;
#endif

  // Formulate full-space model after lifting of non-polynomial terms and quadratization of polynomials
  MINLP.setup();
  MINLP.propagate_bounds();
  MINLP.lift_polynomial_subexpressions( true );
  MINLP.flatten_linear_functions( true );
  //MINLP.flatten_quadratic_functions( true );
  //MINLP.flatten_polynomial_functions( true );
  MINLP.quadratize_polynomial_functions( true );
  MINLP.export_model( "CS_Xmin_lifted.gms" );

#ifdef CHECK_REFORMULATION
  unsigned const NL = MINLP.lifted_variables().size();
  double Dl[NL];
  mc::FFVar Fl[NL];
  unsigned i = 0;
  for( auto& [j,Fj] : MINLP.lifted_variables() ) Fl[i++] = Fj;
  std::cout << "Lifted Variables: " << NL << std::endl;
  MINLP.dag()->eval( NL, Fl, Dl, NP, MINLP.variables().data(), Dp );
  for( unsigned i=0; i<NL; ++i ) std::cout << Fl[i] << " = " << Dl[i] << std::endl;

  unsigned const NF1 = MINLP.functions().size();
  double Df[NF1];
  std::cout << "Functions (after lifting):" << NF1 << std::endl;
  MINLP.dag()->eval( NF1, MINLP.functions().data(), Df, NP, MINLP.variables().data(), Dp, NL, MINLP.variables().data()+NP, Dl );
  for( unsigned i=0; i<NF1; ++i ) std::cout << MINLP.functions()[i] << " = " << Df[i] << std::endl;
#endif

  return 0;
}

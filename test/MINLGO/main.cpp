#define READ_GAMS    // <-- select test problem here
////////////////////////////////////////////////////////////////////////
//#define MC__MINLGO_SETUP_DEBUG
//#define MC__MINLGO_PREPROCESS_DEBUG
//#define MC__MINLGO_DEBUG
//#define MC__MINLPBND_SHOW_REDUC
//#define MC__MINLPBND_DEBUG_SCQ
//#define MC__REVAL_DEBUG
//#define MC__FFUNC_DEBUG_SIGNOM
//#define MC__SQUAD_DEBUG_REDUC

#include <fstream>
#include <iomanip>

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

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIP;
#elif  MC__USE_IPOPT
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIP;
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
 typedef mc::NLPSLV_SNOPT NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT NLP;
#endif

#include "minlgo.hpp"
 
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
#ifdef MC__USE_CPLEX
  throw std::runtime_error("Error: CPLEX solver not yet implemented");
#endif

  // Global optimization
  mc::MINLGO<I,NLP,MIP> MINLP;

#ifdef READ_GAMS
  MINLP.options.read( "canon.opt" );
//  MINLP.read( "doxydoc.gms" );
//  MINLP.read( "ex1221.gms" );
//  MINLP.read( "ex1222.gms" );
//  MINLP.read( "ex1252a.gms" );
//  MINLP.read( "transswitch0009r.gms" );
//  MINLP.read( "batch0812.gms" );
  MINLP.read( "batch_nc.gms" );
//  MINLP.read( "jit1.gms" );
//  MINLP.read( "ex7_2_2.gms" );
//  MINLP.read( "packing.gms" );
  
#else
  mc::FFGraph DAG;
  const unsigned NP = 4; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, 1, 5, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_NLP::MIN, (P[0]*P[3])*(P[0]+P[1]+P[2])+P[2] ); // objective
  MINLP.add_ctr( mc::BASE_NLP::GE,  (P[0]*P[3])*P[1]*P[2]-25 );          // constraints
  MINLP.add_ctr( mc::BASE_NLP::EQ,  sqr(P[0])+sqr(P[1])+sqr(P[2])+sqr(P[3])-40 );

//  const unsigned NP = 2; mc::FFVar P[NP];
//  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

//  MINLP.set_dag( &DAG );  // DAG
//  MINLP.add_var( P[0], 0, 115.8, 0 ); // decision variables
//  MINLP.add_var( P[1], 0.00001, 30.0, 0 );
//  MINLP.set_obj( mc::BASE_NLP::MIN, 29.4*P[0]+18*P[1] ); // objective
//  MINLP.add_ctr( mc::BASE_NLP::GE,  P[0]-0.2458*sqr(P[0])/P[1]-6 ); // constraints

  MINLP.options.read( "canon.opt" );
//  MINLP.options.DISPLEVEL                   = 1;
//  MINLP.options.CVRTOL                      = 1e-3;
//  MINLP.options.CVATOL                      = 1e-5;
//  MINLP.options.FEASTOL                     = 1e-5;
//  MINLP.options.TIMELIMIT                   = 72e2;
//  MINLP.options.MAXITER                     = 1;
//  MINLP.options.MINLPSLV.MSLOC              = 16;
//  MINLP.options.MINLPBND.NPOLLIFT           = true;
//  MINLP.options.MINLPBND.PSDQUADCUTS        = 2;
//  MINLP.options.MINLPBND.MONSCALE           = true;
//  MINLP.options.MINLPBND.SQUAD.BASIS        = 1;
//  MINLP.options.MINLPPRE.MIPSLV.DISPLEVEL   = 0;
//  MINLP.options.MINLPBND.MIPSLV.OUTPUTFILE  = "main.lp";
#endif

  MINLP.setup();
  MINLP.presolve();
  MINLP.optimize();
  MINLP.stats.display();
  
  return 0;
}

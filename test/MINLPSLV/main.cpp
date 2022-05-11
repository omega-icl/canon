#define READ_GAMS    // <-- select test problem here
////////////////////////////////////////////////////////////////////////
//#define MC__MINLPSLV_DEBUG
//#define MC__REVAL_DEBUG

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

#include "minlpslv.hpp"
 
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  // Local optimization
  mc::MINLPSLV<I,NLP,MIP> MINLP;
  MINLP.options.DISPLEVEL               = 1;
//  MINLP.options.CVRTOL                  = 1e-5;
//  MINLP.options.CVATOL                  = 1e-5;
//  MINLP.options.FEASTOL                 = 1e-5;
//  MINLP.options.FEASPUMP                = true;
//  MINLP.options.INCCUT                  = true;
//  MINLP.options.TIMELIMIT               = 36e2;
//  MINLP.options.MAXITER                 = 1000;
  MINLP.options.MSLOC                   = 16;
//#ifdef MC__USE_SNOPT
  MINLP.options.NLPSLV.DISPLEVEL        = 0;
  MINLP.options.NLPSLV.MAXITER          = 500;
//  MINLP.options.NLPSLV.FEASTOL          = 1e-7;
//  MINLP.options.NLPSLV.OPTIMTOL         = 1e-7;
  MINLP.options.NLPSLV.GRADMETH         = mc::NLPSLV_SNOPT::Options::FAD;
  MINLP.options.NLPSLV.GRADCHECK        = false;
  MINLP.options.NLPSLV.MAXTHREAD        = 0;
//#elif  MC__USE_IPOPT
//  MINLP.options.NLPSLV.DISPLEVEL        = 0;
//  MINLP.options.NLPSLV.MAXITER          = 100;
//  MINLP.options.NLPSLV.FEASTOL          = 1e-8;
//  MINLP.options.NLPSLV.OPTIMTOL         = 1e-8;
//  MINLP.options.NLPSLV.GRADMETH         = mc::NLPSLV_IPOPT::Options::FAD;
//  MINLP.options.NLPSLV.GRADCHECK        = false;
//  MINLP.options.NLPSLV.MAXTHREAD        = 0;
//#endif
//#ifdef MC__USE_GUROBI
//  MINLP.options.MIPSLV.DISPLEVEL        = 0;
//  MINLP.options.MIPSLV.THREADS          = 0;
//  MINLP.options.MIPSLV.MIPRELGAP        = 1e-5;
//  MINLP.options.MIPSLV.OUTPUTFILE       = "main.lp";
//#elif  MC__USE_CPLEX
//  throw std::runtime_error("Error: CPLEX solver not yet implemented");
//#endif

#ifdef READ_GAMS
  MINLP.read( "doxydoc.gms" );
//  MINLP.read( "ex1221.gms" );
//  MINLP.read( "ex1222.gms" );
//  MINLP.read( "ex1252a.gms" );
//  MINLP.read( "transswitch0009r.gms" );

#else
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );
  MINLP.add_var( P[0], 1, 20, 0 );
  MINLP.add_var( P[1], 1, 20, 1 );
  MINLP.set_obj( mc::BASE_NLP::MIN, -6*P[0]-P[1] );
  MINLP.add_ctr( mc::BASE_NLP::LE, 0.3*pow(P[0]-8,2)+0.04*pow(P[1]-6,4)+0.1*exp(2*P[0])/pow(P[1],4)-56 );
  MINLP.add_ctr( mc::BASE_NLP::LE, 1/P[0]+1/P[1]-sqrt(P[0])*sqrt(P[1])+4 );
  MINLP.add_ctr( mc::BASE_NLP::LE, 2*P[0]-5*P[1]+1 );
#endif

  MINLP.setup();
  MINLP.optimize();
  MINLP.stats.display();
  
  return 0;
}

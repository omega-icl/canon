#define TEST40L4       // <-- select test function here

#undef MC__MINLPBND_DEBUG_DRL
#undef MC__MINLPBND_SHOW_REDUC
#undef MC__MINLGO_DEBUG_SBB 

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

#include "mcml.hpp"

#if defined( TEST0 )
double const XL = -10., XU = 10.;
size_t const NX = 2;
std::vector<std::vector<std::vector<double>>> const MLPCOEF =
{ { { 0.0, 0.2, 0.3 },
    { -3.0, 0.5, -0.2 },
    { 0.0, 0.2, -0.4 },
    { 0.0, -0.5, 0.0 }           },
  { { 0.0, 1.0, -1.0, 1.0, 1.0 } } };

#elif defined( TEST30L1 )
double const XL = -3., XU = 3.;
size_t const NX = 2;
#include "ReLUANN_30L1.hpp"

#elif defined( TEST40L4 )
double const XL = -3., XU = 3.;
size_t const NX = 2;
#include "ReLUANN_40L4.hpp"
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
 typedef mc::NLPSLV_SNOPT<mc::FFANN<I,0>,mc::FFGRADANN<I,0>> NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT<mc::FFANN<I,0>,mc::FFGRADANN<I,0>> NLP;
#endif

#include "minlgo.hpp"
typedef mc::MINLGO<I,NLP,MIP,mc::FFANN<I,0>,mc::FFGRADANN<I,0>> MINLP;

int main()
{
  // Create ANN
  mc::ANN<I> f;
  f.options.ACTIV     = mc::ANN<I>::Options::RELU;
  f.options.RELAX     = mc::ANN<I>::Options::ASM;//MC;//AUX;//MCISM;
  f.options.ISMDIV    = 1024;
  f.options.ASMBPS    = 8;
  f.options.ISMCONT   = true;
  f.options.ISMSLOPE  = true;
  f.options.ISMSHADOW = true;
  f.options.CUTSHADOW = false;
  f.options.RELU2ABS  = true;
  f.set( MLPCOEF );

  // Create DAG
  mc::FFGraph<mc::FFANN<I,0>,mc::FFGRADANN<I,0>> DAG;
  mc::FFVar X[NX];
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );
  mc::FFANN<I,0> MLP;
  mc::FFVar F = MLP( 0, NX, X, &f );
  std::cout << DAG;

  // Create optimization model
  MINLP MODEL;
  MODEL.set_dag( &DAG );  // DAG
  MODEL.set_var( NX, X, XL, XU, 0 ); // decision variables
  MODEL.set_obj( mc::BASE_OPT::MIN, F ); // objective

  // Set optimization options
  //MODEL.options.GAMSEXPORT                  = "test_ANN.gms";
  //MODEL.options.PRESOLVE                    = 0;
  MODEL.options.STRATEGY                    = MINLP::Options::SBB;
  MODEL.options.DISPLEVEL                   = 1;
  MODEL.options.CVATOL                      = 1e-4;
  MODEL.options.CVRTOL                      = 1e-4;
  MODEL.options.MAXITER                     = 0;
  MODEL.options.TIMELIMIT                   = 600;
  MODEL.options.MINLPBND.OBBTMAX            = 10;
  MODEL.options.MINLPBND.POLIMG.BREAKPOINT_TYPE = mc::PolBase<I>::Options::CONT;//BIN;//SOS2;
  MODEL.options.MINLPBND.POLIMG.BREAKPOINT_RTOL =
  MODEL.options.MINLPBND.POLIMG.BREAKPOINT_ATOL = 0e0;
  MODEL.options.MINLPBND.MIPSLV.CONTRELAX   = true;
  MODEL.options.MINLPBND.MIPSLV.FEASTOL     = 1e-7;
  MODEL.options.MINLPBND.MIPSLV.OPTIMTOL    = 1e-7;
  MODEL.options.MINLPBND.MIPSLV.DUALRED     = 0;
  MODEL.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  MODEL.options.MINLPBND.MIPSLV.OUTPUTFILE  = "";//"test_ANN.lp";
  MODEL.options.MINLPPRE                    = MODEL.options.MINLPBND;
  MODEL.options.MINLPSLV.NLPSLV.DISPLEVEL   = 0;
  MODEL.options.MINLPSLV.NLPSLV.GRADMETH    = NLP::Options::FSYM;

  // Solve optimization model
  MODEL.setup();
  MODEL.presolve();
  //MODEL.GAMSexport();
  MODEL.optimize();
  MODEL.stats.display();

  return 0;
}

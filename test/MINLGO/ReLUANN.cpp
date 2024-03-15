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
#elif  MC__USE_CPLEX
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIP;
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
 typedef mc::NLPSLV_SNOPT<mc::FFMLP<I,0>,mc::FFGRADMLP<I,0>> NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT<mc::FFMLP<I,0>,mc::FFGRADMLP<I,0>> NLP;
#endif

#include "minlgo.hpp"
typedef mc::MINLGO<I,NLP,MIP,mc::FFMLP<I,0>,mc::FFGRADMLP<I,0>> MINLP;
typedef mc::FFGraph<mc::FFMLP<I,0>,mc::FFGRADMLP<I,0>> DAG;
int main()
{
  // Create ANN
  mc::MLP<I> f;
  f.options.RELAX     = mc::MLP<I>::Options::ASM;//MC;//AUX;//MCISM;
  f.options.ISMDIV    = 1024;
  f.options.ASMBPS    = 8;
  f.options.ISMCONT   = true;
  f.options.ISMSLOPE  = true;
  f.options.ISMSHADOW = true;
  f.options.CUTSHADOW = false;
  f.options.RELU2ABS  = true;
  unsigned l=0;
  for( auto const& layer : MLPCOEF )
    f.append_data( layer, (++l)<MLPCOEF.size()? mc::MLP<I>::Options::RELU:
                                                mc::MLP<I>::Options::LINEAR );

  // Create DAG
  DAG tree;
  mc::FFVar X[NX];
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &tree );
  mc::FFMLP<I,0> MLP;
  mc::FFVar F = MLP( 0, NX, X, &f );
  std::cout << tree;

  // Create optimization model
  MINLP model;
  model.set_dag( &tree );  // DAG
  model.set_var( NX, X, XL, XU, 0 ); // decision variables
  model.set_obj( mc::BASE_OPT::MIN, F ); // objective

  // Set optimization options
  //model.options.GAMSEXPORT                  = "test_ANN.gms";
  //model.options.PRESOLVE                    = 0;
  model.options.STRATEGY                    = MINLP::Options::SBB;
  model.options.DISPLEVEL                   = 1;
  model.options.CVATOL                      = 1e-4;
  model.options.CVRTOL                      = 1e-4;
  model.options.MAXITER                     = 0;
  model.options.TIMELIMIT                   = 600;
  model.options.MINLPBND.OBBTMAX            = 10;
  model.options.MINLPBND.POLIMG.BREAKPOINT_TYPE = mc::PolBase<I>::Options::CONT;//BIN;//SOS2;
  model.options.MINLPBND.POLIMG.BREAKPOINT_RTOL =
  model.options.MINLPBND.POLIMG.BREAKPOINT_ATOL = 0e0;
  model.options.MINLPBND.MIPSLV.CONTRELAX   = true;
  model.options.MINLPBND.MIPSLV.FEASTOL     = 1e-7;
  model.options.MINLPBND.MIPSLV.OPTIMTOL    = 1e-7;
  model.options.MINLPBND.MIPSLV.DUALRED     = 0;
  model.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  model.options.MINLPBND.MIPSLV.OUTPUTFILE  = "";//"test_ANN.lp";
  model.options.MINLPPRE                    = model.options.MINLPBND;
  model.options.MINLPSLV.NLPSLV.DISPLEVEL   = 0;
  model.options.MINLPSLV.NLPSLV.GRADMETH    = NLP::Options::FSYM;

  // Solve optimization model
  model.setup();
  model.presolve();
  //model.GAMSexport();
  model.optimize();
  model.stats.display();

  return 0;
}

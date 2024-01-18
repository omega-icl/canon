#define TEST40L4       // <-- select test function here
#define USE_ASM         // <-- select relaxation approach here: USE_MC / USE_ISM / USE_MCISM / USE_ASM

#undef MC__MINLPBND_DEBUG_DRL
#undef MC__MINLPBND_SHOW_REDUC
#undef MC__MINLGO_DEBUG_SBB 

#include "ReLUANN.hpp"

unsigned const ISMDIV    = 1024;
unsigned const ASMBPS    = 8;
bool const     ISMCONT   = true;
bool const     ISMSLOPE  = true;  
bool const     ISMSHADOW = true;  
bool const     CUTSHADOW = false;  

#if defined( TEST0 )
double const xL = -10., xU = 10.;
unsigned const NP = 2;
std::vector<std::vector<std::vector<double>>> const MLPCOEF =
{ { { 0.0, 0.2, 0.3 },
    { -3.0, 0.5, -0.2 },
    { 0.0, 0.2, -0.4 },
    { 0.0, -0.5, 0.0 }           },
  { { 0.0, 1.0, -1.0, 1.0, 1.0 } } };

#elif defined( TEST30L1 )
double const xL = -3., xU = 3.;
unsigned const NP = 2;
#include "ReLUANN_30L1.hpp"

#elif defined( TEST40L4 )
double const xL = -3., xU = 3.;
unsigned const NP = 2;
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
 typedef mc::NLPSLV_SNOPT<mc::ReLUANNOp> NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT<mc::ReLUANNOp> NLP;
#endif

#include "minlgo.hpp"

int main()
{  
  mc::FFGraph< mc::ReLUANNOp > DAG;
  mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );
  mc::ReLUANNOp MLP;
  MLP.options.ASMBPS    = ASMBPS;
  MLP.options.ISMCONT   = ISMCONT;
  MLP.options.ISMSLOPE  = ISMSLOPE;
  MLP.options.ISMSHADOW = ISMSHADOW;
  MLP.options.CUTSHADOW = CUTSHADOW;
  MLP.set_data( MLPCOEF, ISMDIV );

  mc::MINLGO<I,NLP,MIP,mc::ReLUANNOp> MINLP;
  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, xL, xU, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_OPT::MIN, MLP( NP, P ) ); // objective

  //MINLP.options.GAMSEXPORT                  = "test_ANN.gms";
  //MINLP.options.PRESOLVE                    = 0;
  MINLP.options.STRATEGY                    = mc::MINLGO<I,NLP,MIP,mc::ReLUANNOp>::Options::SBB;
  MINLP.options.DISPLEVEL                   = 1;
  MINLP.options.CVATOL                      = 1e-4;
  MINLP.options.CVRTOL                      = 1e-4;
  MINLP.options.MAXITER                     = 0;
  MINLP.options.TIMELIMIT                   = 60;
  MINLP.options.MINLPBND.OBBTMAX            = 10;
  MINLP.options.MINLPBND.POLIMG.BREAKPOINT_TYPE = mc::PolBase<I>::Options::CONT;//BIN;//SOS2;
  MINLP.options.MINLPBND.POLIMG.BREAKPOINT_RTOL =
  MINLP.options.MINLPBND.POLIMG.BREAKPOINT_ATOL = 0e0;
  MINLP.options.MINLPBND.MIPSLV.CONTRELAX   = ISMCONT;
  MINLP.options.MINLPBND.MIPSLV.FEASTOL     = 1e-7;
  MINLP.options.MINLPBND.MIPSLV.OPTIMTOL    = 1e-7;
  MINLP.options.MINLPBND.MIPSLV.DUALRED     = 0;
  MINLP.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  MINLP.options.MINLPBND.MIPSLV.OUTPUTFILE  = "";//"test_ANN.lp";
  MINLP.options.MINLPPRE                    = MINLP.options.MINLPBND;
  MINLP.options.MINLPSLV.NLPSLV.DISPLEVEL   = 0;
  MINLP.setup();
  MINLP.presolve();
  //MINLP.GAMSexport();
  MINLP.optimize();
  MINLP.stats.display();

  return 0;
}

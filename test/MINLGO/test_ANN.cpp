#define PEAK_RELU_40L4       // <-- select test function here
////////////////////////////////////////////////////////////////////////

#undef MC__MINLPBND_DEBUG_DRL
#undef MC__MINLPBND_SHOW_REDUC
#undef MC__MINLGO_DEBUG_SBB 

#if defined( MC__USE_PROFIL )
 #include "mcprofil.hpp"
 typedef INTERVAL I;
#elif defined( MC__USE_FILIB )
 #include "mcfilib.hpp"
 typedef filib::interval<double,filib::native_switched,filib::i_mode_extended> I;
#elif defined( MC__USE_BOOST )
 #include "mcboost.hpp"
 typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
 typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
 typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
 typedef boost::numeric::interval<double,T_boost_policy> I;
#else
 #include "interval.hpp"
 typedef mc::Interval I;
#endif

#include "ffmlp.hpp"
#include "minlpbnd.hpp"

#if defined( MC__USE_GUROBI )
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIP;
#elif defined( MC__USE_CPLEX )
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIP;
#endif

#if defined( MC__USE_SNOPT )
 #include "nlpslv_snopt.hpp"
 typedef mc::NLPSLV_SNOPT NLP;
#elif defined( MC__USE_IPOPT )
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT NLP;
#endif

#include "minlgo.hpp"
typedef mc::MINLGO<I,NLP,MIP> MINLGO;

////////////////////////////////////////////////////////////////////////

#if defined( TEST_RELU )
size_t const NX = 2;
const double XL[2] = { -2, -2 };	// <-- X range lower bound
const double XU[2] = {  2,  2 };	// <-- X range upper bound
std::vector<std::vector<std::vector<double>>> const MLPCOEF =
{ { { 0.0, 0.2, 0.3 },
    { -3.0, 0.5, -0.2 },
    { 0.0, 0.2, -0.4 },
    { 0.0, -0.5, 0.0 }           },
  { { 0.0, 1.0, -1.0, 1.0, 1.0 } } };

#elif defined( PEAK_RELU_30L1 )
size_t const NX = 2;
const double XL[2] = { -3, -3 };	// <-- X range lower bound
const double XU[2] = {  3,  3 };	// <-- X range upper bound
#include "peak_ReLU_30L1.hpp"

#elif defined( PEAK_RELU_40L4 )
size_t const NX = 2;
const double XL[2] = { -3, -3 };	// <-- X range lower bound
const double XU[2] = {  3,  3 };	// <-- X range upper bound
#include "peak_ReLU_40L4.hpp"

#endif

///////////////////////////////////////////////////////////////////////////////

int main()
{
  try{
    // Create ANN
    mc::MLP<I> NN;
    NN.options.RELU2ABS  = false;
    unsigned l=0;
    for( auto const& layer : MLPCOEF )
      NN.append_data( layer, (++l)<MLPCOEF.size()? NN.RELU: NN.LINEAR );

    // Create DAG
    mc::FFGraph DAG;
    DAG.options.MAXTHREAD = 0;
    std::vector<mc::FFVar> X = DAG.add_vars( NX, "X" );
    mc::FFMLP<I> OpNN;
    OpNN.options.RELAX  = { OpNN.options.PWLS }; //PWLS PWCS MC AUX INT
    OpNN.options.PWCDIV = 32;
    OpNN.options.PWCREL = 1;
    OpNN.options.PWCSUP.USE_SHADOW = 0;
    OpNN.options.PWCSHADOW = 0;
    OpNN.options.PWLINI = 1;
    OpNN.options.PWLREL = 1;
    OpNN.options.PWLSUP.MAX_SUBDIV = 16;
    OpNN.options.PWLSUP.USE_SHADOW = 0;
    OpNN.options.PWLSHADOW = 0;
    std::vector<mc::FFVar> Y{ OpNN( 0, X, &NN, OpNN.COPY ) };

    auto SgY = DAG.subgraph( Y );
    auto StrY = mc::FFExpr::subgraph( &DAG, SgY );
    std::cout << "Y: " << StrY[0] << std::endl;

    MINLGO Model;
    Model.set_dag( &DAG );
    Model.add_var( NX, X.data(), XL, XU );
    Model.set_obj( mc::BASE_OPT::MAX, Y[0] );

    // Set optimization options
    //Model.options.GAMSEXPORT                  = "test_ANN.gms";
    //Model.options.PRESOLVE                    = 0;
    Model.options.STRATEGY                    = Model.options.SBB;
    Model.options.DISPLEVEL                   = 1;
    Model.options.CVATOL                      = 1e-4;
    Model.options.CVRTOL                      = 1e-4;
    Model.options.MAXITER                     = 0;
    Model.options.TIMELIMIT                   = 600;
    Model.options.MINLPBND.RELAXMETH          = { Model.options.MINLPBND.DRL };
    Model.options.MINLPBND.OBBTMAX            = 10;
    Model.options.MINLPBND.POLIMG.BREAKPOINT_TYPE = Model.options.MINLPBND.POLIMG.CONT;//BIN;//SOS2;
    Model.options.MINLPBND.POLIMG.BREAKPOINT_RTOL =
    Model.options.MINLPBND.POLIMG.BREAKPOINT_ATOL = 0e0;
    Model.options.MINLPBND.POLIMG.AGGREG_LQ    = 1;
    Model.options.MINLPBND.MIPSLV.CONTRELAX   = true;
    Model.options.MINLPBND.MIPSLV.FEASTOL     = 1e-7;
    Model.options.MINLPBND.MIPSLV.OPTIMTOL    = 1e-7;
    Model.options.MINLPBND.MIPSLV.DUALRED     = 0;
    Model.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
    Model.options.MINLPBND.MIPSLV.OUTPUTFILE  = "";//"test_ANN.lp";
    Model.options.MINLPPRE                    = Model.options.MINLPBND;
    Model.options.MINLPSLV.NLPSLV.DISPLEVEL   = 0;
    Model.options.MINLPSLV.NLPSLV.GRADMETH    = Model.options.MINLPSLV.NLPSLV.FSYM; //FAD;

    // Solve optimization model
    Model.setup();
    Model.presolve();
    //Model.GAMSexport();
    Model.optimize();
    Model.stats.display();

    return 0;
  }

  catch( mc::FFBase::Exceptions &eObj ){
    std::cerr << "Error " << eObj.ierr()
              << " in factorable function manipulation:" << std::endl
              << eObj.what() << std::endl
              << "Aborts." << std::endl;
    return eObj.ierr();
  }

  catch( MINLGO::Exceptions &eObj ){
    std::cerr << "Error " << eObj.ierr()
              << " in MINLGO solver:" << std::endl
              << eObj.what() << std::endl
              << "Aborts." << std::endl;
    return eObj.ierr();
  }

  return 0;
}


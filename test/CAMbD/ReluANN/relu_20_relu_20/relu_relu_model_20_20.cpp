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

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  // Create optimization model
  MINLP model;

  // Read GAMS model (without MLP1 & MLP2)  
  model.options.read( "canon.opt" );
  std::string gamsfile( "relu_relu_model_supp.gms"); 
  if( !model.read( gamsfile, true, false ) ){//true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return MINLP::STATUS::ABORTED;
  }

  // Locate MLP1 input variables
  #include "Relu_20_20_20.vec.hpp"
  std::vector<mc::FFVar> MLP1in;
  for( auto const& varname : std::list<std::string>( { "inps_qa", "inps_qb", "inps_ra", "inps_rb", "inps_za", "inps_zb" } ) ){
    mc::FFVar const* p1in = model.find_var( varname );
    if( !p1in ) throw std::runtime_error("Error: Variable not found in GAMS model");
    MLP1in.push_back( *p1in );
  }

  // Locate MLP1 output variable
  mc::FFVar const* p1out = model.find_var( "nn_sum_l4_l4n1" );
  if( !p1out ) throw std::runtime_error("Error: Variable not found in GAMS model");
  mc::FFVar MLP1out = *p1out;

  // Create MLP1
  mc::MLP<I> MLP1data;
  MLP1data.options.RELAX     = mc::MLP<I>::Options::ASM;//MC;//AUX;//MCISM;
  MLP1data.options.ISMDIV    = 1024;
  MLP1data.options.ASMBPS    = 8;
  MLP1data.options.ISMCONT   = true;
  MLP1data.options.ISMSLOPE  = true;
  MLP1data.options.ISMSHADOW = true;
  MLP1data.options.CUTSHADOW = false;
  MLP1data.options.RELU2ABS  = true;
  MLP1data.options.SIG2EXP   = true;
  unsigned l1=0;
  for( auto const& layer : MLP1COEF )
    MLP1data.append_data( layer, (++l1)<MLP1COEF.size()? mc::MLP<I>::Options::RELU:
                                                         mc::MLP<I>::Options::LINEAR );

  // Add MLP1 to model
  mc::FFMLP<I,0> MLP1;
  model.add_ctr( MINLP::EQ, MLP1out - MLP1( 0, MLP1in.size(), MLP1in.data(), &MLP1data ) );

  // Locate MLP2 input variables
  #include "Relu_20_20_20_sle.vec.hpp"
  std::vector<mc::FFVar> MLP2in;
  for( auto const& varname : std::list<std::string>( { "sinps_qa", "sinps_qb", "sinps_ra", "sinps_rb", "sinps_za", "sinps_zb" } ) ){
    mc::FFVar const* p2in = model.find_var( varname );
    if( !p2in ) throw std::runtime_error("Error: Variable not found in GAMS model");
    MLP2in.push_back( *p2in );
  }

  // Locate MLP2 output variable
  mc::FFVar const* p2out = model.find_var( "snn_sum_l4_l4n1" );
  if( !p2out ) throw std::runtime_error("Error: Variable not found in GAMS model");
  mc::FFVar MLP2out = *p2out;

  // Create MLP2
  mc::MLP<I> MLP2data;
  MLP2data.options.RELAX     = mc::MLP<I>::Options::AUX;//MC;//ASM;//MCISM;
  MLP2data.options.ISMDIV    = 1024;
  MLP2data.options.ASMBPS    = 8;
  MLP2data.options.ISMCONT   = true;
  MLP2data.options.ISMSLOPE  = true;
  MLP2data.options.ISMSHADOW = true;
  MLP2data.options.CUTSHADOW = false;
  MLP2data.options.RELU2ABS  = true;
  MLP2data.options.SIG2EXP   = true;
  unsigned l2=0;
  for( auto const& layer : MLP2COEF )
    MLP2data.append_data( layer, (++l2)<MLP2COEF.size()? mc::MLP<I>::Options::RELU:
                                                         mc::MLP<I>::Options::LINEAR );

  // Add MLP to model
  mc::FFMLP<I,0> MLP2;
  model.add_ctr( MINLP::EQ, MLP2out - MLP2( 0, MLP2in.size(), MLP2in.data(), &MLP2data ) );
/*
  // Set optimization options
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
  model.options.MINLPBND.MIPSLV.CONTRELAX   = 0;
  model.options.MINLPBND.MIPSLV.FEASTOL     = 1e-5;
  model.options.MINLPBND.MIPSLV.OPTIMTOL    = 1e-5;
  model.options.MINLPBND.MIPSLV.DUALRED     = 0;
  model.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  model.options.MINLPBND.MIPSLV.OUTPUTFILE  = "";//"test_ANN.lp";
  model.options.MINLPPRE                    = model.options.MINLPBND;
  model.options.MINLPSLV.NLPSLV.DISPLEVEL   = 1;
  model.options.MINLPSLV.NLPSLV.GRADMETH    = NLP::Options::FAD;
*/
  // Optimize model
  std::cout << model;
  model.setup();
  std::ostream& os = std::cout;
  int flag = model.presolve( nullptr, nullptr, os ); 
  switch( flag ){
    case MINLP::STATUS::INTERRUPTED:
      std::cerr << "# Exit: GAMS model preprocessing was interrupted" << std::endl;
      return flag;
    case MINLP::STATUS::FAILED:
    case MINLP::STATUS::ABORTED:
      std::cerr << "# Exit: GAMS model preprocessing failed" << std::endl;
      return flag;
    default:
      break;
  }
  //return flag;
  model.GAMSexport( false, os );
  flag = model.optimize( os ); 
  if( model.options.DISPLEVEL >= 1 )
    model.stats.display();

  return flag;  
}

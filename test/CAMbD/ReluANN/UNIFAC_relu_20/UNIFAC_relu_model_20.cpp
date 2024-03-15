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

  // Read GAMS model (without ANN)  
  model.options.read( "canon.opt" );
  std::string gamsfile( "UNIFAC_relu_model_supp.gms"); 
  if( !model.read( gamsfile, true, false ) ){//true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return MINLP::STATUS::ABORTED;
  }

  // Locate MLP input variables
  #include "Relu_20_20_20.vec.hpp"
  std::vector<mc::FFVar> MLPin;
  for( auto const& varname : std::list<std::string>( { "inps_qa", "inps_qb", "inps_ra", "inps_rb", "inps_za", "inps_zb" } ) ){
    mc::FFVar const* pvar = model.find_var( varname );
    if( !pvar ) throw std::runtime_error("Error: Variable not found in GAMS model");
    MLPin.push_back( *pvar );
  }

  // Locate MLP output variable
  mc::FFVar const* pvar = model.find_var( "nn_sum_l4_l4n1" );
  //mc::FFVar const* pvar = model.find_var( "nn_act_l4_l4n1" );
  if( !pvar ) throw std::runtime_error("Error: Variable not found in GAMS model");
  mc::FFVar MLPout = *pvar;

  // Create MLP
  mc::MLP<I> MLPdata;
  MLPdata.options.RELAX     = mc::MLP<I>::Options::MC;//AUX;//MCISM;
  MLPdata.options.ISMDIV    = 1024;
  MLPdata.options.ASMBPS    = 8;
  MLPdata.options.ISMCONT   = true;
  MLPdata.options.ISMSLOPE  = true;
  MLPdata.options.ISMSHADOW = true;
  MLPdata.options.CUTSHADOW = false;
  MLPdata.options.RELU2ABS  = true;
  MLPdata.options.SIG2EXP   = true;
  unsigned l=0;
  for( auto const& layer : MLPCOEF )
    MLPdata.append_data( layer, (++l)<MLPCOEF.size()? mc::MLP<I>::Options::RELU:
//                                                      mc::MLP<I>::Options::SIGMOID );
                                                      mc::MLP<I>::Options::LINEAR );

  // Add MLP to model
  mc::FFMLP<I,0> MLP;
  model.add_ctr( MINLP::EQ, MLPout - MLP( 0, MLPin.size(), MLPin.data(), &MLPdata ) );
  
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

#define USE_PROFIL
#undef  MC__MINLGO_PREPROCESS_DEBUG
#undef  MC__MINLPBND_DEBUG_SCQ
#undef  MC__MINLPBND_SHOW_REDUC
#undef  MC__MINLPBND_SHOW_BREAKPTS
#undef  MC__SQUAD_DEBUG_CHECK

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <csignal>

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
#elif  MC__USE_CPLEX
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
mc::MINLGO<I,NLP,MIP> MINLP;

static
void
signalHandler
( int signum )
{
  std::cout << "Interrupt signal (" << signum << ") received.\n";
  MINLP.interrupt();
  exit( signum );  
}

static
std::string header_usage
( std::string exec )
{
  std::ostringstream os;
  os << "Usage: " << exec << " [OPTION]... GAMSFILE" << std::endl
     << "Run global optimizer CANON on model GAMSFILE." << std::endl
     << std::endl
     << "Mandatory arguments to long options are mandatory for short options too";
  return os.str();
}

int main
( int argc, char* argv[] )
{
  signal( SIGINT, signalHandler );

  ////////////////////////////////////////////////////////////
  // PARSE COMMAND LINE
  opt::options_description cli_main( header_usage( argv[0] ) ); 
  cli_main.add_options()
    ( "option-file,o", opt::value<std::string>(), "load option file" )
    ( "help,h", "display help and exit" )
    ;

  std::string GAMSFILE;
  opt::options_description cli_hidden;
  cli_hidden.add_options()
    ( "gams-model", opt::value<std::string>(&GAMSFILE), "load GAMS file" )
    ;

  opt::options_description const& cli_user = MINLP.options.user_options(); 

  opt::options_description cli_all;
  cli_all.add( cli_main );
  cli_all.add( cli_hidden );
  cli_all.add( cli_user );

  opt::positional_options_description cli_pos;
  cli_pos.add( "gams-model", -1 );

  opt::variables_map map_main;
  try{
    opt::store( opt::command_line_parser( argc, argv ).options( cli_all ).positional( cli_pos ).run(),
                map_main);
    opt::notify( map_main );
  }
  catch( const opt::required_option& e ){
    std::cerr << "Error: " << e.what() << std::endl;
    return mc::MINLGO<I,NLP,MIP>::STATUS::ABORTED;
  }

  if( map_main.count( "help" ) )
  {
    std::cout << cli_main << std::endl
              << cli_user << std::endl;
    return mc::MINLGO<I,NLP,MIP>::STATUS::SUCCESSFUL;
  }

  if( !map_main.count( "gams-model" ) )
  {
    std::cerr << "Error: GAMS file missing" << std::endl;
    return mc::MINLGO<I,NLP,MIP>::STATUS::ABORTED;
  }

  ////////////////////////////////////////////////////////////
  // PARSE OPTION FILE
  std::ofstream logfile;
  if( map_main.count( "option-file" ) )
  {
    std::string optionfilename = map_main["option-file"].as<std::string>();
    if( !MINLP.options.read( optionfilename, logfile ) ){
      std::cerr << "# Exit: Error loading option file " << optionfilename << std::endl;
      return mc::MINLGO<I,NLP,MIP>::STATUS::ABORTED;
    }
  }
      
  ////////////////////////////////////////////////////////////
  // READ AND OPTIMIZE GAMS MODEL
  if( !MINLP.read( GAMSFILE ) ){
    std::cerr << "# Exit: Error reading GAMS file" << std::endl;
    return mc::MINLGO<I,NLP,MIP>::STATUS::ABORTED;
  }
  
  MINLP.setup();
  std::ostream& os = logfile.is_open()? logfile: std::cout;
  int flag = MINLP.presolve( nullptr, nullptr, os ); 
  switch( flag ){
    case mc::MINLGO<I,NLP,MIP>::STATUS::INFEASIBLE:
      if( logfile.is_open() ) logfile.close();
      std::cerr << "# Exit: GAMS model was proven infeasible during preprocessing" << std::endl;
      return flag;
    case mc::MINLGO<I,NLP,MIP>::STATUS::UNBOUNDED:
      if( logfile.is_open() ) logfile.close();
      std::cerr << "# Exit: GAMS model could not be bounded during preprocessing" << std::endl;
      return flag;
    case mc::MINLGO<I,NLP,MIP>::STATUS::INTERRUPTED:
      if( logfile.is_open() ) logfile.close();
      std::cerr << "# Exit: GAMS model preprocessing was interrupted" << std::endl;
      return flag;
    case mc::MINLGO<I,NLP,MIP>::STATUS::FAILED:
    case mc::MINLGO<I,NLP,MIP>::STATUS::ABORTED:
      if( logfile.is_open() ) logfile.close();
      std::cerr << "# Exit: GAMS model preprocessing failed" << std::endl;
      return flag;
    default:
      break;

  }
    
  flag = MINLP.optimize( os ); 
  if( MINLP.options.DISPLEVEL >= 1 )
    MINLP.stats.display();

  if( logfile.is_open() ) logfile.close();
  return flag;  
}

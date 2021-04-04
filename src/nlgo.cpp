#define USE_PROFIL
#undef  MC__NLGO_PREPROCESS_DEBUG
#undef  MC__NLPBND_DEBUG_SCQ
#undef  MC__NLPBND_SHOW_REDUC
#define MC__NLPBND_SHOW_BREAKPTS
#undef  MC__SQUAD_DEBUG_CHECK

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mcprofil.hpp"
typedef INTERVAL I;
#include "nlgo.hpp"

#include <boost/program_options.hpp> 
namespace opt = boost::program_options; 

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
  mc::NLGO<I> NLP;

  ////////////////////////////////////////////////////////////
  // PARSE COMMAND LINE
  opt::options_description cli_main( header_usage( argv[0] ) ); 
  cli_main.add_options()
    ( "option-file,o", opt::value<std::string>(), "load option file" )
    ( "help,h", "display help and exit" )
    ;

  std::string gamsfile;
  opt::options_description cli_hidden;
  cli_hidden.add_options()
    ( "gams-model", opt::value<std::string>(&gamsfile), "load GAMS file" )
    ;
    
  opt::options_description cli_user( "User-defined solver options" ); 
  cli_user.add_options()
    ( "CSALGO",    opt::value<unsigned>(&NLP.options.CSALGO),  "complete search algorithm" )
    ( "PREPROC",   opt::value<bool>(&NLP.options.PREPROC),     "preprocess root node" )
    ( "CVATOL",    opt::value<double>(&NLP.options.CVATOL),    "convergence absolute tolerance" )
    ( "CVRTOL",    opt::value<double>(&NLP.options.CVRTOL),    "convergence relative tolerance" )
    ( "FEASTOL",   opt::value<double>(&NLP.options.FEASTOL),   "feasibility tolerance" )
    ( "MSLOC",     opt::value<unsigned>(&NLP.options.MSLOC),   "multistart local search repeats" )
    ( "TIMELIMIT", opt::value<double>(&NLP.options.TIMELIMIT), "runtime limit" )
    ( "DISPLEVEL", opt::value<int>(&NLP.options.DISPLEVEL),    "general display level" )
    ( "NLPBND.OBBTLIN",     opt::value<unsigned>(&NLP.options.NLPBND.OBBTLIN),  "optimization-based bounds tighteneting approach" )
    ( "NLPBND.OBBTCONT",    opt::value<bool>(&NLP.options.NLPBND.OBBTCONT),     "continuous relaxation for optimization-based bounds tighteneting" )
    ( "NLPBND.OBBTMAX",     opt::value<unsigned>(&NLP.options.NLPBND.OBBTMAX),  "maximum rounds of optimization-based bounds tighteneting" )
    ( "NLPBND.OBBTTHRES",   opt::value<double>(&NLP.options.NLPBND.OBBTTHRES),  "threshold for optimization-based bounds tighteneting repeats" )
    ( "NLPBND.OBBTBKOFF",   opt::value<double>(&NLP.options.NLPBND.OBBTBKOFF),  "backoff for optimization-based bounds tighteneting" )
    ( "NLPBND.OBBTMIG",     opt::value<double>(&NLP.options.NLPBND.OBBTMIG),    "minimum range for optimization-based bounds tighteneting" )
    ( "NLPBND.CPMAX",       opt::value<unsigned>(&NLP.options.NLPBND.CPMAX),    "maximum rounds of constraint propagation" )
    ( "NLPBND.CPTHRES",     opt::value<double>(&NLP.options.NLPBND.CPTHRES),    "threshold for constraint propagation repeats" )
    ( "NLPBND.CPBKOFF",     opt::value<double>(&NLP.options.NLPBND.CPBKOFF),    "backoff for constraint propagation" )
    ( "NLPBND.NPOLLIFT",    opt::value<bool>(&NLP.options.NLPBND.NPOLLIFT),     "reformulate nonpolynomial subexpressions" )
    ( "NLPBND.CMODPROP",    opt::value<unsigned>(&NLP.options.NLPBND.CMODPROP), "maximum order of sparse polynomial model" )
    ( "NLPBND.MONMIG",      opt::value<double>(&NLP.options.NLPBND.CMODEL.MIN_FACTOR), "monomial minimal coefficient in sparse polynomial model" )
    ( "NLPBND.MONBASIS",    opt::value<unsigned>(&NLP.options.NLPBND.SQUAD.BASIS), "monomial basis in sparse quadratic form" )
    ( "NLPBND.MONSCALE",    opt::value<bool>(&NLP.options.NLPBND.MONSCALE),     "monomial scaling in sparse quadratic form" )
    ( "NLPBND.PSDQUADCUTS", opt::value<bool>(&NLP.options.NLPBND.PSDQUADCUTS),  "add PSD cuts within quadratisation" )
    ( "NLPBND.DCQUADCUTS",  opt::value<bool>(&NLP.options.NLPBND.DCQUADCUTS),   "add DC cuts within quadratisation" )
    ( "NLPBND.RRLTCUTS",    opt::value<bool>(&NLP.options.NLPBND.RRLTCUTS),     "add reduced RLT cuts" )
    ( "NLPBND.NCOCUTS",     opt::value<bool>(&NLP.options.NLPBND.NCOCUTS),      "add NCO cuts" )
    ( "NLPBND.NCOMETH",     opt::value<unsigned>(&NLP.options.NLPBND.NCOMETH),  "NCO cut generation method" )
    ( "NLPBND.LINCTRSEP",   opt::value<bool>(&NLP.options.NLPBND.LINCTRSEP),    "separate linear constraints during relaxation" )
    ( "NLPBND.AGGREGLQ",    opt::value<bool>(&NLP.options.NLPBND.POLIMG.AGGREG_LQ), "keep linear and quadratic expressions aggregated" )
    ( "NLPBND.DISPLEVEL",   opt::value<int>(&NLP.options.NLPBND.MIPSLV.DISPLEVEL), "display level of MIP solver" )
    ( "NLPBND.OUTPUTFILE",  opt::value<std::string>(&NLP.options.NLPBND.MIPSLV.OUTPUTFILE), "output file for MIP model" )
    ( "NLPBND.MAXTHREAD",   opt::value<unsigned>(&NLP.options.NLPBND.MIPSLV.THREADS), "set number of threads used by MIP solver" )
    ( "NLPLOC.FEASTOL",     opt::value<double>(&NLP.options.NLPSLV.FEASTOL),    "feasibility tolerance of local NLP solver" )
    ( "NLPLOC.OPTIMTOL",    opt::value<double>(&NLP.options.NLPSLV.OPTIMTOL),   "optimality tolerance of local NLP solver" )
    ( "NLPLOC.MAXITER",     opt::value<int>(&NLP.options.NLPSLV.MAXITER),       "maximal number of iterations of local NLP solver" )
    ( "NLPLOC.DISPLEVEL",   opt::value<int>(&NLP.options.NLPSLV.DISPLEVEL),     "display level of local NLP solver" )
    ( "NLPLOC.RELAXINT",    opt::value<bool>(&NLP.options.NLPSLV.RELAXINT),     "relax the binary/integer variables of local NLP solver" )
    ( "NLPLOC.MAXTHREAD",   opt::value<unsigned>(&NLP.options.NLPSLV.MAXTHREAD), "set number of threads used by local NLP solver" )
    ;

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
    return 2;
  }

  if( map_main.count( "help" ) )
  {
    std::cout << cli_main << std::endl
              << cli_user << std::endl;
    return 1;
  }

  if( !map_main.count( "gams-model" ) )
  {
    std::cerr << "Error: GAMS file missing" << std::endl;
    return 2;
  }

  ////////////////////////////////////////////////////////////
  // PARSE OPTION FILE
  if( map_main.count( "option-file" ) )
  {
    std::string optionfilename = map_main["option-file"].as<std::string>();
    std::ifstream optionfile( optionfilename.c_str() );
    if( optionfile.fail() )
    {
      std::cerr << "Error: cannot open option file " << optionfilename << std::endl;
      return 2;
    }

    opt::variables_map map_user;
    try{
      opt::store( opt::parse_config_file<char>( optionfile, cli_user ), map_user );
      opt::notify( map_user );
    }
    catch( const opt::reading_file& e ){
      std::cerr << "Error: " << e.what() << std::endl;
      return 2;
    }
    catch( const opt::required_option& e ){
      std::cerr << "Error: " << e.what() << std::endl;
      return 2;
    }
    
    if( map_user.count( "TIMELIMIT" ) )
      NLP.options.NLPSLV.TIMELIMIT = NLP.options.NLPBND.TIMELIMIT = NLP.options.TIMELIMIT;
  }

  ////////////////////////////////////////////////////////////
  // LOAD AND OPTIMIZE GAMS MODEL
  if( !NLP.read( gamsfile ) ){
    std::cerr << "Exit: Error reading GAMS file" << std::endl;
    return -2;
  }
  
  NLP.setup();
  if( NLP.options.PREPROC && !NLP.preprocess() ){
    std::cerr << "Exit: Infeasible GAMS mode during preprocessing" << std::endl;
    return -1;
  }
  return 0;
    
  int flag = NLP.optimize(); 
  if( NLP.options.DISPLEVEL >= 1 )
    NLP.stats.display();

  return flag;  
}

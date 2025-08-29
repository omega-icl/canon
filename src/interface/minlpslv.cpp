#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef MC__USE_PROFIL
 #include "mcprofil.hpp"
 typedef INTERVAL I;
#else
 #ifdef MC__USE_FILIB
  #include "mcfilib.hpp"
  typedef filib::interval<double,filib::native_switched,filib::i_mode_extended> I;
 #else
  #ifdef MC__USE_BOOST
   #include "mcboost.hpp"
   typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
   typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
   typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
   typedef boost::numeric::interval<double,T_boost_policy> I;
  #else
   #include "interval.hpp"
   typedef mc::Interval I;
  #endif
 #endif
#endif

#ifdef MC__USE_SNOPT
  #include "nlpslv_snopt.hpp"
  typedef mc::NLPSLV_SNOPT NLPSLV;
#elif  MC__USE_IPOPT
  #include "nlpslv_ipopt.hpp"
  typedef mc::NLPSLV_IPOPT NLPSLV;
#endif
typedef mc::BASE_NLP  BASE;

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIPSLV;
#elif  MC__USE_CPLEX
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIPSLV;
#endif

#include "minlpslv.hpp"
typedef mc::MINLPSLV<I,NLPSLV,MIPSLV> MINLPSLV;

namespace py = pybind11;

void mc_minlpslv( py::module_ &m )
{

py::class_<MINLPSLV,BASE> pyMINLPSLV( m, "MINLPSLV", py::multiple_inheritance() );

pyMINLPSLV
 .def(
   py::init<>()
 )
 .def_readwrite( 
   "options",
   &MINLPSLV::options
 )
#if defined( MC__WITH_GAMS )
 .def(
   "read",
   []( MINLPSLV& self, std::string const& filename, bool const init )
     { return self.read( filename, init ); },
   py::arg("file"),
   py::arg("init")=true,
   "read optimization model from GAMS file"
 )   
#endif
 .def(
   "setup",
   []( MINLPSLV& self ){ self.setup(); },
   "setup optimization model"
 )
 .def(
   "solve",
   []( MINLPSLV& self, std::vector<double> const& xini, std::vector<double> const& pval )
     { return self.optimize( xini.data(), nullptr, pval.data() ); },
   py::arg("ini")=std::vector<double>(),
   py::arg("par")=std::vector<double>(),
   "solve optimization model using local search"
 )
 .def_property_readonly(
   "solution",
   []( MINLPSLV const& self ){ return self.get_incumbent(); },
   py::return_value_policy::reference_internal,
   "incumbent solution of optimization model"
 )
 .def(
   "is_feasible",
   []( MINLPSLV& self, double const& tol ){ return self.is_feasible( tol ); },
   py::return_value_policy::reference_internal,
   "test feasibility of optimization model solution"
 )
 .def_property_readonly(
   "status",
   []( MINLPSLV const& self ){ return self.get_status(); },
   py::return_value_policy::reference_internal,
   "status of optimization model solution"
 )
;

py::enum_<MINLPSLV::STATUS>(pyMINLPSLV, "STATUS")
 .value("SUCCESSFUL", MINLPSLV::STATUS::SUCCESSFUL, "optimal solution found (within required accuracy)" )
 .value("INFEASIBLE", MINLPSLV::STATUS::INFEASIBLE, "model appears to be infeasible" )
 .value("UNBOUNDED", MINLPSLV::STATUS::UNBOUNDED, "model appears to be unbounded" )
 .value("INTERRUPTED", MINLPSLV::STATUS::INTERRUPTED, "resource limit reached" )
 .value("FAILURE", MINLPSLV::STATUS::FAILURE, "search terminated after numerical difficulties" )
 .value("ABORTED", MINLPSLV::STATUS::ABORTED, "critical error encountered" )
 .export_values()
;

py::class_<MINLPSLV::Options> pyMINLPSLVOptions( pyMINLPSLV, "Options" );

pyMINLPSLVOptions
 .def( py::init<>() )
 .def( py::init<MINLPSLV::Options const&>() )
 .def( "reset", []( MINLPSLV::Options& self ){ self.reset(); }, "Reset options to default" )
 .def_readwrite( "SEARCHALG",   &MINLPSLV::Options::SEARCHALG,  "Local search algorithm [Default: OA; Other: BB]" )
 .def_readwrite( "LINMETH",     &MINLPSLV::Options::LINMETH,    "Linearization method [Default: PENAL; Other: CVX]" )
 .def_readwrite( "FEASPUMP",    &MINLPSLV::Options::FEASPUMP,   "Apply feasibility pump strategy in OA algorithm [Default: true]" )
 .def_readwrite( "CORRINC",     &MINLPSLV::Options::CORRINC,    "Correct the incumbent for feasibility using KKT multipliers [Default: true]" )
 .def_readwrite( "ROOTCUT",     &MINLPSLV::Options::ROOTCUT,    "Add cut from root-node relaxation in master problem in OA algorithm [Default: true]" )
 .def_readwrite( "FEASTOL",     &MINLPSLV::Options::FEASTOL,    "Feasibility tolerance [Default: 1e-5]" )
 .def_readwrite( "CVATOL",      &MINLPSLV::Options::CVATOL,     "Convergence absolute tolerance [Default: 1e-3]" )
 .def_readwrite( "CVRTOL",      &MINLPSLV::Options::CVRTOL,     "Convergence relative tolerance [Default: 1e-3]" )
 .def_readwrite( "MAXITER",     &MINLPSLV::Options::MAXITER,    "Maximal number of iterations [Default: 20; 0=no limit]" )
 .def_readwrite( "CPMAX",       &MINLPSLV::Options::CPMAX,      "Maximum number of constraint propagation iterations [Default: 20; 0=no limit]" )
 .def_readwrite( "CPTHRES",     &MINLPSLV::Options::CPTHRES,    "Minimal improvement threshold for constraint propagation iteration [Default: 0.]" )
 .def_readwrite( "PENSOFT",     &MINLPSLV::Options::PENSOFT,    "Penalty weight in soft constraints [Default: 1e3 ]" )
 .def_readwrite( "MSLOC",       &MINLPSLV::Options::MSLOC,      "Number of multistart local search [Default: 8]" )
 .def_readwrite( "TIMELIMIT",   &MINLPSLV::Options::TIMELIMIT,  "Maximum run-time (in seconds) [Default: 600]" )
 .def_readwrite( "DISPLEVEL",   &MINLPSLV::Options::DISPLEVEL,  "Display level during solve [Default: 1]" )
 .def_readwrite( "NLPSLV",      &MINLPSLV::Options::NLPSLV,     "NLP local solver options" )
 .def_readwrite( "MIPSLV",      &MINLPSLV::Options::MIPSLV,     "MIP master solver options" )
 .def_readwrite( "POLIMG",      &MINLPSLV::Options::POLIMG,     "Polyhedral relaxation options" )
;

py::enum_<MINLPSLV::Options::ALGORITHM>(pyMINLPSLVOptions, "ALGORITHM")
 .value("OA", MINLPSLV::Options::ALGORITHM::OA, "Outer-approximation algorithm")
 .value("BB", MINLPSLV::Options::ALGORITHM::BB, "Branch-and-bound algorithm")
 .export_values()
;

py::enum_<MINLPSLV::Options::LINEARIZATION>(pyMINLPSLVOptions, "LINEARIZATION")
 .value("CVX",   MINLPSLV::Options::LINEARIZATION::CVX,   "Direct linearization of cost and constraints at NLP solution point (assumes convexity)")
 .value("PENAL", MINLPSLV::Options::LINEARIZATION::PENAL, "Softening and relaxation of constraints in MIP subproblem (does not assume convexity)")
 .export_values()
;
}


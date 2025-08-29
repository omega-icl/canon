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

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIPSLV;
#elif  MC__USE_CPLEX
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIPSLV;
#endif
typedef mc::BASE_OPT  BASEOPT;

namespace py = pybind11;

void mc_mipslv( py::module_ &m )
{

py::class_<MIPSLV,BASEOPT> pyMIPSLV( m, "MIPSLV", py::multiple_inheritance() );

pyMIPSLV
 .def(
   py::init<>()
 )
 .def_readwrite( 
   "options",
   &MIPSLV::options
 )
 .def(
   "reset",
   []( MIPSLV& self ){ self.reset(); },
   "reset MIP model"
 )
 .def(
   "solve",
   []( MIPSLV& self )
     { return self.solve(); },
   "solve MIP model"
 )
 .def(
   "val_variable",
   []( MIPSLV& self, mc::FFVar const& X )
     { return self.get_variable( X ); },
   "get variable value after MIP solve"
 )
 .def(
   "val_objective",
   []( MIPSLV& self )
     { return self.get_objective(); },
   "get objective value after MIP solve"
 )
 .def(
   "bnd_objective",
   []( MIPSLV& self )
     { return self.get_objective_bound(); },
   "get best objective bound after MIP solve"
 )
 .def_property_readonly(
   "status",
   []( MIPSLV const& self ){ return self.get_status(); },
   py::return_value_policy::reference_internal,
   "status of optimization model solution"
 )
;

py::enum_<MIPSLV::STATUS>(pyMIPSLV, "STATUS")
 .value("OPTIMAL",     MIPSLV::STATUS::OPTIMAL,     "optimal solution found within tolerances" )
 .value("SUBOPTIMAL",  MIPSLV::STATUS::SUBOPTIMAL,  "unable to satisfy optimality tolerances (but sub-optimal solution available)" )
 .value("INFEASIBLE",  MIPSLV::STATUS::INFEASIBLE,  "infeasible (but not unbounded)" )
 .value("INFORUNBND",  MIPSLV::STATUS::INFORUNBND,  "infeasible or unbounded" )
 .value("UNBOUNDED",   MIPSLV::STATUS::UNBOUNDED,   "unbounded" )
 .value("TIMELIMIT",   MIPSLV::STATUS::TIMELIMIT,   "time limit reached" )
 .value("OTHER",       MIPSLV::STATUS::OTHER,       "other status" )
 .export_values()
;

py::class_<MIPSLV::Options> pyMIPSLVOptions( pyMIPSLV, "Options" );

#ifdef MC__USE_GUROBI
pyMIPSLVOptions
 .def( py::init<>() )
 .def( py::init<MIPSLV::Options const&>() )
 .def( "reset", []( MIPSLV::Options& self ){ self.reset(); }, "Reset options to default" )
 .def_readwrite( "ALGO",             &MIPSLV::Options::ALGO,             "Algorithm used to solve continuous models [Default: -1=automatic; Other options: 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex]" )
 .def_readwrite( "PRESOLVE",         &MIPSLV::Options::PRESOLVE,         "Presolve level [Default: -1=automatic; Other options: 0=off, 1=conservative, 2=aggressive]" )
 .def_readwrite( "LPWARMSTART",      &MIPSLV::Options::LPWARMSTART,      "Use of warm start information for LP optimization [Default: 1=use before presolve; Other options: 0=ignore; 2=use after presolve]" )
 .def_readwrite( "CONTRELAX",        &MIPSLV::Options::CONTRELAX,        "Relaxation of binary/integer variables as continuous variables in solve [Default: false=no relaxation; Other options: true=enable continuous relaxation]" )
 .def_readwrite( "DUALRED",          &MIPSLV::Options::DUALRED,          "Perform dual reductions in presolve [Default: 1=enabled; Other option: 0=disabled]" )
 .def_readwrite( "NONCONVEX",        &MIPSLV::Options::NONCONVEX,        "Handling of non-convex quadratic objectives or non-convex quadratic constraints [Default: -1=enabled with convexity detection; Other options: 0=disabled; 1=detection only; 2=enabled]" )
 .def_readwrite( "FEASTOL",          &MIPSLV::Options::FEASTOL,          "Tolerance for constraint feasibility in LP solve [Default: 1e-6]" )
 .def_readwrite( "INTFEASTOL",       &MIPSLV::Options::INTFEASTOL,       "Tolerance for integral feasibility [Default: 1e-5]" )
 .def_readwrite( "OPTIMTOL",         &MIPSLV::Options::OPTIMTOL,         "Tolerance for reduced costs in LP solve [Default: 1e-6]" )
 .def_readwrite( "MIPRELGAP",        &MIPSLV::Options::MIPRELGAP,        "Relative convergence tolerance in MIP solve [Default: 1e-4]" )
 .def_readwrite( "MIPABSGAP",        &MIPSLV::Options::MIPABSGAP,        "Absolute convergence tolerance in MIP solve [Default: 1e-10]" )
 .def_readwrite( "OBBT",             &MIPSLV::Options::OBBT,             "Aggressiveness of Optimality-Based Bound Tightening [Default: -1=automatic; 0=disabled; 1-3=increase aggresiveness]" )
 .def_readwrite( "INTEGRALITYFOCUS", &MIPSLV::Options::INTEGRALITYFOCUS, "Attempt to avoid solutions that exploit integrality tolerances [Default: 0=disabled; 1=enabled]" )
 .def_readwrite( "NUMERICFOCUS",     &MIPSLV::Options::NUMERICFOCUS,     "Attempt to detect and manage numerical issues [Default: 0=automatic; 1-3= increased focus]" )
 .def_readwrite( "SCALEFLAG",        &MIPSLV::Options::SCALEFLAG,        "Handling of model scaling [Default: -1=automatic; 0=disabled; 1-3= increased scaling]" )
 .def_readwrite( "MIPFOCUS",         &MIPSLV::Options::MIPFOCUS,         "High-level MIP solution strategy [Default: 0=automatic; 1=feasibility focus; 2=incumbent focus; 3: bound focus]" )
 .def_readwrite( "HEURISTICS",       &MIPSLV::Options::HEURISTICS,       "Work devoted to MIP heuristic in solve [Default: 0.05]" )
 .def_readwrite( "PRESOS1BIGM",      &MIPSLV::Options::PRESOS1BIGM,      "Threshold for automatic reformulation of SOS1 constraints into binary form [Default: -1=automatic; >0 max big-M value]" )
 .def_readwrite( "PRESOS2BIGM",      &MIPSLV::Options::PRESOS2BIGM,      "Threshold for automatic reformulation of SOS2 constraints into binary form [Default: -1=automatic; >0 max big-M value]" )
 .def_readwrite( "QCPEQFACTOR",      &MIPSLV::Options::QCPEQFACTOR,      "Handling of redundant constraints to describe a factored out quadratic term [Default: 1=enabled; 0=disabled]" )
 .def_readwrite( "FUNCNONLINEAR",    &MIPSLV::Options::FUNCNONLINEAR,    "Handling of general function constraints [Default: 1=nonlinear; 0: piecewise linear]" )
 .def_readwrite( "FUNCMAXVAL",       &MIPSLV::Options::FUNCMAXVAL,       "Maximal allowed x-y values in function constraints [Default: 1e6]" )
 .def_readwrite( "PWLRELGAP",        &MIPSLV::Options::PWLRELGAP,        "Maximal relative error in piecewise-linear approximations of general function constraints [Default: 1e-5]" )
 .def_readwrite( "TIMELIMIT",        &MIPSLV::Options::TIMELIMIT,        "Maximal run-time (in seconds) [Default: 600]" )
 .def_readwrite( "MAXTHREAD",        &MIPSLV::Options::THREADS,          "Maximal number of threads used by MIP solver [Default: 0 (all threads)]" )
 .def_readwrite( "DISPLEVEL",        &MIPSLV::Options::DISPLEVEL,        "Solver output level [Default: 1=solver iterations and results]" )
 .def_readwrite( "LOGFILE",          &MIPSLV::Options::LOGFILE,          "Name of Gurobi log file [Default: -]" )
 .def_readwrite( "OUTPUTFILE",       &MIPSLV::Options::OUTPUTFILE,       "Name of file to be written before solving the model [Default: -]" )
;
#endif

}


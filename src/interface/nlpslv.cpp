#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#ifdef MC__USE_SNOPT
  #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
  #include "nlpslv_ipopt.hpp"
#endif


namespace py = pybind11;

void mc_nlpslv( py::module_ &m )
{
typedef mc::FFGraph FFGraph;
#ifdef MC__USE_SNOPT
  typedef mc::NLPSLV_SNOPT NLPSLV;
#elif  MC__USE_IPOPT
  typedef mc::NLPSLV_IPOPT NLPSLV;
#endif

py::class_<NLPSLV> pyNLPSLV( m, "NLPSLV" );

pyNLPSLV
 .def(
   py::init<>()
 )
 .def_readwrite_static(
   "INF",
   &NLPSLV::INF
 )
 .def_readwrite( 
   "options",
   &NLPSLV::options
 )
#if defined( MC__WITH_GAMS )
 .def(
   "read",
   []( NLPSLV& self, std::string const& filename, bool const init, bool const disp )
     { return self.read( filename, init, disp ); },
   py::arg("file"),
   py::arg("init")=true,
   py::arg("disp")=false,
   "read optimization model from GAMS file"
 )   
#endif
 .def( 
   "set",
   []( NLPSLV& self,  NLPSLV& nlp ){ self.set( nlp ); },
   py::keep_alive<1,2>(),
   "copy optimization model"
 )
 .def( 
   "reset",
   []( NLPSLV& self ){ self.reset(); },
   "reset optimization model"
 )
 .def( 
   "set_dag",
   []( NLPSLV& self,  FFGraph* dag ){ self.set_dag( dag ); },
   py::keep_alive<1,2>(),
   "set DAG"
 )
 .def(
   "dag",
   []( NLPSLV& self ){ return self.dag(); },
   py::return_value_policy::reference_internal,
   "get DAG"
 )
 .def(
   "reset_parameter",
   []( NLPSLV& self ){ self.reset_par(); },
   "reset parameters"
 )
 .def(
   "set_parameter",
   []( NLPSLV& self, std::vector<mc::FFVar> const& par ){ self.set_par( par ); },
   py::arg("par"),
   "set parameters"
 )
 .def(
   "add_parameter",
   []( NLPSLV& self, std::vector<mc::FFVar> const& par ){ self.add_par( par ); },
   py::arg("par"),
   "add parameters"
 )
 .def(
   "reset_variable",
   []( NLPSLV& self ){ self.reset_var(); },
   "reset decision variables"
 )
 .def_property_readonly(
   "all_parameter",
   []( NLPSLV const& self ){ return self.par(); },
   py::return_value_policy::reference_internal,
   "current parameters"
 )
 .def(
   "set_variable",
   []( NLPSLV& self, std::vector<mc::FFVar> const& var, std::vector<double> const& lb,
       std::vector<double> const& ub, std::vector<unsigned> const& typ ){ self.set_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=std::vector<double>(),
   py::arg("ub")=std::vector<double>(),
   py::arg("typ")=std::vector<unsigned>(),
   "set decision variables"
 )
 .def(
   "set_variable",
   []( NLPSLV& self, std::vector<mc::FFVar> const& var, double const& lb,
       double const& ub, unsigned const& typ ){ self.set_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=-NLPSLV::INF,
   py::arg("ub")=NLPSLV::INF,
   py::arg("typ")=0,
   "set decision variables"
 )
 .def(
   "add_variable",
   []( NLPSLV& self, std::vector<mc::FFVar> const& var, std::vector<double> const& lb,
       std::vector<double> const& ub, std::vector<unsigned> const& typ ){ self.add_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=std::vector<double>(),
   py::arg("ub")=std::vector<double>(),
   py::arg("typ")=std::vector<unsigned>(),
   "add decision variables"
 )
 .def(
   "add_variable",
   []( NLPSLV& self, std::vector<mc::FFVar> const& var, double const& lb,
       double const& ub, unsigned const& typ ){ self.add_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=-NLPSLV::INF,
   py::arg("ub")=NLPSLV::INF,
   py::arg("typ")=0,
   "add decision variables"
 )
 .def_property_readonly(
   "all_variable",
   []( NLPSLV const& self ){ return self.var(); },
   py::return_value_policy::reference_internal,
   "current decision variables"
 )
 .def_property_readonly(
   "typ_variable",
   []( NLPSLV const& self ){ return self.vartyp(); },
   py::return_value_policy::reference_internal,
   "current decision variable types"
 )
 .def_property_readonly(
   "lo_variable",
   []( NLPSLV const& self ){ return self.vartyp(); },
   py::return_value_policy::reference_internal,
   "current decision variable lower bounds"
 )
 .def_property_readonly(
   "up_variable",
   []( NLPSLV const& self ){ return self.vartyp(); },
   py::return_value_policy::reference_internal,
   "current decision variable upper bounds"
 )
 .def_property_readonly(
   "all_constraint",
   []( NLPSLV const& self ){ return self.ctr(); },
   py::return_value_policy::reference_internal,
   "current model constraints"
 )
 .def(
   "reset_constraint",
   []( NLPSLV& self ){ self.reset_ctr(); },
   "reset model constraints"
 )
 .def(
   "add_constraint",
   []( NLPSLV& self, NLPSLV::t_CTR const& type, mc::FFVar const& ctr ){ self.add_ctr( type, ctr ); },
   "add constraint"
 )
 .def(
   "add_constraint",
   []( NLPSLV& self, NLPSLV::t_CTR const& type, std::vector<mc::FFVar> const& vctr ){ for( auto const& ctr : vctr ) self.add_ctr( type, ctr ); },
   "add constraints"
 )
 .def_property_readonly(
   "all_objective",
   []( NLPSLV const& self ){ return self.obj(); },
   py::return_value_policy::reference_internal,
   "current model objectives"
 )
 .def(
   "reset_objective",
   []( NLPSLV& self ){ self.reset_ctr(); },
   "reset model objectives"
 )
 .def(
   "set_objective",
   []( NLPSLV& self, NLPSLV::t_OBJ const& type, mc::FFVar const& obj ){ self.set_obj( type, obj ); },
   "set objective"
 )
 .def(
   "setup",
   []( NLPSLV& self ){ self.setup(); },
   "setup optimization model"
 )
 .def(
   "solve",
   []( NLPSLV& self, size_t const& NSAM, std::vector<double> const& pval )
     { return self.solve( NSAM, nullptr, nullptr, pval.data() ); },
   py::arg("nsam"),
   py::arg("par")=std::vector<double>(),
   "solve optimization model using multistart local search"
 )
 .def(
   "solve",
   []( NLPSLV& self, std::vector<double> const& xini, std::vector<double> const& pval )
     { return self.solve( xini.data(), nullptr, nullptr, pval.data() ); },
   py::arg("ini")=std::vector<double>(),
   py::arg("par")=std::vector<double>(),
   "solve optimization model using local search"
 )
 .def_property_readonly(
   "solution",
   []( NLPSLV const& self ){ return self.solution(); },
   py::return_value_policy::reference_internal,
   "current solution of optimization model"
 )
 .def(
   "is_feasible",
   []( NLPSLV& self, double const& tol ){ return self.is_feasible( tol ); },
   py::return_value_policy::reference_internal,
   "test feasibility of optimization model solution"
 )
 .def(
   "is_stationary",
   []( NLPSLV& self, double const& tol ){ return self.is_stationary( tol ); },
   py::return_value_policy::reference_internal,
   "test stationarity of optimization model solution"
 )
 .def_property_readonly(
   "status",
   []( NLPSLV const& self ){ return self.get_status(); },
   py::return_value_policy::reference_internal,
   "status of optimization model solution"
 )
;

py::enum_<NLPSLV::STATUS>(pyNLPSLV, "NLPSLV.STATUS")
 .value("SUCCESSFUL", NLPSLV::STATUS::SUCCESSFUL, "optimal solution found (possibly not within required accuracy)" )
 .value("INFEASIBLE", NLPSLV::STATUS::INFEASIBLE, "model appears to be infeasible" )
 .value("UNBOUNDED", NLPSLV::STATUS::UNBOUNDED, "model appears to be unbounded" )
 .value("INTERRUPTED", NLPSLV::STATUS::INTERRUPTED, "resource limit reached" )
 .value("FAILURE", NLPSLV::STATUS::FAILURE, "search terminated after numerical difficulties" )
 .value("ABORTED", NLPSLV::STATUS::ABORTED, "critical error encountered" )
 .export_values()
;

py::enum_<NLPSLV::t_OBJ>(pyNLPSLV, "NLPSLV.OBJ")
 .value("MIN", NLPSLV::t_OBJ::MIN, "minimization" )
 .value("MAX", NLPSLV::t_OBJ::MAX, "maximization" )
 .export_values()
;

py::enum_<NLPSLV::t_CTR>(pyNLPSLV, "NLPSLV.CTR")
 .value("EQ", NLPSLV::t_CTR::EQ, "equal-to-zero constraint" )
 .value("LE", NLPSLV::t_CTR::LE, "less-than-or-equal-to-zero constraint" )
 .value("GE", NLPSLV::t_CTR::GE, "greater-than-or-equal-to-zero constraint" )
 .export_values()
;

py::class_<mc::SOLUTION_OPT> pyNLPSLVSol( pyNLPSLV, "NLPSLV.Solution" );
pyNLPSLVSol
 .def_readonly( "status", &mc::SOLUTION_OPT::stat, "optimization solver status" )
 .def_readonly( "x",      &mc::SOLUTION_OPT::x,    "variable values" )
 .def_readonly( "ux",     &mc::SOLUTION_OPT::ux,   "variable bound multipliers" )
 .def_readonly( "f",      &mc::SOLUTION_OPT::f,    "function values" )
 .def_readonly( "uf",     &mc::SOLUTION_OPT::uf,   "function multipliers" )
 .def( "__str__",
       []( mc::SOLUTION_OPT& self ){
         std::ostringstream ss;
         ss << self;
         return ss.str();
       }
 )
 .def( "__repr__",
       []( mc::SOLUTION_OPT& self ){
         std::ostringstream ss;
         ss << self;
         return ss.str();
       }
 )
;

py::class_<NLPSLV::Options> pyNLPSLVOptions( pyNLPSLV, "NLPSLV.Options" );

#ifdef MC__USE_SNOPT
pyNLPSLVOptions
 .def( py::init<>() )
 .def( py::init<NLPSLV::Options const&>() )
 .def_readwrite( "FEASTOL",   &NLPSLV::Options::FEASTOL,   "Corresponds to 'Major feasibility tolerance' in snOptA, which specifies how accurately the nonlinear constraints should be satisfied [Default: 1e-7]" )
 .def_readwrite( "OPTIMTOL",  &NLPSLV::Options::OPTIMTOL,  "Corresponds to 'Major optimality tolerance' in snOptA, which specifies the final accuracy of the dual variables [Default: 1e-5]" )
 .def_readwrite( "MAXITER",   &NLPSLV::Options::MAXITER,   "Corresponds to 'Major iterations limit' in snOptA, which is the maximum number of major iterations allowed. It is intended to guard against an excessive number of linearizations of the constraints. If non-positive value given, both feasibility and optimality are checked [Default: 200]" )
 .def_readwrite( "GRADMETH",  &NLPSLV::Options::GRADMETH,  "Specifies the method for computing derivatives, either analytically (FSYM, BSYM), computed using automatic differentiation (FAD, BAD), or estimated using finite differences (FD) [Default: FSYM]" )
 .def_readwrite( "GRADCHECK", &NLPSLV::Options::GRADCHECK, "Corresponds to 'Verify level' in snOptA, which enables finite-difference checks on the derivatives computed by the user-provided routines at the first point that satisfies all bounds and linear constraints [Default: 0]" )
 .def_readwrite( "QPFEASTOL", &NLPSLV::Options::QPFEASTOL, "Corresponds to 'Minor feasibility tolerance' in snOptA, which ensures that all linear constraints eventually satisfy their upper and lower bounds to within this tolerance [Default: 1e-7]" )
 .def_readwrite( "QPMAXITER", &NLPSLV::Options::QPMAXITER, "Corresponds to 'Minor iterations limit' in snOptA. If the number of minor iterations for the optimality phase of the QP subproblem exceeds this value, then all nonbasic QP variables that have not yet moved are frozen at their current values and the reduced QP is solved to optimality [Default: 500]" )
 .def_readwrite( "QPMETH",    &NLPSLV::Options::QPMETH,    "Corresponds to 'QPSolver' in snOptA, which specifies the method used to solve the QP subproblems: Cholesky QP solver (CHOL), conjugate-gradient QP solver (CG), quasi-Newton QP solver (QN) [Default: CHOL]" )
 .def_readwrite( "FEASPB",    &NLPSLV::Options::FEASPB,    "Corresponds to 'Feasible point' in snOptA, which specifies to “Ignore the objective function” while finding afeasible point for the linear and nonlinear constraints [Default: 0]" )
 .def_readwrite( "DISPLEVEL",  &NLPSLV::Options::DISPLEVEL,  "Corresponds to 'Print file' in snOptA, which specifies the file name for the 'Summary file'. Displays to screen if an empty string is passed [Default: -]" )
 .def_readwrite( "LOGFILE",  &NLPSLV::Options::LOGFILE,  "Corresponds to 'Summary file' in snOptA, which specifies whether (>0) or not (<=0) to generate the summary file [Default: 0]" )
 .def_readwrite( "TIMELIMIT", &NLPSLV::Options::TIMELIMIT, "Maximum run-time (in seconds) - this is checked externally to snOptA based on the wall clock [Default: 7200]" )
 .def_readwrite( "MAXTHREAD", &NLPSLV::Options::MAXTHREAD, "Maximum number of threads for multistart solve [Default: 0 (all threads)]" )
;

py::enum_<NLPSLV::Options::QP_STRATEGY>(pyNLPSLVOptions, "NLPSLV.QP_STRATEGY")
 .value("CHOL", NLPSLV::Options::QP_STRATEGY::CHOL, "Cholesky QP solver")
 .value("CG",   NLPSLV::Options::QP_STRATEGY::CG,   "Conjugate-gradient QP solver")
 .value("QN",   NLPSLV::Options::QP_STRATEGY::QN,   "Quasi-Newton QP solver")
 .export_values()
;

#elif  MC__USE_IPOPT
pyNLPSLVOptions
 .def( py::init<>() )
 .def( py::init<NLPSLV::Options const&>() )
 .def_readwrite( "FEASTOL",   &NLPSLV::Options::FEASTOL,   "Corresponds to 'constr_viol_tol' in Ipopt, which specifies the final accuracy on the constraints [Default: 1e-7]" )
 .def_readwrite( "OPTIMTOL",  &NLPSLV::Options::OPTIMTOL,  "Corresponds to 'tol', 'dual_inf_tol' and 'compl_inf_tol' in Ipopt, which specific the final accuracy on the dual and complementarity slackness conditions [Default: 1e-5]" )
 .def_readwrite( "MAXITER",   &NLPSLV::Options::MAXITER,   "Corresponds to 'max_iter' in Ipopt, which is the maximum number of iterations [Default: 200]" )
 .def_readwrite( "GRADMETH",  &NLPSLV::Options::GRADMETH,  "Specifies the method for computing derivatives, either analytically (FSYM, BSYM), computed using automatic differentiation (FAD, BAD), or estimated using finite differences (FD) [Default: FSYM]" )
 .def_readwrite( "HESSMETH",  &NLPSLV::Options::HESSMETH,  "Corresponds to 'hessian_approximation' in  Ipopt, which specifies the method for computing the Hessian, either exactly via symbolic or automatic differentiation (EXACT) or using a limited-memory BFGS update (LBFGS) [Default: LBFGS]" )
 .def_readwrite( "LINMETH",   &NLPSLV::Options::LINMETH,   "Corresponds to 'linear_solver' in Ipopt, which specifies the linear solver [Default: MA57]" )
 .def_readwrite( "GRADCHECK", &NLPSLV::Options::GRADCHECK, "Corresponds to 'derivative-test' in Ipopt, which enables finite-difference checks on the derivatives computed by the user-provided routines, both first- and second-order derivatives [Default: 0]" )
 .def_readwrite( "DISPLEVEL", &NLPSLV::Options::DISPLEVEL, "Corresponds to 'print_level' in Ipopt, which specifies the verbosity level of the solver between 0 (no output) and 12 (maximum verbosity) [Default: 0]" )
 .def_readwrite( "TIMELIMIT", &NLPSLV::Options::TIMELIMIT, "Maximum run-time (in seconds) - this is checked both internally (option 'max_cpu_time' in Ipopt) and externally based on the wall clock [Default: 7200]" )
 .def_readwrite( "MAXTHREAD", &NLPSLV::Options::MAXTHREAD, "Maximum number of threads for multistart solve [Default: 0 (all threads)]" )
;

py::enum_<NLPSLV::Options::HESSIAN_STRATEGY>(pyNLPSLVOptions, "NLPSLV.HESSIAN_STRATEGY")
 .value("EXACT", NLPSLV::Options::HESSIAN_STRATEGY::EXACT, "Exact second derivatives from AD")
 .value("LBFGS", NLPSLV::Options::HESSIAN_STRATEGY::LBFGS, "Limited-memory quasi-Newton approximation")
 .export_values()
;

py::enum_<NLPSLV::Options::LINEAR_SOLVER>(pyNLPSLVOptions, "NLPSLV.LINEAR_SOLVER")
 .value("MA27",    NLPSLV::Options::LINEAR_SOLVER::MA27,    "Harwell routine MA27")
 .value("MA57",    NLPSLV::Options::LINEAR_SOLVER::MA57,    "Harwell routine MA57")
 .value("MA77",    NLPSLV::Options::LINEAR_SOLVER::MA77,    "Harwell routine HSL_MA77")
 .value("MA86",    NLPSLV::Options::LINEAR_SOLVER::MA86,    "Harwell routine HSL_MA86")
 .value("MA97",    NLPSLV::Options::LINEAR_SOLVER::MA97,    "Harwell routine HSL_MA97")
 .value("PARDISO", NLPSLV::Options::LINEAR_SOLVER::PARDISO, "Pardiso package")
 .value("WSMP",    NLPSLV::Options::LINEAR_SOLVER::WSMP,    "WSMP package")
 .value("MUMPS",   NLPSLV::Options::LINEAR_SOLVER::MUMPS,   "MUMPS package")
 .export_values()
;
#endif

py::enum_<NLPSLV::Options::GRADIENT_STRATEGY>(pyNLPSLVOptions, "NLPSLV.GRADIENT_STRATEGY")
 .value("FSYM", NLPSLV::Options::GRADIENT_STRATEGY::FSYM, "Forward symbolic AD")
 .value("BSYM", NLPSLV::Options::GRADIENT_STRATEGY::BSYM, "Backward symbolic AD")
 .value("FAD",  NLPSLV::Options::GRADIENT_STRATEGY::FAD,  "Forward numeric AD")
 .value("BAD",  NLPSLV::Options::GRADIENT_STRATEGY::BAD,  "Backward numeric AD")
 .value("FD",   NLPSLV::Options::GRADIENT_STRATEGY::FD,   "Finite differences")
 .export_values()
;
}


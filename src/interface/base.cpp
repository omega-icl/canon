#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base_nlp.hpp"

namespace py = pybind11;

void mc_base( py::module_ &m )
{

typedef mc::BASE_OPT BASEOPT;

py::class_<BASEOPT> pyBASEOPT( m, "BASEOPT" );

pyBASEOPT
 .def(
   py::init<>()
 )
 .def_readwrite_static(
   "INF",
   &BASEOPT::INF
 )
;

py::enum_<BASEOPT::t_OBJ>(pyBASEOPT, "OBJ")
 .value("MIN", BASEOPT::t_OBJ::MIN, "minimization" )
 .value("MAX", BASEOPT::t_OBJ::MAX, "maximization" )
 .export_values()
;

py::enum_<BASEOPT::t_CTR>(pyBASEOPT, "CTR")
 .value("EQ", BASEOPT::t_CTR::EQ, "equal-to-zero constraint" )
 .value("LE", BASEOPT::t_CTR::LE, "less-than-or-equal-to-zero constraint" )
 .value("GE", BASEOPT::t_CTR::GE, "greater-than-or-equal-to-zero constraint" )
 .export_values()
;

py::class_<mc::SOLUTION_OPT> pyBASESOL( m, "Solution" );

pyBASESOL
 .def_readonly( "status", &mc::SOLUTION_OPT::stat, "optimization solver status" )
 .def_readonly( "p",      &mc::SOLUTION_OPT::p,    "parameter values" )
 .def_readonly( "x",      &mc::SOLUTION_OPT::x,    "decision values" )
 .def_readonly( "ux",     &mc::SOLUTION_OPT::ux,   "decision bound multipliers" )
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

typedef mc::BASE_NLP BASE;

py::class_<BASE,BASEOPT> pyBASE( m, "BASE" );

pyBASE
 .def(
   py::init<>()
 )
 .def( 
   "set",
   []( BASE& self,  BASE& nlp ){ self.set( nlp ); },
   py::keep_alive<1,2>(),
   "copy optimization model"
 )
 .def( 
   "reset",
   []( BASE& self ){ self.reset(); },
   "reset optimization model"
 )
 .def( 
   "set_dag",
   []( BASE& self,  mc::FFGraph* dag ){ self.set_dag( dag ); },
   py::keep_alive<1,2>(),
   "set DAG"
 )
 .def(
   "dag",
   []( BASE& self ){ return self.dag(); },
   py::return_value_policy::reference_internal,
   "get DAG"
 )
 .def(
   "reset_parameter",
   []( BASE& self ){ self.reset_par(); },
   "reset parameters"
 )
 .def(
   "set_parameter",
   []( BASE& self, std::vector<mc::FFVar> const& par ){ self.set_par( par ); },
   py::arg("par"),
   "set parameters"
 )
 .def(
   "add_parameter",
   []( BASE& self, std::vector<mc::FFVar> const& par ){ self.add_par( par ); },
   py::arg("par"),
   "add parameters"
 )
 .def(
   "reset_variable",
   []( BASE& self ){ self.reset_var(); },
   "reset decision variables"
 )
 .def_property_readonly(
   "var_parameter",
   []( BASE const& self ){ return self.par(); },
   py::return_value_policy::reference_internal,
   "parameters"
 )
 .def(
   "set_decision",
   []( BASE& self, std::vector<mc::FFVar> const& var, std::vector<double> const& lb,
       std::vector<double> const& ub, std::vector<unsigned> const& typ ){ self.set_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=std::vector<double>(),
   py::arg("ub")=std::vector<double>(),
   py::arg("typ")=std::vector<unsigned>(),
   "set decision variables"
 )
 .def(
   "set_decision",
   []( BASE& self, std::vector<mc::FFVar> const& var, double const& lb,
       double const& ub, unsigned const& typ ){ self.set_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=-BASE::INF,
   py::arg("ub")=BASE::INF,
   py::arg("typ")=0,
   "set decision variables"
 )
 .def(
   "add_decision",
   []( BASE& self, std::vector<mc::FFVar> const& var, std::vector<double> const& lb,
       std::vector<double> const& ub, std::vector<unsigned> const& typ ){ self.add_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=std::vector<double>(),
   py::arg("ub")=std::vector<double>(),
   py::arg("typ")=std::vector<unsigned>(),
   "add decision variables"
 )
 .def(
   "add_decision",
   []( BASE& self, std::vector<mc::FFVar> const& var, double const& lb,
       double const& ub, unsigned const& typ ){ self.add_var( var, lb, ub, typ ); },
   py::arg("var"),
   py::arg("lb")=-BASE::INF,
   py::arg("ub")=BASE::INF,
   py::arg("typ")=0,
   "add decision variables"
 )
 .def_property_readonly(
   "var_decision",
   []( BASE const& self ){ return self.var(); },
   py::return_value_policy::reference_internal,
   "decision variables"
 )
 .def_property_readonly(
   "typ_decision",
   []( BASE const& self ){ return self.vartyp(); },
   py::return_value_policy::reference_internal,
   "decision variable types"
 )
 .def_property_readonly(
   "lb_decision",
   []( BASE const& self ){ return self.varlb(); },
   py::return_value_policy::reference_internal,
   "decision variable lower bounds"
 )
 .def_property_readonly(
   "ub_decision",
   []( BASE const& self ){ return self.varub(); },
   py::return_value_policy::reference_internal,
   "decision variable upper bounds"
 )
 .def_property_readonly(
   "var_constraint",
   []( BASE const& self ){ return self.ctr(); },
   py::return_value_policy::reference_internal,
   "model constraints"
 )
 .def(
   "reset_constraint",
   []( BASE& self ){ self.reset_ctr(); },
   "reset model constraints"
 )
 .def(
   "add_constraint",
   []( BASE& self, BASE::t_CTR const& type, mc::FFVar const& ctr ){ self.add_ctr( type, ctr ); },
   "add constraint"
 )
 .def(
   "add_constraint",
   []( BASE& self, BASE::t_CTR const& type, std::vector<mc::FFVar> const& vctr ){ for( auto const& ctr : vctr ) self.add_ctr( type, ctr ); },
   "add constraints"
 )
 .def_property_readonly(
   "var_objective",
   []( BASE const& self ){ return self.obj(); },
   py::return_value_policy::reference_internal,
   "model objectives"
 )
 .def(
   "reset_objective",
   []( BASE& self ){ self.reset_ctr(); },
   "reset model objectives"
 )
 .def(
   "set_objective",
   []( BASE& self, BASE::t_OBJ const& type, mc::FFVar const& obj ){ self.set_obj( type, obj ); },
   "set objective"
 )
;

}


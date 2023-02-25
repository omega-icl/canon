#include "minlpbnd.hpp"
#include "interval.hpp"
typedef mc::Interval I;
   
int main()
{
  mc::MINLPBND<mc::FFGraph<>,I> MINLP;

/*
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );
  MINLP.add_var( P[0], 1, 20, 0 );
  MINLP.add_var( P[1], 1, 20, 1 );
  MINLP.set_obj( mc::BASE_OPT::MIN, -6*P[0]-P[1] );
  MINLP.add_ctr( mc::BASE_OPT::LE, 0.3*pow(P[0]-8,2)+0.04*pow(P[1]-6,4)+0.1*exp(2*P[0])/pow(P[1],4)-56 );
  MINLP.add_ctr( mc::BASE_OPT::LE, 1/P[0]+1/P[1]-sqrt(P[0])*sqrt(P[1])+4 );
  MINLP.add_ctr( mc::BASE_OPT::LE, 2*P[0]-5*P[1]+1 );


  mc::FFGraph DAG;
  const unsigned NP = 4; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, 1, 5, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_OPT::MIN, (P[0]*P[3])*(P[0]+P[1]+P[2])+P[2] ); // objective
  MINLP.add_ctr( mc::BASE_OPT::GE,  (P[0]*P[3])*P[1]*P[2]-25 );          // constraints
  MINLP.add_ctr( mc::BASE_OPT::EQ,  sqr(P[0])+sqr(P[1])+sqr(P[2])+sqr(P[3])-40 );

  // Solving for a MIP relaxation in Gurobi without reformulation of the objective or constraints
  MINLP.options.RELAXMETH          = { MINLP.options.ISM };
  MINLP.options.ISMDIV             = 50;
  MINLP.options.ISMCONT            = false;
  MINLP.options.REFORMMETH         = {};
  MINLP.options.MIPSLV.DISPLEVEL   = 1;
  MINLP.options.MIPSLV.OUTPUTFILE  = "doxydoc1.lp";
*/

  std::string gamsfile( "tuncphd_30.gms"); 
  if( !MINLP.read( gamsfile, true ) ){
    std::cerr << "# Exit: Error reading GAMS file " << gamsfile << std::endl;
    return -1;
  }


  MINLP.setup();
  switch( MINLP.relax() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl
                <<"MINLP relaxation bound: " << MINLP.solver()->get_objective() << std::endl;
      //for( unsigned i=0; i<NP; i++ ) 
      //  std::cout << "  " << P[i] << " = " << MINLP.solver()->get_variable( P[i] ) << std::endl;
      MINLP.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }
  
  //return 0;

  // Solving for a MIP relaxation in Gurobi without reformulation of the objective or constraints
  MINLP.options.RELAXMETH          = { MINLP.options.DRL };
  MINLP.options.REFORMMETH         = {};
  MINLP.options.MIPSLV.DISPLEVEL   = 1;
  MINLP.options.MIPSLV.OUTPUTFILE  = "doxydoc2.lp";

  MINLP.setup();
  switch( MINLP.relax() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl
                <<"MINLP relaxation bound: " << MINLP.solver()->get_objective() << std::endl;
      //for( unsigned i=0; i<NP; i++ ) 
      //  std::cout << "  " << P[i] << " = " << MINLP.solver()->get_variable( P[i] ) << std::endl;
      MINLP.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }
  
  //return 0;

  // Solving for a nonconvex MIQCP relaxation in Gurobi after lifting of non-polynomial terms and quadratization of polynomials
  // in the objective and constraints
  MINLP.options.RELAXMETH          = { MINLP.options.SCDRL };
  MINLP.options.REFORMMETH         = {};// MINLP.options.NPOL };
  MINLP.options.LINCTRSEP          = true;
  MINLP.options.CMODPROP           = 4;
  MINLP.options.MIPSLV.OUTPUTFILE  = "doxydoc3.lp";

  MINLP.setup();
  switch( MINLP.relax() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl
                <<"MINLP relaxation bound: " << MINLP.solver()->get_objective() << std::endl;
      //for( unsigned i=0; i<NP; i++ ) 
      //  std::cout << "  " << P[i] << " = " << MINLP.solver()->get_variable( P[i] ) << std::endl;
      MINLP.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }

  //return 0;

  // Solving for a nonconvex MIQCP relaxation in Gurobi after lifting of non-polynomial terms and quadratization of polynomials
  // in the objective and constraints
  MINLP.options.RELAXMETH          = { MINLP.options.DRL };
  //MINLP.options.RELAXMETH          = { MINLP.options.SCQ };
  MINLP.options.REFORMMETH         = { MINLP.options.NPOL, MINLP.options.QUAD };
  MINLP.options.LINCTRSEP          = true;
  MINLP.options.POLIMG.RELAX_QUAD  = 0;
  MINLP.options.POLIMG.RELAX_NLIN  = 0;
  MINLP.options.POLIMG.RELAX_MONOM = 1;
  MINLP.options.MIPSLV.PWLRELGAP   = 1e-6;
  MINLP.options.MIPSLV.FUNCMAXVAL  = 1e12;
  MINLP.options.MIPSLV.OUTPUTFILE  = "doxydoc.lp";
  MINLP.options.MIPQUADCUTS        = true;
  MINLP.options.SQUAD.MIPFIXEDBASIS = false;

  MINLP.setup();
  switch( MINLP.relax() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl << std::scientific << std::setprecision(5)
                <<"MINLP relaxation bound: " << MINLP.solver()->get_objective() << std::endl;
      //for( unsigned i=0; i<NP; i++ ) 
      //  std::cout << "  " << P[i] << " = " << MINLP.solver()->get_variable( P[i] ) << std::endl;
      MINLP.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }


  // Exporting the MIQCP relaxation to a GAMS model
  MINLP.relax( 0, 0, 0, 0, 1, 1, "doxydoc.gms" );
  
  return 0;
}

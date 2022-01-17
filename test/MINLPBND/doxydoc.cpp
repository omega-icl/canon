#include "minlpbnd.hpp"
#include "interval.hpp"
typedef mc::Interval I;
   
int main()
{
  mc::MINLPBND<I> MINLP;

  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );
  MINLP.add_var( P[0], 1, 20, 0 );
  MINLP.add_var( P[1], 1, 20, 1 );
  MINLP.set_obj( mc::BASE_NLP::MIN, -6*P[0]-P[1] );
  MINLP.add_ctr( mc::BASE_NLP::LE, 0.3*pow(P[0]-8,2)+0.04*pow(P[1]-6,4)+0.1*exp(2*P[0])/pow(P[1],4)-56 );
  MINLP.add_ctr( mc::BASE_NLP::LE, 1/P[0]+1/P[1]-sqrt(P[0])*sqrt(P[1])+4 );
  MINLP.add_ctr( mc::BASE_NLP::LE, 2*P[0]-5*P[1]+1 );
  MINLP.setup();

  MINLP.options.REFORMMETH         = {};//{ MINLP.options.NPOL };//, MINLP.options.QUAD };
  MINLP.options.POLIMG.RELAX_QUAD  = 0;
  MINLP.options.POLIMG.RELAX_NLIN  = 0;
  MINLP.options.POLIMG.RELAX_MONOM = 1;
  MINLP.options.MIPSLV.DISPLEVEL   = 1;
  MINLP.options.MIPSLV.OUTPUTFILE  = "doxydoc.lp";
  switch( MINLP.relax() ){
    case mc::MIPSLV_GUROBI<I>::OPTIMAL:
      std::cout << std::endl
                <<"MINLP relaxation bound: " << MINLP.solver()->get_objective() << std::endl;
      for( unsigned i=0; i<NP; i++ ) 
        std::cout << "  " << P[i] << " = " << MINLP.solver()->get_variable( P[i] ) << std::endl;
      MINLP.stats.display();
      break;
    default:
      std::cout << "MINLP relaxation was unsuccessful" << std::endl;
      break;
  }

  return 0;
}

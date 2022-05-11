#include "minlgo.hpp"

int main()
{
  mc::FFGraph DAG;
  const unsigned NP = 4; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  mc::MINLGO MINLP;

  MINLP.set_dag( &DAG );  // DAG
  MINLP.set_var( NP, P, 1, 5, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_NLP::MIN, (P[0]*P[3])*(P[0]+P[1]+P[2])+P[2] ); // objective
  MINLP.add_ctr( mc::BASE_NLP::GE,  (P[0]*P[3])*P[1]*P[2]-25 );          // constraints
  MINLP.add_ctr( mc::BASE_NLP::EQ,  sqr(P[0])+sqr(P[1])+sqr(P[2])+sqr(P[3])-40 );

  std::cout << MINLP;

  // With default options: quadratisation and MIQCQP solution
  MINLP.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  MINLP.setup();
  MINLP.presolve();
  MINLP.optimize();
  MINLP.stats.display();

  // With modified options: hierarchy of piecewise-linear relaxation and MIP solution
  MINLP.options.MAXITER                     = 10;
  MINLP.options.MINLPBND.MIPSLV.DISPLEVEL   = 0;
  MINLP.options.MINLPBND.POLIMG.RELAX_QUAD  = true;
  MINLP.setup();
  MINLP.presolve();
  MINLP.optimize();
  MINLP.stats.display();

  return 0;
}


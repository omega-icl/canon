#include "minlpbnd.hpp"
#include "interval.hpp"
typedef mc::Interval I;
   
int main()
{
  mc::MINLPBND<mc::FFGraph<>,I> MINLP;

  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  MINLP.set_dag( &DAG );
  MINLP.set_var( NP, P, 1, 5, 0 ); // decision variables
  MINLP.set_obj( mc::BASE_OPT::MIN, -6*P[0]-P[1] );
  MINLP.add_ctr( mc::BASE_OPT::EQ, 1/P[0]+1/P[1]-1 );

  // Solving for full-space model after lifting of non-polynomial terms and quadratization of polynomials
  MINLP.options.REFORMMETH         = { MINLP.options.NPOL, MINLP.options.QUAD };

  // Exporting the reduced-space model to GAMS
  MINLP.setup();
  MINLP.write( "doxydoc2a.gms" );

  // Solving for reduced-space model after eliminating variables using invertible equality constraints
  MINLP.options.REFORMMETH         = { MINLP.options.ELIM };

  // Exporting the reduced-space model to GAMS
  MINLP.setup();
  MINLP.write( "doxydoc2b.gms" );

  return 0;
}

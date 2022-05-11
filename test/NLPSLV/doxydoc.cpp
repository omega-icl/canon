#include "nlpslv_snopt.hpp"
  
int main()
{
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

  mc::NLPSLV_SNOPT NLP;
  NLP.set_dag( &DAG );                       // DAG
  NLP.add_var( P[0], 0, 6 );
  NLP.add_var( P[1], 0, 4 );
  NLP.set_obj( mc::BASE_NLP::MAX, P[0]+P[1] );   // objective
  NLP.add_ctr( mc::BASE_NLP::LE, P[0]*P[1]-4. ); // constraints

  NLP.options.DISPLEVEL = 0;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;

  NLP.setup();
  double p0[NP] = { 5., 1. };
  NLP.solve( p0 );
  std::cout << NLP.solution();
  
  return 0;
}

#undef MC__MINLPSLV_DEBUG
#include "minlpslv.hpp"

void
nearest
( unsigned const n, unsigned const* typ, double* val )
{
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = round( val[i] );
  }
}
  
int
main
()
{
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );
/*
  mc::NLPSLV_SNOPT NLP;
  NLP.set_dag( &DAG );
  NLP.add_var( P[0], 1, 20, 0 );
  NLP.add_var( P[1], 1, 20, 1 );
  NLP.set_obj( mc::BASE_OPT::MIN, -6*P[0]-P[1] );
  NLP.add_ctr( mc::BASE_OPT::LE, 0.3*pow(P[0]-8,2)+0.04*pow(P[1]-6,4)+0.1*exp(2*P[0])/pow(P[1],4)-56 );
  NLP.add_ctr( mc::BASE_OPT::LE, 1/P[0]+1/P[1]-sqrt(P[0])*sqrt(P[1])+4 );
  NLP.add_ctr( mc::BASE_OPT::LE, 2*P[0]-5*P[1]+1 );

  NLP.options.DISPLEVEL               = 1;
  NLP.setup();
  NLP.solve();
*/
  mc::MINLPSLV MINLP;
  MINLP.set_dag( &DAG );
  MINLP.add_var( P[0], 1, 20, 0 );
  MINLP.add_var( P[1], 1, 20, 1 );
  MINLP.set_obj( mc::BASE_OPT::MIN, -6*P[0]-P[1] );
  MINLP.add_ctr( mc::BASE_OPT::LE, 0.3*pow(P[0]-8,2)+0.04*pow(P[1]-6,4)+0.1*exp(2*P[0])/pow(P[1],4)-56 );
  MINLP.add_ctr( mc::BASE_OPT::LE, 1/P[0]+1/P[1]-sqrt(P[0])*sqrt(P[1])+4 );
  MINLP.add_ctr( mc::BASE_OPT::LE, 2*P[0]-5*P[1]+1 );

  MINLP.options.LINMETH                 = mc::MINLPSLV<>::Options::CVX;
  MINLP.options.CVRTOL                  =
  MINLP.options.CVATOL                  = 1e-5;
  MINLP.options.DISPLEVEL               = 1;
//  MINLP.options.FEASPUMP                = true;
//  MINLP.options.INCCUT                  = true;
  MINLP.options.MSLOC                   = 4;
  MINLP.options.NLPSLV.DISPLEVEL        = 0;
  MINLP.options.NLPSLV.GRADCHECK        = false;
  MINLP.options.NLPSLV.MAXTHREAD        = 0;
  MINLP.options.MIPSLV.DISPLEVEL        = 0;

  std::cout << MINLP;

  MINLP.setup();
  //MINLP.optimize();
  MINLP.optimize( nullptr, nullptr, nearest );
  //MINLP.stats.display();

  return 0;
}

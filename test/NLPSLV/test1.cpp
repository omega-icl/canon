#define USE_PROFIL
#include <fstream>
#include <iomanip>

#ifdef MC__USE_SNOPT
  #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
  #include "nlpslv_ipopt.hpp"
#endif

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  mc::FFGraph DAG;
  const unsigned NP = 2; mc::FFVar P[NP];
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );
  mc::FFVar RHS( &DAG );

  // Local optimization
#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT NLP;
  NLP.options.DISPLEVEL = 0;
  NLP.options.MAXITER   = 100;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_SNOPT<>::Options::BSYM;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 0;
#else
  mc::NLPSLV_IPOPT NLP;
  NLP.options.DISPLEVEL = 5;
  NLP.options.MAXITER   = 100;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_IPOPT<>::Options::BAD;//BSYM;
  NLP.options.HESSMETH  = mc::NLPSLV_IPOPT<>::Options::EXACT;//LBFGS;//
  NLP.options.GRADCHECK = true;
  NLP.options.MAXTHREAD = 0;
#endif
  NLP.set_dag( &DAG );                            // DAG
  NLP.add_var( P[0], 0., 6. );                    // decision variables
  NLP.add_var( P[1], 0., 4. );
  //NLP.add_par( RHS, 4. );                         // parameters
  NLP.set_obj( mc::BASE_OPT::MAX, P[0]+sqr(P[1]) );    // objective
  //NLP.add_ctr( mc::BASE_OPT::LE, P[0]*P[1]-RHS ); // constraints
  NLP.add_ctr( mc::BASE_OPT::LE, P[0]*P[1]-4 ); // constraints
  NLP.setup();

  double p0[NP] = { 5., 1. };
  //double p0[NP] = { 1., 5. };

  NLP.solve( p0 ); //, Ip );
  //NLP.solve( 10 );//p0 ); //, Ip );
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  NLP.set_obj_lazy( mc::BASE_OPT::MAX, P[0]*P[1] );    // new objective
  NLP.solve( p0 ); //, Ip );
  //NLP.solve( 10 );//, p0 ); //, Ip );
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  NLP.add_ctr_lazy( mc::BASE_OPT::LE, P[0]-P[1] );    // new constraint
  NLP.solve( p0 ); //, Ip );
  //NLP.solve( 10 );//, p0 ); //, Ip );
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  NLP.restore_model();    // Back to original model
  NLP.solve( p0 ); //, Ip );
  //NLP.solve( 10 );//, p0 ); //, Ip );
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  return 0;
}

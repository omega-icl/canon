#include <fstream>
#include <iomanip>
#include "interval.hpp"
#ifdef MC__USE_SNOPT
  #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
  #include "nlpslv_ipopt.hpp"
#endif

// Kun He, Menglong Huangn, Chenkai Yang
// An action-space-based global optimization algorithm for packing circles into a square container
// Computers & Operations Research 58 (2015) 67–74
// http://dx.doi.org/10.1016/j.cor.2014.12.010

const unsigned N = 15;  // Number of circles

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  mc::FFGraph DAG;
  const unsigned NP = 1+2*N;
  mc::FFVar P[NP], &L = P[0], *X = P+1, *Y = X+N;
  for( unsigned i=0; i<NP; i++ ) P[i].set( &DAG );

#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT *NLP = new mc::NLPSLV_SNOPT;
  NLP->options.DISPLEVEL = 1;
  NLP->options.MAXITER   = 200;
  NLP->options.FEASTOL   = 1e-8;
  NLP->options.OPTIMTOL  = 1e-8;
  NLP->options.GRADMETH  = mc::NLPSLV_SNOPT::Options::FAD;
  NLP->options.GRADCHECK = false;
  NLP->options.MAXTHREAD = 8;
#else
  mc::NLPSLV_IPOPT *NLP = new mc::NLPSLV_IPOPT;
  NLP->options.DISPLEVEL = 5;
  NLP->options.MAXITER   = 200;
  NLP->options.FEASTOL   = 1e-8;
  NLP->options.OPTIMTOL  = 1e-8;
  NLP->options.GRADMETH  = mc::NLPSLV_IPOPT::Options::FAD;
  NLP->options.GRADCHECK = false;
  NLP->options.MAXTHREAD = 8;
#endif

  double R[N], LMAX=0.;
  for( unsigned i=0; i<N; i++ ){
    R[i] = (double)i+1;
//    R[i] = std::sqrt(i+1);
    LMAX += R[i];
  }
  
  NLP->set_dag( &DAG );
//  NLP->set_var( NP, P );
  NLP->add_var( L, 0, LMAX );
  NLP->add_var( N, X, -LMAX, LMAX );
  NLP->add_var( N, Y, -LMAX, LMAX );

  NLP->set_obj( mc::BASE_NLP::MIN, L );
  for( unsigned i=0; i<N; i++ ){
    for( unsigned j=i+1; j<N; j++ )
      NLP->add_ctr( mc::BASE_NLP::GE, mc::sqr(X[i]-X[j]) + mc::sqr(Y[i]-Y[j]) - mc::sqr(R[i]+R[j]) );
    NLP->add_ctr( mc::BASE_NLP::LE, X[i] + R[i] - 0.5*L );
    NLP->add_ctr( mc::BASE_NLP::GE, X[i] - R[i] + 0.5*L );
    NLP->add_ctr( mc::BASE_NLP::LE, Y[i] + R[i] - 0.5*L );
    NLP->add_ctr( mc::BASE_NLP::GE, Y[i] - R[i] + 0.5*L );
  }
//  NLP->add_ctr( mc::BASE_NLP::LE, X[0] - X[1] );
//  NLP->add_ctr( mc::BASE_NLP::LE, Y[0] - Y[1] );
  NLP->setup();

//  typedef mc::Interval I;
//  I IP[NP], &IL = IP[0], *IX = IP+1, *IY = IX+N;
//  IL = I( 0, LMAX );
//  for( unsigned i=0; i<N; i++ ){
//    IX[i] = IY[i] = I( -LMAX, LMAX );
//  }

  NLP->solve();

  std::cout << "NLP LOCAL SOLUTION:\n" << NLP->solution();
  std::cout << "FEASIBLE:   " << NLP->is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP->is_stationary( 1e-7 ) << std::endl;

  NLP->options.DISPLEVEL = 0;
  for( NLP->options.MAXTHREAD = 1; NLP->options.MAXTHREAD <= 8; NLP->options.MAXTHREAD++ ){
    double tStart = mc::userclock();
    NLP->solve( 1000 );
    std::cout << "MULTISTART ON " << NLP->options.MAXTHREAD << " THREADS: " << mc::userclock()-tStart << " CPU-sec\n";
  }

  std::cout << "NLP LOCAL SOLUTION: " << NLP->solution();
  std::cout << "FEASIBLE:   " << NLP->is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP->is_stationary( 1e-7 ) << std::endl;

  delete NLP;
  return 0;
}

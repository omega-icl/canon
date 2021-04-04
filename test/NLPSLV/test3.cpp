#include <fstream>
#include <iomanip>
#include "interval.hpp"
#ifdef MC__USE_SNOPT
  #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
  #include "nlpslv_ipopt.hpp"
#endif

// Predrag Milosavljevic, Alejandro G Marchetti, Andrea Cortinovis, Timm Faulwasser, Mehmet Mercangoz, Dominique Bonvin
// Real-time optimization of load sharing for gas compressors in the presence of uncertainty
// Applied Energy 272 (2020) 114883
// http://dx.doi.org/10.1016/j.apenergy.2020.114883

const double R    =  8.314e-3; // Power in MW
const double MW   =  16.04e0;
const double Zin  =  0.90e0;
const double Tin  =  293e0;
const double nv   =  1.27e0;
const double Pin  =  1.05e5;
const double Pout =  1.55e5;

const double kin  =  2.2627e0;
const double kout =  0.9410e0;
const double krec =  0.4238e0;

const double s0   =  -6.64e-1;
const double s1   =   3.81e-2;

const double c0   =   3.41e-1;
const double c1   =   9.66e-3;

const double mapA_a[6] = { 2.6909e+00, -1.3878e-02, -4.0926e-02, 9.8696e-04, -4.1858e-04,  2.4528e-05 };
const double mapA_b[6] = { 5.9193e-01, -2.1319e-03,  2.9338e-01,  2.9788e-03, -2.6030e-05, -1.1791e-01 };

const double mapB_a[6] = { 2.6909e+00*1.05, -1.3878e-02*1.05, -4.0926e-02*1.05, 9.8696e-04, -4.1858e-04,  2.4528e-05 };
const double mapB_b[6] = { 5.9193e-01, -2.1319e-03*1.05,  2.9338e-01*0.95,  2.9788e-03*1.05, -2.6030e-05*1.05, -1.1791e-01*1.05 };

const double mapC_a[6] = { 2.6909e+00, -1.3878e-02, -4.0926e-02, 9.8696e-04, -4.1858e-04*1.1,  2.4528e-05 };
const double mapC_b[6] = { 5.9193e-01*0.95, -2.1319e-03*1.05,  2.9338e-01,  2.9788e-03*0.95, -2.6030e-05, -1.1791e-01*0.95 };

const double mapM_a[6] = { 2.6909e+00*1.05, -1.3878e-02*1.05, -4.0926e-02*1.05, 9.8696e-04*1.05, -4.1858e-04*1.05,  2.4528e-05*1.05 };
const double mapM_b[6] = { 5.9193e-01*1.05, -2.1319e-03*1.05,  2.9338e-01*1.05,  2.9788e-03*1.05, -2.6030e-05*1.05, -1.1791e-01*1.05 };

const unsigned NS = 3;
std::vector<double const*> maps_a( { mapA_a, mapB_a, mapC_a } );
std::vector<double const*> maps_b( { mapA_b, mapB_b, mapC_b } );
double Mspdef = 4e2;

////////////////////////////////////////////////////////////////////////

mc::FFVar map_pi( mc::FFVar const& Mc, mc::FFVar const& W, double const* a )
{ return a[0] + a[1]*Mc + a[2]*W + a[3]*Mc*W + a[4]*sqr(Mc) + a[5]*sqr(W); }

mc::FFVar map_eta( mc::FFVar const& Mc, mc::FFVar const& PI, double const* b )
{ return b[0] + b[1]*Mc + b[2]*PI + b[3]*Mc*PI + b[4]*sqr(Mc) + b[5]*sqr(PI); }

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{

#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT *NLP = new mc::NLPSLV_SNOPT;
  NLP->options.DISPLEVEL = 1;
  NLP->options.MAXITER   = 200;
  NLP->options.FEASTOL   = 1e-8;
  NLP->options.OPTIMTOL  = 1e-8;
  NLP->options.GRADMETH  = mc::NLPSLV_SNOPT::Options::FAD;
  NLP->options.MAXTHREAD = 4;
#else
  mc::NLPSLV_IPOPT *NLP = new mc::NLPSLV_IPOPT;
  NLP->options.DISPLEVEL = 1;
  NLP->options.MAXITER   = 200;
  NLP->options.FEASTOL   = 1e-8;
  NLP->options.OPTIMTOL  = 1e-8;
  NLP->options.GRADMETH  = mc::NLPSLV_IPOPT::Options::FAD;
  NLP->options.MAXTHREAD = 4;
#endif

  mc::FFGraph DAG;
  std::vector<mc::FFVar> W(NS), Vrec(NS), Min(NS), Mout(NS), Mrec(NS), Mc(NS),
    Ps(NS), Pd(NS), PI(NS), Ep(NS), Yp(NS), P(NS);
  for( unsigned i=0; i<NS; i++ ){
    W[i].set(&DAG); Vrec[i].set(&DAG); Min[i].set(&DAG); Mout[i].set(&DAG); Mrec[i].set(&DAG); Mc[i].set(&DAG);
    Ps[i].set(&DAG); Pd[i].set(&DAG); PI[i].set(&DAG); Ep[i].set(&DAG); Yp[i].set(&DAG); P[i].set(&DAG);
  }
  mc::FFVar Msp(&DAG);

  NLP->set_dag( &DAG );
  NLP->add_var( W,    1.0e1,  1.0e2 );
  NLP->add_var( Vrec, 0.0e0,  1.0e0 );
  NLP->add_var( Min,  0.0e0,  5.0e2 );
  NLP->add_var( Mout, 0.0e0,  5.0e2 );
  NLP->add_var( Mrec, 0.0e0,  5.0e2 );
  NLP->add_var( Mc,   0.0e0,  5.0e2 );
  NLP->add_var( Ps,   1.0e5,  2.0e5 );
  NLP->add_var( Pd,   1.0e5,  1.0e6 );
  NLP->add_var( PI,   1.0e0,  1.0e1 );
  NLP->add_var( Ep,   1.0e-2, 1.0e0 );
  NLP->add_var( Yp,   0.0e0,  1.0e0 );
  NLP->add_var( P,    0.0e0,  1.0e2 );
  NLP->add_par( Msp,  Mspdef );

  NLP->set_obj( mc::BASE_NLP::MIN, sum( NS, P.data() ) );
  NLP->add_ctr( mc::BASE_NLP::EQ, sum( NS, Mout.data() ) - Msp );
  for( unsigned i=0; i<NS; i++ ){
    NLP->add_ctr( mc::BASE_NLP::LE, PI[i] - s0 - s1*Mc[i] );
    NLP->add_ctr( mc::BASE_NLP::GE, PI[i] - c0 - c1*Mc[i] );
    NLP->add_ctr( mc::BASE_NLP::EQ, Vrec[i] * ( PI[i] - s0 - s1*Mc[i] ) ); // <- complementarity constraint to force Vrec>0 only if surge constraint is active
    NLP->add_ctr( mc::BASE_NLP::EQ, sqr(Min[i]/kin) - (Pin-Ps[i]) );
    NLP->add_ctr( mc::BASE_NLP::EQ, sqr(Mout[i]/kout) - (Pd[i]-Pout) );
    NLP->add_ctr( mc::BASE_NLP::EQ, sqr(Mrec[i]/krec) - sqr(Vrec[i])*(Pd[i]-Ps[i]) );
    NLP->add_ctr( mc::BASE_NLP::EQ, Mout[i] - Min[i] );
    NLP->add_ctr( mc::BASE_NLP::EQ, Mc[i] - Min[i] - Mrec[i] );
    NLP->add_ctr( mc::BASE_NLP::EQ, Pd[i] - PI[i] * Ps[i] );
    NLP->add_ctr( mc::BASE_NLP::EQ, PI[i] - map_pi( Mc[i], W[i], maps_a[i] ) );
    NLP->add_ctr( mc::BASE_NLP::EQ, Ep[i] - map_eta( Mc[i], PI[i], maps_b[i] ) );
    NLP->add_ctr( mc::BASE_NLP::EQ, Yp[i] - ((Zin*R*Tin)/MW)*(nv/(nv-1))*(pow(PI[i],(nv-1)/nv)-1) );
    NLP->add_ctr( mc::BASE_NLP::EQ, P[i] * Ep[i] - Yp[i] * Mc[i] );
    NLP->add_ctr( mc::BASE_NLP::GE, Pin   - Ps[i] );
    NLP->add_ctr( mc::BASE_NLP::GE, Pd[i] - Ps[i] );
    NLP->add_ctr( mc::BASE_NLP::LE, Pout  - Pd[i] );
  }
  NLP->setup();
  NLP->options.DISPLEVEL = 0;
  NLP->solve( 200 );
  std::cout << "NLP LOCAL SOLUTION:\n" << NLP->solution();
  std::cout << "FEASIBLE:   " << NLP->is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP->is_stationary( 1e-7 ) << std::endl;

//  std::cout << std::scientific << std::setprecision(5) << std::right;
//  for( Mspdef=2e2; Mspdef<=5e2; Mspdef+=1e1 ){
//    NLP->set_par( Msp, Mspdef );
//    NLP->options.DISPLEVEL = 0;
//    NLP->setup();
//    NLP->solve( 100 );
//    std::cout << NLP->is_feasible( 1e-7 ) << " " << NLP->is_stationary( 1e-7 ) << std::setw(12) << Msp;
//    for( auto && x : NLP->solution().x ) std::cout << std::setw(12) << x;
//    std::cout << std::endl;
//  }
  
  delete NLP;
  return 0;
}

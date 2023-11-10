#undef MC_MINLPSLV_DEBUG
#undef MC__FFUNC_SFAD_CLEAR

#include <fstream>
#include <iomanip>
#include <armadillo>

#include "ffunc.hpp"

#ifdef MC__USE_PROFIL
 #include "mcprofil.hpp"
 typedef INTERVAL I;
#else
 #ifdef MC__USE_BOOST
  #include "mcboost.hpp"
   typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
   typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
   typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
   typedef boost::numeric::interval<double,T_boost_policy> I;
 #else
  #ifdef MC__USE_FILIB
   #include "mcfilib.hpp"
   typedef filib::interval<double> I;
  #else
   #include "interval.hpp"
   typedef mc::Interval I;
  #endif
 #endif
#endif

////////////////////////////////////////////////////////////////////////
// EXTERNAL OPERATIONS
////////////////////////////////////////////////////////////////////////
namespace mc
{

struct FFDOptBase
{
  // Vector of atom matrices
  static std::vector< arma::mat > M;

  // Number of uncertainty scenarios
  static unsigned nRep;

  // Read atom matrices from file
  static unsigned read
    ( unsigned const dim, std::string filename, bool const reset=true, bool const disp=false )
    {
      if( reset ){
        M.clear();
        nRep = 0;
      }
      ++nRep;
      
      std::ifstream file( filename );
      if( !file ) throw std::runtime_error("Error: Could not open input file\n");
      std::string line;
      unsigned i = 0;
      bool empty = false;
      arma::mat Mi( dim, dim, arma::fill::none );
      while( std::getline( file, line ) ){
        std::istringstream iss( line );
        for( unsigned j=0; j<dim; j++ ){
          //std::cout << "reading (" << i << "," << j << ")" << std::endl;
          if( i >= dim || !(iss >> Mi(i,j) ) ){
            if( j ) throw std::runtime_error("Error: Could not read input file\n");
            empty = true;
            break;
          }
          //std::cout << "reading (" << i << "," << j << "): " << Mi(i,j) << std::endl;
        }
        i++;
        if( empty ){
          if( disp ) std::cout << "Atomic matrix #" << M.size() << ":" << std::endl << Mi;
          M.push_back( Mi );
          i = 0;
          empty = false;
        }
        if( i > dim ) throw std::runtime_error("Error: Could not read input file\n");
      }
      if( i ) M.push_back( Mi );
      return M.size();
    }
};

inline std::vector< arma::mat > FFDOptBase::M;
inline unsigned FFDOptBase::nRep;

class FFDOpt
: public FFOp,
  public FFDOptBase
{
public:
  // Constructors
  FFDOpt
    ()
    : FFOp( (int)EXTERN )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      auto dep = FFDep();
      for( unsigned i=0; i<nVar; ++i ) dep += pVar[i].dep();
      dep.update( FFDep::TYPE::N );
      return **insert_external_operation( *this, 1, dep, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      throw std::runtime_error("Error: No generic implementation for DOpt\n");
    }

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const
    {
      //std::cout << "FFDOpt::eval: double\n"; 
      assert( nRes == 1 && nVar * nRep == M.size() );
      arma::mat Mmat;//( M[0].size(), arma::fill::none );
      double logdet;
      for( unsigned s=0; s<nRep; ++s ){
        for( unsigned i=0; i<nVar; ++i )
          if( !i ) Mmat  = vVar[0] * M[s*nVar+0];
          else     Mmat += vVar[i] * M[s*nVar+i];
        //std::cout << Mmat;
        if( rank( Mmat ) < Mmat.n_rows || !arma::log_det_sympd( logdet, Mmat ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        //std::cout << s << ": " << logdet << "  " << vRes[0] << std::endl;
        if( !s ) vRes[0]  = logdet;
        else     vRes[0] += logdet;
      }
      vRes[0] /= nRep;
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      //std::cout << "FFDOpt::eval: FFVar\n"; 
      vRes[0] = operator()( nVar, vVar );
    }

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
      unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    { return "DOPT"; }
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFDOptGrad
: public FFOp,
  public FFDOptBase
{
public:
  // Constructors
  FFDOptGrad
    ()
    : FFOp( (int)EXTERN+1 )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar )
    const
    {
      auto dep = FFDep();
      for( unsigned i=0; i<nVar; ++i ) dep += pVar[i].dep();
      dep.update( FFDep::TYPE::N );
      return *(insert_external_operation( *this, nVar, dep, nVar, pVar )[idep]);
    }
  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      auto dep = FFDep();
      for( unsigned i=0; i<nVar; ++i ) dep += pVar[i].dep();
      dep.update( FFDep::TYPE::N );
      return insert_external_operation( *this, nVar, dep, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      throw std::runtime_error("Error: No generic implementation for DOptGrad\n");
    }

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const
    {
      //std::cout << "FFDOptGrad::eval: double\n";
      assert( nRes == nVar && nVar * nRep == M.size() );
      arma::mat Mmat, Lmat, Xmat, Ymat;
      for( unsigned s=0; s<nRep; ++s ){
        for( unsigned i=0; i<nVar; ++i )
          if( !i ) Mmat  = vVar[0] * M[s*nVar+0];
          else     Mmat += vVar[i] * M[s*nVar+i];
        //std::cout << Mmat;
        if( !arma::chol( Lmat, Mmat, "lower" ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        //std::cout << Lmat;
        for( unsigned i=0; i<nVar; ++i ){
          if( !solve( Ymat, trimatl(Lmat), M[s*nVar+i] ) )  // indicate that Lmat is lower triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          if( !solve( Xmat, trimatu(trans(Lmat)), Ymat ) )  // indicate that Lmat is lower triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          if( !s ) vRes[i]  = arma::trace( Xmat );
          else     vRes[i] += arma::trace( Xmat );
          //std::cout << "vRes[" << i << "]: " << vRes[i] << std::endl;
        }
      }
      for( unsigned i=0; i<nVar; ++i )
        vRes[i] /= nRep;
      //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nVar );
      //std::cout << "FFDOptGrad::eval: FFVar\n"; 
      FFVar** ppRes = operator()( nVar, vVar );
      for( unsigned j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
    }

  // Properties
  std::string name
    ()
    const
    { return "DOPTGRAD"; }
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline void
FFDOpt::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
  assert( nRes == 1 && nVar * nRep == M.size() );
  //std::cout << "FFDOpt::eval: fadbad::F<FFVar>\n";
  static std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  vRes[0] = operator()( nVar, vVarVal.data() );
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );
  static FFDOptGrad DOptGrad;
  FFVar const*const* vDOptGrad = DOptGrad( nVar, vVarVal.data() );
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *vDOptGrad[i] * vVar[i][j];
    }
  }
}

inline void
FFDOpt::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
  assert( nRes == 1 && nVar * nRep == M.size() );
  //std::cout << "FFDOpt::eval: fadbad::F<T>\n";
  static std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  static double vResVal;
  eval( 1, &vResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = vResVal;
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );
  static FFDOptGrad DOptGrad;
  static std::vector<double> vDOptGrad( nVar ); 
  DOptGrad.eval( nVar, vDOptGrad.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j] == 0. ) continue;
      vRes[0][j] += vDOptGrad[i] * vVar[i][j];
    }
  }
}

class FFSum
: public FFOp
{
public:
  // Constructors
  FFSum
    ()
    : FFOp( (int)EXTERN+2 )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      auto dep = FFDep();
      for( unsigned i=0; i<nVar; ++i ) dep += pVar[i].dep();
      dep.update( FFDep::TYPE::L );
      return **insert_external_operation( *this, 1, dep, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      for( unsigned j=0; j<nRes; ++j )
        for( unsigned i=0; i<nVar; ++i )
          if( !i ) vRes[j]  = vVar[i];
          else     vRes[j] += vVar[i];
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      //std::cout << "FFDOpt::eval: FFVar\n"; 
      vRes[0] = operator()( nVar, vVar );
    }

  void eval
    ( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
      unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      std::cout << "FFSum::eval: fadbad::F<T>\n";
      static std::vector<FFVar> vVarVal( nVar );
      for( unsigned i=0; i<nVar; ++i )
        vVarVal[i] = vVar[i].val();
      static FFVar vResVal;
      eval( 1, &vResVal, nVar, vVarVal.data(), nullptr );
      vRes[0] = vResVal;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0].setDepend( vVar[i] );
      for( unsigned j=0; j<vRes[0].size(); ++j ){
        vRes[0][j] = 0.;
        for( unsigned i=0; i<nVar; ++i ){
          if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
          vRes[0][j] += vVar[i][j];
        }
      }
    }

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      std::cout << "FFSum::eval: fadbad::F<T>\n";
      static std::vector<double> vVarVal( nVar );
      for( unsigned i=0; i<nVar; ++i )
        vVarVal[i] = vVar[i].val();
      static double vResVal;
      eval( 1, &vResVal, nVar, vVarVal.data(), nullptr );
      vRes[0] = vResVal;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0].setDepend( vVar[i] );
      for( unsigned j=0; j<vRes[0].size(); ++j ){
        vRes[0][j] = 0.;
        for( unsigned i=0; i<nVar; ++i ){
          if( vVar[i][j] == 0. ) continue;
          vRes[0][j] += vVar[i][j];
        }
      }
    }

  // Properties
  std::string name
    ()
    const
    { return "SUM"; }
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

}

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIP;
#elif  MC__USE_IPOPT
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIP;
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
 typedef mc::NLPSLV_SNOPT< mc::FFDOpt, mc::FFDOptGrad, mc::FFSum > NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT< mc::FFDOpt, mc::FFDOptGrad, mc::FFSum > NLP;
#endif

#include "minlpslv.hpp"

////////////////////////////////////////////////////////////////////////
// APPORTIONMENT
////////////////////////////////////////////////////////////////////////
void
apportion
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

  std::cout << "Initial efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    if( val[i] > TOLZERO ) std::cout << "X0[" << i << "]: " << val[i] << std::endl;
  }

  double sum = 0.;
  unsigned supp = 0;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    sum += val[i];
    if( val[i] >= TOLZERO ) supp++;
  }
  sum = std::round( sum );

  static std::vector<double> intval( n );
  for( double ratio=1.; ratio>0.1; ){
    double intsum = 0.;
    for( unsigned i=0; i<n; ++i ){
      if( !typ[i] ) continue;
      intval[i] = (val[i]<TOLZERO? 0: (supp<=sum? std::ceil( ratio*val[i] ): std::round( ratio*val[i] )));
      intsum += intval[i];
    }
    if( sum < intsum )
      ratio *= 0.9;
    else if( sum > intsum )
      ratio /= 0.95;
    else
      break;
  }

  std::cout << "Apportioned efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
  }
}
void
effrounding
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

  std::cout << "Initial efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    if( val[i] > TOLZERO ) std::cout << "X0[" << i << "]: " << val[i] << std::endl;
  }

  double sum = 0.;
  unsigned supp = 0;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    sum += val[i];
    if( val[i] >= TOLZERO ) supp++;
  }
  sum = std::round( sum );

  static std::vector<double> intval( n );
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    intval[i] = std::ceil( (1.-supp/(2*sum)) * val[i] );
  }

  for( ; ; ){
    std::cout << "Intermediate efforts:" << std::endl;
    double intsum = 0.;
    for( unsigned i=0; i<n; ++i ){
      if( !typ[i] ) continue;
      intsum += intval[i];
      if( val[i] > TOLZERO ) std::cout << "X1[" << i << "]: " << intval[i] << std::endl;
    }
    if( std::fabs( sum - intsum ) < TOLZERO ) break;
    if( sum > intsum ){
      int imin = -1;
      double effmin = 1.;
      for( unsigned i=0; i<n; ++i ){
        if( !typ[i] || val[i] < TOLZERO ) continue;
        if( intval[i]/val[i] < effmin ){
          imin = i;
          effmin = intval[i]/val[i];
        }
      }
      assert( imin >= 0 );
      intval[imin] += 1;
    }
    else{
      int imax = -1;
      double effmax = 1.;
      for( unsigned i=0; i<n; ++i ){
        if( !typ[i] || val[i] < TOLZERO ) continue;
        if( intval[i]/val[i] > effmax ){
          imax   = i;
          effmax = intval[i]/val[i];
        }
      }
      assert( imax >= 0 );
      intval[imax] -= 1;
    }
  }
  
  std::cout << "Apportioned efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
  }
  //{ int dum; std::cout << "ENTER <1>"; std::cin >> dum; }
}

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  const unsigned NS = mc::FFDOptBase::read( 4, "fim/ester_1080.fim" );
  //const unsigned NS = mc::FFDOptBase::read( 5, "fim/han_5000.fim" );
  //const unsigned NS = mc::FFDOptBase::read( 4, "fim/doe_1000.fim" ); 
  //const unsigned NS = mc::FFDOptBase::read( 4, "fim/doe_50.fim" ); 

  mc::FFGraph< mc::FFDOpt, mc::FFDOptGrad, mc::FFSum > DAG;
  mc::FFVar S[NS];
  double S0[NS];
  for( unsigned int i=0; i<NS; i++ ){
    S[i].set( &DAG );
    S0[i] = 1./NS;
  }
  mc::FFDOpt DOpt;
  mc::FFSum  Sum;

  mc::MINLPSLV<I,NLP,MIP,mc::FFDOpt,mc::FFDOptGrad,mc::FFSum> MINLP;
  MINLP.options.SEARCHALG               = mc::MINLPSLV<I,NLP,MIP,mc::FFDOpt,mc::FFDOptGrad,mc::FFSum>::Options::OA;
  MINLP.options.DISPLEVEL               = 1;
  MINLP.options.CVRTOL                  = 1e-5;
  MINLP.options.CVATOL                  = 1e-9;
  MINLP.options.FEASTOL                 = 1e-5;
  MINLP.options.FEASPUMP                = 0;
  MINLP.options.ROOTCUT                 = 1;
  MINLP.options.TIMELIMIT               = 6e2;
  MINLP.options.LINMETH                 = mc::MINLPSLV<I,NLP,MIP,mc::FFDOpt,mc::FFDOptGrad,mc::FFSum>::Options::CVX;
  MINLP.options.MAXITER                 = 30;
  MINLP.options.MSLOC                   = 1;
#ifdef MC__USE_SNOPT
  MINLP.options.NLPSLV.DISPLEVEL        = 0;
  MINLP.options.NLPSLV.MAXITER          = 100;
  MINLP.options.NLPSLV.FEASTOL          = 1e-7;
  MINLP.options.NLPSLV.OPTIMTOL         = 1e-7;
  MINLP.options.NLPSLV.GRADMETH         = NLP::Options::FAD; //SYM;
  MINLP.options.NLPSLV.GRADCHECK        = 0;
  MINLP.options.NLPSLV.MAXTHREAD        = 0;
#elif  MC__USE_IPOPT
  MINLP.options.NLPSLV.DISPLEVEL        = 0;
  MINLP.options.NLPSLV.MAXITER          = 100;
  MINLP.options.NLPSLV.FEASTOL          = 1e-8;
  MINLP.options.NLPSLV.OPTIMTOL         = 1e-8;
  MINLP.options.NLPSLV.GRADMETH         = NLP::Options::FAD;
  //MINLP.options.NLPSLV.GRADCHECK        = 0;
  MINLP.options.NLPSLV.MAXTHREAD        = 0;
#endif
#ifdef MC__USE_GUROBI
  MINLP.options.MIPSLV.DISPLEVEL        = 0;
  MINLP.options.MIPSLV.THREADS          = 0;
  MINLP.options.MIPSLV.MIPRELGAP        = 1e-6;
  MINLP.options.MIPSLV.MIPABSGAP        = 1e-9;
  MINLP.options.MIPSLV.OUTPUTFILE       = "";//"doe.lp";
#elif  MC__USE_CPLEX
  throw std::runtime_error("Error: CPLEX solver not yet implemented");
#endif

  int const NEXP = 5; 
  MINLP.set_dag( &DAG );
  MINLP.set_var( NS, S, 0., NEXP, 1 );
  MINLP.set_obj( mc::BASE_OPT::MAX, DOpt( NS, S ) );
  MINLP.add_ctr( mc::BASE_OPT::EQ, Sum( NS, S ) - NEXP );
  //MINLP.add_ctr( mc::BASE_OPT::EQ, DAG.sum( NS, S ) - NEXP );

  MINLP.setup();
  //MINLP.optimize( S0 );
  //MINLP.optimize( S0, nullptr, apportion );
  MINLP.optimize( S0, nullptr, effrounding );
  MINLP.stats.display();

  std::cout << "Optimal effort:" << std::endl;
  unsigned i=0;
  for( auto const& Xi : MINLP.get_incumbent().x ){
    if( Xi > 1e-1 ) std::cout << "X[" << i << "]: " << Xi << std::endl;
    ++i;
  }
  
  return 0;
}

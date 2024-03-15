//#define MC__NLPSLV_SNOPT_DEBUG_CALLBACK
//#define MC__MINLPSLV_DEBUG_LINEARIZATION
//#define MC__REVAL_DEBUG

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
    ( unsigned const dim, std::string filename, bool const reset=true, unsigned const disp=false )
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
          if( disp > 1 ) std::cout << "reading (" << i << "," << j << "): " << Mi(i,j) << std::endl;
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

template<unsigned int ID>
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

  // Declaration
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID;
      return *(insert_external_operation( *this, nRep, nVar, pVar )[idep]);
    }

  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID;
      return insert_external_operation( *this, nRep, nVar, pVar );
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
      assert( nRes == nRep && nVar * nRep == M.size() );
      arma::mat Mmat;//( M[0].size(), arma::fill::none );
      double logdet;
      for( unsigned s=0; s<nRep; ++s ){
        for( unsigned i=0; i<nVar; ++i )
          if( !i ) Mmat  = vVar[0] * M[s*nVar+0];
          else     Mmat += vVar[i] * M[s*nVar+i];
        //std::cout << Mmat << "  det = " << arma::det( Mmat ) << std::endl;
        if( rank( Mmat ) < Mmat.n_rows || !arma::log_det_sympd( logdet, Mmat ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        //std::cout << s << ": " << logdet << std::endl;
        vRes[s]  = logdet;
      }
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nRep );
      //std::cout << "FFDOpt::eval: FFVar\n"; 
      FFVar** ppRes = operator()( nVar, vVar );
      for( unsigned j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nRep );
      //std::cout << "FFDOpt::eval: FFDep\n"; 
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
    }

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
      unsigned const* mVar )
    const;

  void deriv
    ( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
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

template<unsigned int ID>
class FFDOptGrad
: public FFOp,
  public FFDOptBase
{
public:
  // Constructors
  FFDOptGrad
    ()
    : FFOp( (int)EXTERN )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID+1;
      return *(insert_external_operation( *this, nRep * nVar, nVar, pVar )[idep]);
    }
  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID+1;
      return insert_external_operation( *this, nRep * nVar, nVar, pVar );
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
      assert( nRes == nRep * nVar && nVar * nRep == M.size() );
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
          vRes[s*nVar+i] = arma::trace( Xmat );
          //std::cout << "vRes[" << s << "," << i << "]: " << vRes[s*nVar+i] << std::endl;
        }
      }
      //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nRep * nVar );
      //std::cout << "FFDOptGrad::eval: FFVar\n"; 
      FFVar** ppRes = operator()( nVar, vVar );
      for( unsigned j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nRep * nVar );
      //std::cout << "FFDOpt::eval: FFDep\n"; 
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
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

template<unsigned int ID>
inline void
FFDOpt<ID>::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
  assert( nRes == nRep && nVar * nRep == M.size() );
  //std::cout << "FFDOpt::eval: fadbad::F<FFVar>\n";

  static std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar const*const* vDOpt = operator()( nVar, vVarVal.data() );
  for( unsigned s=0; s<nRep; ++s ){
    vRes[s] = *vDOpt[s];
    for( unsigned i=0; i<nVar; ++i )
      vRes[s].setDepend( vVar[i] );
  }

  static FFDOptGrad<ID> DOptGrad;
  FFVar const*const* vDOptGrad = DOptGrad( nVar, vVarVal.data() ); 
  for( unsigned s=0; s<nRep; ++s ){
    for( unsigned j=0; j<vRes[0].size(); ++j ){
      vRes[s][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
        vRes[s][j] += *vDOptGrad[s*nVar+i] * vVar[i][j];
      }
    }
  }
}

template<unsigned int ID>
inline void
FFDOpt<ID>::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
  assert( nRes == nRep && nVar * nRep == M.size() );
  //std::cout << "FFDOpt::eval: fadbad::F<T>\n";

  static std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  static std::vector<double> vResVal( nRep ); 
  eval( nRep, vResVal.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned s=0; s<nRep; ++s ){
    vRes[s] = vResVal[s];
    for( unsigned i=0; i<nVar; ++i )
      vRes[s].setDepend( vVar[i] );
  }

  static FFDOptGrad<ID> DOptGrad;
  static std::vector<double> vDOptGrad( nRep * nVar ); 
  DOptGrad.eval( nRep * nVar, vDOptGrad.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned s=0; s<nRep; ++s ){
    for( unsigned j=0; j<vRes[0].size(); ++j ){
      vRes[s][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j] == 0. ) continue;
        vRes[s][j] += vDOptGrad[s*nVar+i] * vVar[i][j];
      }
    }
  }
}

template<unsigned int ID>
inline void
FFDOpt<ID>::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
  assert( nRes == nRep && nVar * nRep == M.size() );
  //std::cout << "FFDOpt::deriv: FFVar\n";

  static FFDOptGrad<ID> DOptGrad;
  FFVar const*const* vDOptGrad = DOptGrad( nVar, vVar ); 
  for( unsigned s=0; s<nRep; ++s )
    for( unsigned i=0; i<nVar; ++i )
      vDer[s][i] = *vDOptGrad[s*nVar+i];
}

template<unsigned int ID>
class FFSum
: public FFOp
{
public:
  // Constructors
  FFSum
    ()
    : FFOp( (int)EXTERN )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID;
      return **insert_external_operation( *this, 1, nVar, pVar );
    }
  FFVar& operator()
    ( unsigned const nVar, FFVar const*const* pVar )
    const
    {
      info = ID;
      return **insert_external_operation( *this, 1, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      for( unsigned i=0; i<nVar; ++i )
        if( !i ) vRes[0]  = vVar[i];
        else     vRes[0] += vVar[i];
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
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      //std::cout << "FFDOpt::eval: FFDep\n"; 
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::L );
    }

  void eval
    ( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
      unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      //std::cout << "FFSum::eval: fadbad::F<T>\n";
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
      //std::cout << "FFSum::eval: fadbad::F<T>\n";
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

  void deriv
    ( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
    const
    {
      assert( nRes == 1 );
      //std::cout << "FFSum::deriv: FFVar\n";
      for( unsigned i=0; i<nVar; ++i )
        vDer[0][i] = 1;
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

} // end namespace mc

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
 typedef mc::MIPSLV_GUROBI<I> MIP;
#elif  MC__USE_IPOPT
 #include "mipslv_cplex.hpp"
 typedef mc::MIPSLV_CPLEX<I> MIP;
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
 typedef mc::NLPSLV_SNOPT< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFSum<2> > NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFSum<2> > NLP;
#endif

#include "minlpslv.hpp"
typedef mc::MINLPSLV< I, NLP, MIP, mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFSum<2> > MINLP;

typedef mc::FFGraph< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFSum<2> > DAG;

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
  const unsigned nEff = mc::FFDOptBase::read( 4, "fim/ester_1080_1.fim", true );
  for( int s=2; s<=100; ++s )
    mc::FFDOptBase::read( 4, "fim/ester_1080_"+std::to_string(s)+".fim", false );
  const unsigned nRep = mc::FFDOptBase::nRep;
  assert( nEff * nRep == mc::FFDOptBase::M.size() );

  int const nExp = 8;
  DAG tree;
  mc::FFDOpt<0> DOpt;
  mc::FFSum<2> Sum;
  mc::FFVar E[nEff];
  for( unsigned int i=0; i<nEff; i++ )
    E[i].set( &tree );

  double const beta = 9e-1;
  mc::FFVar D[nRep], V( &tree );
  for( unsigned int s=0; s<nRep; s++ )
    D[s].set( &tree );

  std::vector<double> ini( nEff, 1./nEff );
  ini.resize( nEff+nRep+1, 0e0 );

///////////////////////
// RISK-NEUTRAL DESIGN

  MINLP doe;
  doe.options.SEARCHALG               = MINLP::Options::OA;
  doe.options.DISPLEVEL               = 1;
  doe.options.CVRTOL                  = 1e-5;
  doe.options.CVATOL                  = 1e-9;
  doe.options.FEASTOL                 = 1e-5;
  doe.options.FEASPUMP                = 0;
  doe.options.ROOTCUT                 = 1;
  doe.options.TIMELIMIT               = 6e2;
  doe.options.LINMETH                 = MINLP::Options::CVX;
  doe.options.MAXITER                 = 40;
  doe.options.MSLOC                   = 1;
  doe.options.CPMAX                   = 5;
#ifdef MC__USE_SNOPT
  doe.options.NLPSLV.DISPLEVEL        = 0;
  doe.options.NLPSLV.MAXITER          = 100;
  doe.options.NLPSLV.FEASTOL          = 1e-7;
  doe.options.NLPSLV.OPTIMTOL         = 1e-7;
  doe.options.NLPSLV.GRADMETH         = NLP::Options::FAD;
  doe.options.NLPSLV.GRADCHECK        = 0;
  doe.options.NLPSLV.MAXTHREAD        = 0;
#elif  MC__USE_IPOPT
  doe.options.NLPSLV.DISPLEVEL        = 0;
  doe.options.NLPSLV.MAXITER          = 100;
  doe.options.NLPSLV.FEASTOL          = 1e-8;
  doe.options.NLPSLV.OPTIMTOL         = 1e-8;
  doe.options.NLPSLV.GRADMETH         = NLP::Options::FAD;
  //doe.options.NLPSLV.GRADCHECK        = 0;
  doe.options.NLPSLV.MAXTHREAD        = 0;
#endif
#ifdef MC__USE_GUROBI
  doe.options.MIPSLV.DISPLEVEL        = 0;
  doe.options.MIPSLV.THREADS          = 0;
  doe.options.MIPSLV.MIPRELGAP        = 1e-6;
  doe.options.MIPSLV.MIPABSGAP        = 1e-9;
  doe.options.MIPSLV.OUTPUTFILE       = "";//"doe.lp";
#elif  MC__USE_CPLEX
  throw std::runtime_error("Error: CPLEX solver not yet implemented");
#endif

  doe.set_dag( &tree );
  doe.set_var( nEff, E, 0., nExp, 1 );
  doe.set_obj( mc::BASE_OPT::MAX, Sum( nRep, DOpt( nEff, E ) ) / nRep );
  doe.add_ctr( mc::BASE_OPT::EQ, Sum( nEff, E ) - nExp );

  doe.setup();
  //doe.optimize( ini.data() );
  //doe.optimize( ini.data(), nullptr, apportion );
  doe.optimize( ini.data(), nullptr, effrounding );
  doe.stats.display();

  std::cout << "Optimal efforts:" << std::endl;
  unsigned i=0;
  for( auto const& Xi : doe.get_incumbent().x ){
    if( Xi > 1e-1 ) std::cout << "X[" << i << "]: " << Xi << std::endl;
    ++i;
  }
  //return 0;
  
///////////////////////
// RISK-AVERSE DESIGN

  MINLP doe2;
  doe2.options = doe.options;

  doe2.set_dag( &tree );
  doe2.set_var( nEff, E, 0., nExp, 1 );
  doe2.add_var( nRep, D, 0., 1e2 );
  doe2.add_var( V, 0., 1e2 );
  doe2.set_obj( mc::BASE_OPT::MAX, V - Sum( nRep, D ) / ((1-beta) * nRep) );
  doe2.add_ctr( mc::BASE_OPT::EQ, Sum( nEff, E ) - nExp );
  for( unsigned int s=0; s<nRep; s++ )
    doe2.add_ctr( mc::BASE_OPT::LE, V - D[s] - DOpt( s, nEff, E ) );

  doe2.setup();
  //doe2.optimize( ini.data() );
  //doe2.optimize( ini.data(), nullptr, apportion );
  doe2.optimize( ini.data(), nullptr, effrounding );
  doe2.stats.display();

  std::cout << "Optimal efforts:" << std::endl;
  i=0;
  for( auto const& Xi : doe2.get_incumbent().x ){
    if( i >= nEff ) break;
    if( Xi > 1e-1 ) std::cout << "X[" << i << "]: " << Xi << std::endl;
    ++i;
  }

  std::vector<I> bnd( nEff+nRep+1 );
  for( unsigned i=0; i<nEff; ++i ){
    ini[i] = doe2.get_incumbent().x[i];
    bnd[i] = ini[i];
  }
  doe.local_solver().options.DISPLEVEL = 1;
  doe.local_solver().solve( ini.data(), bnd.data() );
  std::cout << "Suboptimal average:" << doe.local_solver().solution().f[0] << std::endl;

  for( unsigned i=0; i<nEff; ++i ){
    ini[i] = doe.get_incumbent().x[i];
    bnd[i] = ini[i];
  }
  for( unsigned i=nEff; i<nEff+nRep+1; ++i ){
    ini[i] = 0;
    bnd[i] = I(0e0,1e2);
  }
  doe2.local_solver().options.DISPLEVEL = 1;
  doe2.local_solver().solve( ini.data(), bnd.data() );
  std::cout << "Suboptimal risk:" << doe2.local_solver().solution().f[0] << std::endl;

  return 0;
}

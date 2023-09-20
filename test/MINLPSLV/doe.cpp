//#define MC__MINLPSLV_DEBUG

#include <fstream>
#include <iomanip>

#include "ffunc.hpp"
#include "mclapack.hpp"

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
  static std::vector< CPPL::dsymatrix > _A;

  // Read atom matrices from file
  static unsigned read
    ( unsigned const dim, std::string filename, bool const disp=false )
    {
      CPPL::dsymatrix Ai( dim );
      _A.clear();
      std::ifstream file( filename );
      if( !file ) throw std::runtime_error("Error: Could not open input file\n");
      std::string line;
      unsigned i = 0;
      bool empty = false;
      while( std::getline( file, line ) ){
        std::istringstream iss( line );
        for( unsigned j=0; j<dim; j++ ){
          if( !(iss >> Ai(i,j) ) ){
            if( j ) throw std::runtime_error("Error: Could not read input file\n");
            empty = true;
            break;
          }
          //std::cout << "reading (" << i << "," << j << "): " << Ai(i,j) << std::endl;
        }
        i++;
        if( empty ){
          if( disp ) std::cout << "Atomic matrix #" << _A.size() << ":" << std::endl << Ai;
          _A.push_back( Ai );
          i = 0;
          empty = false;
        }
        if( i > dim ) throw std::runtime_error("Error: Could not read input file\n");
      }
      if( i ) _A.push_back( Ai );
      return _A.size();
    }
};

inline std::vector< CPPL::dsymatrix > FFDOptBase::_A;

class FFDOpt
: public FFOp,
  public FFDOptBase
{
public:
  // Constructors
  FFDOpt
    ()
    : FFOp( (int)EXTERN+4 )
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
      assert( nRes == 1 && nVar == _A.size() && _A.begin() != _A.end() );
      CPPL::dsymatrix Amat( _A[0].n );
      Amat.zero();
      for( unsigned i=0; i<nVar; ++i )
        if( !i ) Amat  = vVar[0] * _A[0];
        else     Amat += vVar[i] * _A[i];
      //std::cout << Amat;
      if( dgeqrf( Amat.to_dgematrix(), vRes[0] ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      vRes[0] = std::log( vRes[0] );
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
    : FFOp( (int)EXTERN+5 )
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
      assert( nRes == nVar && nVar == _A.size() && _A.begin() != _A.end() );
      CPPL::dsymatrix Amat( _A[0].n );
      Amat.zero();
      for( unsigned i=0; i<nVar; ++i )
        if( !i ) Amat  = vVar[0] * _A[0];
        else     Amat += vVar[i] * _A[i];
      //std::cout << Amat;
      // Perform LDL' decomposition
      CPPL::dgematrix Lmat;
      std::vector<int> IPIV;
      if( dsytrf( Amat, Lmat, IPIV ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      CPPL::dgematrix Xmat;
      for( unsigned i=0; i<nVar; ++i ){
        if( dsytrs( Lmat, IPIV, _A[i].to_dgematrix(), Xmat ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        //std::cout << Xmat;
        for( int j=0; j<Xmat.n; ++j )
          if( !j ) vRes[i]  = Xmat(0,0);
          else     vRes[i] += Xmat(j,j);
      }
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
  assert( nRes == 1 && nVar == _A.size() && _A.begin() != _A.end() );
  std::cout << "FFDOpt::eval: fadbad::F<FFVar>\n";
  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  vRes[0] = operator()( nVar, vVarVal.data() );
  FFDOptGrad DOptGrad;
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += DOptGrad( i, nVar, vVarVal.data() ) * vVar[i][j];
      //std::cout << "(" << i << "," << j << ")" << std::endl;
      //if( !i ) vRes[0][j]  = DOptGrad( 0, nVar, vVarVal.data() ) * vVar[0][j];
      //else     vRes[0][j] += DOptGrad( i, nVar, vVarVal.data() ) * vVar[i][j];
    }
  }
}

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
 typedef mc::NLPSLV_SNOPT< mc::FFDOpt, mc::FFDOptGrad > NLP;
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
 typedef mc::NLPSLV_IPOPT< mc::FFDOpt, mc::FFDOptGrad > NLP;
#endif

#include "minlpslv.hpp"

////////////////////////////////////////////////////////////////////////
// APPORTIONMENT
////////////////////////////////////////////////////////////////////////
void
nearest
( unsigned const n, unsigned const* typ, double* val )
{
  for( unsigned i=0; i<n; ++i )
    val[i] = round( val[i] );
}

void
apportion
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

  double sum = 0.;
  unsigned supp = 0;
  for( unsigned i=0; i<n; ++i ){
    sum += val[i];
    if( val[i] >= TOLZERO ) supp++;
  }
  sum = std::round( sum );

  std::vector<double> intval( n );
  for( double ratio=1.; ratio>0.1; ){
    double intsum = 0.;
    for( unsigned i=0; i<n; ++i ){
      intval[i] = (val[i]<TOLZERO? 0: (supp<=sum? std::ceil( ratio*val[i] ): std::round( ratio*val[i] )));
      intsum += intval[i];
    }
    if( sum < intsum )
      ratio *= 0.9;
    else if( sum > intsum )
      ratio /= 0.9;
    else
      break;
  }

  for( unsigned i=0; i<n; ++i ) val[i] = intval[i];
}

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  mc::FFGraph< mc::FFDOpt, mc::FFDOptGrad > DAG;
  const unsigned NS = mc::FFDOptBase::read( 4, "doe_1000.fim" ); 
  mc::FFVar S[NS];
  double S0[NS];
  for( unsigned int i=0; i<NS; i++ ){
    S[i].set( &DAG );
    S0[i] = 1./NS;
  }
  mc::FFDOpt DOpt;

  mc::MINLPSLV<I,NLP,MIP,mc::FFDOpt,mc::FFDOptGrad> MINLP;
  MINLP.options.DISPLEVEL               = 1;
  MINLP.options.CVRTOL                  = 1e-5;
  MINLP.options.CVATOL                  = 1e-5;
  MINLP.options.FEASTOL                 = 1e-5;
//  MINLP.options.FEASPUMP                = 1;
  MINLP.options.INCCUT                  = 0;
  MINLP.options.ROOTCUT                 = 1;
  MINLP.options.TIMELIMIT               = 6e2;
  MINLP.options.LINMETH                 = mc::MINLPSLV<I,NLP,MIP,mc::FFDOpt,mc::FFDOptGrad>::Options::CVX;
  MINLP.options.MAXITER                 = 20;
  MINLP.options.MSLOC                   = 1;
#ifdef MC__USE_SNOPT
  MINLP.options.NLPSLV.DISPLEVEL        = 1;
  MINLP.options.NLPSLV.MAXITER          = 100;
  MINLP.options.NLPSLV.FEASTOL          = 1e-8;
  MINLP.options.NLPSLV.OPTIMTOL         = 1e-8;
  MINLP.options.NLPSLV.GRADMETH         = NLP::Options::FAD;
  //MINLP.options.NLPSLV.GRADCHECK        = 0;
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
  MINLP.options.MIPSLV.MIPRELGAP        = 1e-5;
  MINLP.options.MIPSLV.OUTPUTFILE       = "doe.lp";
#elif  MC__USE_CPLEX
  throw std::runtime_error("Error: CPLEX solver not yet implemented");
#endif

  int const NEXP = 4; 
  MINLP.set_dag( &DAG );
  MINLP.set_var( NS, S, 0., NEXP, 1 );
  MINLP.set_obj( mc::BASE_OPT::MAX, DOpt( NS, S ) );
  MINLP.add_ctr( mc::BASE_OPT::EQ, DAG.sum( NS, S ) - NEXP );

  MINLP.setup();
  //MINLP.optimize( S0 );
  //MINLP.optimize( S0, nullptr, nearest );
  MINLP.optimize( S0, nullptr, apportion );
  MINLP.stats.display();

  return 0;
}

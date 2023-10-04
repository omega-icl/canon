#include <fstream>
#include <iomanip>

#include "mclapack.hpp"
#include "interval.hpp"
#ifdef MC__USE_SNOPT
  #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
  #include "nlpslv_ipopt.hpp"
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
  for( unsigned j=0; j<vRes[0].size(); ++j )
    for( unsigned i=0; i<nVar; ++i )
      if( !i ) vRes[0][j]  = DOptGrad( 0, nVar, vVarVal.data() ) * vVar[0][j];
      else     vRes[0][j] += DOptGrad( i, nVar, vVarVal.data() ) * vVar[i][j];
}

}

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  mc::FFGraph< mc::FFDOpt, mc::FFDOptGrad > DAG;
  const unsigned NS = mc::FFDOptBase::read( 4, "test4.fim" ); 
  mc::FFVar S[NS];
  double S0[NS];
  for( unsigned int i=0; i<NS; i++ ){
    S[i].set( &DAG );
    S0[i] = 1./NS;
  }
  mc::FFDOpt DOpt;

#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT< mc::FFDOpt, mc::FFDOptGrad > NLP;
  NLP.options.DISPLEVEL = 1;
  NLP.options.MAXITER   = 200;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_SNOPT< mc::FFDOpt, mc::FFDOptGrad >::Options::FSYM;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 0;
#else
  mc::NLPSLV_IPOPT< mc::FFDOpt, mc::FFDOptGrad > NLP;
  NLP.options.DISPLEVEL = 5;
  NLP.options.MAXITER   = 200;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_IPOPT< mc::FFDOpt, mc::FFDOptGrad >::Options::FAD;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 0;
#endif
 
  int const NEXP = 4; 
  NLP.set_dag( &DAG );
  NLP.set_var( NS, S, 0., NEXP );
  NLP.set_obj( mc::BASE_OPT::MAX, DOpt( NS, S ) );
  NLP.add_ctr( mc::BASE_OPT::EQ, DAG.sum( NS, S ) - NEXP );

  NLP.setup();
  NLP.solve( S0 );

  std::cout << "NLP LOCAL SOLUTION:\n" << NLP.solution();
  std::cout << "FEASIBLE:   " << NLP.is_feasible( 1e-7 )   << std::endl;
  std::cout << "STATIONARY: " << NLP.is_stationary( 1e-7 ) << std::endl;

  return 0;
}

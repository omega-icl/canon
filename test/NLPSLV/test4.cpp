#define  USE_ARMADILLO

#include <fstream>
#include <iomanip>

#ifdef USE_ARMADILLO
  #include <armadillo>
#else
  #include "mclapack.hpp"
#endif

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
#ifdef USE_ARMADILLO
  static std::vector< arma::mat > M;
#else
  static std::vector< CPPL::dsymatrix > M;
#endif

  // Read atom matrices from file
  static unsigned read
    ( unsigned const dim, std::string filename, bool const disp=false )
    {
#ifdef USE_ARMADILLO
      arma::mat Mi( dim, dim, arma::fill::none );
#else
      CPPL::dsymatrix Mi( dim );
#endif
      M.clear();
      std::ifstream file( filename );
      if( !file ) throw std::runtime_error("Error: Could not open input file\n");
      std::string line;
      unsigned i = 0;
      bool empty = false;
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

#ifdef USE_ARMADILLO
inline std::vector< arma::mat > FFDOptBase::M;
#else
inline std::vector< CPPL::dsymatrix > FFDOptBase::M;
#endif

template<unsigned int ID>
class FFDOpt
: public FFOp,
  public FFDOptBase
{
public:
  // Construction
  FFDOpt
    ()
    : FFOp( (int)EXTERN )
    {}

  // Definition
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID;
      return **insert_external_operation( *this, 1, nVar, pVar );
    }

  // Evaluation
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
      std::cout << "FFDOpt::eval: double\n"; 
      assert( nRes == 1 && nVar == M.size() && M.begin() != M.end() );

#ifdef USE_ARMADILLO
      arma::mat Mmat;
#else
      CPPL::dsymatrix Mmat( M[0].n );
      Mmat.zero();
#endif
      for( unsigned i=0; i<nVar; ++i )
        if( !i ) Mmat  = vVar[0] * M[0];
        else     Mmat += vVar[i] * M[i];
      //std::cout << Mmat;

#ifdef USE_ARMADILLO
      if( !arma::log_det_sympd( vRes[0], Mmat ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#else
      if( dgeqrf( Mmat.to_dgematrix(), vRes[0] ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      vRes[0] = std::log( vRes[0] );
#endif
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      std::cout << "FFDOpt::eval: FFVar\n"; 
      vRes[0] = operator()( nVar, vVar );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == 1 );
      std::cout << "FFDOpt::eval: FFDep\n"; 

      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
    }

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const;

  // Differentiation
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
  // Construction
  FFDOptGrad
    ()
    : FFOp( (int)EXTERN )
    {}

  // Definition
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID+1;
      return *(insert_external_operation( *this, nVar, nVar, pVar )[idep]);
    }
  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID+1;
      return insert_external_operation( *this, nVar, nVar, pVar );
    }

  // Evaluation
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
      std::cout << "FFDOptGrad::eval: double\n"; 
      assert( nRes == nVar && nVar == M.size() && M.begin() != M.end() );

#ifdef USE_ARMADILLO
      arma::mat Mmat;
#else
      CPPL::dsymatrix Mmat( M[0].n );
      Mmat.zero();
#endif
      for( unsigned i=0; i<nVar; ++i )
        if( !i ) Mmat  = vVar[0] * M[0];
        else     Mmat += vVar[i] * M[i];
      //std::cout << Mmat;

#ifdef USE_ARMADILLO
      arma::mat Lmat, Ymat, Gmat;
      if( !arma::chol( Lmat, Mmat, "lower" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      //std::cout << Lmat;
      for( unsigned i=0; i<nVar; ++i ){
        if( !solve( Ymat, trimatl(Lmat), M[i] ) )  // indicate that Lmat is lower triangular
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        if( !solve( Gmat, trimatu(trans(Lmat)), Ymat ) )  // indicate that Lmat is lower triangular
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        vRes[i] = arma::trace( Gmat );
      }
#else
      CPPL::dgematrix Lmat;
      std::vector<int> IPIV;
      if( dsytrf( Mmat, Lmat, IPIV ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      CPPL::dgematrix Gmat;
      for( unsigned i=0; i<nVar; ++i ){
        if( dsytrs( Lmat, IPIV, M[i].to_dgematrix(), Gmat ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        //std::cout << Gmat;
        for( int k=0; k<Gmat.n; ++k )
          if( !k ) vRes[i]  = Gmat(0,0);
          else     vRes[i] += Gmat(k,k);
      }
#endif
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nVar );
      std::cout << "FFDOptGrad::eval: FFVar\n"; 

      FFVar** ppRes = operator()( nVar, vVar );
      for( unsigned j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nVar );
      std::cout << "FFDOptGrad::eval: FFDep\n"; 

      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
    }

  // Differentiation
  void deriv
    ( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
    const;

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
class FFDOptHess
: public FFOp,
  public FFDOptBase
{
public:
  // Construction
  FFDOptHess
    ()
    : FFOp( (int)EXTERN )
    {}

  // Definition
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID+2;
      return *(insert_external_operation( *this, nVar*nVar, nVar, pVar )[idep]);
    }
    
  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      info = ID+2;
      return insert_external_operation( *this, nVar*nVar, nVar, pVar );
    }

  // Evaluation
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      throw std::runtime_error("Error: No generic implementation for DOptHess\n");
    }

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const
    {
      std::cout << "FFDOptHess::eval: double\n"; 
      assert( nRes == nVar*nVar && nVar == M.size() && M.begin() != M.end() );

#ifdef USE_ARMADILLO
      arma::mat Mmat;
#else
      CPPL::dsymatrix Mmat( M[0].n );
      Mmat.zero();
#endif
      for( unsigned i=0; i<nVar; ++i )
        if( !i ) Mmat  = vVar[0] * M[0];
        else     Mmat += vVar[i] * M[i];
      //std::cout << Mmat;

#ifdef USE_ARMADILLO
      arma::mat Lmat, Ymat, Gmat, Hmat;
      if( !arma::chol( Lmat, Mmat, "lower" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      //std::cout << Lmat;
      for( unsigned i=0; i<nVar; ++i ){
        if( !solve( Ymat, trimatl(Lmat), M[i] ) )  // indicate that Lmat is lower triangular
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        if( !solve( Gmat, trimatu(trans(Lmat)), Ymat ) )  // indicate that Lmat is lower triangular
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        for( unsigned j=0; j<=i; ++j ){
          if( !solve( Ymat, trimatl(Lmat), M[j] * Gmat ) )  // indicate that Lmat is lower triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          if( !solve( Hmat, trimatu(trans(Lmat)), Ymat ) )  // indicate that Lmat is lower triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          vRes[i*nVar+j] = - arma::trace( Hmat );
          if( j<i ) vRes[i+j*nVar] = vRes[i*nVar+j];
        }
      }
#else
      CPPL::dgematrix Lmat;
      std::vector<int> IPIV;
      if( dsytrf( Mmat, Lmat, IPIV ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      CPPL::dgematrix Gmat;
      for( unsigned i=0; i<nVar; ++i ){
        if( dsytrs( Lmat, IPIV, M[i].to_dgematrix(), Gmat ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        for( unsigned j=0; j<=i; ++j ){
          CPPL::dgematrix Hmat, Rmat = M[j] * Gmat;
          if( dsytrs( Lmat, IPIV, Rmat, Hmat ) )
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          //std::cout << Hmat;
          for( int k=0; k<Hmat.n; ++k )
            if( !k ) vRes[i*nVar+j] = -Hmat(0,0);
            else     vRes[i*nVar+j] -= Hmat(k,k);
          if( j<i ) vRes[i+j*nVar] = vRes[i*nVar+j];
        }
      }
#endif
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nVar*nVar );
      std::cout << "FFDOptHess::eval: FFVar\n"; 

      FFVar** ppRes = operator()( nVar, vVar );
      for( unsigned j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
      assert( nRes == nVar*nVar );
      std::cout << "FFDOptHess::eval: FFDep\n"; 

      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
    }

  // Properties
  std::string name
    ()
    const
    { return "DOPTHESS"; }
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

template<unsigned int ID>
inline void
FFDOpt<ID>::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
  assert( nRes == 1 && nVar == M.size() && M.begin() != M.end() );
  std::cout << "FFDOpt::eval: fadbad::F<double>\n";

  std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double vResVal;
  eval( 1, &vResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = vResVal;
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFDOptGrad<ID> DOptGrad;
  std::vector<double> vDOptGrad( nVar ); 
  DOptGrad.eval( nVar, vDOptGrad.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j] == 0. ) continue;
      vRes[0][j] += vDOptGrad[i] * vVar[i][j];
    }
  }
}

template<unsigned int ID>
inline void
FFDOpt<ID>::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
  assert( nRes == 1 && nVar == M.size() && M.begin() != M.end() );
  std::cout << "FFDOpt::deriv: FFVar\n";

  FFDOptGrad<ID> DOptGrad;
  for( unsigned i=0; i<nVar; ++i )
    vDer[0][i] = DOptGrad( i, nVar, vVar );
}

template<unsigned int ID>
inline void
FFDOptGrad<ID>::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
  assert( nRes == nVar && nVar == M.size() && M.begin() != M.end() );
  std::cout << "FFDOptGrad::deriv: FFVar\n";

  FFDOptHess<ID> DOptHess;
  for( unsigned i=0; i<nVar; ++i )
    for( unsigned j=0; j<nVar; ++j )
      vDer[i][j] = DOptHess( i*nVar+j, nVar, vVar );
}

}

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  mc::FFGraph< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFDOptHess<0> > DAG;
  const unsigned NP = 4;
  const unsigned NS = mc::FFDOptBase::read( NP, "test4.fim" );//, true ); 
  mc::FFVar S[NS];
  double S0[NS];
  for( unsigned int i=0; i<NS; i++ ){
    S[i].set( &DAG );
    S0[i] = 1./NS;
  }
  mc::FFDOpt<0> DOpt;

#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFDOptHess<0> > NLP;
  NLP.options.DISPLEVEL = 1;
  NLP.options.MAXITER   = 200;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_SNOPT< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFDOptHess<0> >::Options::FSYM;
  NLP.options.GRADCHECK = false;
  NLP.options.MAXTHREAD = 0;
#else
  mc::NLPSLV_IPOPT< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFDOptHess<0> > NLP;
  NLP.options.DISPLEVEL = 5;
  NLP.options.MAXITER   = 200;
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = mc::NLPSLV_IPOPT< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFDOptHess<0> >::Options::FSYM;//BSYM;
  NLP.options.HESSMETH  = mc::NLPSLV_IPOPT< mc::FFDOpt<0>, mc::FFDOptGrad<0>, mc::FFDOptHess<0> >::Options::LBFGS;//EXACT;
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

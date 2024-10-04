// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef CANON__FFDOE_HPP
#define CANON__FFDOE_HPP

#include <fstream>
#include <iomanip>
#include <armadillo>

#define MC__FFBRCRIT_LOG
#undef  MC__FFDCRIT_EIG
#define MC__FFFIMCrit_CHECK

namespace mc
{

////////////////////////////////////////////////////////////////////////
// EXTERNAL OPERATIONS
////////////////////////////////////////////////////////////////////////
struct FFDOEBase
{
  // Criterion type
  enum TYPE{
    AOPT=0,
    DOPT,
    EOPT,
    BROPT
  };

  // Selected DOE criterion
  static TYPE type;

  // Selected parameter scaling
  static arma::mat scaling;

  // Set input scaling
  static void set_scaling
    ( std::vector<double> const& vscaling )
    {
      if( vscaling.size() )
        scaling = arma::inv( arma::diagmat( arma::vec( vscaling ) ) );
      else
        scaling.reset();
      //std::cout << scaling;
    }

  //! @brief Selected output variances
  static arma::mat sigmayinv;

  // Set output variance
  static void set_noise
    ( std::vector<double> const& voutvar )
    {
      if( voutvar.size() )
        sigmayinv = arma::inv( arma::diagmat( arma::vec( voutvar ) ) );
      else
        sigmayinv.reset();
      //std::cout << sigmayinv;
    }

  // Selected parameter weights
  static arma::vec weighting;

  // Set input scaling
  static void set_weighting
    ( std::vector<double> const& vweighting )
    {
      if( vweighting.size() )
        weighting = arma::vec( vweighting );
      else
        weighting.reset();
      //std::cout << weighting;
    }

  // Selected parameter weights
  static std::set<std::pair<unsigned,unsigned>>* parsubset;

  // Compute atomic Bayes Risk
  static double atom_BR
    ( std::vector< arma::vec > const& yj, std::vector< arma::vec > const& yk,
      std::vector<double> const& eff )
    {
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      for( unsigned i=0; i<eff.size(); ++i ){
        arma::vec const& Ejk  = yj.at(i) - yk.at(i);
        if( !sigmayinv.empty() ) Et_Vinv_E += eff[i] * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += eff[i] * Ejk.t() * Ejk;
      }
      return std::exp( -0.125 * Et_Vinv_E(0,0) );
    }
};

inline FFDOEBase::TYPE FFDOEBase::type = FFDOEBase::DOPT;
inline arma::mat FFDOEBase::scaling;
inline arma::vec FFDOEBase::weighting;
inline arma::mat FFDOEBase::sigmayinv;
inline std::set<std::pair<unsigned,unsigned>>* FFDOEBase::parsubset = nullptr;

////////////////////////////////////////////////////////////////////////

class FFDOECrit
: public FFOp,
  public FFDOEBase
{
public:

  // Default Constructor
  FFDOECrit
    ()
    : FFOp( EXTERN )
    {}

  // Define operation
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      return **insert_external_operation( *this, 1, nVar, pVar );
    }

  FFVar& operator()
    ( unsigned const nVar, FFVar const*const* ppVar )
    const
    {
      return **insert_external_operation( *this, 1, nVar, ppVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFDOECrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFDOECRIT_CHECK
      assert( nRes == 1 );
#endif
      vRes[0] = operator()( nVar, vVar );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFDOECRIT_CHECK
      assert( nRes == 1 );
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
    }

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

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
    { 
      switch( FFDOEBase::type ){
        case AOPT: return "-tr Inv";
        case DOPT: return "log Det";
        case EOPT: return "min Eig";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradDOECrit
: public FFOp,
  public FFDOEBase
{
public:

  // Default Constructor
  FFGradDOECrit
    ()
    : FFOp( EXTERN )
    {}

  // Define operation
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar )
    const
    {
      return *(insert_external_operation( *this, nVar, nVar, pVar )[idep]);
    }

  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar )
    const
    {
      return insert_external_operation( *this, nVar, nVar, pVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradDOECrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADDOECRIT_CHECK
      assert( nRes == nVar );
#endif
      FFVar** ppRes = operator()( nVar, vVar );
      for( unsigned j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADDOECRIT_CHECK
      assert( nRes == nVar );
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
    }

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      switch( FFDOEBase::type ){
        case AOPT: return "-Grad tr Inv";
        case DOPT: return "Grad log Det";
        case EOPT: return "Grad min Eig";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline void
FFDOECrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFDOECrit::eval: double\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif
      
  unsigned int nDim = std::round( std::sqrt(2*nVar+0.25) - 0.5 );
  arma::mat FIM( nDim, nDim, arma::fill::none );
  for( unsigned i=0, l=0; i<nDim; ++i )
    for( unsigned j=i; j<nDim; ++j, ++l )
      if( i == j ) FIM(i,i) = vVar[l]; 
      else         FIM(i,j) = FIM(j,i) = vVar[l];
  if( scaling.n_elem ) FIM = scaling * FIM * scaling;
#ifdef MC__FFDOECRIT_DEBUG
  std::cout << "FIM:\n" << FIM;
  std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

  switch( FFDOEBase::type ){
    case AOPT:
    {
      arma::vec FIMEIGVAL;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      vRes[0] = 0.;
      for( unsigned k=0; k<nDim; ++k )
        vRes[0] -= 1./FIMEIGVAL(k);
      break;
    }
    
    case DOPT:
    {
#if defined( MC__FFDCRIT_EIG )
      arma::vec FIMEIGVAL;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      vRes[0] = 0.;
      for( unsigned k=0; k<nDim; ++k )
        vRes[0] += std::log( FIMEIGVAL(k) );
#else
      if( arma::rank( FIM ) < nDim || !arma::log_det_sympd( vRes[0], FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#endif
      break;
    }
    
    case EOPT:
    {
      arma::vec FIMEIGVAL;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      vRes[0] = FIMEIGVAL(0);
      break;
    }
    default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
  }
#ifdef MC__FFDOECRIT_DEBUG
  std::cout << name() << ": " << vRes[0] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradDOECrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADDOECRIT_TRACE
  std::cout << "FFGradDOECrit::eval: double\n"; 
#endif
#ifdef MC__FFGRADDOECRIT_CHECK
  assert( nRes == nVar );
#endif

  unsigned int nDim = std::round( std::sqrt(2*nVar+0.25) - 0.5 );
  arma::mat FIM( nDim, nDim, arma::fill::none );
  for( unsigned i=0, l=0; i<nDim; ++i )
    for( unsigned j=i; j<nDim; ++j, ++l )
      if( i == j ) FIM(i,i) = vVar[l]; 
      else         FIM(i,j) = FIM(j,i) = vVar[l];
  if( scaling.n_elem ) FIM = scaling * FIM * scaling;
#ifdef MC__FFGRADDOECRIT_DEBUG
  std::cout << "FIM: " << FIM;
  std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

  switch( FFDOEBase::type ){
    case AOPT:
    {
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
      std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
      for( unsigned i=0, l=0; i<nDim; ++i )
        for( unsigned j=i; j<nDim; ++j, ++l ){
          vRes[l] = 0.;
          for( unsigned k=0; k<nDim; ++k )
            if( scaling.n_elem )
              vRes[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(i,k) )
                       * (scaling(i,i)*scaling(j,j)) / (FIMEIGVAL(k)*FIMEIGVAL(k));
            else
              vRes[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(j,k) )
                       / (FIMEIGVAL(k)*FIMEIGVAL(k));
        }
      break;
    }
    
    case DOPT:
    {
#if defined( MC__FFDCRIT_EIG )
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
      std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
      for( unsigned i=0, l=0; i<nDim; ++i )
        for( unsigned j=i; j<nDim; ++j, ++l ){
          vRes[l] = 0.;
          for( unsigned k=0; k<nDim; ++k )
            if( scaling.n_elem )
              vRes[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(j,k) )
                       * (scaling(i,i)*scaling(j,j)) / FIMEIGVAL(k);
            else
              vRes[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(j,k) )
                       / FIMEIGVAL(k);
        }
#else
      arma::mat L, X, Y, E( nDim, nDim, arma::fill::none );
      if( !arma::chol( L, FIM, "lower" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM Cholesky decomposition:\n" << L;
#endif
      for( unsigned i=0, l=0; i<nDim; ++i ){
        for( unsigned j=i; j<nDim; ++j, ++l ){
          E.zeros();
          E(i,j) = E(j,i) = ( scaling.n_elem? scaling(i,i)*scaling(j,j): 1. );
          if( !solve( Y, trimatl(L), E ) )  // indicate that L is lower triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          if( !solve( X, trimatu(trans(L)), Y ) )  // indicate that L^T is upper triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          vRes[l] = arma::trace( X );
        }
      }
#endif
      break;
    }

    case EOPT:
    {
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM min eigenvalue: "  << FIMEIGVAL(0) << std::endl;
      std::cout << "FIM min eigenvector: " << trans(FIMEIGVEC.col(0));
#endif
      for( unsigned i=0, l=0; i<nDim; ++i )
        for( unsigned j=i; j<nDim; ++j, ++l )
          if( scaling.n_elem )
            vRes[l] = (i==j? FIMEIGVEC(i,0)*FIMEIGVEC(i,0): 2*FIMEIGVEC(i,0)*FIMEIGVEC(j,0) )
                    * (scaling(i,i)*scaling(j,j));
          else
            vRes[l] = (i==j? FIMEIGVEC(i,0)*FIMEIGVEC(i,0): 2*FIMEIGVEC(i,0)*FIMEIGVEC(j,0) );
      break;
    }
    default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
  }
#ifdef MC__FFGRADDOECRIT_DEBUG
  for( unsigned i=0, l=0; i<nDim; ++i )
    for( unsigned j=i; j<nDim; ++j, ++l )
      std::cout << name() << " [" << i << "," << j << "]: " << vRes[l] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFDOECrit::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFDOECrit::eval: fadbad::F<FFVar>\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  vRes[0] = operator()( nVar, vVarVal.data() );
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradDOECrit GradDOECrit;
  FFVar const*const* vGradDOECrit = GradDOECrit( nVar, vVarVal.data() ); 
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *vGradDOECrit[i] * vVar[i][j];
    }
  }
}

inline void
FFDOECrit::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFDOECrit::eval: fadbad::F<double>\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif

  std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double ResVal; 
  eval( 1, &ResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = ResVal;
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradDOECrit GradDOECrit;
  std::vector<double> vGradDOECrit( nVar ); 
  GradDOECrit.eval( nVar, vGradDOECrit.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j] == 0. ) continue;
      vRes[0][j] += vGradDOECrit[i] * vVar[i][j];
    }
  }
}

inline void
FFDOECrit::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFDOECrit::deriv:\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif

  FFGradDOECrit GradDOECrit;
  FFVar const*const* vGradDOECrit = GradDOECrit( nVar, vVar );
  for( unsigned i=0; i<nVar; ++i )
    vDer[0][i] = *vGradDOECrit[i];
}

////////////////////////////////////////////////////////////////////////

class FFDOEEff
: public FFOp,
  public FFDOEBase
{
public:

  // Default Constructor
  FFDOEEff
    ()
    : FFOp( EXTERN )
    {}

  // Define operation
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::mat > >* vFIM )
    const
    {
#ifdef MC__FFDOEEFF_CHECK
      assert( vFIM );
#endif
      data = vFIM; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      unsigned const nRes = vFIM->size();
      return *(insert_external_operation( *this, nRes, nVar, pVar )[idep]);
    }

  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::mat > >* vFIM )
    const
    {
#ifdef MC__FFDOEEFF_CHECK
      assert( vFIM );
#endif
      data = vFIM; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      unsigned const nRes = vFIM->size();
      return insert_external_operation( *this, nRes, nVar, pVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFDOEEff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFDOEEFF_TRACE
      std::cout << "FFDOEEff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
    }
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

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
    { 
      switch( FFDOEBase::type ){
        case AOPT: return "-tr Inv";
        case DOPT: return "log Det";
        case EOPT: return "min Eig";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradDOEEff
: public FFOp,
  public FFDOEBase
{
public:

  // Default Constructor
  FFGradDOEEff
    ()
    : FFOp( EXTERN )
    {}

  // Define operation
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::mat > >* vFIM )
    const
    {
#ifdef MC__FFGRADDOEEFF_CHECK
      assert( vFIM );
#endif
      data = vFIM; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      unsigned const nRes = vFIM->size();
      return *(insert_external_operation( *this, nRes * nVar, nVar, pVar )[idep]);
    }
  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::mat > >* vFIM )
    const
    {
#ifdef MC__FFGRADDOEEFF_CHECK
      assert( vFIM );
#endif
      data = vFIM; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      unsigned const nRes = vFIM->size();
      return insert_external_operation( *this, nRes * nVar, nVar, pVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradDOEEff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADDOEEFF_TRACE
      std::cout << "FFGradDOEEff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j )
        vRes[j] = vRes[0];
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      switch( FFDOEBase::type ){
        case AOPT: return "-Grad tr Inv";
        case DOPT: return "Grad log Det";
        case EOPT: return "Grad min Eig";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline void
FFDOEEff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFDOEEff::eval: FFVar\n";
#endif
#ifdef MC__FFDOEEFF_CHECK
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size() && nVar == vFIM->back().size() );
#endif

  //FFVar** ppRes = operator()( nVar, vVar, vFIM );
  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFGradDOEEff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADDOEEFF_TRACE
  std::cout << "FFGradDOEEff::eval: FFVar\n";
#endif
#ifdef MC__FFGRADDOEEFF_CHECK
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size()*nVar && nVar == vFIM->back().size() );
#endif

  //FFVar** ppRes = operator()( nVar, vVar, vFIM );
  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFDOEEff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFDOEEff::eval: double\n";
#endif
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
#ifdef MC__FFDOEEFF_CHECK
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size() && nVar == vFIM->back().size() );
#endif

  arma::mat FIM;
  for( unsigned s=0; s<nRes; ++s ){
    for( unsigned i=0; i<nVar; ++i )
      if( !i ) FIM  = vVar[0] * vFIM->at(s).at(0);
      else     FIM += vVar[i] * vFIM->at(s).at(i);
    if( scaling.n_elem ) FIM = scaling * FIM * scaling;
#ifdef MC__FFDOEEFF_DEBUG
    std::cout << "FIM: " << FIM;
    std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

    switch( FFDOEBase::type ){
      case AOPT:
      {
        arma::vec FIMEIGVAL;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        vRes[s] = 0.;
        for( unsigned k=0; k<FIM.n_rows; ++k )
          vRes[s] -= 1./FIMEIGVAL(k);
        break;
      }
      
      case DOPT:
      {
#if defined( MC__FFDCRIT_EIG )
        arma::vec FIMEIGVAL;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        vRes[s] = 0.;
        for( unsigned k=0; k<FIM.n_rows; ++k )
          vRes[s] += std::log( FIMEIGVAL(k) );
#else
        if( arma::rank( FIM ) < FIM.n_rows || !arma::log_det_sympd( vRes[s], FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#endif
        break;
      }
      
      case EOPT:
      {
        arma::vec FIMEIGVAL;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        vRes[s] = FIMEIGVAL(0);
        break;
      }
      default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
    }
#ifdef MC__FFDOEEFF_DEBUG
    std::cout << name() << " [" << s << "]: " << vRes[s] << std::endl;
#endif
  }
#ifdef MC__FFDOEEFF_DEBUG
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradDOEEff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADDOEEFF_TRACE
  std::cout << "FFGradDOEEff::eval: double\n";
#endif
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
#ifdef MC__FFGRADDOEEFF_CHECK
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size()*nVar && nVar == vFIM->back().size() );
#endif

  unsigned const nUnc = vFIM->size();
  arma::mat FIM, FIMi;
  for( unsigned s=0; s<nUnc; ++s ){
    for( unsigned i=0; i<nVar; ++i )
      if( !i ) FIM  = vVar[0] * vFIM->at(s).at(0);
      else     FIM += vVar[i] * vFIM->at(s).at(i);
    if( scaling.n_elem ) FIM = scaling * FIM * scaling;
#ifdef MC__FFGRADDOEEFF_DEBUG
    std::cout << "FIM: " << FIM;
    std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

    switch( FFDOEBase::type ){
      case AOPT:
      {
        arma::vec FIMEIGVAL;
        arma::mat FIMEIGVEC;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
        std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
        std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
        for( unsigned i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = scaling * vFIM->at(s).at(i) * scaling;
          else                 FIMi = vFIM->at(s).at(i);
          vRes[s*nVar+i] = 0.;
          for( unsigned k=0; k<FIM.n_rows; ++k ){
            arma::mat const& Et_FIM_E = FIMEIGVEC.col(k).t() * FIMi * FIMEIGVEC.col(k); 
            vRes[s*nVar+i] += Et_FIM_E(0,0) / (FIMEIGVAL(k)*FIMEIGVAL(k));
          }
#ifdef MC__FFGRADDOEEFF_DEBUG
          std::cout << name() << " [" << s << "," << i << "]: " << vRes[s*nVar+i] << std::endl;
#endif
        }
        break;
      }
      
      case DOPT:
      {
#if defined( MC__FFDCRIT_EIG )
        arma::vec FIMEIGVAL;
        arma::mat FIMEIGVEC;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOEEFF_DEBUG
        std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
        std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
        for( unsigned i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = scaling * vFIM->at(s).at(i) * scaling;
          else                 FIMi = vFIM->at(s).at(i);
          vRes[s*nVar+i] = 0.;
          for( unsigned k=0; k<FIM.n_rows; ++k ){
            arma::mat const& Et_FIM_E = FIMEIGVEC.col(k).t() * FIMi * FIMEIGVEC.col(k); 
            vRes[s*nVar+i] += Et_FIM_E(0,0) / FIMEIGVAL(k);
          }
#ifdef MC__FFGRADDOEEFF_DEBUG
          std::cout << name() << " [" << s << "," << i << "]: " << vRes[s*nVar+i] << std::endl;
#endif
        }
#else
        arma::mat L, X, Y;
        if( !arma::chol( L, FIM, "lower" ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOEEFF_DEBUG
        std::cout << "FIM Cholesky decomposition: " << L;
#endif
        for( unsigned i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = scaling * vFIM->at(s).at(i) * scaling;
          else                 FIMi = vFIM->at(s).at(i);
          if( !arma::solve( Y, trimatl(L), FIMi ) )  // indicate that L is lower triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          if( !arma::solve( X, trimatu(trans(L)), Y ) )  // indicate that L^T is upper triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          vRes[s*nVar+i] = arma::trace( X );
#ifdef MC__FFGRADDOEEFF_DEBUG
          std::cout << name() << " [" << s << "," << i << "]: " << vRes[s*nVar+i] << std::endl;
#endif
        }
#endif
        break;
      }
      
      case EOPT:
      {
        arma::vec FIMEIGVAL;
        arma::mat FIMEIGVEC;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOEEFF_DEBUG
        std::cout << "FIM min eigenvalue: "  << FIMEIGVAL(0) << std::endl;
        std::cout << "FIM min eigenvector: " << FIMEIGVEC.col(0);
#endif
        for( unsigned i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = scaling * vFIM->at(s).at(i) * scaling;
          else                 FIMi = vFIM->at(s).at(i);
          arma::mat const& Et_FIM_E = FIMEIGVEC.col(0).t() * FIMi * FIMEIGVEC.col(0); 
          vRes[s*nVar+i] = Et_FIM_E(0,0);
#ifdef MC__FFGRADDOEEFF_DEBUG
          std::cout << name() << " [" << s << "," << i << "]: " << vRes[s*nVar+i] << std::endl;
#endif
        }
        break;
      }
      default: throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
    }
  }
#ifdef MC__FFGRADDOEEFF_DEBUG
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFDOEEff::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFDOEEff::eval: fadbad::F<FFVar>\n";
#endif
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
#ifdef MC__FFDOEEFF_CHECK
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size()*nVar && nVar == vFIM->back().size() );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar const*const* vDOpt = operator()( nVar, vVarVal.data(), vFIM );
  for( unsigned s=0; s<nRes; ++s ){
    vRes[s] = *vDOpt[s];
    for( unsigned i=0; i<nVar; ++i )
      vRes[s].setDepend( vVar[i] );
  }

  FFGradDOEEff DOptGrad;
  FFVar const*const* vDOptGrad = DOptGrad( nVar, vVarVal.data(), vFIM ); 
  for( unsigned s=0; s<nRes; ++s ){
    for( unsigned j=0; j<vRes[0].size(); ++j ){
      vRes[s][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
        vRes[s][j] += *vDOptGrad[s*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFDOEEff::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFDOEEff::eval: fadbad::F<double>\n";
#endif

  std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  std::vector<double> vResVal( nRes ); 
  eval( nRes, vResVal.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned s=0; s<nRes; ++s ){
    vRes[s] = vResVal[s];
    for( unsigned i=0; i<nVar; ++i )
      vRes[s].setDepend( vVar[i] );
  }

  FFGradDOEEff DOptGrad;
  DOptGrad.data = data;
  std::vector<double> vDOptGrad( nRes * nVar ); 
  DOptGrad.eval( nRes * nVar, vDOptGrad.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned s=0; s<nRes; ++s ){
    for( unsigned j=0; j<vRes[0].size(); ++j ){
      vRes[s][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j] == 0. ) continue;
        vRes[s][j] += vDOptGrad[s*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFDOEEff::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFDOEEff::deriv\n";
#endif
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
#ifdef MC__FFDOEEFF_CHECK
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size() && nVar == vFIM->back().size() );
#endif

  FFGradDOEEff DOptGrad;
  FFVar const*const* vDOptGrad = DOptGrad( nVar, vVar, vFIM ); 
  for( unsigned s=0; s<nRes; ++s )
    for( unsigned i=0; i<nVar; ++i )
      vDer[s][i] = *vDOptGrad[s*nVar+i];
}

////////////////////////////////////////////////////////////////////////

class FFBRCrit
: public FFOp,
  public FFDOEBase
{
public:

  // Default Constructor
  FFBRCrit
    ()
    : FFOp( EXTERN )
    {}

  static size_t nUNC;
  static size_t nOUT;

  // Define operation
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar, std::map<unsigned,double>* mEFF,
      unsigned int nUNC, unsigned int nOUT )
    const
    {
#ifdef MC__FFBRCRIT_CHECK
      assert( mEFF );
#endif
      data = mEFF; // no local copy - make sure mEFF isn't going out of scope!
      owndata = false;
      this->nUNC = nUNC;
      this->nOUT = nOUT;
#ifdef MC__FFBRCRIT_CHECK
      assert( nVar == mEFF->size()*nUNC*nOUT );
#endif
      return **insert_external_operation( *this, 1, nVar, pVar );
    }

  FFVar& operator()
    ( unsigned const nVar, FFVar const*const* ppVar, std::map<unsigned,double>* mEFF,
      unsigned int nUNC, unsigned int nOUT )
    const
    {
#ifdef MC__FFBRCRIT_CHECK
      assert( mEFF );
#endif
      data = mEFF; // no local copy - make sure mEFF isn't going out of scope!
      owndata = false;
      this->nUNC = nUNC;
      this->nOUT = nOUT;
#ifdef MC__FFBRCRIT_CHECK
      assert( nVar == mEFF->size()*nUNC*nOUT );
#endif
      return **insert_external_operation( *this, 1, nVar, ppVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFBRCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFBRCRIT_TRACE
      std::cout << "FFBRCrit::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
    }
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const;
    
  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
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
    { 
      switch( FFDOEBase::type ){
        case BROPT: return "Bayes Risk";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline size_t FFBRCrit::nUNC = 0;
inline size_t FFBRCrit::nOUT = 0;

class FFGradBRCrit
: public FFOp,
  public FFDOEBase
{
public:

  // Default constructor
  FFGradBRCrit
    ()
    : FFOp( EXTERN )
    {}

  static size_t nUNC;
  static size_t nOUT;

  // Define operation
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar, std::map<unsigned,double>* mEFF,
      unsigned int nUNC, unsigned int nOUT )
    const
    {
#ifdef MC__FFGRADBRCRIT_CHECK
      assert( mEFF );
#endif
      data = mEFF; // no local copy - make sure mEFF isn't going out of scope!
      owndata = false;
      this->nUNC = nUNC;
      this->nOUT = nOUT;
#ifdef MC__FFBRCRIT_CHECK
      assert( nVar == mEFF->size()*nUNC*nOUT );
#endif
      return *(insert_external_operation( *this, nVar, nVar, pVar )[idep]);
    }
    
  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar, std::map<unsigned,double>* mEFF,
      unsigned int nUNC, unsigned int nOUT )
    const
    {
#ifdef MC__FFGRADBRCRIT_CHECK
      assert( mEFF );
#endif
      data = mEFF; // no local copy - make sure mEFF isn't going out of scope!
      owndata = false;
      this->nUNC = nUNC;
      this->nOUT = nOUT;
#ifdef MC__FFBRCRIT_CHECK
      assert( nVar == mEFF->size()*nUNC*nOUT );
#endif
      return insert_external_operation( *this, nVar, nVar, pVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradBRCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADBRCRIT_TRACE
      std::cout << "FFGradBRCrit::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j )
        vRes[j] = vRes[0];
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      switch( FFDOEBase::type ){
        case BROPT: return "Grad Bayes Risk";
        default:    throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline size_t FFGradBRCrit::nUNC = 0;
inline size_t FFGradBRCrit::nOUT = 0;

inline void
FFBRCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::eval: FFVar\n";
#endif
  //std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  //assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == 1 );
  assert( nRes == 1 );
#endif

  //vRes[0] = operator()( nVar, vVar, mEFF, nUNC, nOUT );
  vRes[0] = **insert_external_operation( *this, 1, nVar, vVar );
}

inline void
FFGradBRCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADBRCRIT_TRACE
  std::cout << "FFGradBRCrit::eval: FFVar\n";
#endif
  //std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  //assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == nVar );
  assert( nRes == nVar );
#endif

  //FFVar** ppRes = operator()( nVar, vVar, mEFF, nUNC, nOUT );
  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFBRCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::eval: double\n";
#endif
  std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == 1 );
#endif
  size_t const inc = mEFF->size()*nOUT;

  auto BRval = [&]( unsigned j, unsigned k, double& res ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      size_t pj = j*inc;
      size_t pk = k*inc;
      size_t pi = 0;
      for( auto const& [Id,Eff] : *mEFF ){
//        std::cout << "y[" << j << "][" << pi << "] = " << arma::vec( const_cast<double*>(vVar+pj+pi), nOUT, false );
//        std::cout << "y[" << k << "][" << pi << "] = " << arma::vec( const_cast<double*>(vVar+pk+pi), nOUT, false );
        arma::vec const& Eijk = arma::vec( const_cast<double*>(vVar+pj+pi), nOUT, false )
                              - arma::vec( const_cast<double*>(vVar+pk+pi), nOUT, false );
        if( !sigmayinv.empty() )
          Et_Vinv_E += Eff * Eijk.t() * sigmayinv * Eijk;
        else
          Et_Vinv_E += Eff * Eijk.t() * Eijk;
        pi += nOUT;
      }
      double BRjk = std::exp( -0.125 * Et_Vinv_E(0,0) );
//      std::cout << "BR[" << j << "," << k << "] = " << BRjk << std::endl; 
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
      res += BRjk;
//      std::cout << "res[" << j << "," << k << "] = " << res << std::endl; 
  };

  vRes[0] = 0.;

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
      BRval( j, k, vRes[0] );
 
  // Use full set of uncertainty scenarios
  else
    for( unsigned j=0; j<nUNC-1; ++j )
      for( unsigned k=j+1; k<nUNC; ++k )
        BRval( j, k, vRes[0] );

#ifdef MC__FFBRCRIT_LOG
  vRes[0] = std::log( vRes[0] );
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << name() << " [" << 0 << "]: " << vRes[0] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradBRCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADBRCRIT_TRACE
  std::cout << "FFGradBRCrit::eval: double\n";
#endif
  std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == 1 );
#endif
  size_t const inc = mEFF->size()*nOUT;

#ifdef MC__FFBRCRIT_LOG
  auto BRder = [&]( unsigned j, unsigned k, double& crit, arma::vec& grad, arma::vec& tmp ){
#else
  auto BRder = [&]( unsigned j, unsigned k, arma::vec& grad, arma::vec& tmp ){
#endif
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      size_t pj = j*inc;
      size_t pk = k*inc;
      size_t pi = 0;
      for( auto const& [Id,Eff] : *mEFF ){
//        std::cout << "y[" << j << "][" << pi << "] = " << arma::vec( const_cast<double*>(vVar+pj+pi), nOUT, false );
//        std::cout << "y[" << k << "][" << pi << "] = " << arma::vec( const_cast<double*>(vVar+pk+pi), nOUT, false );
        arma::vec const& Eijk = arma::vec( const_cast<double*>(vVar+pj+pi), nOUT, false )
                              - arma::vec( const_cast<double*>(vVar+pk+pi), nOUT, false );
        if( !sigmayinv.empty() ){
          tmp.subvec(pi,pi+nOUT-1) = sigmayinv * Eijk;
          Et_Vinv_E += Eff * Eijk.t() * tmp.subvec(pi,pi+nOUT-1);
          tmp.subvec(pi,pi+nOUT-1) *= Eff/4;
        }
        else{
          Et_Vinv_E += Eff * Eijk.t() * Eijk;
          tmp.subvec(pi,pi+nOUT-1) = (Eff/4) * Eijk;
        }
        pi += nOUT;
      }
      double BRjk = std::exp( -0.125 * Et_Vinv_E(0,0) );
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
#ifdef MC__FFBRCRIT_LOG
      crit += BRjk;
#endif
      grad.subvec(pj,pj+inc-1) -= tmp * BRjk;
      grad.subvec(pk,pk+inc-1) += tmp * BRjk;
  };

#ifdef MC__FFBRCRIT_LOG
  double BRCrit = 0.;
#endif
  arma::vec GradBR( vRes, nRes, false );
  GradBR.zeros();
  arma::vec GradBRjk( inc, arma::fill::none );

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
#ifdef MC__FFBRCRIT_LOG
      BRder( j, k, BRCrit, GradBR, GradBRjk );
#else
      BRder( j, k, GradBR, GradBRjk );
#endif
 
  // Use full set of uncertainty scenarios
  else
    for( unsigned j=0; j<nUNC-1; ++j )
      for( unsigned k=j+1; k<nUNC; ++k )
#ifdef MC__FFBRCRIT_LOG
        BRder( j, k, BRCrit, GradBR, GradBRjk );
#else
        BRder( j, k, GradBR, GradBRjk );
#endif

#ifdef MC__FFBRCRIT_LOG
  GradBR /= BRCrit;
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << GradBR;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFBRCrit::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::eval: fadbad::F<FFVar>\n";
#endif
  std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == 1 );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  vRes[0] = operator()( nVar, vVarVal.data(), mEFF, nUNC, nOUT );
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradBRCrit GradBRCrit;
  FFVar const*const* vGradBRCrit = GradBRCrit( nVar, vVarVal.data(), mEFF, nUNC, nOUT ); 
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *vGradBRCrit[i] * vVar[i][j];
    }
  }
}

inline void
FFBRCrit::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::eval: fadbad::F<double>\n";
#endif

  std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double ResVal; 
  eval( 1, &ResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = ResVal;
  for( unsigned i=0; i<nVar; ++i ){
    vRes[0].setDepend( vVar[i] );
//    std::cout << "vVar[" << i << "] = " << vVarVal[i] << std::endl;
  }
  
  FFGradBRCrit GradBRCrit;
  GradBRCrit.nUNC = nUNC;
  GradBRCrit.nOUT = nOUT;
  GradBRCrit.data = data;
  std::vector<double> vGradBRCrit( nVar ); 
  GradBRCrit.eval( nVar, vGradBRCrit.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j] == 0. ) continue;
      vRes[0][j] += vGradBRCrit[i] * vVar[i][j];
    }
  }
}

inline void
FFBRCrit::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::deriv\n";
#endif
  std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == 1 );
#endif

  FFGradBRCrit GradBRCrit;
  FFVar const*const* vGradBRCrit = GradBRCrit( nVar, vVar, mEFF, nUNC, nOUT );
  for( unsigned i=0; i<nVar; ++i )
    vDer[0][i] = *vGradBRCrit[i];
}

////////////////////////////////////////////////////////////////////////

class FFBREff
: public FFOp,
  public FFDOEBase
{
public:
  // Constructors
  FFBREff
    ()
    : FFOp( EXTERN )
    {}

  // Declaration
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::vec > >* vOUT )
    const
    {
#ifdef MC__FFBREFF_CHECK
      assert( vOUT );
#endif
      data = vOUT; // no local copy - make sure vOUT isn't going out of scope!
      owndata = false;
      return **insert_external_operation( *this, 1, nVar, pVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFBREff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFBREFF_TRACE
      std::cout << "FFBREff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
    }
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const;
    
  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
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
    { 
      switch( FFDOEBase::type ){
        case BROPT: return "Bayes Risk";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradBREff
: public FFOp,
  public FFDOEBase
{
public:
  // Constructors
  FFGradBREff
    ()
    : FFOp( EXTERN )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::vec > >* vOUT )
    const
    {
#ifdef MC__FFGRADBREFF_CHECK
      assert( vOUT );
#endif
      data = vOUT; // no local copy - make sure vOUT isn't going out of scope!
      owndata = false;
      return *(insert_external_operation( *this, nVar, nVar, pVar )[idep]);
    }
  FFVar** operator()
    ( unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::vec > >* vOUT )
    const
    {
#ifdef MC__FFGRADBREFF_CHECK
      assert( vOUT );
#endif
      data = vOUT; // no local copy - make sure vOUT isn't going out of scope!
      owndata = false;
      return insert_external_operation( *this, nVar, nVar, pVar );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradBREff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADBREFF_TRACE
      std::cout << "FFGradBREff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j )
        vRes[j] = vRes[0];
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      switch( FFDOEBase::type ){
        case BROPT: return "Grad Bayes Risk";
        default:    throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline void
FFBREff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::eval: FFVar\n";
#endif
  //std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFBREFF_CHECK
  //assert( vOUT && !vOUT->empty() && nRes == 1 && nVar == vOUT->back().size() );
  assert( nRes == 1 );
#endif

  //vRes[0] = operator()( nVar, vVar, vOUT );
  vRes[0] = **insert_external_operation( *this, 1, nVar, vVar );
}

inline void
FFGradBREff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADBREFF_TRACE
  std::cout << "FFGradBREff::eval: FFVar\n";
#endif
  //std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFGRADBREFF_CHECK
  //assert( vOUT && !vOUT->empty() && nRes == nVar && nVar == vOUT->back().size() );
  assert( nRes == nVar );
#endif

  //FFVar** ppRes = operator()( nVar, vVar, vOUT );
  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );;
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFBREff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::eval: double\n";
#endif
  std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFBREFF_CHECK
  assert( vOUT && !vOUT->empty() && nRes == 1 && nVar == vOUT->back().size() );
#endif

  auto BRval = [&]( unsigned j, unsigned k, double& res ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i] == 0. ) continue;
        arma::vec const& Ejk  = vOUT->at(j).at(i) - vOUT->at(k).at(i);
        if( !sigmayinv.empty() ) Et_Vinv_E += vVar[i] * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += vVar[i] * Ejk.t() * Ejk;
      }
      if( !weighting.empty() ) res += std::sqrt( weighting(j)*weighting(k) ) * std::exp( -0.125 * Et_Vinv_E(0,0) );
      else                     res += std::exp( -0.125 * Et_Vinv_E(0,0) );
  };

  vRes[0] = 0.;

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
      BRval( j, k, vRes[0] );
 
  // Use full set of uncertainty scenarios
  else
    for( unsigned j=0; j<vOUT->size()-1; ++j )
      for( unsigned k=j+1; k<vOUT->size(); ++k )
        BRval( j, k, vRes[0] );

#ifdef MC__FFBRCRIT_LOG
  vRes[0] = std::log( vRes[0] );
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << name() << " [" << 0 << "]: " << vRes[0] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradBREff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADBREFF_TRACE
  std::cout << "FFGradBREff::eval: double\n";
#endif
  std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFGRADBREFF_CHECK
  assert( vOUT && !vOUT->empty() && nRes == nVar && nVar == vOUT->back().size() );
#endif

#ifdef MC__FFBRCRIT_LOG
  auto BRder = [&]( unsigned j, unsigned k, double& crit, arma::vec& grad, arma::vec& tmp ){
#else
  auto BRder = [&]( unsigned j, unsigned k, arma::vec& grad, arma::vec& tmp ){
#endif
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      for( unsigned i=0; i<nVar; ++i ){
        arma::vec const& Ejk   = vOUT->at(j).at(i) - vOUT->at(k).at(i);
        if( !sigmayinv.empty() ){
          tmp.subvec(i,i) = -0.125 * Ejk.t() * sigmayinv * Ejk;
          Et_Vinv_E += vVar[i] * tmp(i);
        }
        else{
          tmp.subvec(i,i) = -0.125 * Ejk.t() * Ejk;
          Et_Vinv_E += vVar[i] * tmp(i);
        }
      }
      double BRjk = std::exp( Et_Vinv_E(0,0) );
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
#ifdef MC__FFBRCRIT_LOG
      crit += BRjk;
#endif
      grad += tmp * BRjk;
  };

#ifdef MC__FFBRCRIT_LOG
  double BRCrit = 0.;
#endif
  arma::vec GradBR( vRes, nRes, false );
  GradBR.zeros();
  arma::vec GradBRjk( nVar, arma::fill::none );

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
#ifdef MC__FFBRCRIT_LOG
      BRder( j, k, BRCrit, GradBR, GradBRjk );
#else
      BRder( j, k, GradBR, GradBRjk );
#endif
 
  // Use full set of uncertainty scenarios
  else
    for( unsigned j=0; j<vOUT->size()-1; ++j )
      for( unsigned k=j+1; k<vOUT->size(); ++k )
#ifdef MC__FFBRCRIT_LOG
        BRder( j, k, BRCrit, GradBR, GradBRjk );
#else
        BRder( j, k, GradBR, GradBRjk );
#endif

#ifdef MC__FFBRCRIT_LOG
  GradBR /= BRCrit;
#endif

#ifdef MC__FFBREFF_DEBUG
  for( unsigned i=0; i<nVar; ++i )
    std::cout << name() << " [" << i << "]: " << vRes[i] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}
/*
inline void
FFGradBREff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADBREFF_TRACE
  std::cout << "FFGradBREff::eval: double\n";
#endif
  std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFGRADBREFF_CHECK
  assert( vOUT && !vOUT->empty() && nRes == nVar && nVar == vOUT->back().size() );
#endif

#ifdef MC__FFBRCRIT_LOG
  double BRCrit = 0.;
#endif
  arma::vec GradBR( vRes, nRes, false );
  GradBR.zeros();
  arma::vec GradBRjk( nVar, arma::fill::none );
  for( unsigned j=0; j<vOUT->size(); ++j ){
    for( unsigned k=j+1; k<vOUT->size(); ++k ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      for( unsigned i=0; i<nVar; ++i ){
        arma::vec const& Ejk   = vOUT->at(j).at(i) - vOUT->at(k).at(i);
        if( !sigmayinv.empty() ){
          GradBRjk.subvec(i,i) = -0.125 * Ejk.t() * sigmayinv * Ejk;
          Et_Vinv_E += vVar[i] * GradBRjk(i);
        }
        else{
          GradBRjk.subvec(i,i) = -0.125 * Ejk.t() * Ejk;
          Et_Vinv_E += vVar[i] * GradBRjk(i);
        }
      }
      double BRjk = std::exp( Et_Vinv_E(0,0) );
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
#ifdef MC__FFBRCRIT_LOG
      BRCrit += BRjk;
#endif
      GradBR += GradBRjk * BRjk;
    }
  }
#ifdef MC__FFBRCRIT_LOG
  GradBR /= BRCrit;
#endif

#ifdef MC__FFBREFF_DEBUG
  for( unsigned i=0; i<nVar; ++i )
    std::cout << name() << " [" << i << "]: " << vRes[i] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}
*/
inline void
FFBREff::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::eval: fadbad::F<FFVar>\n";
#endif
  std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFGRADBREFF_CHECK
  assert( vOUT && !vOUT->empty() && nRes == 1 && nVar == vOUT->back().size() );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  vRes[0] = operator()( nVar, vVarVal.data(), vOUT );
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradBREff GradBREff;
  FFVar const*const* vGradBREff = GradBREff( nVar, vVarVal.data(), vOUT ); 
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *vGradBREff[i] * vVar[i][j];
    }
  }
}

inline void
FFBREff::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::eval: fadbad::F<double>\n";
#endif

  std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double ResVal; 
  eval( 1, &ResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = ResVal;
  for( unsigned i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradBREff GradBREff;
  GradBREff.data = data;
  std::vector<double> vGradBREff( nVar ); 
  GradBREff.eval( nVar, vGradBREff.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j] == 0. ) continue;
      vRes[0][j] += vGradBREff[i] * vVar[i][j];
    }
  }
}

inline void
FFBREff::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::deriv\n";
#endif
  std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFGRADBREFF_CHECK
  assert( vOUT && !vOUT->empty() && nRes == 1 && nVar == vOUT->back().size() );
#endif

  FFGradBREff GradBREff;
  FFVar const*const* vGradBREff = GradBREff( nVar, vVar, vOUT );
  for( unsigned i=0; i<nVar; ++i )
    vDer[0][i] = *vGradBREff[i];
}

////////////////////////////////////////////////////////////////////////

class FFFIM
: public FFOp
{
private:
  // Number of parameters
  mutable size_t _nP;
  // Number of outputs
  mutable size_t _nY;

public:
  // Default constructor
  FFFIM
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFFIM
    ( FFFIM const& Op )
    : FFOp( Op ),
      _nP( Op._nP ),
      _nY( Op._nY )
    {}

  // Define operation
  FFVar& operator()
    ( unsigned const idep, size_t const nP, size_t const nY, FFVar const* Yp, std::vector<double>* Yvar )
    const
    {
#ifdef MC__FFFIM_CHECK
      assert( nP && nY && ( Yvar.empty() || Yvar.size() == nY ) );
#endif
      data = Yvar; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      _nP = nP;
      _nY = nY;
      return *(insert_external_operation( *this, nP*(nP+1)/2, nP*nY, Yp )[idep]);
    }

  FFVar** operator()
    ( size_t const nP, size_t const nY, FFVar const* Yp, std::vector<double>* Yvar )
    const
    {
#ifdef MC__FFFIM_CHECK
      assert( nP && nY && ( Yvar.empty() || Yvar.size() == nY ) );
#endif
      data = Yvar; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      _nP = nP;
      _nY = nY;
      return insert_external_operation( *this, nP*(nP+1)/2, nP*nY, Yp );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
      else if( idU == typeid( FFExpr ) )
        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFFIM::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  template< typename G >
  void eval
    ( unsigned const nRes, G* vRes, unsigned const nVar, G const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFFIM_TRACE
      std::cout << "FFFIM::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
    }
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

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
    { 
      return "FIM";
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradFIM
: public FFOp
{

  friend class FFFIM;

private:
  // Number of parameters
  mutable size_t _nP;
  // Number of outputs
  mutable size_t _nY;

public:

  // Default Constructor
  FFGradFIM
    ()
    : FFOp( EXTERN )
    {}

  // Copy constructor
  FFGradFIM
    ( FFGradFIM const& Op )
    : FFOp( Op ),
      _nP( Op._nP ),
      _nY( Op._nY )
    {}

  // Define operation
  FFVar& operator()
    ( unsigned const idep, size_t const nP, size_t const nY, FFVar const* Yp, std::vector<double>* Yvar )
    const
    {
#ifdef MC__FFGRADFIM_CHECK
      assert( nP && nY && ( Yvar.empty() || Yvar.size() == nY ) );
#endif
      data = Yvar; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      _nP = nP;
      _nY = nY;
      return *(insert_external_operation( *this, nP*(nP+1)/2*nP*nY, nP*nY, Yp )[idep]);
    }

  FFVar** operator()
    ( size_t const nP, size_t const nY, FFVar const* Yp, std::vector<double>* Yvar )
    const
    {
#ifdef MC__FFGRADFIM_CHECK
      assert( nP && nY && ( Yvar.empty() || Yvar.size() == nY ) );
#endif
      data = Yvar; // no local copy - make sure vFIM isn't going out of scope!
      owndata = false;
      _nP = nP;
      _nY = nY;
      return insert_external_operation( *this, nP*(nP+1)/2*nP*nY, nP*nY, Yp );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradFIM::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADFIM_TRACE
      std::cout << "FFGradFIM::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( unsigned j=1; j<nRes; ++j )
        vRes[j] = vRes[0];
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;
    
  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      return "Grad FIM";
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

template< typename G >
inline void
FFFIM::eval
( unsigned const nRes, G* vRes, unsigned const nVar, G const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFFIM_TRACE
  std::cout << "FFFIM::eval: Generic\n";
#endif
  std::vector<double> const& Yvar = *static_cast<std::vector<double>*>( data );
#ifdef MC__FFFIM_CHECK
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  for( unsigned k=0; k<_nY; k++ )
    for( unsigned i=0, ij=0; i<_nP; ++i )
      for( unsigned j=i; j<_nP; ++j, ++ij ){
        if( Yvar.size() == _nY ){
          if( !k ) vRes[ij]  = (vVar[_nY*i+k] * vVar[_nY*j+k]) / Yvar[k];
          else     vRes[ij] += (vVar[_nY*i+k] * vVar[_nY*j+k]) / Yvar[k];
        }
        else{
          if( !k ) vRes[ij]  = vVar[_nY*i+k] * vVar[_nY*j+k];
          else     vRes[ij] += vVar[_nY*i+k] * vVar[_nY*j+k];
        }
      }
}

inline void
FFFIM::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFFIM_TRACE
  std::cout << "FFFIM::eval: FFVar\n";
#endif
#ifdef MC__FFFIM_CHECK
  std::vector<double>* Yvar = static_cast<std::vector<double>*>( data );
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFGradFIM::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADFIM_TRACE
  std::cout << "FFGradFIM::eval: FFVar\n";
#endif
#ifdef MC__FFGRADFIM_CHECK
  std::vector<double>* Yvar = static_cast<std::vector<double>*>( data );
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFFIM::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFFIM_TRACE
  std::cout << "FFFIM::eval: double\n";
#endif
  std::vector<double> const& Yvar = *static_cast<std::vector<double>*>( data );
#ifdef MC__FFFIM_CHECK
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  for( unsigned k=0; k<_nY; k++ )
    for( unsigned i=0, ij=0; i<_nP; ++i )
      for( unsigned j=i; j<_nP; ++j, ++ij ){
        if( !k )           vRes[ij]  = 0.;
        if( Yvar.empty() ) vRes[ij] +=  vVar[_nY*i+k] * vVar[_nY*j+k];
        else               vRes[ij] += (vVar[_nY*i+k] * vVar[_nY*j+k]) / Yvar[k];
      }
#ifdef MC__FFFIM_DEBUG
  for( unsigned i=0, ij=0; i<_nP; ++i )
    for( unsigned j=i; j<_nP; ++j, ++ij )
      std::cout << "FIM[" << ij << "] = " << vRes[ij] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradFIM::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADDOEEFF_TRACE
  std::cout << "FFGradFIM::eval: double\n";
#endif
  std::vector<double> const& Yvar = *static_cast<std::vector<double>*>( data );
#ifdef MC__FFFIM_CHECK
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  for( unsigned l=0, kl=0; l<_nP; ++l )
    for( unsigned k=0; k<_nY; ++k, ++kl )
      for( unsigned i=0, ij=0; i<_nP; ++i )
        for( unsigned j=i; j<_nP; ++j, ++ij ){
          if( l == i && i == j ) vRes[ij*nVar+kl] = 2*vVar[_nY*i+k];
          else if( l == i )      vRes[ij*nVar+kl] = vVar[_nY*j+k];
          else if( l == j )      vRes[ij*nVar+kl] = vVar[_nY*i+k];
          else{                  vRes[ij*nVar+kl] = 0.; continue; }
          if( !Yvar.empty() )    vRes[ij*nVar+kl] /= Yvar[k];
#ifdef MC__FFGRADFIM_DEBUG
          std::cout << "Grad FIM[" << ij << "][" << kl << "] = " << vRes[ij*nVar+kl] << std::endl;
#endif
        }
#ifdef MC__FFFIM_DEBUG
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFFIM::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFIM_TRACE
  std::cout << "FFFIM::eval: fadbad::F<FFVar>\n";
#endif
  std::vector<double>& Yvar = *static_cast<std::vector<double>*>( data );
#ifdef MC__FFFIM_CHECK
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar const*const* vResVal = insert_external_operation( *this, nRes, nVar, vVarVal.data() );
  for( unsigned k=0; k<nRes; ++k ){
    vRes[k] = *vResVal[k];
    for( unsigned i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
  }

  FFGradFIM OpGradFIM;
  FFVar const*const* vGradFIM = OpGradFIM( _nP, _nY, vVarVal.data(), &Yvar ); 
  for( unsigned k=0; k<nRes; ++k ){
    for( unsigned j=0; j<vRes[0].size(); ++j ){
      vRes[k][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
        vRes[k][j] += *vGradFIM[k*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFFIM::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFIM_TRACE
  std::cout << "FFFIM::eval: fadbad::F<double>\n";
#endif
#ifdef MC__FFFIM_CHECK
  std::vector<double> const& Yvar = *static_cast<std::vector<double>*>( data );
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  std::vector<double> vResVal( nRes ); 
  eval( nRes, vResVal.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned k=0; k<nRes; ++k ){
    vRes[k] = vResVal[k];
    for( unsigned i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
  }

  FFGradFIM OpGradFIM;
  OpGradFIM.data = data;
  OpGradFIM._nP = _nP;
  OpGradFIM._nY = _nY;
  std::vector<double> vGradFIM( nRes * nVar ); 
  OpGradFIM.eval( nRes * nVar, vGradFIM.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned k=0; k<nRes; ++k ){
    for( unsigned j=0; j<vRes[0].size(); ++j ){
      vRes[k][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j] == 0. ) continue;
        vRes[k][j] += vGradFIM[k*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFFIM::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFFIM_TRACE
  std::cout << "FFFIM::deriv\n";
#endif
  std::vector<double>& Yvar = *static_cast<std::vector<double>*>( data );
#ifdef MC__FFFIM_CHECK
  assert( _nP && _nY && ( Yvar.empty() || Yvar.size() == _nY ) );
#endif

  FFGradFIM OpGradFIM;
  FFVar const*const* vGradFIM = OpGradFIM( _nP, _nY, vVar, &Yvar ); 
  for( unsigned k=0; k<nRes; ++k )
    for( unsigned i=0; i<nVar; ++i )
      vDer[k][i] = *vGradFIM[k*nVar+i];
}

////////////////////////////////////////////////////////////////////////

class FFFIMCrit
: public FFOp,
  public FFDOEBase
{
private:

  // DAG of FIM
  mutable FFGraph* _DAG;
  // Parameters
  std::vector<FFVar> const* _FPAR;
  // Controls
  std::vector<FFVar> const* _FCON;
  // FIM entries
  std::vector<FFVar> const* _FFIM;
  // efforts
  std::map<unsigned,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;

  // Number of parameters
  size_t _np;
  // Number of controls
  size_t _nc;
  // Number of outputs
  size_t _ny;
  // Number of scenarios
  size_t _ns;
  // Number of experiments
  size_t _ne;

  // control values
  mutable std::vector<double> _DCON;
  // FIM values
  mutable std::vector<std::vector<double>> _DFIM;

  // Subgraph
  mutable FFSubgraph _sgFIM;
  // Work storage
  mutable std::vector<double> _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<double>> _wkThd;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
#ifdef MC__FFFIMCrit_CHECK
  assert( dag && par->size() && con->size() && fim->size() && eff->size() && vpar->size() );
#endif

      _DAG  = dag;
      _FPAR = par;
      _FCON = con;
      _FFIM = fim;
      _EFF  = eff;
      _DPAR = vpar;

      _np = _FPAR->size();
      _nc = _FCON->size();
      _ny = std::round( std::sqrt(2*_FFIM->size()+0.25) - 0.5 );
      _ns = _DPAR->size();
      _ne = _EFF->size();
    }

  // Default constructor
  FFFIMCrit
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFFIMCrit
    ( FFFIMCrit const& Op )
    : FFOp( Op ),
      _DAG( Op._DAG ),
      _FPAR( Op._FPAR ),
      _FCON( Op._FCON ),
      _FFIM( Op._FFIM ),
      _EFF( Op._EFF ),
      _DPAR( Op._DPAR ),
      _np( Op._np ),
      _nc( Op._nc ),
      _ny( Op._ny ),
      _ns( Op._ns ),
      _ne( Op._ne )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
#ifdef MC__FFFIMCrit_CHECK
      assert( idep < _ns );
#endif
      set( dag, par, con, fim, eff, vpar );
      return *(insert_external_operation( *this, _ns, _nc*_ne, coneff )[idep]);

    }

  FFVar** operator()
    ( FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
      set( dag, par, con, fim, eff, vpar );
      return insert_external_operation( *this, _ns, _nc*_ne, coneff );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFFIMCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

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
    { 
      switch( FFDOEBase::type ){
        case AOPT: return "-tr Inv";
        case DOPT: return "log Det";
        case EOPT: return "min Eig";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradFIMCrit
: public FFOp,
  public FFDOEBase
{
private:

  // DAG of FIM
  mutable FFGraph* _DAG;
  // Parameters
  std::vector<FFVar> const* _FPAR;
  // Controls
  std::vector<FFVar> const* _FCON;
  // FIM entries
  std::vector<FFVar> const* _FFIM;
  // efforts
  std::map<unsigned,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;
  // parameter scenarios
  std::vector<std::vector<fadbad::F<double>>> _FDPAR;

  // Number of parameters
  size_t _np;
  // Number of controls
  size_t _nc;
  // Number of outputs
  size_t _ny;
  // Number of scenarios
  size_t _ns;
  // Number of experiments
  size_t _ne;

  // control values
  mutable std::vector<std::vector<fadbad::F<double>>> _FDCON;
  // FIM values
  mutable std::vector<std::vector<fadbad::F<double>>> _FDFIM;

  // Subgraph
  mutable FFSubgraph _sgFIM;
  // Work storage
  mutable std::vector<fadbad::F<double>> _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<fadbad::F<double>>> _wkThd;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
#ifdef MC__FFFIMCrit_CHECK
  assert( dag && par->size() && con->size() && fim->size() && eff->size() && vpar->size() );
#endif

      _DAG  = dag;
      _FPAR = par;
      _FCON = con;
      _FFIM = fim;
      _EFF  = eff;
      _DPAR = vpar;

      _np = _FPAR->size();
      _nc = _FCON->size();
      _ny = std::round( std::sqrt(2*fim->size()+0.25) - 0.5 );
      _ns = _DPAR->size();
      _ne = _EFF->size();

      _FDPAR.resize( _ns );
      for( size_t s=0; s<_ns; ++s )
        _FDPAR[s].assign( _DPAR->at(s).cbegin(), _DPAR->at(s).cend() );

      _FDCON.resize( _ne );
      for( size_t e=0, ec=0; e<_ne; ++e ){
        _FDCON[e].assign( _nc, 0. );
        for( size_t c=0; c<_nc; ++c, ++ec ){
          _FDCON[e][c].diff( ec, _nc*_ne );
#ifdef MC__FFDOECRIT_DEBUG
          std::cout << "_FDCON[" << e << "][" << c << "].diff(" << ec << "," << _nc*_ne << ")\n";
#endif
        }
      }
    }

  // Default constructor
  FFGradFIMCrit
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFGradFIMCrit
    ( FFGradFIMCrit const& Op )
    : FFOp( Op ),
      _DAG( Op._DAG ),
      _FPAR( Op._FPAR ),
      _FCON( Op._FCON ),
      _FFIM( Op._FFIM ),
      _EFF( Op._EFF ),
      _DPAR( Op._DPAR ),
      _FDPAR( Op._FDPAR ),
      _np( Op._np ),
      _nc( Op._nc ),
      _ny( Op._ny ),
      _ns( Op._ns ),
      _ne( Op._ne ),
      _FDCON( Op._FDCON )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
#ifdef MC__FFFIMCrit_CHECK
      assert( idep < _ns*_nc*_ne );
#endif
      set( dag, par, con, fim, eff, vpar );
      return *(insert_external_operation( *this, _ns*_nc*_ne, _nc*_ne, coneff )[idep]);
    }

  FFVar** operator()
    ( FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
      set( dag, par, con, fim, eff, vpar );
      return insert_external_operation( *this, _ns*_nc*_ne, _nc*_ne, coneff );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradFIMCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      switch( FFDOEBase::type ){
        case AOPT: return "-Grad tr Inv";
        case DOPT: return "Grad log Det";
        case EOPT: return "Grad min Eig";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline void
FFFIMCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFIMCRIT_TRACE
  std::cout << "FFFIMCrit::eval: FFVar\n"; 
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFFIMCrit::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFIMCRIT_TRACE
  std::cout << "FFFIMCrit::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}

inline void
FFGradFIMCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFIMCRIT_TRACE
  std::cout << "FFGradFIMCrit::eval: FFVar\n"; 
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFGradFIMCrit::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFIMCRIT_TRACE
  std::cout << "FFGradFIMCrit::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}

inline void
FFFIMCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFIMCRIT_TRACE
  std::cout << "FFFIMCrit::eval: double\n"; 
#endif
#ifdef MC__FFFIMCRIT_CHECK
  assert( nRes == _ns && nVar == _nc*_ne );
#endif

  // Get FIM entries for each scenario and each experiment
  _DFIM.resize( _ns );
  for( auto& DFIMs : _DFIM )
    DFIMs.assign( _FFIM->size(), 0. );

  double const* pCON = vVar;
  for( auto const& [id,eff] : *_EFF ){
    _DCON.assign( pCON, pCON+_nc );
    //_DAG->veval( _sgFIM, _wkD, *_FFIM, _DFIM, *_FPAR, *_DPAR, *_FCON, _DCON, &eff );
    _DAG->veval( _sgFIM, _wkD, _wkThd, *_FFIM, _DFIM, *_FPAR, *_DPAR, *_FCON, _DCON, &eff );
    pCON += _nc;
  }

  // Calculate FIM-based criteria in each uncertainty scenario
  FFDOECrit OpDOECrit;
  for( size_t s=0; s<_ns; ++s ){
    OpDOECrit.eval( 1, &vRes[s], _DFIM[s].size(), _DFIM[s].data(), nullptr );
#ifdef MC__FFDOECRIT_DEBUG
    std::cout << name() << "[" << s << "] = " << vRes[s] << std::endl;
    //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
  }
}

inline void
FFGradFIMCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADFIMCRIT_TRACE
  std::cout << "FFGradFIMCrit::eval: double\n"; 
#endif
#ifdef MC__FFGRADFIMCRIT_CHECK
  assert( nRes == nVar*_ns && nVar = _nc*_ne );
#endif

  // Get FIM entry derivatives for each scenario and each experiment
  _FDFIM.resize( _ns );
  for( auto& FDFIMs : _FDFIM )
    FDFIMs.assign( _FFIM->size(), 0. );

  double const* pCON = vVar;
  size_t e = 0;
  for( auto const& [id,eff] : *_EFF ){
    for( size_t c=0; c<_nc; ++c )
      _FDCON[e][c].x() = pCON[c]; // does not change differential variables
    //_DAG->veval( _sgFIM, _wkD, *_FFIM, _FDFIM, *_FPAR, _FDPAR, *_FCON, _FDCON[e], &eff );
    _DAG->veval( _sgFIM, _wkD, _wkThd, *_FFIM, _FDFIM, *_FPAR, _FDPAR, *_FCON, _FDCON[e], &eff );
#ifdef MC__FFDOECRIT_DEBUG
    for( size_t k=0; k<_FFIM->size(); ++k ){
      std::cout << "_FDFIM[0][" << k << "] =";
      for( size_t i=0; i<_FDFIM.back()[k].size(); ++i )
        std::cout << "  " << _FDFIM.back()[k].deriv(i);
      std::cout << std::endl;
    }
#endif
    pCON += _nc;
    ++e;
  }
  //{ int dum; std::cout << "Press 1"; std::cin >> dum; }

  // Calculate FIM-based criteria in each uncertainty scenario
  FFDOECrit OpDOECrit;
  fadbad::F<double> FRes;
  for( size_t s=0; s<_ns; ++s ){
    OpDOECrit.eval( 1, &FRes, _FDFIM[s].size(), _FDFIM[s].data(), nullptr );
    for( size_t ec=0; ec<_ne*_nc; ++ec ){
      vRes[ec+_ne*_nc*s] = FRes.deriv( ec ); 
      //vRes[ec*_ns+s] = FRes.deriv( ec ); 
#ifdef MC__FFDOECRIT_DEBUG
      std::cout << name() << "[" << s << "][" << ec << "] = " << FRes.deriv( ec ) << std::endl;
    //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
    }
  }
}

inline void
FFFIMCrit::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFFIMCrit::eval: fadbad::F<FFVar>\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == _ns );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar const*const* ppResVal = insert_external_operation( *this, nRes, nVar, vVarVal.data() );

  FFGradFIMCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FFIM, _EFF, _DPAR );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nRes*nVar, nVar, vVarVal.data() );
  for( unsigned k=0; k<nRes; ++k ){
    vRes[k] = *ppResVal[k];
    for( unsigned i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
    for( unsigned j=0; j<vRes[k].size(); ++j ){
      vRes[k][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
        //vRes[k][j] += *ppResDer[k+nRes*i] * vVar[i][j];
        vRes[k][j] += *ppResDer[k*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFFIMCrit::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFFIMCrit::eval: fadbad::F<double>\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == _ns );
#endif

  std::vector<double> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  std::vector<double> vResVal( nRes ); 
  eval( nRes, vResVal.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned k=0; k<nRes; ++k ){
    vRes[k] = vResVal[k];
    for( unsigned i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
  }
  
  FFGradFIMCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FFIM, _EFF, _DPAR );
  std::vector<double> vResDer( nRes*nVar ); 
  OpResDer.eval( nRes*nVar, vResDer.data(), nVar, vVarVal.data(), nullptr );
  for( unsigned k=0; k<nRes; ++k ){
    for( unsigned j=0; j<vRes[k].size(); ++j ){
      vRes[k][j] = 0.;
      for( unsigned i=0; i<nVar; ++i ){
        if( vVar[i][j] == 0. ) continue;
        //vRes[k][j] += vResDer[k+nRes*i] * vVar[i][j];
        vRes[k][j] += vResDer[k*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFFIMCrit::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFFIMCrit::deriv:\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == _ns );
#endif

  FFGradFIMCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FFIM, _EFF, _DPAR );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nRes*nVar, nVar, vVar );
  for( unsigned k=0; k<nRes; ++k )
    for( unsigned i=0; i<nVar; ++i )
      //vDer[k][i] = *ppResDer[k+nRes*i];
      vDer[k][i] = *ppResDer[k*nVar+i];
}

////////////////////////////////////////////////////////////////////////

class FFBRISKCrit
: public FFOp,
  public FFDOEBase
{
private:

  // DAG of outputs
  mutable FFGraph* _DAG;
  // parameters
  std::vector<FFVar> const* _FPAR;
  // controls
  std::vector<FFVar> const* _FCON;
  // outputs
  std::vector<FFVar> const* _FOUT;
  // efforts
  std::map<unsigned,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;

  // Number of parameters
  size_t _np;
  // Number of controls
  size_t _nc;
  // Number of outputs
  size_t _ny;
  // Number of scenarios
  size_t _ns;
  // Number of experiments
  size_t _ne;

  // control values
  mutable std::vector<double> _DCON;
  // output values
  mutable std::vector<std::vector<double>> _DOUT;

  // Subgraph
  mutable FFSubgraph _sgOUT;
  // Work storage
  mutable std::vector<double> _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<double>> _wkThd;

  // Evaluation of Bayes risk from output values
  void _BRval
    ( double& BR, std::vector<std::vector<double>>& DOUT )
    const;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
#ifdef MC__FFBRISKCRIT_CHECK
  assert( dag && par->size() && con->size() && out->size() && eff->size() && vpar->size() );
#endif

      _DAG  = dag;
      _FPAR = par;
      _FCON = con;
      _FOUT = out;
      _EFF  = eff;
      _DPAR = vpar;

      _np = _FPAR->size();
      _nc = _FCON->size();
      _ny = _FOUT->size();
      _ns = _DPAR->size();
      _ne = _EFF->size();
    }

  // Default constructor
  FFBRISKCrit
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFBRISKCrit
    ( FFBRISKCrit const& Op )
    : FFOp( Op ),
      _DAG( Op._DAG ),
      _FPAR( Op._FPAR ),
      _FCON( Op._FCON ),
      _FOUT( Op._FOUT ),
      _EFF( Op._EFF ),
      _DPAR( Op._DPAR ),
      _np( Op._np ),
      _nc( Op._nc ),
      _ny( Op._ny ),
      _ns( Op._ns ),
      _ne( Op._ne )
    {}

  // Define operation
  FFVar& operator()
    ( FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
      set( dag, par, con, out, eff, vpar );
      return **insert_external_operation( *this, 1, _nc*_ne, coneff );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFBRISKCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

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
    { 
      switch( FFDOEBase::type ){
        case BROPT: return "Bayes Risk";
        default:    throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradBRISKCrit
: public FFOp,
  public FFDOEBase
{
private:

  // DAG of outputs
  mutable FFGraph* _DAG;
  // parameters
  std::vector<FFVar> const* _FPAR;
  // controls
  std::vector<FFVar> const* _FCON;
  // outputs
  std::vector<FFVar> const* _FOUT;
  // efforts
  std::map<unsigned,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;
  // parameter scenarios
  std::vector<std::vector<fadbad::F<double>>> _FDPAR;

  // Number of parameters
  size_t _np;
  // Number of controls
  size_t _nc;
  // Number of outputs
  size_t _ny;
  // Number of scenarios
  size_t _ns;
  // Number of experiments
  size_t _ne;

  // control values
  mutable std::vector<std::vector<fadbad::F<double>>> _FDCON;
  // output values
  mutable std::vector<std::vector<fadbad::F<double>>> _FDOUT;

  // Subgraph
  mutable FFSubgraph _sgOUT;
  // Work storage
  mutable std::vector<fadbad::F<double>> _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<fadbad::F<double>>> _wkThd;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
#ifdef MC__FFBRISKCRIT_CHECK
  assert( dag && par->size() && con->size() && out->size() && eff->size() && vpar->size() );
#endif

      _DAG  = dag;
      _FPAR = par;
      _FCON = con;
      _FOUT = out;
      _EFF  = eff;
      _DPAR = vpar;

      _np = _FPAR->size();
      _nc = _FCON->size();
      _ny = _FOUT->size();
      _ns = _DPAR->size();
      _ne = _EFF->size();

      _FDPAR.resize( _ns );
      for( size_t s=0; s<_ns; ++s )
        _FDPAR[s].assign( _DPAR->at(s).cbegin(), _DPAR->at(s).cend() );

      _FDCON.resize( _ne );
      for( size_t e=0, ec=0; e<_ne; ++e ){
        _FDCON[e].assign( _nc, 0. );
        for( size_t c=0; c<_nc; ++c, ++ec ){
          _FDCON[e][c].diff( ec, _nc*_ne );
#ifdef MC__FFBRISKCRIT_DEBUG
          std::cout << "_FDCON[" << e << "][" << c << "].diff(" << ec << "," << _nc*_ne << ")\n";
#endif
        }
      }
    }

  // Default constructor
  FFGradBRISKCrit
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFGradBRISKCrit
    ( FFGradBRISKCrit const& Op )
    : FFOp( Op ),
      _DAG( Op._DAG ),
      _FPAR( Op._FPAR ),
      _FCON( Op._FCON ),
      _FOUT( Op._FOUT ),
      _EFF( Op._EFF ),
      _DPAR( Op._DPAR ),
      _FDPAR( Op._FDPAR ),
      _np( Op._np ),
      _nc( Op._nc ),
      _ny( Op._ny ),
      _ns( Op._ns ),
      _ne( Op._ne ),
      _FDCON( Op._FDCON )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
#ifdef MC__FFBRISKCRIT_CHECK
      assert( idep < _nc*_ne );
#endif
      set( dag, par, con, out, eff, vpar );
      return *(insert_external_operation( *this, _nc*_ne, _nc*_ne, coneff )[idep]);
    }

  FFVar** operator()
    ( FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<unsigned,double> const* eff, std::vector<std::vector<double>> const* vpar )
    {
      set( dag, par, con, out, eff, vpar );
      return insert_external_operation( *this, _nc*_ne, _nc*_ne, coneff );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
//      else if( idU == typeid( SLiftVar ) )
//        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
//      else if( idU == typeid( FFExpr ) )
//        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradBRISKCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      switch( FFDOEBase::type ){
        case BROPT: return "Grad Bayes Risk";
        default:    throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline void
FFBRISKCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFBRISKCrit::eval: FFVar\n"; 
#endif

  vRes[0] = **insert_external_operation( *this, nRes, nVar, vVar );
}

inline void
FFBRISKCrit::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFBRISKCrit::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
}

inline void
FFGradBRISKCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFGradBRISKCrit::eval: FFVar\n"; 
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFGradBRISKCrit::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFGradBRISKCrit::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( unsigned j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}

inline void
FFBRISKCrit::_BRval
( double& BR, std::vector<std::vector<double>>& DOUT )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRISKCrit::eval: double\n";
#endif
#ifdef MC__FFBRISKCRIT_CHECK
  assert( _EFF && !_EFF->empty() && _DOUT.size() == _ns && _DOUT.front().size == _nc*_ne );
#endif

  auto BRappend = [&]( size_t j, size_t k, double& BR ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      size_t pi = 0;
      for( auto const& [id,eff] : *_EFF ){
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "y[" << j << "][" << pi << "] = " << arma::vec( DOUT[j].data()+pi, _ny, false );
        std::cout << "y[" << k << "][" << pi << "] = " << arma::vec( DOUT[k].data()+pi, _ny, false );
#endif
        arma::vec const& Eijk = arma::vec( DOUT[j].data()+pi, _ny, false )
                              - arma::vec( DOUT[k].data()+pi, _ny, false );
        if( !sigmayinv.empty() )
          Et_Vinv_E += eff * Eijk.t() * sigmayinv * Eijk;
        else
          Et_Vinv_E += eff * Eijk.t() * Eijk;
        pi += _ny;
      }
      double BRjk = std::exp( -0.125 * Et_Vinv_E(0,0) );
#ifdef MC__FFBRCRIT_DEBUG
      std::cout << "BR[" << j << "," << k << "] = " << BRjk << std::endl; 
#endif
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
      BR += BRjk;
#ifdef MC__FFBRCRIT_DEBUG
      std::cout << "BR[" << j << "," << k << "] = " << BR << std::endl; 
#endif
  };

  BR = 0.;

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
      BRappend( j, k, BR );
 
  // Use full set of uncertainty scenarios
  else
    for( unsigned j=0; j<_ns-1; ++j )
      for( unsigned k=j+1; k<_ns; ++k )
        BRappend( j, k, BR );

#ifdef MC__FFBRCRIT_LOG
  BR = std::log( BR );
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << name() << BR << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFBRISKCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFBRISKCrit::eval: double\n"; 
#endif
#ifdef MC__FFBRISKCRIT_CHECK
  assert( nRes == 1 && nVar == _nc*_ne );
#endif

  // Get outputs for each scenario and each experiment
  _DOUT.resize( _ns );
  for( auto& DOUTs : _DOUT )
    DOUTs.assign( _FOUT->size(), 0. );

  double const* pCON = vVar;
  for( auto const& [id,eff] : *_EFF ){
    _DCON.assign( pCON, pCON+_nc );
    //_DAG->veval( _sgOUT, _wkD, *_FOUT, _DOUT, *_FPAR, *_DPAR, *_FCON, _DCON, &eff );
    _DAG->veval( _sgOUT, _wkD, _wkThd, *_FOUT, _DOUT, *_FPAR, *_DPAR, *_FCON, _DCON, &eff );
    pCON += _nc;
  }

  // Calculate Bayes risk-based criteria in each uncertainty scenario
  _BRval( vRes[0], _DOUT );
#ifdef MC__FFDOECRIT_DEBUG
  std::cout << name() << " = " << vRes[0] << std::endl;
    //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradBRISKCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADFIMCRIT_TRACE
  std::cout << "FFGradBRISKCrit::eval: double\n"; 
#endif
#ifdef MC__FFGRADFIMCRIT_CHECK
  assert( nRes == nVar && nVar = _nc*_ne );
#endif
  /*
  // Get FIM entry derivatives for each scenario and each experiment
  _FDFIM.resize( _ns );
  for( auto& FDFIMs : _FDFIM )
    FDFIMs.assign( _FFIM->size(), 0. );

  double const* pCON = vVar;
  size_t e = 0;
  for( auto const& [id,eff] : *_EFF ){
    for( size_t c=0; c<_nc; ++c )
      _FDCON[e][c].x() = pCON[c]; // does not change differential variables
    //_DAG->veval( _sgFIM, _wkD, *_FFIM, _FDFIM, *_FPAR, _FDPAR, *_FCON, _FDCON[e], &eff );
    _DAG->veval( _sgFIM, _wkD, _wkThd, *_FFIM, _FDFIM, *_FPAR, _FDPAR, *_FCON, _FDCON[e], &eff );
#ifdef MC__FFDOECRIT_DEBUG
    for( size_t k=0; k<_FFIM->size(); ++k ){
      std::cout << "_FDFIM[0][" << k << "] =";
      for( size_t i=0; i<_FDFIM.back()[k].size(); ++i )
        std::cout << "  " << _FDFIM.back()[k].deriv(i);
      std::cout << std::endl;
    }
#endif
    pCON += _nc;
    ++e;
  }
  //{ int dum; std::cout << "Press 1"; std::cin >> dum; }

  // Calculate FIM-based criteria in each uncertainty scenario
  FFDOECrit OpDOECrit;
  fadbad::F<double> FRes;
  for( size_t s=0; s<_ns; ++s ){
    OpDOECrit.eval( 1, &FRes, _FDFIM[s].size(), _FDFIM[s].data(), nullptr );
    for( size_t ec=0; ec<_ne*_nc; ++ec ){
      vRes[ec+_ne*_nc*s] = FRes.deriv( ec ); 
      //vRes[ec*_ns+s] = FRes.deriv( ec ); 
#ifdef MC__FFDOECRIT_DEBUG
      std::cout << name() << "[" << s << "][" << ec << "] = " << FRes.deriv( ec ) << std::endl;
    //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
    }
  }
  */
}

inline void
FFBRISKCrit::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFBRISKCrit::eval: fadbad::F<FFVar>\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( unsigned i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar ResVal = **insert_external_operation( *this, 1, nVar, vVarVal.data() );

  FFGradBRISKCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FOUT, _EFF, _DPAR );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVarVal.data() );
  vRes[0] = ResVal;
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );
  for( size_t j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( size_t i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *ppResDer[i] * vVar[i][j];
    }
  }
}

inline void
FFBRISKCrit::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFBRISKCrit::eval: fadbad::F<double>\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif

  std::vector<double> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double ResVal(0.); 
  eval( 1, &ResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = ResVal;
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );
  
  FFGradBRISKCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FOUT, _EFF, _DPAR );
  std::vector<double> vResDer( nVar ); 
  OpResDer.eval( nVar, vResDer.data(), nVar, vVarVal.data(), nullptr );
  for( size_t j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( size_t i=0; i<nVar; ++i ){
      if( vVar[i][j] == 0. ) continue;
      vRes[0][j] += vResDer[i] * vVar[i][j];
    }
  }
}

inline void
FFBRISKCrit::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFBRISKCrit::deriv:\n"; 
#endif
#ifdef MC__FFBRISKCRIT_CHECK
  assert( nRes == 1 );
#endif

  FFGradBRISKCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FOUT, _EFF, _DPAR );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVar );
  for( size_t i=0; i<nVar; ++i )
    vDer[0][i] = *ppResDer[i];
}

} // end namespace mc

#endif

// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MBDOESLV Model-based Design of Experiments with MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 1.0
\date 2024
\bug No known bugs.

*/

#ifndef MC__MBDOESLV_HPP
#define MC__MDBOESLV_HPP

#include <fstream>
#include <iomanip>
#include <armadillo>

#if defined( MC__USE_PROFIL )
 #include "mcprofil.hpp"
#elif defined( MC__USE_BOOST )
 #include "mcboost.hpp"
#elif defined( MC__USE_FILIB )
 #include "mcfilib.hpp"
#else
 #include "interval.hpp"
#endif

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
#elif  MC__USE_IPOPT
 #include "mipslv_cplex.hpp"
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
#endif

#include "minlpslv.hpp"

#include "base_mbdoe.hpp"

#define MC__FFBRCRIT_LOG

////////////////////////////////////////////////////////////////////////
// EXTERNAL OPERATIONS
////////////////////////////////////////////////////////////////////////
namespace mc
{

struct DOEBase
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

};

////////////////////////////////////////////////////////////////////////

inline DOEBase::TYPE DOEBase::type = DOEBase::DOPT;
inline arma::mat DOEBase::scaling;
inline arma::vec DOEBase::weighting;
inline arma::mat DOEBase::sigmayinv;

template<unsigned int ID>
class FFDOECrit
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFDOECrit
    ()
    : FFOp( (int)EXTERN )
    {}

  // Declaration
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar )
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
      throw std::runtime_error("Error: No generic implementation for FFDOECrit\n");
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
      switch( DOEBase::type ){
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

template<unsigned int ID>
class FFGradDOECrit
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFGradDOECrit
    ()
    : FFOp( (int)EXTERN )
    {}

  // Functor
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

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      throw std::runtime_error("Error: No generic implementation for FFGradDOECrit\n");
    }

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

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

  // Properties
  std::string name
    ()
    const
    {
      switch( DOEBase::type ){
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

template<unsigned int ID>
inline void
FFDOECrit<ID>::eval
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

  switch( DOEBase::type ){
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
      if( arma::rank( FIM ) < nDim || !arma::log_det_sympd( vRes[0], FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
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

template<unsigned int ID>
inline void
FFGradDOECrit<ID>::eval
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

  switch( DOEBase::type ){
    case AOPT:
    {
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM ) )
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
              vRes[l] += (i==j? FIMEIGVEC(k,i)*FIMEIGVEC(k,i): 2*FIMEIGVEC(k,i)*FIMEIGVEC(k,j) )
                       * (scaling(i,i)*scaling(j,j)) / (FIMEIGVAL(k)*FIMEIGVAL(k));
            else
              vRes[l] += (i==j? FIMEIGVEC(k,i)*FIMEIGVEC(k,i): 2*FIMEIGVEC(k,i)*FIMEIGVEC(k,j) )
                       / (FIMEIGVAL(k)*FIMEIGVAL(k));
        }
      break;
    }
    case DOPT:
    {
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
      break;
    }
    case EOPT:
    {
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < nDim || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM min eigenvalue: "  << FIMEIGVAL(0) << std::endl;
      std::cout << "FIM min eigenvector: " << FIMEIGVEC.col(0);
#endif
      for( unsigned i=0, l=0; i<nDim; ++i )
        for( unsigned j=i; j<nDim; ++j, ++l )
          if( scaling.n_elem )
            vRes[l] = (i==j? FIMEIGVEC(0,i)*FIMEIGVEC(0,i): 2*FIMEIGVEC(0,i)*FIMEIGVEC(0,j) )
                    * (scaling(i,i)*scaling(j,j));
          else
            vRes[l] = (i==j? FIMEIGVEC(0,i)*FIMEIGVEC(0,i): 2*FIMEIGVEC(0,i)*FIMEIGVEC(0,j) );
      break;
    }
    default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
  }
#ifdef MC__FFGRADDOECRIT_DEBUG
  for( unsigned i=0, l=0; i<nDim; ++i ){
    for( unsigned j=i; j<nDim; ++j, ++l ){
      std::cout << "grad " << name() << " [" << i << "," << j << "]: " << vRes[l] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

template<unsigned int ID>
inline void
FFDOECrit<ID>::eval
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

  FFGradDOECrit<ID> GradDOECrit;
  FFVar const*const* vGradDOECrit = GradDOECrit( nVar, vVarVal.data() ); 
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *vGradDOECrit[i] * vVar[i][j];
    }
  }
}

template<unsigned int ID>
inline void
FFDOECrit<ID>::eval
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

  FFGradDOECrit<ID> GradDOECrit;
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

template<unsigned int ID>
inline void
FFDOECrit<ID>::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFDOECrit::deriv:\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif

  FFGradDOECrit<ID> GradDOECrit;
  FFVar const*const* vGradDOECrit = GradDOECrit( nVar, vVar );
  for( unsigned i=0; i<nVar; ++i )
    vDer[0][i] = *vGradDOECrit[i];
}

////////////////////////////////////////////////////////////////////////

template<unsigned int ID>
class FFDOEEff
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFDOEEff
    ()
    : FFOp( (int)EXTERN )
    {}

  // Declaration
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::mat > >* vFIM )
    const
    {
#ifdef MC__FFDOEEFF_CHECK
      assert( vFIM );
#endif
      data = vFIM; // no local copy - make sure vFIM isn't going out of scope!
      info = ID;
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
      info = ID;
      unsigned const nRes = vFIM->size();
      return insert_external_operation( *this, nRes, nVar, pVar );
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
      switch( DOEBase::type ){
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

template<unsigned int ID>
class FFGradDOEEff
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFGradDOEEff
    ()
    : FFOp( (int)EXTERN )
    {}

  // Functor
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar, std::vector< std::vector< arma::mat > >* vFIM )
    const
    {
#ifdef MC__FFGRADDOEEFF_CHECK
      assert( vFIM );
#endif
      data = vFIM; // no local copy - make sure vFIM isn't going out of scope!
      info = ID+1;
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
      info = ID+1;
      unsigned const nRes = vFIM->size();
      return insert_external_operation( *this, nRes * nVar, nVar, pVar );
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
      switch( DOEBase::type ){
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

template<unsigned int ID>
inline void
FFDOEEff<ID>::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFDOEEff::eval: FFVar\n";
#endif
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
#ifdef MC__FFDOEEFF_CHECK
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size() && nVar == vFIM->back().size() );
#endif

  FFVar** ppRes = operator()( nVar, vVar, vFIM );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

template<unsigned int ID>
inline void
FFGradDOEEff<ID>::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADDOEEFF_TRACE
  std::cout << "FFGradDOEEff::eval: FFVar\n";
#endif
  std::vector< std::vector< arma::mat > >* vFIM = static_cast<std::vector< std::vector< arma::mat > >*>( data );
#ifdef MC__FFGRADDOEEFF_CHECK
  assert( vFIM && !vFIM->empty() && nRes == vFIM->size()*nVar && nVar == vFIM->back().size() );
#endif

  FFVar** ppRes = operator()( nVar, vVar, vFIM );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

template<unsigned int ID>
inline void
FFDOEEff<ID>::eval
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

    switch( DOEBase::type ){
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
        if( arma::rank( FIM ) < FIM.n_rows || !arma::log_det_sympd( vRes[s], FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
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

template<unsigned int ID>
inline void
FFGradDOEEff<ID>::eval
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

    switch( DOEBase::type ){
      case AOPT:
      {
        arma::vec FIMEIGVAL;
        arma::mat FIMEIGVEC;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
        std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
        std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
        for( unsigned i=0; i<nVar; ++i ){
          vRes[s*nVar+i] = 0.;
          if( scaling.n_elem ) FIMi = scaling * vFIM->at(s).at(i) * scaling;
          else                 FIMi = vFIM->at(s).at(i);
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
        break;
      }
      case EOPT:
      {
        arma::vec FIMEIGVAL;
        arma::mat FIMEIGVEC;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
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
      default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
    }
  }
#ifdef MC__FFGRADDOEFF_DEBUG
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

template<unsigned int ID>
inline void
FFDOEEff<ID>::eval
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

  FFGradDOEEff<ID> DOptGrad;
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

template<unsigned int ID>
inline void
FFDOEEff<ID>::eval
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

  FFGradDOEEff<ID> DOptGrad;
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

template<unsigned int ID>
inline void
FFDOEEff<ID>::deriv
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

  FFGradDOEEff<ID> DOptGrad;
  FFVar const*const* vDOptGrad = DOptGrad( nVar, vVar, vFIM ); 
  for( unsigned s=0; s<nRes; ++s )
    for( unsigned i=0; i<nVar; ++i )
      vDer[s][i] = *vDOptGrad[s*nVar+i];
}

////////////////////////////////////////////////////////////////////////

template<unsigned int ID>
class FFBRCrit
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFBRCrit
    ()
    : FFOp( (int)EXTERN )
    {}

  static size_t nUNC;
  static size_t nOUT;

  // Declaration
  FFVar& operator()
    ( unsigned const nVar, FFVar const* pVar, std::map<unsigned,double>* mEFF,
      unsigned int nUNC, unsigned int nOUT )
    const
    {
#ifdef MC__FFBRCRIT_CHECK
      assert( mEFF );
#endif
      data = mEFF; // no local copy - make sure mEFF isn't going out of scope!
      info = ID;
      this->nUNC = nUNC;
      this->nOUT = nOUT;
#ifdef MC__FFBRCRIT_CHECK
      assert( nVar == mEFF->size()*nUNC*nOUT );
#endif
      return **insert_external_operation( *this, 1, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      throw std::runtime_error("Error: No generic overload for FFBRCrit\n");
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
      switch( DOEBase::type ){
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

template<unsigned int ID> inline size_t FFBRCrit<ID>::nUNC = 0;
template<unsigned int ID> inline size_t FFBRCrit<ID>::nOUT = 0;

template<unsigned int ID>
class FFGradBRCrit
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFGradBRCrit
    ()
    : FFOp( (int)EXTERN )
    {}

  static size_t nUNC;
  static size_t nOUT;

  // Functor
  FFVar& operator()
    ( unsigned const idep, unsigned const nVar, FFVar const* pVar, std::map<unsigned,double>* mEFF,
      unsigned int nUNC, unsigned int nOUT )
    const
    {
#ifdef MC__FFGRADBRCRIT_CHECK
      assert( mEFF );
#endif
      data = mEFF; // no local copy - make sure mEFF isn't going out of scope!
      info = ID+1;
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
      info = ID+1;
      this->nUNC = nUNC;
      this->nOUT = nOUT;
#ifdef MC__FFBRCRIT_CHECK
      assert( nVar == mEFF->size()*nUNC*nOUT );
#endif
      return insert_external_operation( *this, nVar, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      throw std::runtime_error("Error: No generic overload for FFGradBRCrit\n");
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
      switch( DOEBase::type ){
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

template<unsigned int ID> inline size_t FFGradBRCrit<ID>::nUNC = 0;
template<unsigned int ID> inline size_t FFGradBRCrit<ID>::nOUT = 0;

template<unsigned int ID>
inline void
FFBRCrit<ID>::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::eval: FFVar\n";
#endif
  std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == 1 );
#endif

  vRes[0] = operator()( nVar, vVar, mEFF, nUNC, nOUT );
}

template<unsigned int ID>
inline void
FFGradBRCrit<ID>::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADBRCRIT_TRACE
  std::cout << "FFGradBRCrit::eval: FFVar\n";
#endif
  std::map<unsigned,double>* mEFF = static_cast<std::map<unsigned,double>*>( data );
#ifdef MC__FFBRCRIT_CHECK
  assert( mEFF && !mEFF->empty() && nVar == mEFF->size()*nOUT*nUNC && nRes == nVar );
#endif

  FFVar** ppRes = operator()( nVar, vVar, mEFF, nUNC, nOUT );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

template<unsigned int ID>
inline void
FFBRCrit<ID>::eval
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

  vRes[0] = 0.;
  size_t const inc = mEFF->size()*nOUT;
  size_t pj = 0;
  for( unsigned j=0; j<nUNC; ++j, pj+=inc ){
    size_t pk = pj + inc;
    for( unsigned k=j+1; k<nUNC; ++k, pk+=inc ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
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
      vRes[0] += BRjk;
//      std::cout << "vRes[" << j << "," << k << "] = " << vRes[0] << std::endl; 
    }
  }
#ifdef MC__FFBRCRIT_LOG
  vRes[0] = std::log( vRes[0] );
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << name() << " [" << 0 << "]: " << vRes[0] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

template<unsigned int ID>
inline void
FFGradBRCrit<ID>::eval
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

#ifdef MC__FFBRCRIT_LOG
  double BRCrit = 0.;
#endif
  size_t const inc = mEFF->size()*nOUT;
  arma::vec GradBR( vRes, nRes, false );
  GradBR.zeros();
  arma::vec GradBRjk( inc, arma::fill::none );
  size_t pj = 0;
  for( unsigned j=0; j<nUNC; ++j, pj+=inc ){
    size_t pk = pj + inc;
    for( unsigned k=j+1; k<nUNC; ++k, pk+=inc ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      size_t pi = 0;
      for( auto const& [Id,Eff] : *mEFF ){
//        std::cout << "y[" << j << "][" << pi << "] = " << arma::vec( const_cast<double*>(vVar+pj+pi), nOUT, false );
//        std::cout << "y[" << k << "][" << pi << "] = " << arma::vec( const_cast<double*>(vVar+pk+pi), nOUT, false );
        arma::vec const& Eijk = arma::vec( const_cast<double*>(vVar+pj+pi), nOUT, false )
                              - arma::vec( const_cast<double*>(vVar+pk+pi), nOUT, false );
        if( !sigmayinv.empty() ){
//          arma::vec& SEijk = GradBRjk.subvec(pi,pi+nOUT-1);
          GradBRjk.subvec(pi,pi+nOUT-1) = sigmayinv * Eijk;
          Et_Vinv_E += Eff * Eijk.t() * GradBRjk.subvec(pi,pi+nOUT-1);
          GradBRjk.subvec(pi,pi+nOUT-1) *= Eff/4;
        }
        else{
          Et_Vinv_E += Eff * Eijk.t() * Eijk;
          GradBRjk.subvec(pi,pi+nOUT-1) = (Eff/4) * Eijk;
        }
        pi += nOUT;
      }
      double BRjk = std::exp( -0.125 * Et_Vinv_E(0,0) );
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
#ifdef MC__FFBRCRIT_LOG
      BRCrit += BRjk;
#endif
      GradBR.subvec(pj,pj+inc-1) -= GradBRjk * BRjk;
      GradBR.subvec(pk,pk+inc-1) += GradBRjk * BRjk;
    }
  }
#ifdef MC__FFBRCRIT_LOG
  GradBR /= BRCrit;
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << GradBR;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

template<unsigned int ID>
inline void
FFBRCrit<ID>::eval
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

  FFGradBRCrit<ID> GradBRCrit;
  FFVar const*const* vGradBRCrit = GradBRCrit( nVar, vVarVal.data(), mEFF, nUNC, nOUT ); 
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *vGradBRCrit[i] * vVar[i][j];
    }
  }
}

template<unsigned int ID>
inline void
FFBRCrit<ID>::eval
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
  
  FFGradBRCrit<ID> GradBRCrit;
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

template<unsigned int ID>
inline void
FFBRCrit<ID>::deriv
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

  FFGradBRCrit<ID> GradBRCrit;
  FFVar const*const* vGradBRCrit = GradBRCrit( nVar, vVar, mEFF, nUNC, nOUT );
  for( unsigned i=0; i<nVar; ++i )
    vDer[0][i] = *vGradBRCrit[i];
}

////////////////////////////////////////////////////////////////////////

template<unsigned int ID>
class FFBREff
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFBREff
    ()
    : FFOp( (int)EXTERN )
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
      info = ID;
      return **insert_external_operation( *this, 1, nVar, pVar );
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
      switch( DOEBase::type ){
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

template<unsigned int ID>
class FFGradBREff
: public FFOp,
  public DOEBase
{
public:
  // Constructors
  FFGradBREff
    ()
    : FFOp( (int)EXTERN )
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
      info = ID+1;
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
      info = ID+1;
      return insert_external_operation( *this, nVar, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
      throw std::runtime_error("Error: No generic implementation for FFGradBREff\n");
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
      switch( DOEBase::type ){
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

template<unsigned int ID>
inline void
FFBREff<ID>::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::eval: FFVar\n";
#endif
  std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFBREFF_CHECK
  assert( vOUT && !vOUT->empty() && nRes == 1 && nVar == vOUT->back().size() );
#endif

  vRes[0] = operator()( nVar, vVar, vOUT );
}

template<unsigned int ID>
inline void
FFGradBREff<ID>::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADBREFF_TRACE
  std::cout << "FFGradBREff::eval: FFVar\n";
#endif
  std::vector< std::vector< arma::vec > >* vOUT = static_cast<std::vector< std::vector< arma::vec > >*>( data );
#ifdef MC__FFGRADBREFF_CHECK
  assert( vOUT && !vOUT->empty() && nRes == nVar && nVar == vOUT->back().size() );
#endif

  FFVar** ppRes = operator()( nVar, vVar, vOUT );
  for( unsigned j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

template<unsigned int ID>
inline void
FFBREff<ID>::eval
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

  vRes[0] = 0.;
  for( unsigned j=0; j<vOUT->size(); ++j ){
    for( unsigned k=j+1; k<vOUT->size(); ++k ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      for( unsigned i=0; i<nVar; ++i ){
        arma::vec const& Ejk  = vOUT->at(j).at(i) - vOUT->at(k).at(i);
        if( !sigmayinv.empty() ) Et_Vinv_E += vVar[i] * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += vVar[i] * Ejk.t() * Ejk;
      }
      if( !weighting.empty() ) vRes[0] += std::sqrt( weighting(j)*weighting(k) ) * std::exp( -0.125 * Et_Vinv_E(0,0) );
      else                     vRes[0] += std::exp( -0.125 * Et_Vinv_E(0,0) );
    }
  }
#ifdef MC__FFBRCRIT_LOG
  vRes[0] = std::log( vRes[0] );
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << name() << " [" << 0 << "]: " << vRes[0] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

template<unsigned int ID>
inline void
FFGradBREff<ID>::eval
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

/*
  for( unsigned i=0; i<nVar; ++i )
    vRes[i] = 0.;
  for( unsigned j=0; j<vOUT->size(); ++j ){
    for( unsigned k=j+1; k<vOUT->size(); ++k ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      for( unsigned i=0; i<nVar; ++i ){
        arma::vec const& Ejk  = vOUT->at(j).at(i) - vOUT->at(k).at(i);
        if( !sigmayinv.empty() ) Et_Vinv_E += vVar[i] * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += vVar[i] * Ejk.t() * Ejk;
      }
#ifdef MC__FFBRCRIT_LOG
      if( !weighting.empty() ) BRCrit += std::sqrt( weighting(j)*weighting(k) ) * std::exp( -0.125 * Et_Vinv_E(0,0) );
      else                     BRCrit += std::exp( -0.125 * Et_Vinv_E(0,0) );
#endif
      arma::mat der(1,1,arma::fill::none);
      for( unsigned i=0; i<nVar; ++i ){
        arma::vec const& Ejk  = vOUT->at(j).at(i) - vOUT->at(k).at(i);
        if( !sigmayinv.empty() ) der = -0.125 * Ejk.t() * sigmayinv * Ejk * std::exp( -0.125 * Et_Vinv_E(0,0) );
        else                     der = -0.125 * Ejk.t() * Ejk * std::exp( -0.125 * Et_Vinv_E(0,0) );
        if( !weighting.empty() ) vRes[i] += std::sqrt( weighting(j)*weighting(k) ) * der(0,0);
        else                     vRes[i] += der(0,0);
      }
    }
  }
#ifdef MC__FFBRCRIT_LOG
  for( unsigned i=0; i<nVar; ++i )
    vRes[i] /= BRCrit;
#endif
*/
#ifdef MC__FFBREFF_DEBUG
  for( unsigned i=0; i<nVar; ++i )
    std::cout << name() << " [" << i << "]: " << vRes[i] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

template<unsigned int ID>
inline void
FFBREff<ID>::eval
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

  FFGradBREff<ID> GradBREff;
  FFVar const*const* vGradBREff = GradBREff( nVar, vVarVal.data(), vOUT ); 
  for( unsigned j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( unsigned i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *vGradBREff[i] * vVar[i][j];
    }
  }
}

template<unsigned int ID>
inline void
FFBREff<ID>::eval
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

  FFGradBREff<ID> GradBREff;
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

template<unsigned int ID>
inline void
FFBREff<ID>::deriv
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

  FFGradBREff<ID> GradBREff;
  FFVar const*const* vGradBREff = GradBREff( nVar, vVar, vOUT );
  for( unsigned i=0; i<nVar; ++i )
    vDer[0][i] = *vGradBREff[i];
}

////////////////////////////////////////////////////////////////////////

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
    ( unsigned const nVar, FFVar const* pVar, double* wVar=nullptr )
    const
    {
      data = wVar; // no local copy - make sure wVar isn't going out of scope!
      info = ID;
      return **insert_external_operation( *this, 1, nVar, pVar );
    }
    
  FFVar& operator()
    ( unsigned const nVar, FFVar const*const* pVar, double* wVar=nullptr )
    const
    {
      data = wVar; // no local copy - make sure wVar isn't going out of scope!
      info = ID;
      return **insert_external_operation( *this, 1, nVar, pVar );
    }

  // Evaluation overloads
  template< typename T >
  void eval
    ( unsigned const nRes, T* vRes, unsigned const nVar, T const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFSUM_TRACE
      std::cout << "FFSum::eval: T\n"; 
#endif
#ifdef MC__FFSUM_CHECK
      assert( nRes == 1 );
#endif
      if( !data ){
        for( unsigned i=0; i<nVar; ++i )
          if( !i ) vRes[0]  = vVar[0];
          else     vRes[0] += vVar[i];
      }
      else{
        double const* wVar = static_cast<double const*>( data );
        for( unsigned i=0; i<nVar; ++i )
          if( !i ) vRes[0]  = wVar[0] * vVar[0];
          else     vRes[0] += wVar[i] * vVar[i];     
      }
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFSUM_TRACE
      std::cout << "FFSum::eval: FFVar\n"; 
#endif
#ifdef MC__FFSUM_CHECK
      assert( nRes == 1 );
#endif
      double* wVar = static_cast<double*>( data );
      vRes[0] = operator()( nVar, vVar, wVar );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFSUM_TRACE
      std::cout << "FFSum::eval: FFDep\n"; 
#endif
#ifdef MC__FFSUM_CHECK
      assert( nRes == 1 );
#endif
      vRes[0] = 0;
      for( unsigned i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::L );
    }

  void eval
    ( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
      unsigned const* mVar )
    const
    {
#ifdef MC__FFSUM_TRACE
      std::cout << "FFSum::eval: fadbad::F<T>\n";
#endif
#ifdef MC__FFSUM_CHECK
      assert( nRes == 1 );
#endif
      double* wVar = static_cast<double*>( data );

      std::vector<FFVar> vVarVal( nVar );
      for( unsigned i=0; i<nVar; ++i )
        vVarVal[i] = vVar[i].val();
      FFVar vResVal;
      eval( 1, &vResVal, nVar, vVarVal.data(), nullptr );
      vRes[0] = vResVal;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0].setDepend( vVar[i] );

      for( unsigned j=0; j<vRes[0].size(); ++j ){
        vRes[0][j] = 0.;
        for( unsigned i=0; i<nVar; ++i ){
          if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
          vRes[0][j] += wVar? wVar[i]*vVar[i][j]: vVar[i][j];
        }
      }
    }

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const
    {
#ifdef MC__FFSUM_TRACE
      std::cout << "FFSum::eval: fadbad::F<double>\n";
#endif
#ifdef MC__FFSUM_CHECK
      assert( nRes == 1 );
#endif
      double* wVar = static_cast<double*>( data );

      std::vector<double> vVarVal( nVar );
      if( vVarVal.size() < nVar ) vVarVal.resize( nVar );
      for( unsigned i=0; i<nVar; ++i )
        vVarVal[i] = vVar[i].val();
      double vResVal = 0.;
      eval( 1, &vResVal, nVar, vVarVal.data(), nullptr );
      vRes[0] = vResVal;
      for( unsigned i=0; i<nVar; ++i )
        vRes[0].setDepend( vVar[i] );

      for( unsigned j=0; j<vRes[0].size(); ++j ){
        vRes[0][j] = 0.;
        for( unsigned i=0; i<nVar; ++i ){
          if( vVar[i][j] == 0. ) continue;
          vRes[0][j] += wVar? wVar[i]*vVar[i][j]: vVar[i][j];
        }
      }
    }

  void deriv
    ( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
    const
    {
#ifdef MC__FFSUM_TRACE
      std::cout << "FFSum::deriv: FFVar\n";
#endif
#ifdef MC__FFSUM_CHECK
      assert( nRes == 1 );
#endif
      double* wVar = static_cast<double*>( data );

      for( unsigned i=0; i<nVar; ++i )
        vDer[0][i] = wVar? wVar[i]: 1.;
    }

  // Properties
  std::string name
    ()
    const
    { return "Sum"; }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

////////////////////////////////////////////////////////////////////////
// APPORTIONMENT
////////////////////////////////////////////////////////////////////////
void
apportion
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

#ifdef MC__MBDOE_SHOW_APPORTION
  std::cout << "Initial efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    if( val[i] > TOLZERO ) std::cout << "X0[" << i << "]: " << val[i] << std::endl;
  }
#endif

  double sum = 0.;
  unsigned supp = 0;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    sum += val[i];
    if( val[i] >= TOLZERO ) supp++;
  }
  sum = std::round( sum );

  std::vector<double> intval( n );
  unsigned iround = 0;
  for( double ratio=1.; ratio>0.1 && iround<100; ++iround ){
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

#ifdef MC__MBDOE_SHOW_APPORTION
  std::cout << "Apportioned efforts:" << std::endl;
#endif
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
#ifdef MC__MBDOE_SHOW_APPORTION
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
#endif
  }
}

void
effrounding
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

#ifdef MC__MBDOE_SHOW_APPORTION
  std::cout << "Initial efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    if( val[i] > TOLZERO ) std::cout << "X0[" << i << "]: " << val[i] << std::endl;
  }
#endif

  double sum = 0.;
  unsigned supp = 0;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    sum += val[i];
    if( val[i] >= TOLZERO ) supp++;
  }
  sum = std::round( sum );

  std::vector<double> intval( n );
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    intval[i] = std::ceil( (1.-supp/(2*sum)) * val[i] );
  }

  for( ; ; ){
    //std::cout << "Intermediate efforts:" << std::endl;
    double intsum = 0.;
    for( unsigned i=0; i<n; ++i ){
      if( !typ[i] ) continue;
      intsum += intval[i];
      //if( val[i] > TOLZERO ) std::cout << "X1[" << i << "]: " << intval[i] << std::endl;
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
  
#ifdef MC__MBDOE_SHOW_APPORTION
  std::cout << "Apportioned efforts:" << std::endl;
#endif
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
#ifdef MC__MBDOE_SHOW_APPORTION
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
#endif
  }
}

//! @brief C++ class for MBDoE solution using MC++
////////////////////////////////////////////////////////////////////////
//! mc::MBDOESLV is a C++ class for solving problems in model-based
//! design of experiments using MC++
////////////////////////////////////////////////////////////////////////
template <typename... ExtOps>
class MBDOESLV
: public virtual BASE_MBDOE<ExtOps...>
{
public:

  using BASE_MBDOE<ExtOps...>::dag;
  using BASE_MBDOE<ExtOps...>::set_dag;

  using BASE_MBDOE<ExtOps...>::np;
  using BASE_MBDOE<ExtOps...>::set_parameters;
  
  using BASE_MBDOE<ExtOps...>::nc;
  using BASE_MBDOE<ExtOps...>::set_controls;

  using BASE_MBDOE<ExtOps...>::ny;
  using BASE_MBDOE<ExtOps...>::set_model;

protected:

#if defined( MC__USE_PROFIL )
 typedef ::INTERVAL I;
#elif defined( MC__USE_BOOST )
 typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
 typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
 typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
 typedef boost::numeric::interval<double,T_boost_policy> I;
#elif defined( MC__USE_FILIB )
 typedef filib::interval<double> I;
#else
 typedef Interval I;
#endif

  typedef FFGraph< ExtOps... > DAG;
  typedef FFGraph< FFODE<0>, FFGRADODE<0>, FFDOECrit<2>, FFGradDOECrit<2>, FFDOEEff<4>, FFGradDOEEff<4>, FFSum<6>, FFBRCrit<7>, FFGradBRCrit<7>, FFBREff<9>, FFGradBREff<9>, ExtOps... > DAGDOE;

  typedef ODESLVS_CVODES< ExtOps... > IVPODE;

#if defined( MC__USE_GUROBI )
  typedef MIPSLV_GUROBI<I> MIP;
#elif defined( MC__USE_CPLEX )
  typedef MIPSLV_CPLEX<I> MIP;
#endif

#if defined( MC__USE_SNOPT )
  typedef NLPSLV_SNOPT< FFODE<0>, FFGRADODE<0>, FFDOECrit<2>, FFGradDOECrit<2>, FFDOEEff<4>, FFGradDOEEff<4>, FFSum<6>, FFBRCrit<7>, FFGradBRCrit<7>, FFBREff<9>, FFGradBREff<9> > NLP;
#elif defined( MC__USE_IPOPT )
  typedef NLPSLV_IPOPT< FFODE<0>, FFGRADODE<0>, FFDOECrit<2>, FFGradDOECrit<2>, FFDOEEff<4>, FFGradDOEEff<4>, FFSum<6>, FFBRCrit<7>, FFGradBRCrit<7>, FFBREff<9>, FFGradBREff<9> > NLP;
#endif
  typedef MINLPSLV< I, NLP, MIP, FFODE<0>, FFGRADODE<0>, FFDOECrit<2>, FFGradDOECrit<2>, FFDOEEff<4>, FFGradDOEEff<4>, FFSum<6>, FFBRCrit<7>, FFGradBRCrit<7>, FFBREff<9>, FFGradBREff<9> > MINLP;

  using BASE_MBDOE<ExtOps...>::_ny;
  using BASE_MBDOE<ExtOps...>::_np;
  using BASE_MBDOE<ExtOps...>::_nc;
  using BASE_MBDOE<ExtOps...>::_vPARVAL;
  using BASE_MBDOE<ExtOps...>::_vPARWEI;
  using BASE_MBDOE<ExtOps...>::_vPARSCA;
  using BASE_MBDOE<ExtOps...>::_vCONLB;
  using BASE_MBDOE<ExtOps...>::_vCONUB;
  using BASE_MBDOE<ExtOps...>::_vOUTVAR;
  
private:

  //! @brief vector of SNOPTA workers
  //std::vector<WORKER_MBDOE<ExtOps...>*> _worker;

  //! @brief DAG of model
  DAG* _dag;

  //! @brief DAG of IVP-ODE model
  IVPODE* _ivpode;

  //! @brief DAG of MBDOE problems
  DAGDOE* _dagdoe;

  //! @brief local copy of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief local copy of experimental controls
  std::vector<FFVar> _vCON;

  //! @brief vector of experimental control samples
  std::vector<std::vector<double>> _vCONSAM;

  //! @brief local copy of model outputs
  std::vector<FFVar> _vOUT;
  
  //! @brief vector of FIM entries
  std::vector<FFVar> _vFIM;

  //! @brief output subgraph
  mc::FFSubgraph _sgOUT;

  //! @brief work array for output evaluations
  std::vector<double> _wkOUT;

  //! @brief FIM values
  std::vector<double> _dOUT;
  
  // Vector of atom matrices
  std::vector< std::vector< arma::vec > > _vOUTSAM;

  //! @brief FIM subgraph
  mc::FFSubgraph _sgFIM;

  //! @brief work array for FIM evaluations
  std::vector<double> _wkFIM;

  //! @brief FIM values
  std::vector<double> _dFIM;
  
  // Vector of atom matrices
  std::vector< std::vector< arma::mat > > _vFIMSAM;

  //! @brief local copy of IVP-ODE time variable
  std::vector<FFVar> _vT;

  //! @brief local copy of IVP-ODE state variable
  std::vector<FFVar> _vX;

  //! @brief local copy of IVP-ODE quadrature variables
  std::vector<FFVar> _vQ;

  //! @brief local copy of IVP-ODE differential equations
  std::vector<std::vector<FFVar>> _vDE;

  //! @brief local copy of IVP-ODE initial conditions
  std::vector<std::vector<FFVar>> _vIC;

  //! @brief local copy of IVP-ODE quadrature equations
  std::vector<std::vector<FFVar>> _vQUAD;

  //! @brief local copy of IVP-ODE state functions
  std::vector<std::vector<FFVar>> _vFCT;

public:
  /** @defgroup MBDOESLV Model-based Design of Experiments using MC++
   *  @{
   */
   
  //! @brief Constructor
  MBDOESLV()
    : _dag(nullptr), _ivpode(nullptr), _dagdoe(nullptr),
      _VOpt(0./0.)
    {}

  //! @brief Destructor
  virtual ~MBDOESLV()
    {
      delete   _dag;
      delete   _dagdoe;
      delete   _ivpode;
    }

  //! @brief MBDOE solver options
  struct Options
  {
    //! @brief Constructor
    Options():
      CRITERION(DOEBase::DOPT), RISK(NEUTRAL), CVARTHRES(0.2), INISAMP(100), MINDIST(1e-6), 
      MAXITER(4), TOLITER(1e-5), DISPLEVEL(0), MAXTHREAD(0), 
      MINLPSLV(), NLPSLV()
      {
#ifdef MC__USE_SNOPT
        NLPSLV.DISPLEVEL            = 0;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-6;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#elif  MC__USE_IPOPT
        NLPSLV.DISPLEVEL            = 0;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-5;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.HESSMETH             = NLP::Options::LBFGS;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#endif
        MINLPSLV.SEARCHALG          = MINLP::Options::OA;
        MINLPSLV.DISPLEVEL          = DISPLEVEL;
        MINLPSLV.CVRTOL             = 1e-6;
        MINLPSLV.CVATOL             = 1e-9;
        MINLPSLV.FEASTOL            = 1e-6;
        MINLPSLV.FEASPUMP           = 0;
        MINLPSLV.ROOTCUT            = 1;
        MINLPSLV.TIMELIMIT          = 36e2;
        MINLPSLV.LINMETH            = MINLP::Options::CVX;
        MINLPSLV.MAXITER            = 40;
        MINLPSLV.MSLOC              = 1;
        MINLPSLV.CPMAX              = 5;
        MINLPSLV.NLPSLV             = NLPSLV;
#ifdef MC__USE_GUROBI
        MINLPSLV.MIPSLV.DISPLEVEL   = 0;
        MINLPSLV.MIPSLV.THREADS     = 0;
        MINLPSLV.MIPSLV.MIPRELGAP   = 1e-6;
        MINLPSLV.MIPSLV.MIPABSGAP   = 1e-9;
        MINLPSLV.MIPSLV.OUTPUTFILE  = "";//"doe.lp";
#elif  MC__USE_CPLEX
        throw std::runtime_error("Error: CPLEX solver not yet implemented");
#endif
        DOEBase::type               = CRITERION;
      }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        CRITERION   = options.CRITERION;
        RISK        = options.RISK;
        CVARTHRES   = options.CVARTHRES;
        INISAMP     = options.INISAMP;
        MINDIST     = options.MINDIST;
        MAXITER     = options.MAXITER;
        TOLITER     = options.TOLITER;
        DISPLEVEL   = options.DISPLEVEL;
        MAXTHREAD   = options.MAXTHREAD;
        MINLPSLV    = options.MINLPSLV;
        NLPSLV      = options.NLPSLV;
        return *this;
      }
    //! @brief Enumeration type for risk attitude
    enum RISK_TYPE{
      NEUTRAL=0, //!< Perform a risk-neutral average design
      AVERSE     //!< Perform a risk-averse CVaR design
    };
    //! @brief Selected DOE criterion
    DOEBase::TYPE            CRITERION;
    //! @brief Selected risk attitude
    RISK_TYPE                RISK;
    //! @brief Percentile threshold for CVaR calcualtion
    double                   CVARTHRES;
    //! @brief Initial sampling size of experimental control space
    unsigned                 INISAMP;
    //! @brief Minimal relative mean-absolute distance between support points after refinement
    double                   MINDIST;
   //! @brief Maximal iteration of effort-based and gradient-based solves
    int                      MAXITER;
   //! @brief Stopping tolerance for effort-based and gradient-based iteration
    double                   TOLITER;
    //! @brief Verbosity level
    int                      DISPLEVEL;
    //! @brief Maximum number of threads for control space sampling
    unsigned                 MAXTHREAD;
    
    //! @brief MINLP effort-based solver options
    typename MINLP::Options  MINLPSLV;
    //! @brief NLP gradient-based solver options
    typename NLP::Options    NLPSLV;
  } options;

  //! @brief MBDOE solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for MBDOESLV exception handling
    enum TYPE{
      BADSIZE=0,    //!< Inconsistent dimensions
      BADIVP,       //!< Misspecified IVP-ODE
      NOMODEL,	    //!< unspecified model
      INTERN=-33    //!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr(){ return _ierr; }
    //! @brief Inline function returning the error description
    std::string what(){
      switch( _ierr ){
        case BADSIZE:
          return "MBDOESLV::Exceptions  Inconsistent dimensions";
        case BADIVP:
          return "MBDOESLV::Exceptions  Misspecified IVP-ODE model";
        case NOMODEL:
          return "MBDOESLV::Exceptions  Unspecified model";
        case INTERN:
        default:
          return "MBDOESLV::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief Setup MBDOE problem before solution
  bool setup
    ();

  //! @brief Evaluate performance of experimental campaign
  std::pair<double,bool> evaluate_design
    ( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type="",
      std::ostream& os=std::cout );

  //! @brief Generate FIM samples for <a>NSAM</a> initial supports
  bool sample_supports
    ( unsigned const NSAM, std::ostream& os=std::cout );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports
  void effort_solve
    ( unsigned const NEXP, std::map<unsigned,double> const& EIni = std::map<unsigned,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports 
  void gradient_solve
    ( std::map<unsigned,double> const& EOpt, bool const update=true, std::ostream& os=std::cout );

  //! @brief Solve combined effort- and gradient-based experiment designwith <a>NEXP</a> supports 
  void combined_solve
    ( unsigned const NEXP, std::ostream& os=std::cout );

  //! @brief Export effort, support and fim to file
  bool file_export
    ( std::string const& name );

  //! @brief Retrieve optimized efforts  
  std::map<unsigned,double> const& efforts
    ()
    const
    { return _EOpt; }

  //! @brief Retrieve optimized supports
  std::map<unsigned,std::vector<double>> const& supports
    ()
    const
    { return _SOpt; }
  
  //! @brief Retrieve optimized criterion
  double criterion
    ()
    const
    { return _VOpt; }

  //! @brief Retrieve optimized campaign
  std::multimap<double,std::vector<double>> campaign
    ()
    const;

protected:

  //! @brief current optimal criterion
  double _VOpt;

  //! @brief map of current optimal efforts
  std::map<unsigned,double> _EOpt;
  
  //! @brief map of current optimal supports
  std::map<unsigned,std::vector<double>> _SOpt;
  
  //! @brief vector of current optimal values for risk-averse variables
  std::vector<double> _ROpt;

  //! @brief Create local copy of output model for output prediction
  void _setup_out
    ();

  //! @brief Create local copy of output model for FIM prediction
  void _setup_fim
    ();

  //! @brief Create local copy of IVP-ODE model for output prediction
  void _setup_ivp_out
    ( IVPODE const& IVP );

  //! @brief Create local copy of IVP-ODE model for FIM prediction
  void _setup_ivp_fim
    ( IVPODE const& IVP );

  //! @brief Set parameter values to scenario <a>dPAR</a>
  void _set_parameters
    ( double const* dPAR, std::ostream& os=std::cout );

  //! @brief Generate output samples for <a>NSAM</a> initial supports
  bool _sample_out
    ( unsigned const NSAM, std::ostream& os=std::cout );

  //! @brief Append output under uncertainty scenario iUNC
  bool _append_out
    ( double const* Control, unsigned const iUnc=0, std::ostream& os=std::cout );

  //! @brief Generate FIM samples for <a>NSAM</a> initial supports
  bool _sample_fim
    ( unsigned const NSAM, std::ostream& os=std::cout );

  //! @brief Append FIM (columnwise, lower-triangular) under uncertainty scenario iUNC
  bool _append_fim
    ( double const* Control, unsigned const iUnc=0, std::ostream& os=std::cout );

  //! @brief Evaluate Bayesian risk of experimental campaign
  std::pair<double,bool> _evaluate_design_br
    ( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os );

  //! @brief Evaluate FIM-based criterion of experimental campaign
  std::pair<double,bool> _evaluate_design_fim
    ( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports to minimize Bayesian risk
  void _effort_minimize_br
    ( unsigned const NEXP, std::map<unsigned,double> const& EIni = std::map<unsigned,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports to maximize FIM
  void _effort_maximize_fim
    ( unsigned const NEXP, std::map<unsigned,double> const& EIni = std::map<unsigned,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports to minimize Bayesian risk
  void _gradient_minimize_br
    ( std::map<unsigned,double> const& EOpt, bool const update=true, std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports to maximize FIM
  void _gradient_maximize_fim
    ( std::map<unsigned,double> const& EOpt, bool const update=true, std::ostream& os=std::cout );

  //! @brief Build FIM for gradient-based search
  void _build_fim
    ( std::vector<FFVar>& vFIM, std::vector<FFVar>& CTOT, std::map<unsigned,double> const& EOpt,
      std::ostream& os=std::cout );

  //! @brief Build Bayesian risk for gradient-based search
//  void _build_br
//    ( std::vector<FFVar>& BRCRIT, std::vector<double>& WCRIT, std::vector<FFVar>& CTOT,
//      std::map<unsigned,double> const& EOpt, std::ostream& os );
  void _build_br
    ( std::vector<FFVar>& BROUT, std::vector<FFVar>& CTOT, std::map<unsigned,double> const& EOpt,
      std::ostream& os );

  //! @brief Generate samples for refined supports
  bool _update_supports
    ( std::map<unsigned,double> const EOpt, std::map<unsigned,std::vector<double>> const SOpt,
      std::ostream& os=std::cout );

  //! @brief Determine if support <a>supp</a> is redundant with an existing support
  unsigned _redundant_support
    ( std::vector<double> const& supp );

  //! @brief Mean-absolute error between two supports
  double _mae_support
    ( std::vector<double> const& s1, std::vector<double> const& s2 );

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit, std::map<unsigned,double> const& eff,
      std::map<unsigned,std::vector<double>> const& supp, std::ostream& os=std::cout )
    const;

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit, std::multimap<double,std::vector<double>> const& campaign,
      std::ostream& os=std::cout )
    const;
};

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::setup
()
{
  // Case of algebraic model
  if( _ny ){
    switch( options.CRITERION ){
     case DOEBase::BROPT:
      _setup_out();
      break;
     case DOEBase::AOPT:
     case DOEBase::DOPT:
     case DOEBase::EOPT:
     default:
      _setup_fim();
      break;
    }
  }
  
  // Case of dynamic model
  else if( BASE_MBDOE<ExtOps...>::_ivpode ){ 
    switch( options.CRITERION ){
     case DOEBase::BROPT:
      _setup_ivp_out( *BASE_MBDOE<ExtOps...>::_ivpode );
      break;
     case DOEBase::AOPT:
     case DOEBase::DOPT:
     case DOEBase::EOPT:
     default:
      _setup_ivp_fim( *BASE_MBDOE<ExtOps...>::_ivpode );
      break;
    }
  }

  // Missing model
  else
    throw Exceptions( Exceptions::NOMODEL );

  return true;
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_setup_out
()
{
  if( !_ny || _ny != BASE_MBDOE<ExtOps...>::_vOUT.size() || !BASE_MBDOE<ExtOps...>::_vCON.size() )
    throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;

  _vCON.resize( _nc );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _nc, BASE_MBDOE<ExtOps...>::_vCON.data(), _vCON.data() );
  _vPAR.resize( _np );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _np, BASE_MBDOE<ExtOps...>::_vPAR.data(), _vPAR.data() );
  _vOUT.resize( _ny );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _ny, BASE_MBDOE<ExtOps...>::_vOUT.data(), _vOUT.data() );
  _dOUT.resize( _vOUT.size() );
  
#ifdef MC__MBDOE_SETUP_DEBUG
  _sgOUT = _dag->subgraph( _vOUT.size(), _vOUT.data() );
  std::vector<FFExpr> exOUT = FFExpr::subgraph( _dag, _sgOUT ); 
  for( unsigned i=0; i<_ny; ++i )
    std::cout << "OUT[" << i << "] = " << exOUT[i] << std::endl;
#endif
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_setup_fim
()
{
  if( !_ny || _ny != BASE_MBDOE<ExtOps...>::_vOUT.size() || !BASE_MBDOE<ExtOps...>::_vCON.size() )
    throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;

  _vCON.resize( _nc );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _nc, BASE_MBDOE<ExtOps...>::_vCON.data(), _vCON.data() );
  _vPAR.resize( _np );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _np, BASE_MBDOE<ExtOps...>::_vPAR.data(), _vPAR.data() );
  _vOUT.resize( _ny );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _ny, BASE_MBDOE<ExtOps...>::_vOUT.data(), _vOUT.data() );

  FFVar* y_p = _dag->FAD( _ny, _vOUT.data(), _np, _vPAR.data(), true ); // Jacobian in dense format
  _vFIM.assign( _np*(_np+1)/2, 0. );
  for( unsigned k=0; k<_ny; k++ )
    for( unsigned i=0, ij=0; i<_np; ++i )
      for( unsigned j=i; j<_np; ++j, ++ij ){
        if( _vOUTVAR.size() == _ny )
          _vFIM[ij] += (y_p[_ny*i+k] * y_p[_ny*j+k]) / _vOUTVAR[k];
        else
          _vFIM[ij] += y_p[_ny*i+k] * y_p[_ny*j+k];
      }
  _dFIM.resize( _vFIM.size() );
  delete[] y_p;
  
#ifdef MC__MBDOE_SETUP_DEBUG
  _sgFIM = _dag->subgraph( _vFIM.size(), _vFIM.data() );
  std::vector<FFExpr> exFIM = FFExpr::subgraph( _dag, _sgFIM ); 
  for( unsigned i=0, ij=0; i<_np; ++i )
    for( unsigned j=i; j<_np; ++j, ++ij )
      std::cout << "FIM[" << i << "][" << j << "] = " << exFIM[ij] << std::endl;
#endif
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_setup_ivp_out
( IVPODE const& IVP )
{
  if( !IVP.nf() || (!IVP.nx() && !IVP.nq()) || !IVP.np() || IVP.nx() != IVP.nx0() )
    throw Exceptions( Exceptions::BADIVP );

  delete _dag; _dag = new DAG;
  delete _ivpode; _ivpode = new IVPODE;
  _ivpode->options = IVP.options;
  _ivpode->set_dag( _dag );
  
  _vPAR.resize( _np );
  for( unsigned i=0; i<_np; ++i ) BASE_MBDOE<ExtOps...>::_vPAR[i].unset();
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _np, BASE_MBDOE<ExtOps...>::_vPAR.data(), _vPAR.data() );

  _vCON.resize( _nc );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _nc, BASE_MBDOE<ExtOps...>::_vCON.data(), _vCON.data() );
  _ivpode->set_parameter( _vCON );

//  unsigned ns = IVP.nsmax();
  _vT.clear();
  if( IVP.var_time() ){
    _vT.resize( 1 );
    _dag->insert( IVP.dag(), 1, IVP.var_time(), _vT.data() );
  }
  _ivpode->set_time( IVP.val_stage(), _vT.data() );

  unsigned nx = IVP.nx();
  _vX.resize( nx ); // states and state sensitivities wrt parameters
  _dag->insert( IVP.dag(), nx, IVP.var_state().data(), _vX.data() );
  _ivpode->set_state( _vX );

  _vDE.clear();
  _vDE.reserve( IVP.eqn_differential().size() );
  for( auto const& de0 : IVP.eqn_differential() ){
    std::vector<FFVar> de( nx );
    _dag->insert( IVP.dag(), nx, de0.data(), de.data() );
    _vDE.push_back( de );
#ifdef MC__MBDOE_SETUP_DEBUG
    FFSubgraph sgDE = _dag->subgraph( nx, _vDE.back().data() );
    std::vector<FFExpr> exprDE = FFExpr::subgraph( _dag, sgDE ); 
    for( unsigned j=0; j<nx; ++j )
        std::cout << "DE[" << _vDE.size()-1 << "][" << j << "] = " << exprDE[j] << std::endl;
#endif
  }
  _ivpode->set_differential( _vDE );

  assert( IVP.nx0() == nx );
  _vIC.clear();
  _vIC.reserve( IVP.eqn_initial().size() );
  for( auto const& ic0 : IVP.eqn_initial() ){
    std::vector<FFVar> ic( nx );
    _dag->insert( IVP.dag(), nx, ic0.data(), ic.data() );
    _vIC.push_back( ic );
  }
  _ivpode->set_initial( _vIC );

  unsigned nq = IVP.nq();
  _vQUAD.clear();
  _vQ.resize( nq );
  if( nq ){
    _dag->insert( IVP.dag(), nq, IVP.var_quadrature().data(), _vQ.data() );
    _vQUAD.reserve( IVP.eqn_quadrature().size() );
    for( auto const& quad0 : IVP.eqn_quadrature() ){
      std::vector<FFVar> quad( nq );
      _dag->insert( IVP.dag(), nq, quad0.data(), quad.data() );
      _vQUAD.push_back( quad );
    }
    _ivpode->set_quadrature( _vQUAD, _vQ );
  }

  unsigned nf = IVP.nf();
  _vFCT.clear();
  _vFCT.reserve( IVP.eqn_function().size() );
  for( auto const& fct0 : IVP.eqn_function() ){
    std::vector<FFVar> fct( nf );
    _dag->insert( IVP.dag(), nf, fct0.data(), fct.data() );
    _vFCT.push_back( fct );
  }
  _ivpode->set_function( _vFCT );
  _dOUT.resize( nf );
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_setup_ivp_fim
( IVPODE const& IVP )
{
  if( !IVP.nf() || (!IVP.nx() && !IVP.nq()) || !IVP.np() || IVP.nx() != IVP.nx0() )
    throw Exceptions( Exceptions::BADIVP );

  delete _dag; _dag = new DAG;
  delete _ivpode; _ivpode = new IVPODE;
  _ivpode->options = IVP.options;
  _ivpode->set_dag( _dag );
  mc::FFVar One = 1.;
  
  _vPAR.resize( _np );
  for( unsigned i=0; i<_np; ++i ) BASE_MBDOE<ExtOps...>::_vPAR[i].unset();
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _np, BASE_MBDOE<ExtOps...>::_vPAR.data(), _vPAR.data() );

  _vCON.resize( _nc );
  _dag->insert( BASE_MBDOE<ExtOps...>::_dag, _nc, BASE_MBDOE<ExtOps...>::_vCON.data(), _vCON.data() );
  _ivpode->set_parameter( _vCON );

  // Stages and stage times
//  unsigned ns = IVP.nsmax();
  _vT.clear();
  if( IVP.var_time() ){
    _vT.resize( 1 );
    _dag->insert( IVP.dag(), 1, IVP.var_time(), _vT.data() );
  }
  _ivpode->set_time( IVP.val_stage(), _vT.data() );

  // States and differential equations
  unsigned nx = IVP.nx();
  _vX.resize( nx*(1+_np) ); // states and state sensitivities wrt parameters
  _dag->insert( IVP.dag(), nx, IVP.var_state().data(), _vX.data() );
  for( unsigned i=nx; i<nx*(1+_np); ++i ) _vX[i].set( _dag );
  _ivpode->set_state( _vX );

  _vDE.clear();
  _vDE.reserve( IVP.eqn_differential().size() );
  for( auto const& de0 : IVP.eqn_differential() ){
    std::vector<FFVar> de( nx*(1+_np) );
    _dag->insert( IVP.dag(), nx, de0.data(), de.data() );
    for( unsigned i=0; i<_np; ++i ){
      mc::FFVar* de_i = _dag->DFAD( nx, de.data(), nx, _vX.data(), _vX.data()+nx*(1+i), 1, _vPAR.data()+i, &One );
      for( unsigned j=0; j<nx; ++j ) de[nx*(1+i)+j] = de_i[j];
      delete[] de_i;
    }
    _vDE.push_back( de );   
#ifdef MC__MBDOE_SETUP_DEBUG
    FFSubgraph sgDE = _dag->subgraph( nx*(1+_np), _vDE.back().data() );
    std::vector<FFExpr> exprDE = FFExpr::subgraph( _dag, sgDE ); 
    for( unsigned i=0, ij=0; i<_np+1; ++i )
      for( unsigned j=0; j<nx; ++j, ++ij )
        std::cout << "DE[" << _vDE.size()-1 << "][" << i << "][" << j << "] = " << exprDE[ij] << std::endl;
#endif
  }
  _ivpode->set_differential( _vDE );

  // Initial conditions
  assert( IVP.nx0() == nx );
  _vIC.clear();
  _vIC.reserve( IVP.eqn_initial().size() );
  for( auto const& ic0 : IVP.eqn_initial() ){
    std::vector<FFVar> ic( nx*(1+_np) );
    _dag->insert( IVP.dag(), nx, ic0.data(), ic.data() );
    for( unsigned i=0; i<_np; ++i ){
      mc::FFVar* ic_i = _dag->DFAD( nx, ic.data(), 1, _vPAR.data()+i, &One );
      for( unsigned j=0; j<nx; ++j ) ic[nx*(1+i)+j] = ic_i[j];
      delete[] ic_i;
    }
    _vIC.push_back( ic );
#ifdef MC__MBDOE_SETUP_DEBUG
    FFSubgraph sgIC = _dag->subgraph( nx*(1+_np), _vIC.back().data() );
    std::vector<FFExpr> exprIC = FFExpr::subgraph( _dag, sgIC ); 
    for( unsigned i=0, ij=0; i<_np+1; ++i )
      for( unsigned j=0; j<nx; ++j, ++ij )
        std::cout << "IC[" << _vIC.size()-1 << "][" << i << "][" << j << "] = " << exprIC[ij] << std::endl;
#endif
  }
  _ivpode->set_initial( _vIC );

  // Quadrature variables and expressions
  unsigned nq = IVP.nq();
  _vQ.resize( nq*(1+_np) );
  _vQUAD.clear();
  if( nq ){
    _dag->insert( IVP.dag(), nq, IVP.var_quadrature().data(), _vQ.data() );
    _vQUAD.reserve( IVP.eqn_quadrature().size() );
    for( auto const& quad0 : IVP.eqn_quadrature() ){
      std::vector<FFVar> quad( nq*(1+_np) );
      _dag->insert( IVP.dag(), nq, quad0.data(), quad.data() );
      for( unsigned i=0; i<_np; ++i ){
        mc::FFVar* quad_i = _dag->DFAD( nq, quad.data(), nx, _vX.data(), _vX.data()+nx*(1+i), 1, _vPAR.data()+i, &One );
        for( unsigned j=0; j<nq; ++j ) quad[nq*(1+i)+j] = quad_i[j];
        delete[] quad_i;
      }
      _vQUAD.push_back( quad );
#ifdef MC__MBDOE_SETUP_DEBUG
      FFSubgraph sgQUAD = _dag->subgraph( nq*(1+_np), _vQUAD.back().data() );
      std::vector<FFExpr> exprQUAD = FFExpr::subgraph( _dag, sgQUAD ); 
      for( unsigned i=0, ij=0; i<_np+1; ++i )
        for( unsigned j=0; j<nq; ++j, ++ij )
          std::cout << "QUAD[" << _vQUAD.size()-1 << "][" << i << "][" << j << "] = " << exprQUAD[ij] << std::endl;
#endif
    }
    _ivpode->set_quadrature( _vQUAD, _vQ );
  }

  // State functions
  unsigned nf = IVP.nf();
  _vFCT.clear();
  _vFCT.reserve( IVP.eqn_function().size() );
  for( auto const& fct0 : IVP.eqn_function() ){
    std::vector<FFVar> fct( nf );
    _dag->insert( IVP.dag(), nf, fct0.data(), fct.data() );
    std::vector< FFVar* > fct_p( _np );
    for( unsigned i=0; i<_np; ++i )
      fct_p[i] = _dag->DFAD( nf, fct.data(), nx, _vX.data(), _vX.data()+nx*(1+i), 1, _vPAR.data()+i, &One );
    std::vector<FFVar> fim( _np*(_np+1)/2 );
    for( unsigned k=0; k<nf; k++ )
      for( unsigned i=0, ij=0; i<_np; ++i )
        for( unsigned j=i; j<_np; ++j, ++ij ){
          if( !k ){
            if( _vOUTVAR.size() == nf )
              fim[ij] += (fct_p[i][0] * fct_p[j][0]) / _vOUTVAR[k];
            else
              fim[ij] += fct_p[i][0] * fct_p[j][0];
          }
          else{
            if( _vOUTVAR.size() == nf )
              fim[ij] += (fct_p[i][k] * fct_p[j][k]) / _vOUTVAR[k];
            else
              fim[ij] += fct_p[i][k] * fct_p[j][k];
          }
        }
    for( auto& fct_i : fct_p ) delete[] fct_i;
    _vFCT.push_back( fim );
#ifdef MC__MBDOE_SETUP_DEBUG
    FFSubgraph sgFCT = _dag->subgraph( _np*(1+_np)/2, _vFCT.back().data() );
    std::vector<FFExpr> exprFCT = FFExpr::subgraph( _dag, sgFCT ); 
    for( unsigned i=0, ij=0; i<_np; ++i )
      for( unsigned j=i; j<_np; ++j, ++ij )
        std::cout << "FCT[" << _vFCT.size()-1 << "][" << i << "][" << j << "] = " << exprFCT[ij] << std::endl;
#endif
  }
  _ivpode->set_function( _vFCT );
  _dFIM.resize( _np*(_np+1)/2 );
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_set_parameters
( double const* dPAR, std::ostream& os )
{
  for( unsigned i=0; i<_np; ++i )
    _vPAR[i].set( dPAR[i] );

  if( _ivpode ) _ivpode->setup();
}

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::sample_supports
( unsigned const NSAM, std::ostream& os )
{
  if( options.DISPLEVEL )
    os << "** GENERATING SUPPORT SAMPLES" << std::endl;

  // Control samples
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( _nc );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( 0 );

  _vCONSAM.clear();
  _vCONSAM.reserve( NSAM );
  for( unsigned s=0; s<NSAM; ++s ){
    _vCONSAM.push_back( std::vector<double>( _nc ) );
    for( unsigned i=0; i<_nc; i++ )
      _vCONSAM.back()[i] = _vCONLB[i] + ( _vCONUB[i] - _vCONLB[i] ) * gen();
  }

  // Observation samples
  switch( options.CRITERION ){
    case DOEBase::BROPT:
      return _sample_out( NSAM, os );
      
    case DOEBase::AOPT:
    case DOEBase::DOPT:
    case DOEBase::EOPT:
    default:
      return _sample_fim( NSAM, os );
  }
}

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::_sample_out
( unsigned const NSAM, std::ostream& os )
{
  // Response samples
  _vOUTSAM.clear();
  _vOUTSAM.reserve( _vPARVAL.size() );
  auto itPARVAL = _vPARVAL.cbegin();
  for( unsigned k=0; k<_vPARVAL.size(); ++k, ++itPARVAL ){
    // Set parameter scenario
    _set_parameters( itPARVAL->data(), os );

    // Compute responses at control samples
    _vOUTSAM.push_back( std::vector< arma::vec >() );
    for( unsigned s=0; s<NSAM; ++s ){
      if( !_append_out( _vCONSAM[s].data(), k ) )
        return false;
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  return true;
}

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::_append_out
( double const* Control, unsigned const iUnc, std::ostream& os )
{
  if( _ny )
    try{ _dag->eval( _sgOUT, _wkOUT, _vOUT.size(), _vOUT.data(), _dOUT.data(), _nc, _vCON.data(), Control ); }
    catch(...){ return false; }

  else if( _ivpode )
    try{ _ivpode->solve_state( Control ); 
         _dOUT = _ivpode->val_function(); }
//    try{ _ivpode->states( Control, nullptr, _dOUT.data() ); }
    catch(...){ return false; }

  else
    throw Exceptions( Exceptions::NOMODEL );

  arma::vec OUT( _dOUT );
#ifdef MC__MBDOE_SAMPLE_DEBUG
  std::cout << "OUT[" << iUnc << "][" << _vOUTSAM[iUnc].size() << "]:" << std::endl << OUT;
#endif

  if( _vOUTSAM[iUnc].size() && arma::size( _vOUTSAM[iUnc].back() ) != arma::size( OUT ) )
    throw Exceptions( Exceptions::BADSIZE );
  _vOUTSAM[iUnc].push_back( OUT );
  
  return true;
}

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::_sample_fim
( unsigned const NSAM, std::ostream& os )
{
  // FIM samples
  _vFIMSAM.clear();
  _vFIMSAM.reserve( _vPARVAL.size() );
  auto itPARVAL = _vPARVAL.cbegin();
  for( unsigned k=0; k<_vPARVAL.size(); ++k, ++itPARVAL ){
    // Set parameter scenario
    _set_parameters( itPARVAL->data(), os );

    // Compute FIMs at control samples
    _vFIMSAM.push_back( std::vector< arma::mat >() );
    for( unsigned s=0; s<NSAM; ++s ){
      if( !_append_fim( _vCONSAM[s].data(), k ) )
        return false;
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  return true;
}

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::_append_fim
( double const* Control, unsigned const iUnc, std::ostream& os )
{
  if( _ny )
    try{ _dag->eval( _sgFIM, _wkFIM, _vFIM.size(), _vFIM.data(), _dFIM.data(), _nc, _vCON.data(), Control ); }
    catch(...){ return false; }

  else if( _ivpode )
    try{ _ivpode->solve_state( Control ); 
         _dFIM = _ivpode->val_function(); }
    //try{ _ivpode->states( Control, nullptr, _dFIM.data() ); }
    catch(...){ return false; }

  else
    throw Exceptions( Exceptions::NOMODEL );

  arma::mat FIM( _np, _np, arma::fill::none );
  for( unsigned i=0, l=0; i<_np; ++i )
    for( unsigned j=i; j<_np; ++j, ++l )
      if( i == j ) FIM(i,i) = _dFIM[l]; 
      else         FIM(i,j) = FIM(j,i) = _dFIM[l];
#ifdef MC__MBDOE_SAMPLE_DEBUG
  std::cout << "FIM[" << iUnc << "][" << _vFIMSAM[iUnc].size() << "]:" << std::endl << FIM;
#endif

  if( _vFIMSAM[iUnc].size() && arma::size( _vFIMSAM[iUnc].back() ) != arma::size( FIM ) )
    throw Exceptions( Exceptions::BADSIZE );
  _vFIMSAM[iUnc].push_back( FIM );
  
  return true;
}

template <typename... ExtOps>
inline
unsigned
MBDOESLV<ExtOps...>::_redundant_support
( std::vector<double> const& suppref )
{
  unsigned pos=0;
  for( auto const& supp : _vCONSAM ){
    if( _mae_support( supp, suppref ) < options.MINDIST )
      return pos;
    ++pos;
  }
  return pos;
}

template <typename... ExtOps>
inline
double
MBDOESLV<ExtOps...>::_mae_support
( std::vector<double> const& s1, std::vector<double> const& s2 )
{
  if( s1.size() != s2.size() || s1.size() != _vCONLB.size() )
    return 0./0.; // NaN
  double mae=0;
  for( auto it1 = s1.cbegin(), it2 = s2.cbegin(), itLB = _vCONLB.cbegin(), itUB = _vCONUB.cbegin();
       it1 != s1.end();
       ++it1, ++it2, ++itLB, ++itUB )
    mae += std::fabs( *it1 - *it2 ) / std::fabs( *itUB - *itLB );
  mae /= s1.size();
#ifdef MC__MBDOE_SAMPLE_DEBUG
  std::cout << "mae: " << std::scientific << std::setprecision(7) << mae << std::endl;
#endif
  return mae;
}

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::_update_supports
( std::map<unsigned,double> const EOpt, std::map<unsigned,std::vector<double>> const SOpt,
  std::ostream& os )
{
  if( options.DISPLEVEL )
    os << "** REFINING SUPPORT SAMPLES" << std::endl;

  size_t posSupp = _vCONSAM.size(), newSupp = 0;
  auto itE = EOpt.cbegin();
  auto itS = SOpt.cbegin();
  _EOpt.clear();
  _SOpt.clear();
  for( ; itS != SOpt.cend(); ++itS, ++itE ){
    auto const& eff  = itE->second; 
    auto const& supp = itS->second;
    unsigned pos = _redundant_support( supp );
    // Refined support is redundant
    if( pos < _vCONSAM.size() ){
      if( options.DISPLEVEL > 1 )
        os << "   REFINED SUPPORT REDUNDANT WITH #" << pos << std::endl;
      auto itR = _EOpt.find( pos );
      // Redundant support not present
      if( itR == _EOpt.end() ){
        _EOpt[pos] = eff;
        _SOpt[pos] = supp;
      }
      // Redundant support already present
      else
        _EOpt[pos] += eff;      
    }
    // Refined support is distinct
    else{
      _EOpt[_vCONSAM.size()] = eff;
      _SOpt[_vCONSAM.size()] = supp;
      _vCONSAM.push_back( supp );
      ++newSupp;
    }
  }

  auto itPARVAL = _vPARVAL.cbegin();
  for( unsigned k=0; k<_vPARVAL.size(); ++k, ++itPARVAL ){
    // Set parameter scenario
    _set_parameters( itPARVAL->data(), os );

    // Compute FIMs at refined controls
    for( unsigned s=posSupp; s<posSupp+newSupp; ++s ){
      switch( options.CRITERION ){
        case DOEBase::BROPT:
          if( !_append_out( _vCONSAM[s].data(), k ) )
            return false;
          break;
      
        case DOEBase::AOPT:
        case DOEBase::DOPT:
        case DOEBase::EOPT:
        default:
          if( !_append_fim( _vCONSAM[s].data(), k ) )
            return false;
          break;
      }
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  return true;
}

template <typename... ExtOps>
inline
bool
MBDOESLV<ExtOps...>::file_export
( std::string const& name )
{
  auto itPARVAL = _vPARVAL.cbegin();
  for( unsigned s=0; s<_vPARVAL.size(); ++s, ++itPARVAL ){
    std::ofstream ofile( name + "_" + std::to_string(s) + ".log" );
    if( !ofile ) return false;
    
    ofile << std::scientific << std::setprecision(6);
    for( unsigned k=0; k<_vCONSAM.size(); ++k ){
      for( unsigned i=0; i<itPARVAL->size(); ++i )
        ofile << (*itPARVAL)[i] << "  ";

      for( unsigned i=0; i<_vCONSAM[k].size(); ++i )
        ofile << _vCONSAM[k][i] << "  ";

      ofile << ( _EOpt.count(k)? _EOpt[k]: 0 ) << "  ";

      switch( options.CRITERION ){
        case DOEBase::BROPT:
          for( unsigned i=0; i<_vOUTSAM[s][k].n_rows; ++i )
            ofile << _vOUTSAM[s][k](i) << "  ";
          break;
          
        case DOEBase::AOPT:
        case DOEBase::DOPT:
        case DOEBase::EOPT:
        default:
          for( unsigned i=0; i<_vFIMSAM[s][k].n_rows; ++i )
            for( unsigned j=i; j<_vFIMSAM[s][k].n_cols; ++j )
              ofile << _vFIMSAM[s][k](i,j) << "  ";
          break;
      }
      ofile << std::endl;
    }
  }
  return true;
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::combined_solve
( unsigned const NEXP, std::ostream& os )
{
  _EOpt.clear();
  double VLast;
  for( int it=0; ; ){
    effort_solve( NEXP, _EOpt, os );
    if( it && std::fabs( VLast - _VOpt ) < options.TOLITER * std::fabs( VLast + _VOpt ) / 2 ){
      if( options.DISPLEVEL )
        os << "** CONVERGENCE TOLERANCE SATISFIED" << std::endl;
      break;
    }

    gradient_solve( _EOpt, true, os );
    VLast = _VOpt;
    if( ++it >= options.MAXITER ){
      if( options.DISPLEVEL )
        os << "** MAXIMUM ITERATION LIMIT REACHED" << std::endl;
      break;
    }
  }
  
  //unsigned k=0;
  //for( auto const& mat : _vFIMSAM[0] )
  //  std::cout << "FIM " << k++ << ":" << mat;
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::effort_solve
( unsigned const NEXP, std::map<unsigned,double> const& EIni, std::ostream& os )
{
  // Observation samples
  switch( options.CRITERION ){
    case DOEBase::BROPT:
      return _effort_minimize_br( NEXP, EIni, os );
      
    case DOEBase::AOPT:
    case DOEBase::DOPT:
    case DOEBase::EOPT:
    default:
      return _effort_maximize_fim( NEXP, EIni, os );
  }
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_effort_minimize_br
( unsigned const NEXP, std::map<unsigned,double> const& EIni, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAGDOE;
  mc::FFBREff<9> OpDOECrit;
  mc::FFSum<6> Sum;

  unsigned const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF( NSUPP );
  for( auto& Ek : EFF )
    Ek.set( _dagdoe );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, (double)NEXP/(double)NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [isupp,eff] : EIni )
      E0[isupp] = eff;
  }
  
  // Convex MINLP optimization
  MINLP doe;
  DOEBase::set_weighting( _vPARWEI );
  DOEBase::set_scaling( _vPARSCA );
  DOEBase::set_noise( _vOUTVAR );
  DOEBase::type = options.CRITERION;
  doe.options   = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( NSUPP, EFF.data(), 0e0, NEXP, 1 ); 
  doe.set_obj( mc::BASE_OPT::MIN, OpDOECrit( NSUPP, EFF.data(), &_vOUTSAM ) );
  doe.add_ctr( mc::BASE_OPT::EQ, Sum( NSUPP, EFF.data() ) - (int)NEXP );  

  doe.setup();
  //doe.optimize( E0.data() );
  //doe.optimize( E0.data(), nullptr, apportion );
  doe.optimize( E0.data(), nullptr, effrounding );

  if( options.DISPLEVEL > 1 )
    doe.stats.display();

  _EOpt.clear();
  _SOpt.clear();
  _VOpt = mc::BASE_OPT::BASE_OPT::INF;
  if( doe.get_status() == MINLP::SUCCESSFUL ){
    unsigned isupp = 0;
    for( auto const& Ek : doe.get_incumbent().x ){
      if( Ek > 1e-3 ){
        _EOpt[isupp] = Ek;
        _SOpt[isupp] = _vCONSAM[isupp];
      }
      if( ++isupp >= NSUPP )
        break;
    }
    _VOpt = doe.get_incumbent().f[0];
  }

  if( options.DISPLEVEL )
    _display_design( "EFFORT-BASED EXACT DESIGN", _VOpt, _EOpt, _SOpt, os ); 
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_effort_maximize_fim
( unsigned const NEXP, std::map<unsigned,double> const& EIni, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAGDOE;
  mc::FFDOEEff<4> OpDOECrit;
  mc::FFSum<6> Sum;

  unsigned const NUNC  = _vPARVAL.size();
  unsigned const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF( NSUPP );
  for( auto& Ek : EFF )
    Ek.set( _dagdoe );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, (double)NEXP/(double)NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [isupp,eff] : EIni )
      E0[isupp] = eff;
  }
  
  // Convex MINLP optimization
  MINLP doe;
  DOEBase::type = options.CRITERION;
  DOEBase::set_scaling( _vPARSCA );
  doe.options   = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( NSUPP, EFF.data(), 0e0, NEXP, 1 );
 
  switch( options.RISK){
    case Options::NEUTRAL:
    {
      doe.set_obj( mc::BASE_OPT::MAX, Sum( NUNC, OpDOECrit( NSUPP, EFF.data(), &_vFIMSAM ), _vPARWEI.data() ) );
      //doe.set_obj( mc::BASE_OPT::MAX, Sum( NUNC, OpDOECrit( NSUPP, EFF.data(), &_vFIMSAM ) ) / NUNC );
      doe.add_ctr( mc::BASE_OPT::EQ, Sum( NSUPP, EFF.data() ) - (int)NEXP );
      break;
    }
    case Options::AVERSE:
    {
      std::vector<FFVar> DELTA( NUNC );
      FFVar VaR( _dagdoe );
      for( auto& Dk : DELTA )
        Dk.set( _dagdoe );
      E0.resize( NSUPP+NUNC+1, 0e0 );
      doe.add_var( NUNC, DELTA.data(), 0e0 );//, 1e2 );
      doe.add_var( VaR );//, -1e2, 1e2 );
      doe.set_obj( mc::BASE_OPT::MAX, VaR - Sum( NUNC, DELTA.data(), _vPARWEI.data() ) / options.CVARTHRES );
      //doe.set_obj( mc::BASE_OPT::MAX, VaR - Sum( NUNC, DELTA.data() ) / (options.CVARTHRES * NUNC) );
      doe.add_ctr( mc::BASE_OPT::EQ, Sum( NSUPP, EFF.data() ) - (int)NEXP );
      for( unsigned s=0; s<NUNC; s++ )
        doe.add_ctr( mc::BASE_OPT::LE, VaR - DELTA[s] - OpDOECrit( s, NSUPP, EFF.data(), &_vFIMSAM ) );
      break;
    }
  }
  
  doe.setup();
  //doe.optimize( E0.data() );
  //doe.optimize( E0.data(), nullptr, apportion );
  doe.optimize( E0.data(), nullptr, effrounding );

  if( options.DISPLEVEL > 1 )
    doe.stats.display();

  _EOpt.clear();
  _SOpt.clear();
  _ROpt.clear();
  _VOpt = mc::BASE_OPT::BASE_OPT::INF;
  if( doe.get_status() == MINLP::SUCCESSFUL ){
    unsigned isupp = 0;
    for( auto const& Ek : doe.get_incumbent().x ){
      if( isupp >= NSUPP ){
        _ROpt.push_back( Ek );
      }
      else if( Ek > 1e-3 ){
        _EOpt[isupp] = Ek;
        _SOpt[isupp] = _vCONSAM[isupp];
      }
      ++isupp;
    }
    _VOpt = doe.get_incumbent().f[0];
  }

  if( options.DISPLEVEL )
    _display_design( "EFFORT-BASED EXACT DESIGN", _VOpt, _EOpt, _SOpt, os ); 
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_build_br
( std::vector<FFVar>& BROUT, std::vector<FFVar>& CTOT, std::map<unsigned,double> const& EOpt,
  std::ostream& os )
{
  unsigned const NOUT = _dOUT.size();
  unsigned const NEFF = EOpt.size();
  unsigned const NUNC = _vPARVAL.size();
  BROUT.clear();
  BROUT.reserve( NUNC*NEFF*NOUT );

  if( _ny ){

    std::vector<FFVar> vCref( CTOT.size() );
    std::vector<FFVar> vOUTref( NOUT );

    for( unsigned s=0; s<NUNC; ++s ){
      // Set parameter scenario
      _set_parameters( _vPARVAL[s].data(), os );
#ifdef MC__MBDOE_SOLVE_DEBUG
      std::cout << "PARVAL[" << s << "]:";
      for( auto val : _vPARVAL[s] )
        std::cout << "  " << val;
      std::cout <<std::endl;
#endif

      // Define reference outputs in current scenario
      _dagdoe->insert( _dag, _nc, _vCON.data(), vCref.data() );
      _dagdoe->insert( _dag, NOUT, _vOUT.data(), vOUTref.data() );

      // Define outputs in current scenario for each support
      FFVar* Cndx = CTOT.data();
      for( unsigned k=0; k<NEFF; ++k, Cndx+=_nc ){
        FFVar* vOUTcomp = _dagdoe->compose( NOUT, vOUTref.data(), _nc, vCref.data(), Cndx );
        BROUT.insert( BROUT.end(), vOUTcomp, vOUTcomp+NOUT );
        delete[] vOUTcomp;
//        for( unsigned i=0; i<NOUT; ++i )
//          std::cout << "BROUT[" << s << "][" << k << "][" << i << "] -> " << BROUT[BROUT.size()-NOUT+i] << std::endl;
      }
    }
  }

  else if( _ivpode ){

    for( unsigned s=0; s<NUNC; ++s ){
      // Set parameter scenario
      _set_parameters( _vPARVAL[s].data(), os );
#ifdef MC__MBDOE_SOLVE_DEBUG
      std::cout << "PARVAL[" << s << "]:";
      for( auto val : _vPARVAL[s] )
        std::cout << "  " << val;
      std::cout <<std::endl;
#endif

      // Define reference outputs in current scenario
      FFVar* Cndx = CTOT.data();
      mc::FFODE<0> OpODE;
      for( unsigned k=0; k<NEFF; ++k, Cndx+=_nc ){
        IVPODE* pivpode = _ivpode;
        for( unsigned i=0; i<NOUT; ++i ){
          FFVar** vOUTODE = OpODE( _nc, Cndx, pivpode );
          BROUT.push_back( *(vOUTODE[i]) );
#ifdef MC__MBDOE_SOLVE_DEBUG
          std::cout << "  " << BROUT.back();
#endif
        }
#ifdef MC__MBDOE_SOLVE_DEBUG
        std::cout << std::endl;
#endif
      }
    }
  }
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_build_fim
( std::vector<FFVar>& vFIM, std::vector<FFVar>& CTOT, std::map<unsigned,double> const& EOpt,
  std::ostream& os )
{
  unsigned const NELE = _np*(_np+1)/2;
  vFIM.assign( NELE, 0. );
  FFVar* Cndx = CTOT.data();

  if( _ny ){
    std::vector<FFVar> vFIMref( NELE ), vCref( CTOT.size() );
    // Define atom matrices in current scenario
    _dagdoe->insert( _dag, _nc, _vCON.data(), vCref.data() );
    _dagdoe->insert( _dag, NELE, _vFIM.data(), vFIMref.data() );
    for( auto const& [ndx,eff] : EOpt ){
      mc::FFVar* pFIMndx = _dagdoe->compose( NELE, vFIMref.data(), _nc, vCref.data(), Cndx );
      for( unsigned int ij=0; ij<NELE; ++ij ){
        vFIM[ij] += eff * pFIMndx[ij];
      }
#ifdef MC__MBDOE_SOLVE_DEBUG
      _dagdoe->output( _dagdoe->subgraph( NELE, pFIMndx ), " A["+std::to_string(ndx)+"]" );
      //{ int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
      Cndx += _nc;
      delete[] pFIMndx;
    }
#ifdef MC__MBDOE_SOLVE_DEBUG
    _dagdoe->output( _dagdoe->subgraph( NELE, vFIM.data() ), " FIM" );
    //{ int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
  }

  else if( _ivpode ){
    mc::FFODE<0> OpODE;
    for( auto const& [ndx,eff] : EOpt ){
      IVPODE* pivpode = _ivpode;
      for( unsigned int ij=0; ij<NELE; ++ij ){
        vFIM[ij] += eff * OpODE( ij, _nc, Cndx, pivpode );
#ifdef MC__MBDOE_SOLVE_DEBUG
        _dagdoe->output( _dagdoe->subgraph( 1, &vFIM[ij] ), " FIM["+std::to_string(s)+"]["+std::to_string(ndx)+"]["+std::to_string(i)+"]" );
        //{ int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
      }
      Cndx += _nc;
    }
  }
}

template <typename... ExtOps>
inline
std::multimap<double,std::vector<double>>
MBDOESLV<ExtOps...>::campaign
()
const
{
  assert( _EOpt.size() == _SOpt.size() );
  std::multimap<double,std::vector<double>> C;
  auto iteff = _EOpt.cbegin();
  auto itsup = _SOpt.cbegin();
  for( ; iteff != _EOpt.cend(); ++iteff, ++itsup )
    C.insert( std::make_pair( iteff->second, itsup->second ) );
  return C;
}

template <typename... ExtOps>
inline
std::pair<double,bool>
MBDOESLV<ExtOps...>::evaluate_design
( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os )
{
  // Observation samples
  switch( options.CRITERION ){
    case DOEBase::BROPT:
      return _evaluate_design_br( Campaign, type, os );

    case DOEBase::AOPT:
    case DOEBase::DOPT:
    case DOEBase::EOPT:
    default:
      return _evaluate_design_fim( Campaign, type, os );
  }
}

template <typename... ExtOps>
inline
std::pair<double,bool>
MBDOESLV<ExtOps...>::_evaluate_design_br
( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAGDOE;
  mc::FFBRCrit<7> OpBRCrit;

  // Concatenate experimental controls
  unsigned const NCTOT = _nc * Campaign.size();
  std::vector<FFVar> CTOT(NCTOT), CREF(NCTOT);  // Experimental controls
  std::vector<double> CTOT0(NCTOT), CTOTLB(NCTOT), CTOTUB(NCTOT); 
  for( unsigned int i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::map<unsigned,double> EOpt;
  unsigned ieff = 0;
  double* C0 = CTOT0.data();
  for( auto const& [eff,supp] : Campaign ){
    EOpt[ieff++] = eff;
    for( unsigned int i=0; i<_nc; i++ )
      C0[i]  = supp[i];
    C0  += _nc;
  }

  // Define cost function
  std::vector<FFVar> BROUT;
  _build_br( BROUT, CTOT, EOpt, os );
  
  // Evaluate cost function
  unsigned const NUNC = _vPARVAL.size();
  unsigned const NOUT = _dOUT.size();
  DOEBase::set_weighting( _vPARWEI );
  FFVar& FBR = OpBRCrit( BROUT.size(), BROUT.data(), &EOpt, NUNC, NOUT );
  double DBR;
  std::string header( type.empty()? "DESIGN PERFORMANCE": type + " DESIGN PERFORMANCE" );
  try{
    _dagdoe->eval( 1, &FBR, &DBR, NCTOT, CTOT.data(), CTOT0.data() );
  }
  catch(...){
    if( options.DISPLEVEL )
      _display_design( header, DBR, std::multimap<double,std::vector<double>>(), os ); 
    return std::make_pair( 0./0., false ); // NaN
  }

  if( options.DISPLEVEL )
    _display_design( header, DBR, Campaign, os ); 
  return std::make_pair( DBR, true );
}

template <typename... ExtOps>
inline
std::pair<double,bool>
MBDOESLV<ExtOps...>::_evaluate_design_fim
( std::multimap<double,std::vector<double>> const& Campaign, std::string const& type, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAGDOE;
  DOEBase::type = options.CRITERION;
  mc::FFDOECrit<2> OpDOECrit;
  mc::FFSum<6> Sum;

  // Concatenate experimental controls
  unsigned const NCTOT = _nc * Campaign.size();
  std::vector<FFVar> CTOT(NCTOT), CREF(NCTOT);  // Experimental controls
  std::vector<double> CTOT0(NCTOT), CTOTLB(NCTOT), CTOTUB(NCTOT); 
  for( unsigned int i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  std::map<unsigned,double> EOpt;
  unsigned ieff = 0;
  double* C0 = CTOT0.data();
  for( auto const& [eff,supp] : Campaign ){
    EOpt[ieff++] = eff;
    for( unsigned int i=0; i<_nc; i++ )
      C0[i]  = supp[i];
    C0  += _nc;
  }

  // Define cost function
  unsigned const NUNC = _vPARVAL.size();
  std::vector<FFVar> DOECRIT( NUNC );
  std::vector<FFVar> vFIM;
  auto itPARVAL = _vPARVAL.cbegin();
  for( unsigned s=0; s<_vPARVAL.size(); ++s, ++itPARVAL ){
      // Set parameter scenario
    _set_parameters( itPARVAL->data(), os );
    // Define FIM and DOE criterion in current scenario
    _build_fim( vFIM, CTOT, EOpt, os );
    DOECRIT[s] = OpDOECrit( vFIM.size(), vFIM.data() );
#ifdef MC__MBDOE_SOLVE_DEBUG
    _dagdoe->output( _dagdoe->subgraph( vFIM.size(), vFIM.data() ), " FIM" );
    _dagdoe->output( _dagdoe->subgraph( 1, &DOECRIT[s] ), " J["+std::to_string(s)+"]" );
    //{ int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
  }

  // Evaluate cost function
  FFVar FFIM;
  double DFIM;
  std::string header = ( type.empty()? "DESIGN PERFORMANCE": type + " DESIGN PERFORMANCE" );
  try{
    switch( options.RISK){
      case Options::NEUTRAL:
        FFIM = Sum( NUNC, DOECRIT.data(), _vPARWEI.data() );
        _dagdoe->eval( 1, &FFIM, &DFIM, NCTOT, CTOT.data(), CTOT0.data() );
        break;

      case Options::AVERSE:
      {
        std::vector<double> DA( NUNC );
        _dagdoe->eval( NUNC, DOECRIT.data(), DA.data(), NCTOT, CTOT.data(), CTOT0.data() );
        std::map<double,double> SA;
        for( unsigned s=0; s<NUNC; ++s )
          SA[DA[s]] = _vPARWEI[s];
        double prsum = 0., VaR = 0.;
        for( auto const& [crit,pr] : SA ){
          VaR = crit;
          if( prsum + pr > options.CVARTHRES ) break;
          prsum += pr;
        }
        //std::cout << "VaR = " << VaR << std::endl;
        DFIM = VaR;
        for( auto const& [crit,pr] : SA ){
          if( crit > VaR ) break;
          DFIM -= ( VaR - crit ) * pr / options.CVARTHRES;
        }
        //std::cout << "CVaR = " << DFIM << std::endl;
        break;
      }
    }
  }
  catch(...){
    if( options.DISPLEVEL )
      _display_design( header, DFIM, std::multimap<double,std::vector<double>>(), os ); 
    return std::make_pair( 0./0., false ); // NaN
  }

  if( options.DISPLEVEL )
    _display_design( header, DFIM, Campaign, os ); 
  return std::make_pair( DFIM, true );
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::gradient_solve
( std::map<unsigned,double> const& EOpt, bool const update, std::ostream& os )
{
  // Observation samples
  switch( options.CRITERION ){
    case DOEBase::BROPT:
      return _gradient_minimize_br( EOpt, update, os );
      
    case DOEBase::AOPT:
    case DOEBase::DOPT:
    case DOEBase::EOPT:
    default:
      return _gradient_maximize_fim( EOpt, update, os );
  }
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_gradient_minimize_br
( std::map<unsigned,double> const& EOpt, bool const update, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAGDOE;
  mc::FFBRCrit<7> OpBRCrit;

  // Concatenate experimental controls
  unsigned const NCTOT = _nc * EOpt.size();
  std::vector<FFVar> CTOT(NCTOT), CREF(NCTOT);  // Experimental controls
  std::vector<double> CTOT0(NCTOT), CTOTLB(NCTOT), CTOTUB(NCTOT); 
  for( unsigned int i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  double* C0  = CTOT0.data();
  double* CLB = CTOTLB.data();
  double* CUB = CTOTUB.data();
  for( auto const& [ndx,eff] : EOpt ){
    for( unsigned int i=0; i<_nc; i++ ){
      C0[i]  = _vCONSAM[ndx][i];
      CLB[i] = _vCONLB[i];
      CUB[i] = _vCONUB[i];
    }
    C0  += _nc;
    CLB += _nc;
    CUB += _nc;
  }

  // Define cost function
  std::vector<FFVar> BROUT;
  _build_br( BROUT, CTOT, EOpt, os );
  unsigned const NUNC = _vPARVAL.size();
  unsigned const NOUT = _dOUT.size();
  FFVar& FBR = OpBRCrit( BROUT.size(), BROUT.data(), const_cast<std::map<unsigned,double>*>(&EOpt), NUNC, NOUT );

  // Local NLP optimization
  NLP doeref;
  DOEBase::set_weighting( _vPARWEI );
  DOEBase::set_scaling( _vPARSCA );
  DOEBase::type  = options.CRITERION;
  doeref.options = options.NLPSLV;
  doeref.set_dag( _dagdoe ); // DAG
  doeref.add_var( NCTOT, CTOT.data(), CTOTLB.data(), CTOTUB.data() ); // decision variables
//  doeref.set_obj( mc::BASE_OPT::MIN, Sum( NELE, BRCRIT.data(), WCRIT.data() ) ); // minimize Bayesian risk
  doeref.set_obj( mc::BASE_OPT::MIN, FBR ); // minimize Bayesian risk
  doeref.setup();
  doeref.solve( CTOT0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << doeref.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << doeref.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( update ){
    _SOpt.clear();
    _VOpt = 0./0.;//mc::BASE_OPT::BASE_OPT::INF;
 
    if( doeref.get_status() == NLP::SUCCESSFUL || doeref.get_status() == NLP::FAILURE ){
      unsigned isupp = 0;
      for( auto const& [ndx,eff] : EOpt ){
        double const* dC = doeref.solution().x.data() + isupp*_nc;
        _SOpt[ndx] = std::vector<double>( dC, dC+_nc );
        ++isupp;
      }
      _update_supports( _EOpt, _SOpt, os );
      _VOpt = doeref.solution().f[0];
    }
  }

  if( options.DISPLEVEL )
    _display_design( "GRADIENT-BASED REFINED DESIGN", _VOpt, EOpt, _SOpt, os ); 
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_gradient_maximize_fim
( std::map<unsigned,double> const& EOpt, bool const update, std::ostream& os )
{
  delete _dagdoe; _dagdoe = new DAGDOE;
  mc::FFDOECrit<2> OpDOECrit;
  mc::FFSum<6> Sum;

  // Concatenate experimental controls
  unsigned const NCTOT = _nc * EOpt.size();
  std::vector<FFVar> CTOT(NCTOT), CREF(NCTOT);  // Experimental controls
  std::vector<double> CTOT0(NCTOT), CTOTLB(NCTOT), CTOTUB(NCTOT); 
  for( unsigned int i=0; i<NCTOT; i++ )
    CTOT[i].set( _dagdoe );

  double* C0  = CTOT0.data();
  double* CLB = CTOTLB.data();
  double* CUB = CTOTUB.data();
  for( auto const& [ndx,eff] : EOpt ){
    for( unsigned int i=0; i<_nc; i++ ){
      C0[i]  = _vCONSAM[ndx][i];
      CLB[i] = _vCONLB[i];
      CUB[i] = _vCONUB[i];
    }
    C0  += _nc;
    CLB += _nc;
    CUB += _nc;
  }

  // Define cost function
  unsigned const NUNC = _vPARVAL.size();
  std::vector<FFVar> DOECRIT( NUNC );
  std::vector<FFVar> vFIM;
  auto itPARVAL = _vPARVAL.cbegin();
  for( unsigned s=0; s<_vPARVAL.size(); ++s, ++itPARVAL ){
    // Set parameter scenario
    _set_parameters( itPARVAL->data(), os );
    // Define FIM and DOE criterion in current scenario
    _build_fim( vFIM, CTOT, EOpt, os );
    DOECRIT[s] = OpDOECrit( vFIM.size(), vFIM.data() );
#ifdef MC__MBDOE_SOLVE_DEBUG
    _dagdoe->output( _dagdoe->subgraph( 1, &DOECRIT[s] ), " J["+std::to_string(s)+"]" );
    //{ int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
  }

  // Local NLP optimization
  NLP doeref;
  DOEBase::type  = options.CRITERION;
  DOEBase::set_scaling( _vPARSCA );
  doeref.options = options.NLPSLV;
  doeref.set_dag( _dagdoe ); // DAG
  doeref.add_var( NCTOT, CTOT.data(), CTOTLB.data(), CTOTUB.data() ); // decision variables
 
  switch( options.RISK){
    case Options::NEUTRAL:
    {
      doeref.set_obj( mc::BASE_OPT::MAX, Sum( NUNC, DOECRIT.data(), _vPARWEI.data() ) ); // objective
      //doeref.set_obj( mc::BASE_OPT::MAX, Sum( NUNC, DOECRIT.data() ) / NUNC ); // objective
      break;
    }
    case Options::AVERSE:
    {
      std::vector<FFVar> DELTA( NUNC );
      FFVar VaR( _dagdoe );
      for( auto& Dk : DELTA )
        Dk.set( _dagdoe );
      if( _ROpt.size() == NUNC+1 )
        for( auto const& r0 : _ROpt ) CTOT0.push_back( r0 ); 
      else
        CTOT0.resize( NCTOT+NUNC+1, 0e0 );
      doeref.add_var( NUNC, DELTA.data(), 0e0 );//, 1e2 );
      doeref.add_var( VaR );//, -1e2, 1e2 );
      doeref.set_obj( mc::BASE_OPT::MAX, VaR - Sum( NUNC, DELTA.data(), _vPARWEI.data() ) / options.CVARTHRES );
      //doeref.set_obj( mc::BASE_OPT::MAX, VaR - Sum( NUNC, DELTA.data() ) / (options.CVARTHRES * NUNC) );
      for( unsigned s=0; s<NUNC; s++ )
        doeref.add_ctr( mc::BASE_OPT::LE, VaR - DELTA[s] - DOECRIT[s] );
      break;
    }
  }

  doeref.setup();
  doeref.solve( CTOT0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << doeref.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << doeref.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( update ){
    _SOpt.clear();
    _VOpt = 0./0.;//mc::BASE_OPT::BASE_OPT::INF;
 
    if( doeref.get_status() == NLP::SUCCESSFUL || doeref.get_status() == NLP::FAILURE ){
      double const* dC = doeref.solution().x.data(); //unsigned isupp = 0;
      for( auto const& [ndx,eff] : EOpt ){
        _SOpt[ndx] = std::vector<double>( dC, dC+_nc );
        dC += _nc; //double const* dC = doeref.solution().x.data() + isupp*_nc;
        //++isupp;
      }
      _update_supports( _EOpt, _SOpt, os );
      _VOpt = doeref.solution().f[0];
      if( doeref.solution().x.size() == NCTOT+NUNC+1 )
        _ROpt.assign( dC, dC+NUNC+1 );
    }
  }

  if( options.DISPLEVEL )
    _display_design( "GRADIENT-BASED REFINED DESIGN", _VOpt, EOpt, _SOpt, os ); 
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_display_design
( std::string const& title, double const& crit, std::map<unsigned,double> const& eff,
  std::map<unsigned,std::vector<double>> const& supp, std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( eff.empty() ){
     os << " FAILED" << std::endl;
     return;
  } 
   
  os  << std::scientific << std::setprecision(5) << crit << std::endl;
  for( auto const& [i,s] : supp ){
    os << "   SUPPORT #" << i << ": " << std::fixed << std::setprecision(0) << eff.at(i) << " x [ "
       << std::scientific << std::setprecision(5);
      for( auto Ck : s )
        os << Ck << " ";
      os << "]" << std::endl;
  }
  os << std::endl;
}

template <typename... ExtOps>
inline
void
MBDOESLV<ExtOps...>::_display_design
( std::string const& title, double const& crit, std::multimap<double,std::vector<double>> const& campaign,
  std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( campaign.empty() ){
     os << " FAILED" << std::endl;
     return;
  } 
   
  os  << std::scientific << std::setprecision(5) << crit << std::endl;
  unsigned i=0;
  for( auto const& [eff,supp] : campaign ){
    os << "   SUPPORT #" << i++ << ": " << std::fixed << std::setprecision(0) << eff << " x [ "
       << std::scientific << std::setprecision(5);
      for( auto Ck : supp )
        os << Ck << " ";
      os << "]" << std::endl;
  }
  os << std::endl;
}

} // end namespace mc

#endif

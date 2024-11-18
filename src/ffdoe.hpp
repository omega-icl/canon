// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef CANON__FFDOE_HPP
#define CANON__FFDOE_HPP

#include <fstream>
#include <iomanip>
#include <armadillo>

#include "base_mbdoe.hpp"

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
  // Selected DOE criterion
  static BASE_MBDOE::TYPE type;

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

  // Set input scaling
  static void set_scaling
    ( arma::mat const& mscaling )
    {
      scaling = mscaling;
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
  static std::set<std::pair<size_t,size_t>>* parsubset;

  // Compute atomic Bayes Risk
  static double atom_BR
    ( std::vector< arma::vec > const& yj, std::vector< arma::vec > const& yk,
      std::vector<double> const& eff )
    {
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      for( size_t i=0; i<eff.size(); ++i ){
        auto&& Ejk  = yj.at(i) - yk.at(i);
        if( !sigmayinv.empty() ) Et_Vinv_E += eff[i] * (Ejk.t() * sigmayinv * Ejk);
        else                     Et_Vinv_E += eff[i] * (Ejk.t() * Ejk);
      }
      return std::exp( -0.125 * Et_Vinv_E(0,0) );
    }
};

inline BASE_MBDOE::TYPE FFDOEBase::type = BASE_MBDOE::DOPT;
inline arma::mat FFDOEBase::scaling;
inline arma::vec FFDOEBase::weighting;
inline arma::mat FFDOEBase::sigmayinv;
inline std::set<std::pair<size_t,size_t>>* FFDOEBase::parsubset = nullptr;

////////////////////////////////////////////////////////////////////////

class FFFIMEff
: public FFOp,
  public FFDOEBase
{
private:

  // FIM marices
  std::vector<std::vector<arma::mat>> const* _vFIM;

  // Prior experimental efforts
  std::vector<double> const* _vEFFAP;
    
public:

  void set
    ( std::vector<std::vector<arma::mat>> const* vFIM, std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFFIMEFF_CHECK
      assert( vFIM && vEFFAP );
#endif
      _vFIM   = vFIM;
      _vEFFAP = vEFFAP;
    }

  // Default constructor
  FFFIMEff
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFFIMEff
    ( FFFIMEff const& Op )
    : FFOp( Op ),
      _vFIM( Op._vFIM ),
      _vEFFAP( Op._vEFFAP )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, std::vector<FFVar> const& vVar, std::vector< std::vector< arma::mat > > const* vFIM,
      std::vector<double> const* vEFFAP )
    {
      size_t const nRes = vFIM->size();
#ifdef MC__FFDOEEFF_CHECK
      assert( idep < nRes && vFIM && vEFFAP );
#endif
      set( vFIM, vEFFAP );
      return *(insert_external_operation( *this, nRes, vVar.size(), vVar.data() )[idep]);
    }

  FFVar** operator()
    ( std::vector<FFVar> const& vVar, std::vector< std::vector< arma::mat > > const* vFIM,
      std::vector<double> const* vEFFAP )
    {
      size_t const nRes = vFIM->size();
#ifdef MC__FFDOEEFF_CHECK
      assert( vFIM && vEFFAP );
#endif
      set( vFIM, vEFFAP );
      return insert_external_operation( *this, nRes, vVar.size(), vVar.data() );
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

      throw std::runtime_error( "FFFIMEff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFDOEEFF_TRACE
      std::cout << "FFFIMEff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
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
        case BASE_MBDOE::AOPT: return "-tr Inv";
        case BASE_MBDOE::DOPT: return "log Det";
        case BASE_MBDOE::EOPT: return "min Eig";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradFIMEff
: public FFOp,
  public FFDOEBase
{
private:

  // FIM marices
  std::vector<std::vector<arma::mat>> const* _vFIM;

  // Prior experimental efforts
  std::vector<double> const* _vEFFAP;
    
public:

  void set
    ( std::vector<std::vector<arma::mat>> const* vFIM, std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFFIMEFF_CHECK
      assert( vFIM && vEFFAP );
#endif
      _vFIM   = vFIM;
      _vEFFAP = vEFFAP;
    }

  // Default constructor
  FFGradFIMEff
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFGradFIMEff
    ( FFGradFIMEff const& Op )
    : FFOp( Op ),
      _vFIM( Op._vFIM ),
      _vEFFAP( Op._vEFFAP )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, std::vector<FFVar> const& vVar, std::vector< std::vector< arma::mat > > const* vFIM,
      std::vector<double> const* vEFFAP )
    {
      size_t const nRes = vFIM->size();
#ifdef MC__FFGRADDOEEFF_CHECK
      assert( idep < nRes && vFIM && vEFFAP );
#endif
      set( vFIM, vEFFAP );
      return *(insert_external_operation( *this, nRes*vVar.size(), vVar.size(), vVar.data() )[idep]);
    }

  FFVar** operator()
    ( std::vector<FFVar> const& vVar, std::vector< std::vector< arma::mat > > const* vFIM,
      std::vector<double> const* vEFFAP )
    {
      size_t const nRes = vFIM->size();
#ifdef MC__FFGRADDOEEFF_CHECK
      assert( vFIM && vEFFAP );
#endif
      set( vFIM, vEFFAP );
      return insert_external_operation( *this, nRes*vVar.size(), vVar.size(), vVar.data() );
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

      throw std::runtime_error( "FFGradFIMEff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADDOEEFF_TRACE
      std::cout << "FFGradFIMEff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( size_t i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( size_t j=1; j<nRes; ++j )
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
        case BASE_MBDOE::AOPT: return "-Grad tr Inv";
        case BASE_MBDOE::DOPT: return "Grad log Det";
        case BASE_MBDOE::EOPT: return "Grad min Eig";
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
FFFIMEff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFFIMEff::eval: FFVar\n";
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( size_t j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFGradFIMEff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADDOEEFF_TRACE
  std::cout << "FFGradFIMEff::eval: FFVar\n";
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( size_t j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFFIMEff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFFIMEff::eval: double\n";
#endif
#ifdef MC__FFDOEEFF_CHECK
  assert( _vFIM && !_vFIM->empty() && nRes == _vFIM->size() && nVar == _vFIM->back().size()-_vEFFAP->size() );
#endif

  arma::mat FIM;
  for( size_t s=0; s<nRes; ++s ){
    size_t e=0;
    for( auto const& eff : *_vEFFAP ) // Atomic FIM of prior experiment
      if( !e ) FIM  = eff * _vFIM->at(s).at(e++);
      else     FIM += eff * _vFIM->at(s).at(e++);
    for( size_t i=0; i<nVar; ++i ) // Atomic FIM of new experiment
      if( !e ) FIM  = vVar[0] * _vFIM->at(s).at(e++);
      else     FIM += vVar[i] * _vFIM->at(s).at(e++);
    if( scaling.n_elem ) FIM = arma::trans(scaling) * FIM * scaling;
#ifdef MC__FFDOEEFF_DEBUG
    std::cout << "FIM: " << FIM;
    std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

    switch( FFDOEBase::type ){
      case BASE_MBDOE::AOPT:
      {
        arma::vec FIMEIGVAL;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        vRes[s] = 0.;
        for( size_t k=0; k<FIM.n_rows; ++k )
          vRes[s] -= 1./FIMEIGVAL(k);
        break;
      }
      
      case BASE_MBDOE::DOPT:
      {
#if defined( MC__FFDCRIT_EIG )
        arma::vec FIMEIGVAL;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
        vRes[s] = 0.;
        for( size_t k=0; k<FIM.n_rows; ++k )
          vRes[s] += std::log( FIMEIGVAL(k) );
#else
        if( arma::rank( FIM ) < FIM.n_rows || !arma::log_det_sympd( vRes[s], FIM ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#endif
        break;
      }
      
      case BASE_MBDOE::EOPT:
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
  std::cout << name() << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradFIMEff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADDOEEFF_TRACE
  std::cout << "FFGradFIMEff::eval: double\n";
#endif
#ifdef MC__FFGRADDOEEFF_CHECK
  assert( _vFIM && !_vFIM->empty() && nRes == _vFIM->size()*nVar && nVar == _vFIM->back().size()-_vEFFAP->size() );
#endif

  size_t const nUnc = _vFIM->size();
  size_t const nAP = _vEFFAP->size();
  arma::mat FIM, FIMi;
  for( size_t s=0; s<nUnc; ++s ){
    size_t e=0;
    for( auto const& eff : *_vEFFAP ) // Atomic FIM of prior experiment
      if( !e ) FIM  = eff * _vFIM->at(s).at(e++);
      else     FIM += eff * _vFIM->at(s).at(e++);
    for( size_t i=0; i<nVar; ++i ) // Atomic FIM of new experiment
      if( !e ) FIM  = vVar[0] * _vFIM->at(s).at(e++);
      else     FIM += vVar[i] * _vFIM->at(s).at(e++);
    if( scaling.n_elem ) FIM = arma::trans(scaling) * FIM * scaling;
#ifdef MC__FFGRADDOEEFF_DEBUG
    std::cout << "FIM: " << FIM;
    std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

    switch( FFDOEBase::type ){
      case BASE_MBDOE::AOPT:
      {
        arma::vec FIMEIGVAL;
        arma::mat FIMEIGVEC;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
        std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
        std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
        for( size_t i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = arma::trans(scaling) * _vFIM->at(s).at(nAP+i) * scaling;
          else                 FIMi = _vFIM->at(s).at(nAP+i);
          vRes[s*nVar+i] = 0.;
          for( size_t k=0; k<FIM.n_rows; ++k ){
            arma::mat const& Et_FIM_E = FIMEIGVEC.col(k).t() * FIMi * FIMEIGVEC.col(k); 
            vRes[s*nVar+i] += Et_FIM_E(0,0) / (FIMEIGVAL(k)*FIMEIGVAL(k));
          }
#ifdef MC__FFGRADDOEEFF_DEBUG
          std::cout << name() << " [" << s << "," << i << "]: " << vRes[s*nVar+i] << std::endl;
#endif
        }
        break;
      }
      
      case BASE_MBDOE::DOPT:
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
        for( size_t i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = arma::trans(scaling) * _vFIM->at(s).at(nAP+i) * scaling;
          else                 FIMi = _vFIM->at(s).at(nAP+i);
          vRes[s*nVar+i] = 0.;
          for( size_t k=0; k<FIM.n_rows; ++k ){
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
        for( size_t i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = arma::trans(scaling) * _vFIM->at(s).at(nAP+i) * scaling;
          else                 FIMi = _vFIM->at(s).at(nAP+i);
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
      
      case BASE_MBDOE::EOPT:
      {
        arma::vec FIMEIGVAL;
        arma::mat FIMEIGVEC;
        if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
          throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOEEFF_DEBUG
        std::cout << "FIM min eigenvalue: "  << FIMEIGVAL(0) << std::endl;
        std::cout << "FIM min eigenvector: " << FIMEIGVEC.col(0);
#endif
        for( size_t i=0; i<nVar; ++i ){
          if( scaling.n_elem ) FIMi = arma::trans(scaling) * _vFIM->at(s).at(nAP+i) * scaling;
          else                 FIMi = _vFIM->at(s).at(nAP+i);
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
  std::cout << name() << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFFIMEff::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFFIMEff::eval: fadbad::F<FFVar>\n";
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar const*const* ppResVal = insert_external_operation( *this, nRes, nVar, vVarVal.data() );

  FFGradFIMEff OpResDer;
  OpResDer.set( _vFIM, _vEFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nRes*nVar, nVar, vVarVal.data() );
  for( size_t s=0; s<nRes; ++s ){
    vRes[s] = *ppResVal[s];
    for( size_t i=0; i<nVar; ++i )
      vRes[s].setDepend( vVar[i] );
    for( size_t j=0; j<vRes[s].size(); ++j ){
      vRes[s][j] = 0.;
      for( size_t i=0; i<nVar; ++i ){
        if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
        vRes[s][j] += *ppResDer[s*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFFIMEff::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFFIMEff::eval: fadbad::F<double>\n";
#endif

  std::vector<double> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  std::vector<double> vResVal( nRes ); 
  eval( nRes, vResVal.data(), nVar, vVarVal.data(), nullptr );
  for( size_t s=0; s<nRes; ++s ){
    vRes[s] = vResVal[s];
    for( size_t i=0; i<nVar; ++i )
      vRes[s].setDepend( vVar[i] );
  }

  FFGradFIMEff OpResDer;
  OpResDer.set( _vFIM, _vEFFAP );
  std::vector<double> vResDer( nRes*nVar ); 
  OpResDer.eval( nRes*nVar, vResDer.data(), nVar, vVarVal.data(), nullptr );
  for( size_t s=0; s<nRes; ++s ){
    for( size_t j=0; j<vRes[s].size(); ++j ){
      vRes[s][j] = 0.;
      for( size_t i=0; i<nVar; ++i ){
        if( vVar[i][j] == 0. ) continue;
        vRes[s][j] += vResDer[s*nVar+i] * vVar[i][j];
      }
    }
  }
}

inline void
FFFIMEff::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFDOEEFF_TRACE
  std::cout << "FFFIMEff::deriv\n";
#endif

  FFGradFIMEff OpResDer;
  OpResDer.set( _vFIM, _vEFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nRes*nVar, nVar, vVar ); 
  //std::cout << "FFFIMEff::deriv: " << ppResDer[0]->opdef().first << std::endl;
  for( size_t s=0; s<nRes; ++s )
    for( size_t i=0; i<nVar; ++i )
      vDer[s][i] = *ppResDer[s*nVar+i];
}

////////////////////////////////////////////////////////////////////////

class FFBREff
: public FFOp,
  public FFDOEBase
{
private:

  // FIM marices
  std::vector<std::vector<arma::vec>> const* _vOUT;

  // Prior experimental efforts
  std::vector<double> const* _vEFFAP;
    
public:

  void set
    ( std::vector<std::vector<arma::vec>> const* vOUT, std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFBREFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      _vOUT   = vOUT;
      _vEFFAP = vEFFAP;
    }

  // Default constructor
  FFBREff
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFBREff
    ( FFBREff const& Op )
    : FFOp( Op ),
      _vOUT( Op._vOUT ),
      _vEFFAP( Op._vEFFAP )
    {}

  // Define operation
  FFVar& operator()
    ( std::vector<FFVar> const& vVar, std::vector<std::vector<arma::vec>> const* vOUT,
      std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFBREFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      set( vOUT, vEFFAP );
      return **insert_external_operation( *this, 1, vVar.size(), vVar.data() );
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
      for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
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
        case BASE_MBDOE::BROPT: return "Bayes Risk";
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
private:

  // FIM marices
  std::vector<std::vector<arma::vec>> const* _vOUT;

  // Prior experimental efforts
  std::vector<double> const* _vEFFAP;
    
public:

  void set
    ( std::vector<std::vector<arma::vec>> const* vOUT, std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFGRADBREFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      _vOUT   = vOUT;
      _vEFFAP = vEFFAP;
    }

  // Default constructor
  FFGradBREff
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFGradBREff
    ( FFGradBREff const& Op )
    : FFOp( Op ),
      _vOUT( Op._vOUT ),
      _vEFFAP( Op._vEFFAP )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, std::vector<FFVar> const& vVar, std::vector<std::vector<arma::vec>> const* vOUT,
      std::vector<double> const* vEFFAP )
    {
      size_t const nVar = vVar.size();
#ifdef MC__FFGRADBREFF_CHECK
      assert( idep < nVar && vOUT && vEFFAP );
#endif
      set( vOUT, vEFFAP );
      return *(insert_external_operation( *this, nVar, nVar, vVar.data() )[idep]);
    }

  FFVar** operator()
    ( std::vector<FFVar> const& vVar, std::vector<std::vector<arma::vec>> const* vOUT,
      std::vector<double> const* vEFFAP )
    {
      size_t const nVar = vVar.size();
#ifdef MC__FFGRADBREFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      set( vOUT, vEFFAP );
      return insert_external_operation( *this, nVar, nVar, vVar.data() );
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
      for( size_t i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( size_t j=1; j<nRes; ++j )
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
        case BASE_MBDOE::BROPT: return "Grad Bayes Risk";
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
#ifdef MC__FFBREFF_CHECK
  assert( nRes == 1 );
#endif

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
#ifdef MC__FFGRADBREFF_CHECK
  assert( nRes == nVar );
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );;
  for( size_t j=0; j<nRes; ++j )
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
#ifdef MC__FFBREFF_CHECK
  assert( _vOUT && !_vOUT->empty() && nRes == 1 && nVar == _vOUT->back().size()-_vEFFAP->size() );
#endif

  auto BRval = [&]( size_t j, size_t k, double& res ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      size_t e=0;
      for( auto const& eff : *_vEFFAP ){ // Contributions from prior experiment
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "y[" << e << "][" << j << "] = " << _vOUT->at(j).at(e);
        std::cout << "y[" << e << "][" << k << "] = " << _vOUT->at(k).at(e);
#endif
        arma::vec const& Ejk  = _vOUT->at(j).at(e) - _vOUT->at(k).at(e);
        if( !sigmayinv.empty() ) Et_Vinv_E += eff * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += eff * Ejk.t() * Ejk;
        e++;
      }

      for( size_t i=0; i<nVar; ++i, ++e ){ // Contributions from new experiment
        if( vVar[i] == 0. ) continue;
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "y[" << e << "][" << j << "] = " << _vOUT->at(j).at(e);
        std::cout << "y[" << e << "][" << k << "] = " << _vOUT->at(k).at(e);
#endif
        arma::vec const& Ejk  = _vOUT->at(j).at(e) - _vOUT->at(k).at(e);
        if( !sigmayinv.empty() ) Et_Vinv_E += vVar[i] * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += vVar[i] * Ejk.t() * Ejk;
      }

      if( !weighting.empty() ) res += std::sqrt( weighting(j)*weighting(k) ) * std::exp( -0.125 * Et_Vinv_E(0,0) );
      else                     res += std::exp( -0.125 * Et_Vinv_E(0,0) );
#ifdef MC__FFBRCRIT_DEBUG
      std::cout << "weighting: " << weighting(j) << "  " << weighting(k) << std::endl;
      std::cout << "(" << j << "," << k << "): " << res << "  " << Et_Vinv_E(0,0) << std::endl;
#endif
  };

  vRes[0] = 0.;

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
      BRval( j, k, vRes[0] );
 
  // Use full set of uncertainty scenarios
  else
    for( size_t j=0; j<_vOUT->size()-1; ++j )
      for( size_t k=j+1; k<_vOUT->size(); ++k )
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
#ifdef MC__FFGRADBREFF_CHECK
  assert( _vOUT && !_vOUT->empty() && nRes == nVar && nVar == _vOUT->back().size()-_vEFFAP->size() );
#endif

#ifdef MC__FFBRCRIT_LOG
  auto BRder = [&]( size_t j, size_t k, double& crit, arma::vec& grad, arma::vec& tmp ){
#else
  auto BRder = [&]( size_t j, size_t k, arma::vec& grad, arma::vec& tmp ){
#endif
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);
      size_t e=0;
      for( auto const& eff : *_vEFFAP ){ // Contributions from prior experiment
        arma::vec const& Ejk  = _vOUT->at(j).at(e) - _vOUT->at(k).at(e);
        if( !sigmayinv.empty() ) Et_Vinv_E += eff * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += eff * Ejk.t() * Ejk;
        e++;
      }
      for( size_t i=0; i<nVar; ++i, ++e ){ // Contributions from new experiment
        arma::vec const& Ejk  = _vOUT->at(j).at(e) - _vOUT->at(k).at(e);
        if( !sigmayinv.empty() ){
          tmp.subvec(i,i) = Ejk.t() * sigmayinv * Ejk;
          Et_Vinv_E += vVar[i] * tmp(i);
        }
        else{
          tmp.subvec(i,i) = Ejk.t() * Ejk;
          Et_Vinv_E += vVar[i] * tmp(i);
        }
      }
/*
      for( size_t i=0; i<nVar; ++i ){
        arma::vec const& Ejk   = _vOUT->at(j).at(i) - _vOUT->at(k).at(i);
        if( !sigmayinv.empty() ){
          tmp.subvec(i,i) = -0.125 * Ejk.t() * sigmayinv * Ejk;
          Et_Vinv_E += vVar[i] * tmp(i);
        }
        else{
          tmp.subvec(i,i) = -0.125 * Ejk.t() * Ejk;
          Et_Vinv_E += vVar[i] * tmp(i);
        }
      }
*/
      double BRjk = std::exp( -0.125 * Et_Vinv_E(0,0) );
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
#ifdef MC__FFBRCRIT_LOG
      crit += BRjk;
#endif
      grad += (BRjk * -0.125) * tmp;
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
    for( size_t j=0; j<_vOUT->size()-1; ++j )
      for( size_t k=j+1; k<_vOUT->size(); ++k )
#ifdef MC__FFBRCRIT_LOG
        BRder( j, k, BRCrit, GradBR, GradBRjk );
#else
        BRder( j, k, GradBR, GradBRjk );
#endif

#ifdef MC__FFBRCRIT_LOG
  GradBR /= BRCrit;
#endif

#ifdef MC__FFBREFF_DEBUG
  for( size_t i=0; i<nVar; ++i )
    std::cout << name() << " [" << i << "]: " << vRes[i] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFBREff::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::eval: fadbad::F<FFVar>\n";
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  vRes[0] = *(insert_external_operation( *this, 1, nVar, vVarVal.data() )[0]);
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradBREff OpResDer;
  OpResDer.set( _vOUT, _vEFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVarVal.data() );
  for( size_t j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( size_t i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *ppResDer[i] * vVar[i][j];
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
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double ResVal; 
  eval( 1, &ResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = ResVal;
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradBREff OpResDer;
  OpResDer.set( _vOUT, _vEFFAP );
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
FFBREff::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFBREFF_TRACE
  std::cout << "FFBREff::deriv\n";
#endif

  FFGradBREff OpResDer;
  OpResDer.set( _vOUT, _vEFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVar ); 
  for( size_t i=0; i<nVar; ++i )
    vDer[0][i] = *ppResDer[i];
}

////////////////////////////////////////////////////////////////////////

class FFODISTEff
: public FFOp,
  public FFDOEBase
{
private:

  // Reference support
  size_t _ndxSUPP;

  // Output vectors
  std::vector<std::vector<arma::vec>> const* _vOUT;

  // Prior experimental efforts
  std::vector<double> const* _vEFFAP;
    
public:

  void set
    ( size_t const ndxSUPP, std::vector<std::vector<arma::vec>> const* vOUT, std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFODISTEFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      _ndxSUPP = ndxSUPP;
      _vOUT    = vOUT;
      _vEFFAP  = vEFFAP;
    }

  // Default constructor
  FFODISTEff
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFODISTEff
    ( FFODISTEff const& Op )
    : FFOp( Op ),
      _ndxSUPP( Op._ndxSUPP ),
      _vOUT( Op._vOUT ),
      _vEFFAP( Op._vEFFAP )
    {}

  // Define operation
  FFVar& operator()
    ( std::vector<FFVar> const& vVar, size_t const ndxSUPP,
      std::vector<std::vector<arma::vec>> const* vOUT, std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFODISTEFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      set( ndxSUPP, vOUT, vEFFAP );
      return **insert_external_operation( *this, 1, vVar.size(), vVar.data() );
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

      throw std::runtime_error( "FFODISTEff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFODISTEFF_TRACE
      std::cout << "FFODISTEff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
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
        case BASE_MBDOE::BROPT: return "Operable Span";
        default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradODISTEff
: public FFOp,
  public FFDOEBase
{
private:

  // Reference support
  size_t _ndxSUPP;

  // FIM marices
  std::vector<std::vector<arma::vec>> const* _vOUT;

  // Prior experimental efforts
  std::vector<double> const* _vEFFAP;
    
public:

  void set
    ( size_t const ndxSUPP, std::vector<std::vector<arma::vec>> const* vOUT, std::vector<double> const* vEFFAP )
    {
#ifdef MC__FFGRADODISTEFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      _ndxSUPP = ndxSUPP;
      _vOUT    = vOUT;
      _vEFFAP  = vEFFAP;
    }

  // Default constructor
  FFGradODISTEff
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFGradODISTEff
    ( FFGradODISTEff const& Op )
    : FFOp( Op ),
      _ndxSUPP( Op._ndxSUPP ),
      _vOUT( Op._vOUT ),
      _vEFFAP( Op._vEFFAP )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, std::vector<FFVar> const& vVar, size_t const ndxSUPP,
      std::vector<std::vector<arma::vec>> const* vOUT, std::vector<double> const* vEFFAP )
    {
      size_t const nVar = vVar.size();
#ifdef MC__FFGRADODISTEFF_CHECK
      assert( idep < nVar && vOUT && vEFFAP );
#endif
      set( ndxSUPP, vOUT, vEFFAP );
      return *(insert_external_operation( *this, nVar, nVar, vVar.data() )[idep]);
    }

  FFVar** operator()
    ( std::vector<FFVar> const& vVar, size_t const ndxSUPP,
      std::vector<std::vector<arma::vec>> const* vOUT, std::vector<double> const* vEFFAP )
    {
      size_t const nVar = vVar.size();
#ifdef MC__FFGRADODISTEFF_CHECK
      assert( vOUT && vEFFAP );
#endif
      set( ndxSUPP, vOUT, vEFFAP );
      return insert_external_operation( *this, nVar, nVar, vVar.data() );
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

      throw std::runtime_error( "FFGradODISTEff::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const
    {
#ifdef MC__FFGRADODISTEFF_TRACE
      std::cout << "FFGradODISTEff::eval: FFDep\n";
#endif
      vRes[0] = 0;
      for( size_t i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( size_t j=1; j<nRes; ++j )
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
        case BASE_MBDOE::BROPT: return "Grad Operable Span";
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
FFODISTEff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFODISTEFF_TRACE
  std::cout << "FFODISTEff::eval: FFVar\n";
#endif
#ifdef MC__FFODISTEFF_CHECK
  assert( nRes == 1 );
#endif

  vRes[0] = **insert_external_operation( *this, 1, nVar, vVar );
}

inline void
FFGradODISTEff::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRAODISTREFF_TRACE
  std::cout << "FFGradODISTEff::eval: FFVar\n";
#endif
#ifdef MC__FFGRADODISTEFF_CHECK
  assert( nRes == nVar );
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );;
  for( size_t j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFODISTEff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFODISTEFF_TRACE
  std::cout << "FFODISTEff::eval: double\n";
#endif
#ifdef MC__FFODISTEFF_CHECK
  assert( _vOUT && !_vOUT->empty() && nRes == 1 && nVar == _vOUT->back().size()-_vEFFAP->size() );
#endif

  auto OperDist = [&]( size_t e1, size_t e2 ) -> double {
      double Dist = 0.;
      for( size_t j=0; j<_vOUT->size(); ++j ){
#ifdef MC__FFODISTEFF_DEBUG
        std::cout << "y[" << e1 << "][" << j << "] = " << _vOUT->at(j).at(e1);
        std::cout << "y[" << e2 << "][" << j << "] = " << _vOUT->at(j).at(e2);
#endif
        arma::vec const& dOUT12  = _vOUT->at(j).at(e1) - _vOUT->at(j).at(e2);
        arma::mat norm12(1,1,arma::fill::none);
        if( !sigmayinv.empty() ) norm12 = dOUT12.t() * sigmayinv * dOUT12;
        else                     norm12 = dOUT12.t() * dOUT12;
        if( !weighting.empty() ) Dist += weighting(j) * norm12(0,0);
        else                     Dist += norm12(0,0);
      }
      return Dist;
  };

  vRes[0] = 0.;
  size_t e=0;
  for( ; e<_vEFFAP->size(); ++e ) // Contributions from prior experiment
    vRes[0] += OperDist( _ndxSUPP, e );
  for( size_t i=0; i<nVar; ++i, ++e ){ // Contributions from new experiment
    if( e == _ndxSUPP ) continue;
    vRes[0] += vVar[i] * OperDist( _ndxSUPP, e );
  }

#ifdef MC__FFODISTEFF_DEBUG
  std::cout << name() << " [" << 0 << "]: " << vRes[0] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradODISTEff::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
const
{
#ifdef MC__FFGRADODISTEFF_TRACE
  std::cout << "FFGradODISTEff::eval: double\n";
#endif
#ifdef MC__FFGRADODISTEFF_CHECK
  assert( _vOUT && !_vOUT->empty() && nRes == nVar && nVar == _vOUT->back().size()-_vEFFAP->size() );
#endif

  auto OperDist = [&]( size_t e1, size_t e2 ) -> double {
      double Dist = 0.;
      for( size_t j=0; j<_vOUT->size(); ++j ){
#ifdef MC__FFGRADODISTEFF_DEBUG
        std::cout << "y[" << e1 << "][" << j << "] = " << _vOUT->at(j).at(e1);
        std::cout << "y[" << e2 << "][" << j << "] = " << _vOUT->at(j).at(e2);
#endif
        arma::vec const& dOUT12  = _vOUT->at(j).at(e1) - _vOUT->at(j).at(e2);
        arma::mat norm12(1,1,arma::fill::none);
        if( !sigmayinv.empty() ) norm12 = dOUT12.t() * sigmayinv * dOUT12;
        else                     norm12 = dOUT12.t() * dOUT12;
        if( !weighting.empty() ) Dist += weighting(j) * norm12(0,0);
        else                     Dist += norm12(0,0);
      }
      return Dist;
  };

  size_t e=_vEFFAP->size();
  for( size_t i=0; i<nVar; ++i, ++e ) // Contributions from new experiment
    vRes[i] = ( e==_ndxSUPP? 0.: OperDist( _ndxSUPP, e ) );

#ifdef MC__FFGRADODISTEFF_DEBUG
  for( size_t i=0; i<nVar; ++i )
    std::cout << name() << " [" << i << "]: " << vRes[i] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFODISTEff::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFODISTEFF_TRACE
  std::cout << "FFODISTEff::eval: fadbad::F<FFVar>\n";
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  vRes[0] = *(insert_external_operation( *this, 1, nVar, vVarVal.data() )[0]);
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradODISTEff OpResDer;
  OpResDer.set( _ndxSUPP, _vOUT, _vEFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVarVal.data() );
  for( size_t j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( size_t i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *ppResDer[i] * vVar[i][j];
    }
  }
}

inline void
FFODISTEff::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFODISTEFF_TRACE
  std::cout << "FFODISTEff::eval: fadbad::F<double>\n";
#endif

  std::vector<double> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double ResVal; 
  eval( 1, &ResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = ResVal;
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );

  FFGradODISTEff OpResDer;
  OpResDer.set( _ndxSUPP, _vOUT, _vEFFAP );
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
FFODISTEff::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFODISTEFF_TRACE
  std::cout << "FFODISTEff::deriv\n";
#endif

  FFGradODISTEff OpResDer;
  OpResDer.set( _ndxSUPP, _vOUT, _vEFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVar ); 
  for( size_t i=0; i<nVar; ++i )
    vDer[0][i] = *ppResDer[i];
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
  std::map<size_t,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;

  // Prior experimental efforts
  std::vector<double> const* _EFFAP;
  // Prior experimental FIM marices
  std::vector<std::vector<arma::mat>> const* _FIMAP;

  // Number of parameters
  size_t _np;
  // Number of controls
  size_t _nc;
  // Number of outputs
//  size_t _ny;
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
  
  // Evaluation of DOE criterion from FIM values
  void _FIMCrit
    ( double& crit, std::vector<double> const& vfim )
    const;
    
public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::mat>> const* fimap, std::vector<double> const* effap )
    {
#ifdef MC__FFFIMCrit_CHECK
  assert( dag && par->size() && con->size() && fim->size() && eff->size() && vpar->size() );
#endif

      _DAG   = dag;
      _FPAR  = par;
      _FCON  = con;
      _FFIM  = fim;
      _EFF   = eff;
      _DPAR  = vpar;
      _EFFAP = effap;
      _FIMAP = fimap;

      _np = _FPAR->size();
      _nc = _FCON->size();
//      _ny = std::round( std::sqrt(2*_FFIM->size()+0.25) - 0.5 );
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
      _EFFAP( Op._EFFAP ),
      _FIMAP( Op._FIMAP ),
      _np( Op._np ),
      _nc( Op._nc ),
//      _ny( Op._ny ),
      _ns( Op._ns ),
      _ne( Op._ne )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::mat>> const* fimap, std::vector<double> const* effap )
    {
#ifdef MC__FFFIMCrit_CHECK
      assert( idep < _ns );
#endif
      set( dag, par, con, fim, eff, vpar, fimap, effap );
      return *(insert_external_operation( *this, _ns, _nc*_ne, coneff )[idep]);

    }

  FFVar** operator()
    ( FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::mat>> const* fimap, std::vector<double> const* effap )
    {
      set( dag, par, con, fim, eff, vpar, fimap, effap );
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
        case BASE_MBDOE::AOPT: return "-tr Inv";
        case BASE_MBDOE::DOPT: return "log Det";
        case BASE_MBDOE::EOPT: return "min Eig";
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
  std::map<size_t,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;
  // parameter scenarios
  std::vector<std::vector<fadbad::F<double>>> _FDPAR;

  // Prior experimental efforts
  std::vector<double> const* _EFFAP;
  // Prior experimental FIM marices
  std::vector<std::vector<arma::mat>> const* _FIMAP;

  // Number of parameters
  size_t _np;
  // Number of controls
  size_t _nc;
  // Number of outputs
//  size_t _ny;
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
  // Intermediate derivatives
  mutable std::vector<double> _GradCrit;

  // Evaluation of DOE criterion derivatives from FIM values
  void _FIMDerCrit
    ( std::vector<double>& dercrit, std::vector<fadbad::F<double>> const& vFfim )
    const;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::mat>> const* fimap, std::vector<double> const* effap )
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
      _EFFAP = effap;
      _FIMAP = fimap;

      _np = _FPAR->size();
      _nc = _FCON->size();
//      _ny = std::round( std::sqrt(2*fim->size()+0.25) - 0.5 );
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
      _EFFAP( Op._EFFAP ),
      _FIMAP( Op._FIMAP ),
      _np( Op._np ),
      _nc( Op._nc ),
//      _ny( Op._ny ),
      _ns( Op._ns ),
      _ne( Op._ne ),
      _FDCON( Op._FDCON )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::mat>> const* fimap, std::vector<double> const* effap )
    {
#ifdef MC__FFFIMCrit_CHECK
      assert( idep < _ns*_nc*_ne );
#endif
      set( dag, par, con, fim, eff, vpar, fimap, effap );
      return *(insert_external_operation( *this, _ns*_nc*_ne, _nc*_ne, coneff )[idep]);
    }

  FFVar** operator()
    ( FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* fim,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::mat>> const* fimap, std::vector<double> const* effap )
    {
      set( dag, par, con, fim, eff, vpar, fimap, effap );
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
        case BASE_MBDOE::AOPT: return "-Grad tr Inv";
        case BASE_MBDOE::DOPT: return "Grad log Det";
        case BASE_MBDOE::EOPT: return "Grad min Eig";
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
  for( size_t j=0; j<nRes; ++j )
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
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
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
  for( size_t j=0; j<nRes; ++j )
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
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}


inline void
FFFIMCrit::_FIMCrit
( double& crit, std::vector<double> const& vfim )
const
{
#ifdef MC__FFFIMCRIT_TRACE
  std::cout << "FFDOECrit::_FIMCrit\n"; 
#endif

  arma::mat FIM( _np, _np, arma::fill::none );
  for( size_t i=0, l=0; i<_np; ++i )
    for( size_t j=i; j<_np; ++j, ++l )
      if( i == j ) FIM(i,i) = vfim[l]; 
      else         FIM(i,j) = FIM(j,i) = vfim[l];
  if( scaling.n_elem ) FIM = arma::trans(scaling) * FIM * scaling;
#ifdef MC__FFDOECRIT_DEBUG
  std::cout << "FIM:\n" << FIM;
  std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

  switch( FFDOEBase::type ){
    case BASE_MBDOE::AOPT:
    {
      arma::vec FIMEIGVAL;
      if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      crit = 0.;
      for( size_t k=0; k<FIM.n_rows; ++k )
        crit -= 1./FIMEIGVAL(k);
      break;
    }
    
    case BASE_MBDOE::DOPT:
    {
#if defined( MC__FFDCRIT_EIG )
      arma::vec FIMEIGVAL;
      if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      crit = 0.;
      for( size_t k=0; k<FIM.n_rows; ++k )
        crit += std::log( FIMEIGVAL(k) );
      std::cout << "FIMEIGVAL: " << arma::trans( FIMEIGVAL ) << std::endl;
#else
      if( arma::rank( FIM ) < FIM.n_rows || !arma::log_det_sympd( crit, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#endif
      break;
    }
    
    case BASE_MBDOE::EOPT:
    {
      arma::vec FIMEIGVAL;
      if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIM ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      crit = FIMEIGVAL(0);
      break;
    }
    default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
  }
#ifdef MC__FFFIMCRIT_DEBUG
  std::cout << name() << ": " << crit << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradFIMCrit::_FIMDerCrit
( std::vector<double>& dercrit, std::vector<fadbad::F<double>> const& vfim )
const
{
#ifdef MC__FFGRADDOECRIT_TRACE
  std::cout << "FFDOECrit::_FIMDerCrit\n"; 
#endif

  arma::mat FIM( _np, _np, arma::fill::none );
  for( size_t i=0, l=0; i<_np; ++i )
    for( size_t j=i; j<_np; ++j, ++l )
      if( i == j ) FIM(i,i) = vfim[l].val(); 
      else         FIM(i,j) = FIM(j,i) = vfim[l].val();
  if( scaling.n_elem ) FIM = arma::trans(scaling) * FIM * scaling;
#ifdef MC__FFGRADDOECRIT_DEBUG
  std::cout << "FIM: " << FIM;
  std::cout << "rank: " << arma::rank( FIM ) << std::endl;
#endif

  switch( FFDOEBase::type ){
    case BASE_MBDOE::AOPT:
    {
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
      std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
      for( size_t i=0, l=0; i<_np; ++i )
        for( size_t j=i; j<_np; ++j, ++l ){
          dercrit[l] = 0.;
          if( scaling.n_elem ){
            FIM.zeros(_np,_np); // repurpose the FIM matrix
            FIM(i,j) = ( i==j? 1: FIM(j,i) = 1 );
            for( size_t k=0; k<scaling.n_cols; ++k ){
              arma::mat&& FIMEIGVECk = scaling * FIMEIGVEC.col(k);
              arma::mat&& FIMPROJ = FIMEIGVECk.t() * FIM * FIMEIGVECk;
              dercrit[l] += FIMPROJ(0,0) / (FIMEIGVAL(k)*FIMEIGVAL(k));
            }
          }
          else
            for( size_t k=0; k<_np; ++k )
              dercrit[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(j,k) )
                       / (FIMEIGVAL(k)*FIMEIGVAL(k));
          //for( size_t k=0; k<_np; ++k )
          //  if( scaling.n_elem ){
          //    dercrit[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(i,k) )
          //             * (scaling(i,i)*scaling(j,j)) / (FIMEIGVAL(k)*FIMEIGVAL(k));
          //  }
          //  else
          //    dercrit[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(j,k) )
          //             / (FIMEIGVAL(k)*FIMEIGVAL(k));
        }
      break;
    }
    
    case BASE_MBDOE::DOPT:
    {
#if defined( MC__FFDCRIT_EIG )
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM eigenvalues: "  << FIMEIGVAL;
      std::cout << "FIM eigenvectors: " << FIMEIGVEC;
#endif
      for( size_t i=0, l=0; i<_np; ++i )
        for( size_t j=i; j<_np; ++j, ++l ){
          dercrit[l] = 0.;
          if( scaling.n_elem ){
            FIM.zeros(_np,_np); // repurpose the FIM matrix
            FIM(i,j) = ( i==j? 1: FIM(j,i) = 1 );
            for( size_t k=0; k<scaling.n_cols; ++k ){
              arma::mat&& FIMEIGVECk = scaling * FIMEIGVEC.col(k);
              arma::mat&& FIMPROJ = FIMEIGVECk.t() * FIM * FIMEIGVECk;
              dercrit[l] += FIMPROJ(0,0) / FIMEIGVAL(k);
            }
          }
          else
            for( size_t k=0; k<_np; ++k )
              dercrit[l] += (i==j? FIMEIGVEC(i,k)*FIMEIGVEC(i,k): 2*FIMEIGVEC(i,k)*FIMEIGVEC(j,k) )
                       / FIMEIGVAL(k);
        }
#else
      arma::mat L, X, Y;
      if( !arma::chol( L, FIM, "lower" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM Cholesky decomposition:\n" << L;
#endif
      for( size_t i=0, l=0; i<_np; ++i ){
        for( size_t j=i; j<_np; ++j, ++l ){
          FIM.zeros(_np,_np); // repurpose the FIM matrix
          FIM(i,j) = ( i==j? 1: FIM(j,i) = 1 );
          if( scaling.n_elem ){
            arma::mat&& FIMPROJ = trans(scaling) * FIM * scaling;
#ifdef MC__FFGRADDOECRIT_DEBUG
            std::cout << "FIM derivative projection:\n" << FIMPROJ;
#endif
            if( !solve( Y, trimatl(L), FIMPROJ ) )  // indicate that L is lower triangular
              throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          }
          else{
            if( !solve( Y, trimatl(L), FIM ) )  // indicate that L is lower triangular
              throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          }
          if( !solve( X, trimatu(trans(L)), Y ) )  // indicate that L^T is upper triangular
            throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
          dercrit[l] = arma::trace( X );
        }
      }
#endif
      break;
    }

    case BASE_MBDOE::EOPT:
    {
      arma::vec FIMEIGVAL;
      arma::mat FIMEIGVEC;
      if( arma::rank( FIM ) < FIM.n_rows || !arma::eig_sym( FIMEIGVAL, FIMEIGVEC, FIM, "std" ) )
        throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
#ifdef MC__FFGRADDOECRIT_DEBUG
      std::cout << "FIM min eigenvalue: "  << FIMEIGVAL(0) << std::endl;
      std::cout << "FIM min eigenvector: " << trans(FIMEIGVEC.col(0));
#endif
     for( size_t i=0, l=0; i<_np; ++i )
        for( size_t j=i; j<_np; ++j, ++l )
          if( scaling.n_elem ){
            FIM.zeros(_np,_np); // repurpose the FIM matrix
            FIM(i,j) = ( i==j? 1: FIM(j,i) = 1 );
            arma::mat&& FIMEIGVEC0 = scaling * FIMEIGVEC.col(0);
            arma::mat&& FIMPROJ = FIMEIGVEC0.t() * FIM * FIMEIGVEC0;
            dercrit[l] = FIMPROJ(0,0);
            //dercrit[l] = (i==j? FIMEIGVEC(i,0)*FIMEIGVEC(i,0): 2*FIMEIGVEC(i,0)*FIMEIGVEC(j,0) )
            //        * (scaling(i,i)*scaling(j,j));
          }
          else
            dercrit[l] = (i==j? FIMEIGVEC(i,0)*FIMEIGVEC(i,0): 2*FIMEIGVEC(i,0)*FIMEIGVEC(j,0) );
      break;
    }
    default:   throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
  }
#ifdef MC__FFGRADDOECRIT_DEBUG
  for( size_t i=0, l=0; i<_np; ++i )
    for( size_t j=i; j<_np; ++j, ++l )
      std::cout << name() << " [" << i << "," << j << "]: " << dercrit[l] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
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
  for( size_t s=0; s<_ns; ++s ){
    _DFIM[s].assign( _FFIM->size(), 0. );
    for( size_t i=0, l=0; i<_np; ++i )
      for( size_t j=i; j<_np; ++j, ++l ){
        size_t e=0;
        //std::cout << "FIMAP[" << s << "][" << e << "]:" << std::endl << _FIMAP->at(s).at(e);
        for( auto const& eff : *_EFFAP ) // Atomic FIM of prior experiment
          _DFIM[s][l] += eff * (_FIMAP->at(s).at(e++))(i,j);
      }
  }

  double const* pCON = vVar;
  for( auto const& [id,eff] : *_EFF ){
    _DCON.assign( pCON, pCON+_nc );
    //_DAG->veval( _sgFIM, _wkD, *_FFIM, _DFIM, *_FPAR, *_DPAR, *_FCON, _DCON, &eff );
    _DAG->veval( _sgFIM, _wkD, _wkThd, *_FFIM, _DFIM, *_FPAR, *_DPAR, *_FCON, _DCON, &eff ); // Append effort x FIM
    pCON += _nc;
  }

  // Calculate FIM-based criteria in each uncertainty scenario
  for( size_t s=0; s<_ns; ++s ){
    _FIMCrit( vRes[s], _DFIM[s] );
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
  for( size_t s=0; s<_ns; ++s ){
    _FDFIM[s].assign( _FFIM->size(), 0. );
    for( size_t i=0, l=0; i<_np; ++i )
      for( size_t j=i; j<_np; ++j, ++l ){
        size_t e=0;
        //std::cout << "FIMAP[" << s << "][" << e << "]:" << std::endl << _FIMAP->at(s).at(e);
        for( auto const& eff : *_EFFAP ) // Atomic FIM of prior experiment
          _FDFIM[s][l].x() += eff * (_FIMAP->at(s).at(e++))(i,j);
      }
  }

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
  for( size_t s=0; s<_ns; ++s ){
    _GradCrit.resize( _FDFIM[s].size() );
    _FIMDerCrit( _GradCrit, _FDFIM[s] );
    for( size_t ec=0; ec<_ne*_nc; ++ec ){
      vRes[ec+_ne*_nc*s] = 0.; 
      for( size_t i=0; i<_FDFIM[s].size(); ++i )
        vRes[ec+_ne*_nc*s] += _GradCrit[i] * _FDFIM[s][i].deriv( ec ); 
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
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar const*const* ppResVal = insert_external_operation( *this, nRes, nVar, vVarVal.data() );

  FFGradFIMCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FFIM, _EFF, _DPAR, _FIMAP, _EFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nRes*nVar, nVar, vVarVal.data() );
  for( size_t k=0; k<nRes; ++k ){
    vRes[k] = *ppResVal[k];
    for( size_t i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
    for( size_t j=0; j<vRes[k].size(); ++j ){
      vRes[k][j] = 0.;
      for( size_t i=0; i<nVar; ++i ){
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
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  std::vector<double> vResVal( nRes ); 
  eval( nRes, vResVal.data(), nVar, vVarVal.data(), nullptr );
  for( size_t k=0; k<nRes; ++k ){
    vRes[k] = vResVal[k];
    for( size_t i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
  }
  
  FFGradFIMCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FFIM, _EFF, _DPAR, _FIMAP, _EFFAP );
  std::vector<double> vResDer( nRes*nVar ); 
  OpResDer.eval( nRes*nVar, vResDer.data(), nVar, vVarVal.data(), nullptr );
  for( size_t k=0; k<nRes; ++k ){
    for( size_t j=0; j<vRes[k].size(); ++j ){
      vRes[k][j] = 0.;
      for( size_t i=0; i<nVar; ++i ){
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
  OpResDer.set( _DAG, _FPAR, _FCON, _FFIM, _EFF, _DPAR, _FIMAP, _EFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nRes*nVar, nVar, vVar );
  for( size_t k=0; k<nRes; ++k )
    for( size_t i=0; i<nVar; ++i )
      //vDer[k][i] = *ppResDer[k+nRes*i];
      vDer[k][i] = *ppResDer[k*nVar+i];
}

////////////////////////////////////////////////////////////////////////

class FFBRCrit
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
  std::map<size_t,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;

  // Prior experimental efforts
  std::vector<double> const* _EFFAP;
  // Prior experimental output vectors
  std::vector<std::vector<arma::vec>> const* _OUTAP;

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
  mutable std::vector<std::vector<std::vector<double>>> _DOUT; // _nc x _ns x _ny

  // Subgraph
  mutable FFSubgraph _sgOUT;
  // Work storage
  mutable std::vector<double> _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<double>> _wkThd;

  // Evaluation of Bayes risk from output values
  void _BRval
    ( double& BR, std::vector<std::vector<std::vector<double>>>& DOUT )
    const;
    
public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::vec>> const* outap, std::vector<double> const* effap )
    {
#ifdef MC__FFBRCRIT_CHECK
  assert( dag && par->size() && con->size() && out->size() && eff->size() && vpar->size() );
#endif

      _DAG  = dag;
      _FPAR = par;
      _FCON = con;
      _FOUT = out;
      _EFF  = eff;
      _DPAR = vpar;
      _OUTAP = outap;
      _EFFAP = effap;

      _np = _FPAR->size();
      _nc = _FCON->size();
      _ny = _FOUT->size();
      _ns = _DPAR->size();
      _ne = _EFF->size();
    }

  // Default constructor
  FFBRCrit
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFBRCrit
    ( FFBRCrit const& Op )
    : FFOp( Op ),
      _DAG( Op._DAG ),
      _FPAR( Op._FPAR ),
      _FCON( Op._FCON ),
      _FOUT( Op._FOUT ),
      _EFF( Op._EFF ),
      _DPAR( Op._DPAR ),
      _EFFAP( Op._EFFAP ),
      _OUTAP( Op._OUTAP ),
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
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::vec>> const* outap, std::vector<double> const* effap )
    {
      set( dag, par, con, out, eff, vpar, outap, effap );
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

      throw std::runtime_error( "FFBRCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
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
        case BASE_MBDOE::BROPT: return "Bayes Risk";
        default:    throw FFBase::Exceptions( FFBase::Exceptions::EXTERN );
      }
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

class FFGradBRCrit
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
  std::map<size_t,double> const* _EFF;
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;
  // parameter scenarios
  std::vector<std::vector<fadbad::F<double>>> _FDPAR;

  // Prior experimental efforts
  std::vector<double> const* _EFFAP;
  // Prior experimental output vectors
  std::vector<std::vector<arma::vec>> const* _OUTAP;

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
  mutable std::vector<std::vector<std::vector<fadbad::F<double>>>> _FDOUT;

  // Subgraph
  mutable FFSubgraph _sgOUT;
  // Work storage
  mutable std::vector<fadbad::F<double>> _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<fadbad::F<double>>> _wkThd;

  // Evaluation of Bayes risk derivatives from output values
  void _BRder
    ( double* gradBR, std::vector<std::vector<std::vector<fadbad::F<double>>>>& FDOUT )
    const;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::vec>> const* outap, std::vector<double> const* effap )
    {
#ifdef MC__FFBRCRIT_CHECK
  assert( dag && par->size() && con->size() && out->size() && eff->size() && vpar->size() );
#endif

      _DAG  = dag;
      _FPAR = par;
      _FCON = con;
      _FOUT = out;
      _EFF  = eff;
      _DPAR = vpar;
      _OUTAP = outap;
      _EFFAP = effap;

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
  FFGradBRCrit
    ()
    : FFOp( EXTERN )
    {}
    
  // Copy constructor
  FFGradBRCrit
    ( FFGradBRCrit const& Op )
    : FFOp( Op ),
      _DAG( Op._DAG ),
      _FPAR( Op._FPAR ),
      _FCON( Op._FCON ),
      _FOUT( Op._FOUT ),
      _EFF( Op._EFF ),
      _DPAR( Op._DPAR ),
      _FDPAR( Op._FDPAR ),
      _EFFAP( Op._EFFAP ),
      _OUTAP( Op._OUTAP ),
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
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::vec>> const* outap, std::vector<double> const* effap )
    {
#ifdef MC__FFBRISKCRIT_CHECK
      assert( idep < _nc*_ne );
#endif
      set( dag, par, con, out, eff, vpar, outap, effap );
      return *(insert_external_operation( *this, _nc*_ne, _nc*_ne, coneff )[idep]);
    }

  FFVar** operator()
    ( FFVar const* coneff, FFGraph* dag, std::vector<FFVar> const* par,
      std::vector<FFVar> const* con, std::vector<FFVar> const* out,
      std::map<size_t,double> const* eff, std::vector<std::vector<double>> const* vpar,
      std::vector<std::vector<arma::vec>> const* outap, std::vector<double> const* effap)
    {
      set( dag, par, con, out, eff, vpar, outap, effap );
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

      throw std::runtime_error( "FFGradBRCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
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
        case BASE_MBDOE::BROPT: return "Grad Bayes Risk";
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
FFBRCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::eval: FFVar\n"; 
#endif

  vRes[0] = **insert_external_operation( *this, nRes, nVar, vVar );
}

inline void
FFBRCrit::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
}

inline void
FFGradBRCrit::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFGradBRCrit::eval: FFVar\n"; 
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( size_t j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFGradBRCrit::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFGradBRCrit::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}

inline void
FFBRCrit::_BRval
( double& BR, std::vector<std::vector<std::vector<double>>>& DOUT )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBRCrit::_BRval\n";
#endif
#ifdef MC__FFBRISKCRIT_CHECK
  assert( _EFF && !_EFF->empty() && DOUT.size() == _ne && DOUT.front().size == _ns && DOUT.front().front().size == _ny );
#endif

  auto BRappend = [&]( size_t j, size_t k, double& BR ){
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);

      size_t e = 0;
      for( auto const& eff : *_EFFAP ){ // Contributions from prior experiment
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "yAP[" << e << "][" << j << "] = " << _OUTAP->at(j).at(e);
        std::cout << "yAP[" << e << "][" << k << "] = " << _OUTAP->at(k).at(e);
#endif
        arma::vec const& Ejk  = _OUTAP->at(j).at(e) - _OUTAP->at(k).at(e);
        if( !sigmayinv.empty() ) Et_Vinv_E += eff * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += eff * Ejk.t() * Ejk;
        e++;
      }

      e = 0;
      for( auto const& [id,eff] : *_EFF ){ // Contributions from new experiment
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "y[" << e << "][" << j << "] = " << arma::vec( DOUT[e][j].data(), _ny, false );
        std::cout << "y[" << e << "][" << k << "] = " << arma::vec( DOUT[e][k].data(), _ny, false );
#endif
        arma::vec const& Ejk = arma::vec( DOUT[e][j].data(), _ny, false )
                             - arma::vec( DOUT[e][k].data(), _ny, false );
        if( !sigmayinv.empty() ) Et_Vinv_E += eff * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += eff * Ejk.t() * Ejk;
        e++;
      }
      
      double BRjk = std::exp( -0.125 * Et_Vinv_E(0,0) );
#ifdef MC__FFBRCRIT_DEBUG
      std::cout << "BR[" << j << "," << k << "] = " << BRjk << std::endl; 
#endif
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
      BR += BRjk;
#ifdef MC__FFBRCRIT_DEBUG
      std::cout << "weighting: " << weighting(j) << "  " << weighting(k) << std::endl;
      std::cout << "BR[" << j << "," << k << "] = " << BR << "  " << Et_Vinv_E(0,0) << std::endl; 
#endif
  };

  BR = 0.;

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
      BRappend( j, k, BR );
 
  // Use full set of uncertainty scenarios
  else
    for( size_t j=0; j<_ns-1; ++j )
      for( size_t k=j+1; k<_ns; ++k )
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
FFBRCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFBRCrit::eval: double\n"; 
#endif
#ifdef MC__FFBRISKCRIT_CHECK
  assert( nRes == 1 && nVar == _nc*_ne );
#endif

  // Get outputs for each scenario and each experiment
  //_DOUT.resize( _ne );
  _DOUT.assign( _ne, std::vector<std::vector<double>>( _ns, std::vector<double>( _ny, 0. ) ) );

  double const* pCON = vVar;
  for( size_t e=0; e<_ne; ++e ){
    _DCON.assign( pCON, pCON+_nc );
#ifdef MC__FFBRCRIT_DEBUG
    std::cout << "c[" << e << "] = " << arma::vec( _DCON.data(), _nc, false );
#endif
    //_DAG->veval( _sgOUT, _wkD, *_FOUT, _DOUT[e], *_FPAR, *_DPAR, *_FCON, _DCON );
    _DAG->veval( _sgOUT, _wkD, _wkThd, *_FOUT, _DOUT[e], *_FPAR, *_DPAR, *_FCON, _DCON );
#ifdef MC__FFBRCRIT_DEBUG
    for( size_t j=0; j<_ns; ++j )
      std::cout << "y[" << e << "][" << j << "] = " << arma::vec( _DOUT[e][j].data(), _ny, false );
#endif
    pCON += _nc;
  }

  // Calculate Bayes risk-based criterion
  _BRval( vRes[0], _DOUT );
#ifdef MC__FFBRISKCRIT_DEBUG
  std::cout << name() << " = " << vRes[0] << std::endl;
    //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradBRCrit::_BRder
( double* pgradBR, std::vector<std::vector<std::vector<fadbad::F<double>>>>& FDOUT )
const
{
#ifdef MC__FFBRCRIT_TRACE
  std::cout << "FFBGradRISKCrit::_BRder\n";
#endif
#ifdef MC__FFBRISKCRIT_CHECK
  assert( _EFF && !_EFF->empty() && FDOUT.size() == _ne && FDOUT.front().size == _ns && FDOUT.front().front().size == _ny );
#endif

#ifdef MC__FFBRCRIT_LOG
  auto BRappend = [&]( size_t j, size_t k, double& BR, arma::vec& gradBR, arma::mat& dBRdy ){
#else
  auto BRappend = [&]( size_t j, size_t k, arma::vec& gradBR, arma::mat& dBRdy ){
#endif
      arma::mat Et_Vinv_E(1,1,arma::fill::zeros);

      size_t e = 0;
      for( auto const& eff : *_EFFAP ){ // Contributions from prior experiment
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "yAP[" << e << "][" << j << "] = " << _OUTAP->at(j).at(e);
        std::cout << "yAP[" << e << "][" << k << "] = " << _OUTAP->at(k).at(e);
#endif
        arma::vec const& Ejk  = _OUTAP->at(j).at(e) - _OUTAP->at(k).at(e);
        if( !sigmayinv.empty() ) Et_Vinv_E += eff * Ejk.t() * sigmayinv * Ejk;
        else                     Et_Vinv_E += eff * Ejk.t() * Ejk;
        e++;
      }

      e = 0;
      arma::vec Ejk( _ny );
      for( auto const& [id,eff] : *_EFF ){
        for( size_t i=0; i<_ny; ++i )
          Ejk(i) = FDOUT[e][j][i].x() - FDOUT[e][k][i].x();
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "E_" << e << "[" << j << "][" << k << "] = " << Ejk;
#endif
        if( !sigmayinv.empty() ){
          dBRdy.unsafe_col(e) = sigmayinv * Ejk;
          Et_Vinv_E += eff * Ejk.t() * dBRdy.unsafe_col(e);
          dBRdy.unsafe_col(e) *= eff/4;
        }
        else{
          Et_Vinv_E += eff * Ejk.t() * Ejk;
          dBRdy.unsafe_col(e) = (eff/4) * Ejk;
        }
        e++;
      }

      double BRjk = std::exp( -0.125 * Et_Vinv_E(0,0) );
      if( !weighting.empty() ) BRjk *= std::sqrt( weighting(j)*weighting(k) );
#ifdef MC__FFBRCRIT_LOG
      BR += BRjk;
#endif
      for( e=0; e<_ne; ++e )
        for( size_t i=0; i<_ny; ++i )
          gradBR -= ( dBRdy(i,e) * BRjk )
                  * ( arma::vec( &FDOUT[e][j][i].d(0), _nc*_ne, false )
                    - arma::vec( &FDOUT[e][k][i].d(0), _nc*_ne, false ) );
  };

#ifdef MC__FFBRCRIT_LOG
  double BR = 0.;
#endif
  arma::vec gradBR( pgradBR, _nc*_ne, false );
  gradBR.zeros();
  arma::mat dBRdy( _ny, _ne, arma::fill::none );

  // Use subset of uncertainty scenarios
  if( parsubset && !parsubset->empty() )
    for( auto const& [j,k] : *parsubset )
#ifdef MC__FFBRCRIT_LOG
      BRappend( j, k, BR, gradBR, dBRdy );
#else
      BRappend( j, k, gradBR, dBRdy );
#endif
 
  // Use full set of uncertainty scenarios
  else
    for( size_t j=0; j<_ns-1; ++j )
      for( size_t k=j+1; k<_ns; ++k )
#ifdef MC__FFBRCRIT_LOG
        BRappend( j, k, BR, gradBR, dBRdy );
#else
        BRappend( j, k, gradBR, dBRdy );
#endif

#ifdef MC__FFBRCRIT_LOG
  gradBR /= BR;
#endif

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << gradBR;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradBRCrit::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADFIMCRIT_TRACE
  std::cout << "FFGradBRCrit::eval: double\n"; 
#endif
#ifdef MC__FFGRADFIMCRIT_CHECK
  assert( nRes == nVar && nVar = _nc*_ne );
#endif

  // Get FIM entry derivatives for each scenario and each experiment
  //_FDOUT.resize( _ne );
  _FDOUT.assign( _ne, std::vector<std::vector<fadbad::F<double>>>( _ns, std::vector<fadbad::F<double>>( _ny, 0. ) ) );

  double const* pCON = vVar;
  for( size_t e=0; e<_ne; ++e ){
    for( size_t c=0; c<_nc; ++c )
      _FDCON[e][c].x() = pCON[c]; // does not change differential variables
    //_DAG->veval( _sgOUT, _wkD, *_FOUT, _FDOUT[e], *_FPAR, _FDPAR, *_FCON, _FDCON[e] );
    _DAG->veval( _sgOUT, _wkD, _wkThd, *_FOUT, _FDOUT[e], *_FPAR, _FDPAR, *_FCON, _FDCON[e] );
#ifdef MC__FFBRISKCRIT_DEBUG
    for( size_t k=0; k<_ny; ++k ){
      std::cout << "_FDOUT[" << e << "][" << _ns-1 << "][" << k << "] =";
      for( size_t i=0; i<_FDOUT[e].back()[k].size(); ++i )
        std::cout << "  " << _FDOUT[e].back()[k].deriv(i);
      std::cout << std::endl;
    }
    { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
    pCON += _nc;
  }

  // Calculate Bayes risk-based criterion derivatives
  _BRder( vRes, _FDOUT );
#ifdef MC__FFDOECRIT_DEBUG
  for( size_t i=0; i<nVar; ++i )
    std::cout << name() << "[" << i << "] = " << vRes[i] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFBRCrit::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFBRCrit::eval: fadbad::F<FFVar>\n"; 
#endif
#ifdef MC__FFDOECRIT_CHECK
  assert( nRes == 1 );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar ResVal = **insert_external_operation( *this, 1, nVar, vVarVal.data() );

  FFGradBRCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FOUT, _EFF, _DPAR, _OUTAP, _EFFAP );
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
FFBRCrit::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFDOECRIT_TRACE
  std::cout << "FFBRCrit::eval: fadbad::F<double>\n"; 
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
  
  FFGradBRCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FOUT, _EFF, _DPAR, _OUTAP, _EFFAP );
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
FFBRCrit::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFBRISKCRIT_TRACE
  std::cout << "FFBRCrit::deriv:\n"; 
#endif
#ifdef MC__FFBRISKCRIT_CHECK
  assert( nRes == 1 );
#endif

  FFGradBRCrit OpResDer;
  OpResDer.set( _DAG, _FPAR, _FCON, _FOUT, _EFF, _DPAR, _OUTAP, _EFFAP );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVar );
  for( size_t i=0; i<nVar; ++i )
    vDer[0][i] = *ppResDer[i];
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
    ( size_t const idep, size_t const nP, size_t const nY, FFVar const* Yp, std::vector<double>* Yvar )
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
      for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
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
    ( size_t const idep, size_t const nP, size_t const nY, FFVar const* Yp, std::vector<double>* Yvar )
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
      for( size_t i=0; i<nVar; ++i )
        vRes[0] += vVar[i];
      vRes[0].update( FFDep::TYPE::N );
      for( size_t j=1; j<nRes; ++j )
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

  for( size_t k=0; k<_nY; k++ )
    for( size_t i=0, ij=0; i<_nP; ++i )
      for( size_t j=i; j<_nP; ++j, ++ij ){
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
  for( size_t j=0; j<nRes; ++j )
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
  for( size_t j=0; j<nRes; ++j )
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

  for( size_t k=0; k<_nY; k++ )
    for( size_t i=0, ij=0; i<_nP; ++i )
      for( size_t j=i; j<_nP; ++j, ++ij ){
        if( !k )           vRes[ij]  = 0.;
        if( Yvar.empty() ) vRes[ij] +=  vVar[_nY*i+k] * vVar[_nY*j+k];
        else               vRes[ij] += (vVar[_nY*i+k] * vVar[_nY*j+k]) / Yvar[k];
      }
#ifdef MC__FFFIM_DEBUG
  for( size_t i=0, ij=0; i<_nP; ++i )
    for( size_t j=i; j<_nP; ++j, ++ij )
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

  for( size_t l=0, kl=0; l<_nP; ++l )
    for( size_t k=0; k<_nY; ++k, ++kl )
      for( size_t i=0, ij=0; i<_nP; ++i )
        for( size_t j=i; j<_nP; ++j, ++ij ){
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
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar const*const* vResVal = insert_external_operation( *this, nRes, nVar, vVarVal.data() );
  for( size_t k=0; k<nRes; ++k ){
    vRes[k] = *vResVal[k];
    for( size_t i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
  }

  FFGradFIM OpGradFIM;
  FFVar const*const* vGradFIM = OpGradFIM( _nP, _nY, vVarVal.data(), &Yvar ); 
  for( size_t k=0; k<nRes; ++k ){
    for( size_t j=0; j<vRes[0].size(); ++j ){
      vRes[k][j] = 0.;
      for( size_t i=0; i<nVar; ++i ){
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
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  std::vector<double> vResVal( nRes ); 
  eval( nRes, vResVal.data(), nVar, vVarVal.data(), nullptr );
  for( size_t k=0; k<nRes; ++k ){
    vRes[k] = vResVal[k];
    for( size_t i=0; i<nVar; ++i )
      vRes[k].setDepend( vVar[i] );
  }

  FFGradFIM OpGradFIM;
  OpGradFIM.data = data;
  OpGradFIM._nP = _nP;
  OpGradFIM._nY = _nY;
  std::vector<double> vGradFIM( nRes * nVar ); 
  OpGradFIM.eval( nRes * nVar, vGradFIM.data(), nVar, vVarVal.data(), nullptr );
  for( size_t k=0; k<nRes; ++k ){
    for( size_t j=0; j<vRes[0].size(); ++j ){
      vRes[k][j] = 0.;
      for( size_t i=0; i<nVar; ++i ){
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
  for( size_t k=0; k<nRes; ++k )
    for( size_t i=0; i<nVar; ++i )
      vDer[k][i] = *vGradFIM[k*nVar+i];
}

} // end namespace mc

#endif

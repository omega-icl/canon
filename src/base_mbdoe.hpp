// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef CANON__BASE_MBDOE_HPP
#define CANON__BASE_MBDOE_HPP

#undef  CANON__DEBUG__BASE_MBDOE

#include <assert.h>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include "ffunc.hpp"
#include "odeslvs_cvodes.hpp"

namespace mc
{
//! @brief C++ base class for defining of model-based design of experiment problems
////////////////////////////////////////////////////////////////////////
//! mc::BASE_MBDOE is a C++ base class for defining the controls,
//! parameters and outputs participating in model-based design of
//! experiment (MBDoE) problems
////////////////////////////////////////////////////////////////////////
class BASE_MBDOE
{
protected:
  //! @brief pointer to DAG of equation
  FFGraph* _dag;

  //! @brief Size of model output
  size_t _ny;

  //! @brief Size of model parameter
  size_t _np;

  //! @brief Size of experimental control
  size_t _nc;

  //! @brief vector of model outputs
  std::vector<FFVar> _vOUT;

  //! @brief vector of model output variances
  std::vector<double> _vOUTVAR;

  //! @brief vector of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief list of model parameter values
  std::vector<std::vector<double>> _vPARVAL;

  //! @brief list of model parameter weights
  std::vector<double> _vPARWEI;

  //! @brief vector fo model parameter scaling factors
  std::vector<double> _vPARSCA;

  //! @brief vector of experimental controls
  std::vector<FFVar> _vCON;

  //! @brief vector of experimental control lower bounds
  std::vector<double> _vCONLB;

  //! @brief vector of experimental control upper bounds
  std::vector<double> _vCONUB;

public:
  //! @brief Class constructor
  BASE_MBDOE()
    : _dag(nullptr), _ny(0), _np(0), _nc(0)
    {}

  //! @brief Class destructor
  virtual ~BASE_MBDOE()
    {}

  //! @brief Get pointer to DAG
  FFGraph const& dag()
    const
    { return *_dag; }

  //! @brief Set pointer to DAG
  void set_dag
    ( FFGraph& dag )
    { _dag = &dag; }

  //! @brief Get number of model outputs
  size_t ny
    ()
    const
    { return _ny; }

  //! @brief Get number of experimental controls
  size_t nc
    ()
    const
    { return _nc; }

  //! @brief Get number of model parameters
  size_t np
    ()
    const
    { return _np; }

  //! @brief Set model outputs
  void set_model
    ( std::vector<FFVar> const& Y, std::vector<double> const& varY=std::vector<double>() )
    {
      assert( !Y.empty() );
      _ny = Y.size();
      _vOUT = Y;
      _vOUTVAR = varY;
    }

  //! @brief Set nominal model parameters
  void set_parameters
    ( std::vector<FFVar> const& P, std::vector<double> const& valP,
      std::vector<double> const& scaP=std::vector<double>() )
    {
      assert( !P.empty() && valP.size() == P.size() );
      _np   = P.size();
      _vPAR = P;
      _vPARVAL.clear();
      _vPARVAL.push_back( valP );
      _vPARWEI.assign( 1, 1. );

      assert( scaP.empty() || scaP.size() == _np );
      _vPARSCA = scaP;
    }

  //! @brief Set list of model parameters
  void set_parameters
    ( std::vector<FFVar> const& P, std::list<std::vector<double>> const& l_valP,
      std::vector<double> const& scaP=std::vector<double>() )
    {
      assert( !P.empty() && !l_valP.empty() );
      _np   = P.size();
      _vPAR = P;
      _vPARVAL.clear();
      for( auto const& valP : l_valP ){
        assert( valP.size() == _np );
        _vPARVAL.push_back( valP );
      }
      _vPARWEI.assign( _vPARVAL.size(), 1/(double)l_valP.size() ); // equal frequencies

      assert( scaP.empty() || scaP.size() == _np );
      _vPARSCA = scaP;
    }

  //! @brief Set list of model parameters
  void set_parameters
    ( std::vector<FFVar> const& P, std::list<std::pair<std::vector<double>,double>> const& l_valP,
      std::vector<double> const& scaP=std::vector<double>() )
    {
      assert( !P.empty() && !l_valP.empty() );
      _np   = P.size();
      _vPAR = P;

      _vPARWEI.clear();
      double prTot = 0.;
      for( auto const& [valP,prP] : l_valP ){
        assert( prP > 0 );
        prTot += prP;
      }

      _vPARVAL.clear();
      for( auto const& [valP,prP] : l_valP ){
        assert( valP.size() == _np );
        _vPARVAL.push_back( valP );
        _vPARWEI.push_back( prP/prTot );
      }

      assert( scaP.empty() || scaP.size() == _np );
      _vPARSCA = scaP;
    }

  //! @brief Set experimental controls
  void set_controls
    ( std::vector<FFVar> const& C, std::vector<double> const& CLB, std::vector<double> const& CUB )
    {
      assert( !C.empty() && CLB.size() == C.size() && CUB.size() == C.size() );
      _nc     = C.size();
      _vCON   = C;
      _vCONLB = CLB;
      _vCONUB = CUB;
    }

  //! @brief Set uniform sample within bounds
  static std::list<std::vector<double>> uniform_sample
    ( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB );

  //! @brief Round fractional experimental efforts to nearest integer
  static void effort_rounding
    ( unsigned const n, unsigned const* typ, double* val );

  //! @brief Apportion fractional experimental efforts
  static void effort_apportion
    ( unsigned const n, unsigned const* typ, double* val );

private:
  //! @brief Private methods to block default compiler methods
  BASE_MBDOE( BASE_MBDOE const& ) = delete;
  BASE_MBDOE& operator=( BASE_MBDOE const& ) = delete;
};

inline std::list<std::vector<double>>
BASE_MBDOE::uniform_sample
( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB )
{
  assert( NSAM && LB.size() && LB.size() == UB.size() );
  size_t NDIM = LB.size();

  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( NDIM );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( 0 );

  std::list<std::vector<double>> LSAM;
  for( size_t s=0; s<NSAM; ++s ){
    LSAM.push_back( std::vector<double>( NDIM ) );
    for( size_t k=0; k<NDIM; k++ )
      LSAM.back()[k] = LB[k] + ( UB[k] - LB[k] ) * gen();
  }

  return LSAM;
}

inline void
BASE_MBDOE::effort_apportion
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

#ifdef CANON__MBDOE_SHOW_APPORTION
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

#ifdef CANON__MBDOE_SHOW_APPORTION
  std::cout << "Apportioned efforts:" << std::endl;
#endif
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
#ifdef CANON__MBDOE_SHOW_APPORTION
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
#endif
  }
}

inline void
BASE_MBDOE::effort_rounding
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

#ifdef CANON__MBDOE_SHOW_APPORTION
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
  
#ifdef CANON__MBDOE_SHOW_APPORTION
  std::cout << "Rounded efforts:" << std::endl;
#endif
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
#ifdef CANON__MBDOE_SHOW_APPORTION
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
#endif
  }
}

} // end namescape mc

#endif


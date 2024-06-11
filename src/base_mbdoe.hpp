// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__BASE_MBDOE_HPP
#define MC__BASE_MBDOE_HPP

#undef  MC__DEBUG__BASE_MBDOE

#include <assert.h>
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
template <typename... ExtOps>
class BASE_MBDOE
{
protected:
  //! @brief pointer to DAG of equation
  FFGraph<ExtOps...>* _dag;

  //! @brief pointer to DAG of equation
  mc::ODESLVS_CVODES<ExtOps...>* _ivpode;

  //! @brief Size of model output
  unsigned _ny;

  //! @brief Size of model parameter
  unsigned _np;

  //! @brief Size of experimental control
  unsigned _nc;

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
    : _dag(nullptr), _ivpode(nullptr), _ny(0), _np(0), _nc(0)
    {}

  //! @brief Class destructor
  virtual ~BASE_MBDOE()
    {}

  //! @brief Get pointer to DAG
  FFGraph<ExtOps...>* dag()
    const
    { return _dag; }

  //! @brief Set pointer to DAG
  void set_dag
    ( FFGraph<ExtOps...>* dag )
    { assert( dag );
      _dag = dag; }

  //! @brief Get number of model outputs
  unsigned ny
    ()
    const
    { return _ny; }

  //! @brief Get number of experimental controls
  unsigned nc
    ()
    const
    { return _nc; }

  //! @brief Get number of model parameters
  unsigned np
    ()
    const
    { return _np; }

  //! @brief Set model outputs
  void set_model
    ( unsigned const ny, FFVar const* Y, double const* varY=nullptr )
    {
      assert( ny && Y );
      _ny = ny;
      _vOUT.assign( Y, Y+ny );
      if( varY ) _vOUTVAR.assign( varY, varY+ny );
      _ivpode = nullptr;
    }

  //! @brief Set model outputs
  void set_model
    ( mc::ODESLVS_CVODES<ExtOps...>* ivpode, double const* varY=nullptr )
    {
      assert( ivpode );
      _ivpode = ivpode;
      _dag = ivpode->dag();
      if( varY ) _vOUTVAR.assign( varY, varY+ivpode->nf() );
      _ny = 0;
      _vOUT.clear();
    }

  //! @brief Set nominal model parameters
  void set_parameters
    ( unsigned const np, FFVar const* P, double const* valP, double const* scaP=nullptr )
    {
      assert( np && P && valP );
      _np = np;
      _vPAR.assign( P, P+np );
      _vPARVAL.clear();
      _vPARVAL.push_back( std::vector<double>( valP, valP+np ) );
      _vPARWEI.assign( 1, 1. );
      if( scaP ) _vPARSCA.assign( scaP, scaP+np );
      else       _vPARSCA.clear();
    }

  //! @brief Set list of model parameters
  void set_parameters
    ( unsigned const np, FFVar const* P, std::list<double const*> const& l_valP, double const* scaP=nullptr )
    {
      assert( np && P && !l_valP.empty() );
      _np = np;
      _vPAR.assign( P, P+np );
      _vPARVAL.clear();
      for( auto const& valP : l_valP )
        _vPARVAL.push_back( std::vector<double>( valP, valP+np ) );
      _vPARWEI.assign( _vPARVAL.size(), 1/(double)l_valP.size() ); // equal frequencies
      if( scaP ) _vPARSCA.assign( scaP, scaP+np );
      else       _vPARSCA.clear();
    }

  //! @brief Set list of model parameters
  void set_parameters
    ( unsigned const np, FFVar const* P, std::list<std::pair<double const*,double>> const& l_valP, double const* scaP=nullptr )
    {
      assert( np && P && !l_valP.empty() );
      _np = np;
      _vPAR.assign( P, P+np );
      _vPARVAL.clear();
      _vPARWEI.clear();
      double prTot = 0.;
      for( auto const& [valP,prP] : l_valP )
        prTot += prP;
      for( auto const& [valP,prP] : l_valP ){
        _vPARVAL.push_back( std::vector<double>( valP, valP+np ) );
        _vPARWEI.push_back( prP/prTot );
      }
      if( scaP ) _vPARSCA.assign( scaP, scaP+np );
      else       _vPARSCA.clear();
    }

  //! @brief Set experimental controls
  void set_controls
    ( const unsigned nc, const FFVar*C, const double*CLB, const double*CUB )
    {
      assert( nc && C && CLB && CUB );
      _nc = nc;
      _vCON.assign( C, C+nc );
      _vCONLB.assign( CLB, CLB+nc );
      _vCONUB.assign( CUB, CUB+nc );
    }

protected:
  //! @brief Private methods to block default compiler methods
  BASE_MBDOE( BASE_MBDOE<ExtOps...> const& ) = delete;
  BASE_MBDOE<ExtOps...>& operator=( BASE_MBDOE<ExtOps...> const& ) = delete;
};

} // end namescape mc

#endif


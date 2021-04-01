// Copyright (C) 2014 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__BASE_NLP_HPP
#define MC__BASE_NLP_HPP

#include "base_opt.hpp"
#include "base_ae.hpp"

namespace mc
{
//! @brief C++ base class for the definition of nonlinear programs
////////////////////////////////////////////////////////////////////////
//! mc::BASE_NLP is a C++ base class for definition of the variables,
//! objective and constraints participating in nonlinear programs.
////////////////////////////////////////////////////////////////////////
class BASE_NLP:
  public virtual BASE_OPT,
  public virtual BASE_AE
{
public:
  //! @brief Class constructor
  BASE_NLP()
    : BASE_OPT(), BASE_AE()
    {}

  //! @brief Class destructor
  virtual ~BASE_NLP()
    {}

  //! @brief Get constraints
  const std::tuple< std::vector<t_CTR>, std::vector<FFVar>, std::vector<FFVar>, std::vector<bool> >& ctr() const
    { return _ctr; }

  //! @brief Reset constraints
  void reset_ctr()
    { std::get<0>(_ctr).clear(); std::get<1>(_ctr).clear(); std::get<2>(_ctr).clear(); std::get<3>(_ctr).clear(); }

  //! @brief Add constraint
  void add_ctr
    ( t_CTR const type, FFVar const& ctr, bool const is_redundant=false )
    { std::get<0>(_ctr).push_back( type );
      std::get<1>(_ctr).push_back( ctr );
      std::get<2>(_ctr).push_back( FFVar( ctr.dag() ) );
      std::get<3>(_ctr).push_back( is_redundant ); }

  //! @brief Get objective
  std::tuple< std::vector<t_OBJ>, std::vector<FFVar>, std::vector<FFVar> > const& obj() const
    { return _obj; }

  //! @brief Set objective
  void set_obj
    ( const t_OBJ type, const FFVar&obj )
    { std::get<0>(_obj).clear(); std::get<0>(_obj).push_back( type );
      std::get<1>(_obj).clear(); std::get<1>(_obj).push_back( obj );
      std::get<2>(_obj).clear(); 
      if( obj.dag() != nullptr )
        std::get<2>(_obj).push_back( FFVar( obj.dag() ) );
      else
        std::get<2>(_obj).push_back( FFVar( 0. ) );
    }
    
  //! @brief Copy equations
  void set
    ( BASE_NLP const& nlp )
    { BASE_AE::set(nlp); _ctr = nlp._ctr; _obj = nlp._obj; }

protected:
  //! @brief constraints (types, constraint variables, constraint multipliers)
  std::tuple< std::vector<t_CTR>, std::vector<FFVar>, std::vector<FFVar>, std::vector<bool> > _ctr;

  //! @brief objective (type, cost variable, cost multiplier)
  std::tuple< std::vector<t_OBJ>, std::vector<FFVar>, std::vector<FFVar> > _obj;

  //! @brief constraints (types, NCO variables), including dependent equations
  std::tuple< std::vector<t_CTR>, std::vector<FFVar> > _nco;

  //! @brief Get 1st-order necessary conditions for optimality (NCO)
  std::tuple< std::vector<t_CTR>, std::vector<FFVar> > const& nco
    ()
    const
    { return _nco; }

  //! @brief Reset NCO
  void reset_nco
    ()
    { std::get<0>(_nco).clear(); std::get<1>(_nco).clear(); }

  //! @brief Define NCO w.r.t. continuous variables only
  bool set_nco
    ( unsigned const* tvar=nullptr, bool const BADIFF=true );

  //! @brief Private methods to block default compiler methods
  BASE_NLP( BASE_NLP const& );
  BASE_NLP& operator=( BASE_NLP const& );
};

inline bool
BASE_NLP::set_nco
( const unsigned*tvar, bool const BADIFF )
{
  reset_nco();
  if( !std::get<0>(_obj).size() )
    return false;

  // Expressions of Lagrangian function and multiplier scaling condition
  FFVar lagr = 0., scal = -1.;
  switch( std::get<0>(_obj)[0] ){
   case BASE_OPT::MIN:
    lagr += std::get<2>(_obj)[0] * std::get<1>(_obj)[0];
    scal += std::get<2>(_obj)[0];
    break;
   case BASE_OPT::MAX:
    lagr -= std::get<2>(_obj)[0] * std::get<1>(_obj)[0];
    scal += std::get<2>(_obj)[0];
    break;
  }
  for( unsigned ic=0; ic<std::get<0>(_ctr).size(); ++ic ){
    switch( std::get<0>(_ctr)[ic] ){
     case BASE_OPT::EQ:
      lagr += std::get<2>(_ctr)[ic] * std::get<1>(_ctr)[ic];
      scal += sqr( std::get<2>(_ctr)[ic] );
      break;
     case BASE_OPT::LE:
      lagr += std::get<2>(_ctr)[ic] * std::get<1>(_ctr)[ic];
      scal += std::get<2>(_ctr)[ic];
      break;
     case BASE_OPT::GE:
      lagr -= std::get<2>(_ctr)[ic] * std::get<1>(_ctr)[ic];
      scal += std::get<2>(_ctr)[ic];
      break;
    }
  }
  for( unsigned ip=0; ip<_var.size(); ip++ ){
    // Only accout for finite bounds on continuous variables in Lagrangian function
    if( tvar && tvar[ip] ) continue;
    if( _varub[ip] < BASE_OPT::INF ){
      lagr += _varum[ip] * _var[ip];
      scal += _varum[ip];
    }
    if( _varlb[ip] > -BASE_OPT::INF ){
      lagr -= _varlm[ip] * _var[ip];
      scal += _varlm[ip];
    }
  }
  for( unsigned ip=0; ip<_dep.size(); ip++ ){
    // Only accout for finite bounds on continuous dependents in Lagrangian function
    if( tvar && tvar[_var.size()+ip] ) continue;
    if( _depub[ip] < BASE_OPT::INF ){
      lagr += _depum[ip] * _dep[ip];
      scal += _depum[ip];
    }
    if( _deplb[ip] > -BASE_OPT::INF ){
      lagr -= _deplm[ip] * _dep[ip];
      scal += _deplm[ip];
    }
  }
  for( unsigned ie=0; ie<_sys.size(); ++ie ){
    lagr += _sys[ie] * _sysm[ie];
    scal += sqr( _sysm[ie] );
  }
  
  // Multipliers normalization (all multipliers between [0,1])
  std::get<0>(_nco).push_back( BASE_OPT::EQ );
  std::get<1>(_nco).push_back( scal );
#ifdef MC__BASE_NLP__DEBUG
  std::cout << "scaling:";
  _dag->output( _dag->subgraph( 1, &scal ) );
#endif

  // Lagrangian stationarity conditions (w.r.t. continuous variables only)
  std::vector<FFVar> vPCNT;
  for( unsigned ip=0; ip<_var.size(); ip++ ){
    if( tvar && tvar[ip] ) continue;
    vPCNT.push_back( _var[ip] );
  }
  for( unsigned ip=0; ip<_dep.size(); ip++ ){
    if( tvar && tvar[_var.size()+ip] ) continue;
    vPCNT.push_back( _dep[ip] );
  }
  const FFVar* dlagr = nullptr;
  switch( BADIFF ){
   case false: // Forward differentiation
    dlagr = _dag->FAD( 1, &lagr, vPCNT.size(), vPCNT.data() );
    break;
   case true:  // Backward differentiation
    dlagr = _dag->BAD( 1, &lagr, vPCNT.size(), vPCNT.data() );
    break;
  }
  for( unsigned ip=0; ip<vPCNT.size(); ip++ ){
    std::get<0>(_nco).push_back( BASE_OPT::EQ );
    std::get<1>(_nco).push_back( dlagr[ip] );
#ifdef MC__BASE_NLP__DEBUG
    std::cout << "dLagr/d" << vPCNT[ip] << ":";
    _dag->output( _dag->subgraph( 1, dlagr+ip ) );
#endif
  }
  delete[] dlagr;

  // Complementarity slackness conditions (continuous parameters only)
  // TO DO: ONLY ENFORCE IF BOUND IS FINITE!!!
  for( unsigned ip=0; ip<_var.size(); ip++ ){
    if( tvar && tvar[ip] ) continue;
    if( _varub[ip] < BASE_OPT::INF ){
      std::get<0>(_nco).push_back( BASE_OPT::EQ );
      std::get<1>(_nco).push_back( _varum[ip] * ( _var[ip] - _varub[ip] ) );
    }
    if( _varlb[ip] > -BASE_OPT::INF ){
      std::get<0>(_nco).push_back( BASE_OPT::EQ );
      std::get<1>(_nco).push_back( _varlm[ip] * ( _var[ip] - _varlb[ip] ) );
    }
  }
  for( unsigned ip=0; ip<_dep.size(); ip++ ){
    if( tvar && tvar[_var.size()+ip] ) continue;
    if( _varub[ip] < BASE_OPT::INF ){
      std::get<0>(_nco).push_back( BASE_OPT::EQ );
      std::get<1>(_nco).push_back( _depum[ip] * ( _dep[ip] - _depub[ip] ) );
    }
    if( _varlb[ip] > -BASE_OPT::INF ){
      std::get<0>(_nco).push_back( BASE_OPT::EQ );
      std::get<1>(_nco).push_back( _deplm[ip] * ( _dep[ip] - _deplb[ip] ) );
    }
  }
  for( unsigned ic=0; ic<std::get<0>(_ctr).size(); ++ic ){
    switch( std::get<0>(_ctr)[ic] ){
     case BASE_OPT::EQ:
      break;
     case BASE_OPT::LE:
     case BASE_OPT::GE:
      std::get<0>(_nco).push_back( BASE_OPT::EQ );
      std::get<1>(_nco).push_back( std::get<2>(_ctr)[ic] * std::get<1>(_ctr)[ic] );
      break;
    }
  }
  return true;
}

} // end namescape mc

#endif


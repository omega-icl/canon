// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__BASE_NLP_HPP
#define MC__BASE_NLP_HPP

#include <assert.h>
#include "ffunc.hpp"
#include "base_opt.hpp"


namespace mc
{
//! @brief C++ base class for the definition of nonlinear programs
////////////////////////////////////////////////////////////////////////
//! mc::BASE_NLP is a C++ base class for definition of the variables,
//! objective and constraints participating in nonlinear programs.
////////////////////////////////////////////////////////////////////////
template <typename... ExtOps>
class BASE_NLP
: public virtual BASE_OPT
{
protected:
  //! @brief pointer to DAG of equation
  FFGraph<ExtOps...>*    _dag;

  //! @brief parameters
  std::vector<FFVar>     _par;

  //! @brief decision variables
  std::vector<FFVar>     _var;

  //! @brief variable lower bounds
  std::vector<double>    _varlb;

  //! @brief variable upper bounds
  std::vector<double>    _varub;

  //! @brief variable lower bound multipliers
  std::vector<FFVar>     _varlm;

  //! @brief variable upper bound multipliers
  std::vector<FFVar>     _varum;

  //! @brief variable types
  std::vector<unsigned>  _vartyp;

public:
  //! @brief Class constructor
  BASE_NLP
    ()
    : BASE_OPT(),
      _dag( nullptr )
    {}

  //! @brief Class destructor
  virtual ~BASE_NLP
    ()
    {}

  //! @brief Get pointer to DAG
  FFGraph<ExtOps...>* dag
    ()
    const
    { return _dag; }

  //! @brief Set pointer to DAG
  void set_dag
    ( FFGraph<ExtOps...>* dag )
    { _dag = dag; }

  //! @brief Get parameters
  std::vector<FFVar> const& par
    ()
    const
    { return _par; }

  //! @brief Set parameters
  void set_par
    ( std::vector<FFVar> const& par, std::vector<double> const& val=std::vector<double>() )
    { _par = par;
      for( unsigned i=0; i<val.size() && i<_par.size(); i++ ){
        _par[i].set( val[i] );
      }
    }

  //! @brief Add parameters
  void add_par
    ( std::vector<FFVar> const& par, std::vector<double> const& val=std::vector<double>() )
    { _par.insert( _par.end(), par.begin(), par.end() );
      for( unsigned i=0; i<val.size() && i<par.size(); i++ ){
        _par[_par.size()-par.size()+i].set( val[i] );
      }
    }

  //! @brief Set parameters
  void set_par
    ( unsigned const npar, FFVar const* par, double const* val=nullptr )
    { _par.assign( par, par+npar );
      for( unsigned i=0; val && i<_par.size(); i++ ){
        _par[i].set( val[i] );
      }
    }

  //! @brief Add parameters
  void add_par
    ( unsigned const npar, FFVar const* par, double const* val=nullptr )
    { _par.insert( _par.end(), par, par+npar );
      for( unsigned i=0; val && i<npar; i++ ){
        _par[_par.size()-npar+i].set( val[i] );
      }
    }

  //! @brief Set parameters
  void set_par
    ( FFVar const& par )
    { _par.assign( &par, &par+1 );
    }

  //! @brief Set parameters
  void set_par
    ( FFVar const& par, double const& val )
    { _par.assign( &par, &par+1 );
      _par[0].set( val );
    }

  //! @brief Add parameter
  void add_par
    ( FFVar const& par )
    { _par.push_back( par );
    }

  //! @brief Add parameter
  void add_par
    ( FFVar const& par, double const& val )
    { _par.push_back( par );
      _par.back().set( val );
    }

  //! @brief Reset parameters
  void reset_par
    ()
    { _par.clear();
    }

  //! @brief Get decision variables
  std::vector<FFVar> const& var
    ()
    const
    { return _var; }

  //! @brief Get decision variable types
  std::vector<unsigned> const& vartyp
    ()
    const
    { return _vartyp; }

  //! @brief Get decision variable lower bounds
  std::vector<double> const& varlb
    ()
    const
    { return _varlb; }

  //! @brief Get decision variable upper bounds
  std::vector<double> const& varub
    ()
    const
    { return _varub; }

  //! @brief Set decision variables
  void set_var
    ( std::vector<FFVar> const& var,
      std::vector<double> const& lb=std::vector<double>(),
      std::vector<double> const& ub=std::vector<double>(),
      std::vector<unsigned> const& typ=std::vector<unsigned>() )
    { _var = var;
      _varlb  = lb;
      _varub  = ub;
      _vartyp = typ;
      if( _varlb.size()  < _var.size() ) _varlb.insert( _varlb.end(), _var.size()-_varlb.size(), -INF );
      if( _varub.size()  < _var.size() ) _varub.insert( _varub.end(), _var.size()-_varub.size(),  INF );
      if( _vartyp.size() < _var.size() ) _vartyp.insert( _vartyp.end(), _var.size()-_vartyp.size(), 0 );
      _varlm.clear();
      _varum.clear();
      for( unsigned i=0; i<_var.size(); i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Set decision variables
  void set_var
    ( std::vector<FFVar> const& var, double const& lb=-INF, double const& ub=INF, unsigned const typ=0 )
    { _var = var;
      _varlb.assign(  var.size(), lb  );
      _varub.assign(  var.size(), ub  );
      _vartyp.assign( var.size(), typ );
      _varlm.clear();
      _varum.clear();
      for( unsigned i=0; i<var.size(); i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Add decision variables
  void add_var
    ( std::vector<FFVar> const& var,
      std::vector<double> const& lb=std::vector<double>(),
      std::vector<double> const& ub=std::vector<double>(),
      std::vector<unsigned> const& typ=std::vector<unsigned>() )
    { _var.insert( _var.end(), var.begin(), var.end() );
      _varlb.insert( _varlb.end(), lb.begin(), lb.end() );
      _varub.insert( _varub.end(), ub.begin(), ub.end() );
      _vartyp.insert( _vartyp.end(), typ.begin(), typ.end() );
      if( _varlb.size()  < _var.size() ) _varlb.insert( _varlb.end(), _var.size()-_varlb.size(), -INF );
      if( _varub.size()  < _var.size() ) _varub.insert( _varub.end(), _var.size()-_varub.size(),  INF );
      if( _vartyp.size() < _var.size() ) _vartyp.insert( _vartyp.end(), _var.size()-_vartyp.size(), 0 );
      for( unsigned i=0; i<var.size(); i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Add decision variables
  void add_var
    ( std::vector<FFVar> const& var, double const& lb=-INF, double const& ub=INF, unsigned const typ=0 )
    { _var.insert( _var.end(), var.begin(), var.end() );
      _varlb.insert( _varlb.end(), var.size(), lb );
      _varub.insert( _varub.end(), var.size(), ub );
      _vartyp.insert( _vartyp.end(), var.size(), typ );
      for( unsigned i=0; i<var.size(); i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Set decision variables
  void set_var
    ( unsigned const nvar, FFVar const* var, double const* lb, double const* ub=nullptr, const unsigned* typ=nullptr )
    { _var.assign( var, var+nvar );
      if( lb )  _varlb.assign( lb, lb+nvar );
      else      _varlb.assign( nvar,  -INF );
      if( ub )  _varub.assign( ub, ub+nvar );
      else      _varub.assign( nvar,   INF );
      if( typ ) _vartyp.assign( typ, typ+nvar );
      else      _vartyp.assign( nvar,       0 );
      _varlm.clear();
      _varum.clear();
      for( unsigned i=0; i<nvar; i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Set decision variables
  void set_var
    ( unsigned const nvar, FFVar const* var, double const& lb=-INF, double const& ub=INF, unsigned const typ=0 )
    { _var.assign( var, var+nvar );
      _varlb.assign( nvar, lb );
      _varub.assign( nvar, ub );
      _vartyp.assign( nvar, typ );
      _varlm.clear();
      _varum.clear();
      for( unsigned i=0; i<nvar; i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Add decision variables
  void add_var
    ( unsigned const nvar, FFVar const* var, double const* lb, double const* ub=nullptr, const unsigned* typ=nullptr )
    { _var.insert( _var.end(), var, var+nvar );
      if( lb ) _varlb.insert( _varlb.end(), lb, lb+nvar );
      else     _varlb.insert( _varlb.end(), nvar, -INF  );
      if( ub ) _varub.insert( _varub.end(), ub, ub+nvar );
      else     _varub.insert( _varub.end(), nvar,  INF  );
      if( typ ) _vartyp.insert( _vartyp.end(), typ, typ+nvar );
      else      _vartyp.insert( _vartyp.end(), nvar,       0 );
      for( unsigned i=0; i<nvar; i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Add decision variables
  void add_var
    ( unsigned const nvar, FFVar const* var, double const& lb=-INF, double const& ub=INF, unsigned const typ=0 )
    { _var.insert( _var.end(), var, var+nvar );
      _varlb.insert( _varlb.end(), nvar, lb );
      _varub.insert( _varub.end(), nvar, ub );
      _vartyp.insert( _vartyp.end(), nvar, typ );
      for( unsigned i=0; i<nvar; i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Add decision variable
  void add_var
    ( FFVar const& var, double const& lb=-INF, double const& ub=INF, unsigned const typ=0 )
    { _var.push_back( var );
      _varlb.push_back( lb );
      _varub.push_back( ub );
      _vartyp.push_back( typ );
      _varlm.push_back( FFVar( _dag ) );
      _varum.push_back( FFVar( _dag ) );
    }

  //! @brief Reset decision variables
  void reset_var
    ()
    { _var.clear();
      _varlm.clear();
      _varum.clear();
      _varlb.clear();
      _varub.clear();
      _vartyp.clear();
    }

  //! @brief Update decision variable types
  void update_vartyp
    ( unsigned const typ=0 )
    { _vartyp.assign( _vartyp.size(), typ ); }

  //! @brief Get constraints
  std::tuple< std::vector<t_CTR>, std::vector<FFVar>, std::vector<FFVar>, std::vector<bool> > const& ctr() const
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
    ( t_OBJ const type, FFVar const& obj )
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
    { _dag = nlp._dag; //std::cout << "DAG: " << nlp._dag << std::endl;
      _var = nlp._var; _vartyp = nlp._vartyp;
      _varlb = nlp._varlb; _varub = nlp._varub;
      _varlm = nlp._varlm; _varum = nlp._varum;
      _ctr = nlp._ctr; _obj = nlp._obj; }

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
  BASE_NLP( BASE_NLP<ExtOps...> const& );
  BASE_NLP<ExtOps...>& operator=( BASE_NLP<ExtOps...> const& );
};

template <typename... ExtOps>
inline bool
BASE_NLP<ExtOps...>::set_nco
( unsigned const* tvar, bool const BADIFF )
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
#ifdef MC__BASE_NLP__DEBUG
  std::cout << "_var.size = " << _var.size() << ", _varlm.size = " << _varlm.size() << std::endl;
#endif
  assert(_var.size() == _varlm.size() );
  assert(_var.size() == _varum.size() );
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
  FFVar const* dlagr = nullptr;
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


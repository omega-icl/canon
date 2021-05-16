// Copyright (C) 2014 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__BASE_AE_HPP
#define MC__BASE_AE_HPP

#include <assert.h>
#include "ffunc.hpp"
#include "base_opt.hpp"

namespace mc
{
//! @brief C++ base class for defining of parametric nonlinear equations
////////////////////////////////////////////////////////////////////////
//! mc::BASE_AE is a C++ base class for defining parametric algebraic
//! equations, distinguishing between independent and dependent
//! variables
////////////////////////////////////////////////////////////////////////
class BASE_AE:
  protected virtual BASE_OPT
{
protected:
  //! @brief pointer to DAG of equation
  FFGraph* _dag;

  //! @brief parameters
  std::vector<FFVar> _par;

  //! @brief decision variables
  std::vector<FFVar> _var;

  //! @brief dependent variables
  std::vector<FFVar> _dep;

  //! @brief equation system
  std::vector<FFVar> _sys;

  //! @brief variable lower bounds
  std::vector<double> _varlb;

  //! @brief variable upper bounds
  std::vector<double> _varub;

  //! @brief variable lower bound multipliers
  std::vector<FFVar> _varlm;

  //! @brief variable upper bound multipliers
  std::vector<FFVar> _varum;

  //! @brief variable types
  std::vector<unsigned> _vartyp;

  //! @brief dependent lower bounds
  std::vector<double> _deplb;

  //! @brief dependent upper bounds
  std::vector<double> _depub;

  //! @brief dependent lower bound multipliers
  std::vector<FFVar> _deplm;

  //! @brief dependent upper bound multipliers
  std::vector<FFVar> _depum;

  //! @brief equation multipliers
  std::vector<FFVar> _sysm;

  //! @brief Whether the system equations have changed
  bool _newsys;

  //! @brief number of AE block
  unsigned _noblk;

  //! @brief starting position of AE blocks
  std::vector<unsigned> _pblk;

  //! @brief size of AE blocks
  std::vector<unsigned> _nblk;

  //! @brief Current block
  unsigned _iblk;

  //! @brief Structural singularity of system
  bool _singsys;

  //! @brief Linearity of problem w.r.t *all of* the dependents
  bool _linsys;

  //! @brief Linearity of blocks w.r.t. the block variables
  std::vector<bool> _linblk;

  //! @brief Linearity of dependents in blocks
  std::vector<bool> _lindep;

  //! @brief Lower and upper band width of system
  std::pair<long,long> _bwsys;

  //! @brief Lower and upper band width of system blocks
  std::vector< std::pair<long,long> > _bwblk;

  //! @brief variable indices after possible permutation (forward)
  std::vector<unsigned> _fpdep;

  //! @brief variable indices after possible permutation (reverse)
  std::vector<unsigned> _rpdep;

  //! @brief Perform block decomposition of system
  bool set_block
    ( const bool disp=false, std::ostream&os=std::cout );

  //! @brief Reset block decomposition
  bool reset_block
    ();

public:
  /** @ingroup AEBND
   *  @{
   */

  //! @brief Class constructor
  BASE_AE()
    : _dag(nullptr), _newsys(true), _noblk(0), _singsys(false), _linsys(false)
    {}

  //! @brief Class destructor
  virtual ~BASE_AE()
    {}

  //! @brief Get pointer to DAG
  FFGraph* dag() const
    { return _dag; }

  //! @brief Set pointer to DAG
  void set_dag
    ( FFGraph*dag )
    { _dag = dag; }

  //! @brief Get parameters
  std::vector<FFVar> const& par() const
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
    ( unsigned const npar, FFVar const* par, double const* val=0 )
    { _par.assign( par, par+npar );
      for( unsigned i=0; val && i<_par.size(); i++ ){
        _par[i].set( val[i] );
      }
    }

  //! @brief Add parameters
  void add_par
    ( unsigned const npar, FFVar const* par, double const* val=0 )
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
    ( FFVar const& par, double const val )
    { _par.assign( &par, &par+1 );
      _par[0].set( val );
    }

  //! @brief Add parameter
  void add_par
    ( const FFVar&par )
    { _par.push_back( par );
    }

  //! @brief Add parameter
  void add_par
    ( const FFVar&par, const double val )
    { _par.push_back( par );
      _par.back().set( val );
    }

  //! @brief Reset parameters
  void reset_par
    ()
    { //for( auto && p : _par ) p.unset();
      _par.clear();
    }

  //! @brief Get decision variables
  std::vector<FFVar> const& var() const
    { return _var; }

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
    ( std::vector<FFVar> const& var, const double lb=-INF, const double ub=INF, const unsigned typ=0 )
    { _var = var;
      _varlb.assign(  var.size(), lb  );
      _varub.assign(  var.size(), ub  );
      _vartyp.assign( var.size(), typ );
      _varlm.clear();
      _varum.clear();
      for( unsigned i=0; i<_var.size(); i++ ){
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
    ( std::vector<FFVar> const& var, const double lb=-INF, const double ub=INF, const unsigned typ=0 )
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
      for( unsigned i=0; i<_var.size(); i++ ){
        _varlm.push_back( FFVar( _dag ) );
        _varum.push_back( FFVar( _dag ) );
      }
    }

  //! @brief Set decision variables
  void set_var
    ( unsigned const nvar, FFVar const* var, const double lb=-INF, const double ub=INF, const unsigned typ=0 )
    { _var.assign( var, var+nvar );
      _varlb.assign( nvar, lb );
      _varub.assign( nvar, ub );
      _vartyp.assign( nvar, typ );
      _varlm.clear();
      _varum.clear();
      for( unsigned i=0; i<_var.size(); i++ ){
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
    ( unsigned const nvar, FFVar const* var, const double lb=-INF, const double ub=INF, const unsigned typ=0 )
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
    ( const FFVar&var, const double lb=-INF, const double ub=INF, const unsigned typ=0 )
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
    ( const unsigned typ=0 )
    { _vartyp.assign( _vartyp.size(), typ ); }

  //! @brief Get dependent variables
  const std::vector<FFVar>& dep() const
    { return _dep; }

  //! @brief Get equation system
  const std::vector<FFVar>& sys() const
    { return _sys; }

  //! @brief Set dependent variables
  void set_dep
    ( const std::vector<FFVar>&dep, const std::vector<FFVar>&sys,
      std::vector<double> const& lb=std::vector<double>(),
      std::vector<double> const& ub=std::vector<double>() )
    { assert( dep.size()==sys.size() );
      _dep = dep;
      _sys = sys;
      _deplb = lb;
      _depub = ub;
      if( _deplb.size() < _dep.size() ) _deplb.insert( _deplb.end(), _dep.size()-_deplb.size(), -INF );
      if( _depub.size() < _dep.size() ) _depub.insert( _depub.end(), _dep.size()-_depub.size(),  INF );
      _deplm.clear();
      _depum.clear();
      _sysm.clear();
      for( unsigned i=0; i<_dep.size(); i++ ){
        _deplm.push_back( FFVar( _dag ) );
        _depum.push_back( FFVar( _dag ) );
        _sysm.push_back( FFVar( _dag ) );
      }
      _newsys = true;
    }

  //! @brief Set dependent variables
  void set_dep
    ( const unsigned ndep, const FFVar*dep, const FFVar*eq, double const* lb=0, double const* ub=0 )
    { _dep.assign( dep, dep+ndep );
      _sys.assign( eq, eq+ndep );
      if( lb ) _deplb.assign( lb, lb+ndep );
      else     _deplb.assign( ndep, -INF  );
      if( ub ) _depub.assign( ub, ub+ndep );
      else     _depub.assign( ndep,  INF  );
      _deplm.clear();
      _depum.clear();
      _sysm.clear();
      for( unsigned i=0; i<ndep; i++ ){
        _deplm.push_back( FFVar( _dag ) );
        _depum.push_back( FFVar( _dag ) );
        _sysm.push_back( FFVar( _dag ) );
      }
      _newsys = true;
    }

  //! @brief Add dependent variable
  void add_dep
    ( const FFVar&dep, const double lb=-INF, const double ub=INF )
    { _dep.push_back( dep );
      _deplm.push_back( FFVar( _dag ) );
      _depum.push_back( FFVar( _dag ) );
      _deplb.push_back( lb );
      _depub.push_back( ub );
    }
    
  //! @brief Reset dependent variables
  void reset_dep
    ()
    { _dep.clear(); _deplm.clear(); _depum.clear(); _deplb.clear(); _depub.clear(); }

  //! @brief Add algebraic equation
  void add_sys
    ( const FFVar&eq )
    { _sys.push_back( eq );
      _sysm.push_back( FFVar( _dag ) );
      _newsys = true;
    }

  //! @brief Reset algebraic equations
  void reset_sys
    ()
    { _sys.clear(); _sysm.clear(); _newsys = true; }

  //! @brief Copy algebraic system and structure
  void set
    ( BASE_AE const& aes )
    { _dag = aes._dag; //std::cout << "DAG: " << aes._dag << std::endl;
      _var = aes._var; _dep = aes._dep; _sys = aes._sys; _vartyp = aes._vartyp;
      _varlb = aes._varlb; _varub = aes._varub; _varlm = aes._varlm;
      _varum = aes._varum; _deplb  = aes._deplb; _depub = aes._depub;
      _deplm  = aes._deplm; _depum = aes._depum; _sysm  = aes._sysm;
      _newsys = aes._newsys;
      _noblk = aes._noblk; _pblk  = aes._pblk;  _nblk   = aes._nblk;
      _singsys = aes._singsys; _linsys = aes._linsys; _linblk = aes._linblk;
      _lindep = aes._lindep; _bwsys = aes._bwsys; _bwblk = aes._bwblk;
      _fpdep = aes._fpdep; _rpdep = aes._rpdep; }

  //! @brief Number of blocks
  unsigned int noblk
    () const
    { return _noblk; }

  //! @brief Size of block ib
  unsigned int nblk
    ( const unsigned ib ) const
    { return ib<_noblk? _nblk[ib]: 0; }

  //! @brief Current block
  unsigned int iblk
    () const
    { return _iblk; }

  //! @brief Linearity of block ib
  bool linblk
    ( const unsigned ib ) const
    { return ib<_noblk? _linblk[ib]: false; }

  //! @brief Equations in block ib
  FFVar const* eqblk
    ( const unsigned ib ) const
    { return ib<_noblk? _sys.data()+_pblk[ib]: 0; }

  //! @brief Variables in block ib
  FFVar const* depblk
    ( const unsigned ib ) const
    { return ib<_noblk? _dep.data()+_pblk[ib]: 0; }

  //! @brief Linearity of block ib
  bool lindepblk
    ( const unsigned ib, const unsigned j ) const
    { return ib<_noblk && j<_nblk[ib]? _lindep[_pblk[ib]+j]: false; }

  //! @brief Forward permutation of dependent variables
  unsigned int pblk
    ( const unsigned ib ) const
    { return ib<_noblk? _pblk[ib]: 0; }

  //! @brief Forward permutation of dependent variables
  unsigned int fpdep
    ( const unsigned i ) const
    { return i<_dep.size()? _fpdep[i]: 0; }

  //! @brief Reverse permutation of dependent variables
  unsigned int rpdep
    ( const unsigned i ) const
    { return i<_dep.size()? _rpdep[i]: 0; }

  //! @brief Reverse permutation of dependent variables
  unsigned int rpdep
    ( const unsigned ib, const unsigned j ) const
    { return ib<_noblk && j<_nblk[ib]? _rpdep[_pblk[ib]+j]: 0; }
  /** @} */

protected:

  //! @brief Private methods to block default compiler methods
  BASE_AE( BASE_AE const& );
  BASE_AE& operator=( BASE_AE const& );
};

inline bool
BASE_AE::reset_block
()
{
  const unsigned int ndep = _dep.size();
  _noblk = 1;
  _nblk.resize(1); _nblk[0] = ndep;
  _pblk.resize(1); _pblk[0] = 0;
  _fpdep.resize(ndep); _rpdep.resize(ndep);
  for( unsigned int i=0; i<ndep; i++ )
    _fpdep[i] = _rpdep[i] = i;
  return true;
}

inline bool
BASE_AE::set_block
( const bool disp, std::ostream&os )
{
  const unsigned int ndep = _dep.size();
  if( !ndep || _sys.size() != ndep ) return false;
  if( !_newsys ) return true;
  _newsys = false;

  // Perform block lower-triangular decomposition using MC21A/MC13D
  int NB = 1;
  std::vector<int> IPERM(ndep), IOR(ndep), IB(ndep);
  _singsys = !_dag->MC13( ndep, _sys.data(), _dep.data(), IPERM.data(),
    IOR.data(), IB.data(), NB, disp?true:false, os );
  if( _singsys ) return reset_block();

  // Permute order of equation system AND variables in vectors sys and var,
  // now arranged in upper-triangular block form
  // Keep track of forward and reverse permutations in _fpdep and _rpdep
  std::vector<FFVar> sys(ndep), var(ndep);
  _fpdep.resize(ndep); _rpdep.resize(ndep);
  for( unsigned int i=0; i<ndep; i++ ){
    sys[i] = _sys[IPERM[IOR[ndep-i-1]-1]-1];
    var[i] = _dep[IOR[ndep-i-1]-1];
    _fpdep[IOR[i]-1] = ndep-i-1;
    _rpdep[ndep-i-1] = IOR[i]-1;
  }
  _sys = sys;
  _dep = var;

  // Keep track of first row and size of each block in permuted matrix in _pblk and _nblk
  _noblk = NB;
  _nblk.resize(_noblk);
  _pblk.resize(_noblk);
  for( int i=0; i<NB; i++ ){
    _nblk[i] = ( i==NB-1? ndep+1: IB[i+1] ) - IB[i]; 
    _pblk[i] = ndep+1 - IB[i] - _nblk[i];   
  }

  // Systam & block properties (linearity, Jacobian bandwidth)
  var.insert( var.end(), _var.begin(), _var.end() );
  _linblk.resize(_noblk);
  _lindep.resize(ndep);
  _bwblk.resize(_noblk);
  _linsys = true;

  std::vector<FFDep> depsys(ndep), depvar(var.size());
  for( unsigned i=0; i<ndep; i++ ) depvar[i].indep(i);
  //for( unsigned i=0; i<var.size(); i++ ) std::cout << var[i] << ": " << depvar[i] << std::endl;
  _dag->eval( ndep, sys.data(), depsys.data(), var.size(), var.data(),
              depvar.data() );
  _bwsys.first = _bwsys.second = 0;
  for( unsigned i=0; i<ndep; i++ ){
    auto cit = depsys[i].dep().begin();
    for( ; cit != depsys[i].dep().end(); ++cit ){
      if( (*cit).second ) // Detecting overall system linearity
          _linsys = false;
      if( _bwsys.first  < (int)i-(*cit).first ) // Updating lower band width
        _bwsys.first  = (int)i-(*cit).first;
      if( _bwsys.second < (*cit).first-(int)i ) // updating upper band width
        _bwsys.second = (*cit).first-(int)i;
    }
  }

  for( unsigned ib=0; ib<_noblk; ib++ ){
    std::vector<FFDep> depblk(_nblk[ib]), varblk(var.size()-_pblk[ib]);
    for( unsigned i=0; i<_nblk[ib]; i++ ) varblk[i].indep(i);
    _dag->eval( _nblk[ib], sys.data()+_pblk[ib], depblk.data(),
                var.size()-_pblk[ib], var.data()+_pblk[ib], varblk.data() );
    _linblk[ib] = true;
    for( unsigned i=0; i<_nblk[ib]; i++ ) _lindep[_pblk[ib]+i] = true;
    _bwblk[ib].first = _bwblk[ib].second = 0;

    for( unsigned i=0; i<_nblk[ib]; i++ ){
      auto cit = depblk[i].dep().begin();
      for( ; cit != depblk[i].dep().end(); ++cit ){
        if( (*cit).second ) // Detecting block/dependent linearity
          _linblk[ib] = _lindep[_pblk[ib]+(*cit).first] = false;
        if( _bwblk[ib].first  < (int)i-(*cit).first ) // Updating lower band width
          _bwblk[ib].first  = (int)i-(*cit).first;
        if( _bwblk[ib].second < (*cit).first-(int)i ) // updating upper band width
          _bwblk[ib].second = (*cit).first-(int)i;
      }
      //if( !_linblk[ib] ) _linsys = false; // Overall linearity
    }
#ifdef MC__BASE_AE_DEBUG
    std::cout << "BLOCK #" << ib << ": "
              << (_linblk[ib]?"L":"NL") << ", BW "
              << _bwblk[ib].first << "," << _bwblk[ib].second << std::endl;
#endif
  }

  return true;
}

} // end namescape mc

#endif


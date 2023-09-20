// Copyright (C) 2022 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__GAMSWRITER_HPP
#define MC__GAMSWRITER_HPP

#include <stdexcept>
#include <cassert>
#include <fstream>

#include "base_opt.hpp"
#include "polimage.hpp"
#include "ffexpr.hpp"

//#define MC__GAMSWRITER_DEBUG

namespace mc
{
//! @brief C++ class for exporting a CANON reformulated model into GAMS
////////////////////////////////////////////////////////////////////////
//! mc::GAMSWRITER is a C++ class for exporting a reformulated CANON
//! model into GAMS language.
////////////////////////////////////////////////////////////////////////
template< typename T,
          typename... ExtOps >
class GAMSWRITER
: public virtual BASE_OPT
{  
  // Typedef for variable set
  typedef std::map< FFVar const*, std::tuple<unsigned, T const*, double const*>, lt_FFVar > t_DAGVar;
  typedef std::set< PolVar<T> const*, lt_PolVar<T> > t_PolVar;

public:

  //! @brief Model type
  enum MODELTYPE{
     LIN=0,  //!< Linear program
     QUAD,   //!< Quadratically constrained program
     NLIN,   //!< Nonlinear program
  };

protected:

  //! @brief Model type
  MODELTYPE _type;

  //! @brief Objective direction
  t_OBJ _DirObj;

  //! @brief Objective variable
  std::string _VarObj;

  //! @brief Continuous variable declaration
  std::stringstream _CVarDec;

  //! @brief Binary variable declaration
  std::stringstream _BVarDec;

  //! @brief Integer variable declaration
  std::stringstream _IVarDec;

  //! @brief Variable bounds
  std::stringstream _VarBnd;

  //! @brief Variable Fixings
  std::stringstream _VarFix;

  //! @brief Variable levels
  std::stringstream _VarIni;

  //! @brief Equation declaration
  std::stringstream _EqnDec;

  //! @brief Equation definition
  std::stringstream _EqnDef;

  //! @brief Equation counter
  unsigned long _EqnCnt;

  //! @brief set of variables in GAMS model
  t_DAGVar _GAMSdagvar;

  //! @brief vector of variables in GAMS model
  std::vector<FFExpr> _GAMSvar;

  //! @brief vector of functions in GAMS model
  std::vector<FFExpr> _GAMSfun;

  //! @brief set of variables in GAMS model
  t_PolVar _GAMSpolvar;

  //! @brief Polyhedral image environment
  PolBase<T>* _POLenv;

public:

  //! @brief Constructor
  GAMSWRITER
    ()
    : _VarObj( "objvar" ), _EqnCnt( 0 ), _POLenv( nullptr )
    {}

  //! @brief Destructor
  virtual ~GAMSWRITER()
    {}

  //! @brief Reset GAMS model
  void reset
    ();

  //! @brief Write GAMS model to file
  bool write
    ( std::string const filename, bool const optfile=true );

  //! @brief Set variables and cuts in GAMS model
  void set_cuts
    ( PolBase<T>* env, bool const reset_=true );

  //! @brief Set objective variable and direction in GAMS model
  void set_objective
    ( PolVar<T> const& pObj, t_OBJ const& tObj );

  //! @brief Set PolImg variable <a>X</a> with level 
  bool set_variable
    ( PolVar<T> const& X, double const* l );

  //! @brief Set DAG variable <a>X</a> with pointers to lower/upper bounds and level
  void add_variable
    ( FFVar const& Var, unsigned type, T const* Bnd, double const* l );

  //! @brief Set DAG variable <a>X</a> with pointers to lower/upper bounds and level
  void add_variables
    ( unsigned const nVar, FFVar const* Var, unsigned const* type, T const* Bnd, double const* l );

  //! @brief Set DAG function <a>F</a> expressions
  void set_functions
    ( FFGraph<ExtOps...>* pDAG, GAMSWRITER<T,ExtOps...>::MODELTYPE const type, unsigned const nFun,
      FFVar const* Fun, unsigned const nVar, FFVar const* Var );

  //! @brief Set constraints corresponding to DAG functions
  void set_constraints
    ( unsigned const objRow, unsigned const nFun, T const* Fbnd );

  //! @brief Set objective corresponding to DAG functions
  void set_objective
    ( unsigned const objRow, t_OBJ const& tObj );

  //! @brief Real values
  static std::string d2s
    ( double const& c )
    { std::ostringstream ostr;
      ostr << std::setprecision(FFExpr::options.DISPLEN);
      ostr << c;
      return ostr.str(); }


protected:

  //! @brief Append variable to MIP model
  void _add_var
    ( FFVar const* pVar, unsigned type, T const* Bnd, double const* l );

  //! @brief Append variable to MIP model
  void _add_var
    ( PolVar<T> const* pVar );

  //! @brief Append constraint to MIP model
  void _add_cut
    ( PolCut<T> const* pCut );

  //! @brief Append constraint to MIP model
  std::string _lhs_cut
    ( PolCut<T> const* pCut );
    
  //! @brief Write long line with fragments 
  std::string _split_line
    ( std::stringstream& line, unsigned const maxlen=10 );
    
  //! @brief Write long line with breaks 
  std::string _break_line
    ( std::stringstream& line, unsigned const maxlen=79900 );
};

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::reset
()
{
  _type = LIN;
  _EqnCnt = 0;
  _CVarDec.clear(); _CVarDec.str("");
  _BVarDec.clear(); _BVarDec.str("");
  _IVarDec.clear(); _IVarDec.str("");
  _EqnDec.clear();  _EqnDec.str("");
  _EqnDef.clear();  _EqnDef.str(""); _EqnDef << std::setprecision(16);
  _VarBnd.clear();  _VarBnd.str(""); _VarBnd << std::setprecision(16);
  _VarFix.clear();  _VarFix.str(""); _VarFix << std::setprecision(16);
  _VarIni.clear();  _VarIni.str(""); _VarIni << std::setprecision(16);
}

template <typename T, typename... ExtOps>
inline bool
GAMSWRITER<T,ExtOps...>::write
( std::string const filename, bool const optfile )
{
  // Create GAMS file
  std::ofstream GAMSmodel;
  GAMSmodel.open( filename, std::ios::trunc );
  if( !GAMSmodel.is_open() ){
    std::cerr << "Could not create GAMS model. Do you have write permissions in execution directory?"
              << std::endl;
    return false;
  }

  // Determine GAMS model type
  std::string type;
  switch( _type ){
    case LIN  : type = (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MIP"  : "LP") ; break;
    case QUAD : type = (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MIQCP": "QCP"); break;
    case NLIN : type = (_BVarDec.tellp()>0||_IVarDec.tellp()>0? "MINLP": "NLP"); break;
  };

  // Write GAMS model to file
  GAMSmodel << "VARIABLE " << _VarObj << ";" << std::endl << std::endl;
  if( _CVarDec.tellp() > 0 )
    GAMSmodel << "VARIABLES " << _split_line( _CVarDec ) << ";" << std::endl << std::endl;
  if( _BVarDec.tellp() > 0 )
    GAMSmodel << "BINARY VARIABLES " << _split_line( _BVarDec ) << ";" << std::endl << std::endl;
  if( _IVarDec.tellp() > 0 )
    GAMSmodel << "INTEGER VARIABLES " << _split_line( _IVarDec ) << ";" << std::endl << std::endl;
  GAMSmodel << "EQUATIONS " << _split_line( _EqnDec ) << ";" << std::endl << std::endl;
  GAMSmodel << _break_line( _EqnDef ) << std::endl;
  //GAMSmodel << _EqnDef.str() << std::endl;
  if( _VarBnd.tellp() > 0 )
    GAMSmodel << _VarBnd.str() << std::endl;
  if( _VarIni.tellp() > 0 )
    GAMSmodel << _VarIni.str() << std::endl;
  if( _VarFix.tellp() > 0 )
    GAMSmodel << _VarFix.str() << std::endl;
  GAMSmodel << "MODEL canon / ALL /;" << std::endl;
  if( optfile ) GAMSmodel << "canon.OPTFILE = 1;" << std::endl;
  GAMSmodel << "SOLVE canon USING " << type;
  switch( _DirObj ){
    case MIN  : GAMSmodel << " MINIMIZING "; break;
    case MAX  : GAMSmodel << " MAXIMIZING "; break;
  }
  GAMSmodel << _VarObj << ";" << std::endl;

  // Close GAMS file
  GAMSmodel.close();
  return true;
}

template <typename T, typename... ExtOps>
inline std::string
GAMSWRITER<T,ExtOps...>::_split_line
( std::stringstream& line, unsigned const maxlen )
{
  std::stringstream linewithbreaks;
  unsigned pos = 1;
  for( std::string term; std::getline( line, term, ',' ); ++pos ){
    if( pos > 1 )       linewithbreaks << ",";
    if( !(pos%maxlen) ) linewithbreaks << std::endl << "  ";
    linewithbreaks << term;
  }
  return linewithbreaks.str();
}

template <typename T, typename... ExtOps>
inline std::string
GAMSWRITER<T,ExtOps...>::_break_line
( std::stringstream& line, unsigned const maxlen )
{
  std::stringstream linewithbreaks;
  std::string::size_type len = 0;
  std::string fragment;
  while( line >> fragment ){
    len += fragment.size();
    if( fragment.back() == ';' ){
      linewithbreaks << fragment << std::endl;
      len = 0;
      continue;    
    }
    if( len + fragment.size() > maxlen && fragment.front() != '*' ){
      linewithbreaks << std::endl;
      len = 0;
    }
    linewithbreaks << fragment << ' ';
    len += fragment.size()+1;
  }

  return linewithbreaks.str();
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::_add_var
( FFVar const* pVar, unsigned type, T const* Bnd, double const* l )
{
  assert( pVar );

  // Case constant variable
  if( pVar->cst() ){
    _CVarDec << (_CVarDec.tellp()>0?", ":" ") << pVar->name();
    _VarFix  << pVar->name() << ".FX = " << d2s(pVar->num().val()) << ";" << std::endl;
  }

  else{
    switch( type ){
      case 0:
        _CVarDec << (_CVarDec.tellp()>0?", ":" ") << pVar->name();
        if( Bnd && Op<T>::l(*Bnd) == Op<T>::u(*Bnd) )
          _VarBnd << pVar->name() << ".FX = " << d2s(Op<T>::l(*Bnd)) << ";" << std::endl;
	else{
          if( Bnd && Op<T>::l(*Bnd) > -0.999*BASE_OPT::INF )
            _VarBnd << pVar->name() << ".LO = " << d2s(Op<T>::l(*Bnd)) << ";" << std::endl;
          if( Bnd && Op<T>::u(*Bnd) <  0.999*BASE_OPT::INF )
            _VarBnd << pVar->name() << ".UP = " << d2s(Op<T>::u(*Bnd)) << ";" << std::endl;
        }
	break;
	
      case 1:
        if( !Bnd || Op<T>::l(*Bnd) < -0.999*BASE_OPT::INF || Op<T>::u(*Bnd) > 0.999*BASE_OPT::INF )
	  throw std::runtime_error("GAMSWRITER - Error: Discrete variable must be bounded");
        if( Op<T>::l(*Bnd) > -1. && Op<T>::u(*Bnd) < 2. )
          _BVarDec << (_BVarDec.tellp()>0?", ":" ") << pVar->name();
        else{
          _IVarDec << (_IVarDec.tellp()>0?", ":" ") << pVar->name();
          if( !isequal( Op<T>::l(*Bnd), 0. ) )
            _VarBnd << pVar->name() << ".LO = " << d2s(std::ceil(Op<T>::l(*Bnd))) << ";" << std::endl;
          if( !isequal( Op<T>::u(*Bnd), 100. ) )
            _VarBnd << pVar->name() << ".UP = " << d2s(std::floor(Op<T>::u(*Bnd))) << ";" << std::endl;
        }
        break;

      default:
        throw std::runtime_error("GAMSWRITER - Error: Unsupported variable type");
    }
    
    if( l ) _VarIni << pVar->name() << ".L = " << *l << ";" << std::endl;
  }
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::add_variable
( FFVar const& Var, unsigned type, T const* Bnd, double const* l )
{
  auto itv = _GAMSdagvar.find( const_cast<FFVar*>(&Var) );
  if( itv != _GAMSdagvar.end() )
    throw std::runtime_error("GAMSWRITER - Error: Cannot redefine DAG variable");
  _GAMSdagvar[&Var] = std::make_tuple( type, Bnd, l );
  _add_var( &Var, type, Bnd, l );
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::add_variables
( unsigned const nVar, FFVar const* Var, unsigned const* type, T const* Bnd, double const* l )
{
  for( unsigned i=0; i<nVar; ++i )
    add_variable( Var[i], type[i], (Bnd? &Bnd[i]: nullptr), (l? &l[i]: nullptr) );
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::set_functions
( FFGraph<ExtOps...>* pDAG, GAMSWRITER<T,ExtOps...>::MODELTYPE const type, unsigned const nFun,
  FFVar const* Fun, unsigned const nVar, FFVar const* Var )
{
  _type = type;
  _GAMSfun.resize( nFun );
  _GAMSvar.resize( nVar );
  FFExpr::options.LANG = FFExpr::Options::GAMS;
  for( unsigned int i=0; i<nVar; i++ ) _GAMSvar[i].set( Var[i] );
  pDAG->eval( nFun, Fun, _GAMSfun.data(), nVar, Var, _GAMSvar.data() );

#ifdef MC__GAMSWRITER_DEBUG
  for( unsigned int i=0; i<nFun; i++ ){
    pDAG->output( pDAG->subgraph( 1, &Fun[i] ) );    
    std::cout << "Expression of F[" << i << "]: " << _GAMSfun[i] << std::endl;
  }
#endif
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::set_constraints
( unsigned const objRow, unsigned const nFun, T const* Fbnd )
{
  assert( Fbnd );
  for( unsigned i=0; i<nFun; i++ ){
    if( i == objRow ) continue; // Exclude objective row
    if( Op<T>::l(Fbnd[i]) < -0.999*BASE_OPT::INF ){ // lower bound inactive
      assert( Op<T>::u(Fbnd[i]) < 0.999*BASE_OPT::INF );
      _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
      _EqnDef << "E" << _EqnCnt << " .. " << _GAMSfun[i] << " =L= "
              << d2s(Op<T>::u(Fbnd[i])) << ";" << std::endl;
    }
    else if( Op<T>::u(Fbnd[i]) > 0.999*BASE_OPT::INF ){ // upper bound inactive
      assert( Op<T>::l(Fbnd[i]) > -0.999*BASE_OPT::INF );
      _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
      _EqnDef << "E" << _EqnCnt << " .. " << _GAMSfun[i] << " =G= "
              << d2s(Op<T>::l(Fbnd[i])) << ";" << std::endl;
    }
    else if( Op<T>::l(Fbnd[i]) == Op<T>::u(Fbnd[i]) ){ // equality constraints
      _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
      _EqnDef << "E" << _EqnCnt << " .. " << _GAMSfun[i] << " =E= "
              << d2s(Op<T>::l(Fbnd[i])) << ";" << std::endl;
    }
    else{ // two distinct inequality constraints
      _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
      _EqnDef << "E" << _EqnCnt << " .. " << _GAMSfun[i] << " =L= "
              << d2s(Op<T>::u(Fbnd[i])) << ";" << std::endl;
      _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
      _EqnDef << "E" << _EqnCnt << " .. " << _GAMSfun[i] << " =G= "
              << d2s(Op<T>::l(Fbnd[i])) << ";" << std::endl;
    }
  }
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::set_objective
( unsigned const objRow, t_OBJ const& tObj )
{
  _DirObj = tObj;
  _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
  _EqnDef << "E" << _EqnCnt << " .. " << _VarObj << " =E= " << _GAMSfun[objRow] << ";" << std::endl;
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::set_cuts
( PolBase<T>* env, bool const reset_ )
{
  if( reset_ ) reset();

  // Add cuts into GAMS model
  _POLenv = env;
  for( auto && pcut : _POLenv->Cuts() ){
#ifdef MC__GAMSWRITER_DEBUG
    std::cout << *pcut << std::endl;
#endif
    _add_cut( pcut );
#ifdef MC__GAMSWRITER_DEBUG
    std::cout << "VARIABLE " << _VarObj << ";" << std::endl;
    std::cout << "VARIABLES " << _CVarDec.str() << ";" << std::endl << std::endl;
    std::cout << "BINARY VARIABLES " << _BVarDec.str() << ";" << std::endl << std::endl;
    std::cout << "INTEGER VARIABLES " << _IVarDec.str() << ";" << std::endl << std::endl;
    std::cout << "EQUATIONS " << _EqnDec.str() << ";" << std::endl << std::endl;
    std::cout << _EqnDef.str() << std::endl << std::endl;
#endif
  }
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::_add_var
( PolVar<T> const* pVar )
{
  switch( pVar->id().first ){
    case PolVar<T>::AUXCST:
      _CVarDec << (_CVarDec.tellp()>0?", ":" ") << pVar->name();
      _VarFix  << pVar->name() << ".FX = " << d2s(Op<T>::mid(pVar->range())) << ";" << std::endl;
      break;

    case PolVar<T>::VARCONT:
    case PolVar<T>::AUXCONT:
      _CVarDec << (_CVarDec.tellp()>0?", ":" ") << pVar->name();
      if( Op<T>::l(pVar->range()) > -0.999*BASE_OPT::INF )
        _VarBnd << pVar->name() << ".LO = " << d2s(Op<T>::l(pVar->range())) << ";" << std::endl;
      if( Op<T>::u(pVar->range()) <  0.999*BASE_OPT::INF )
        _VarBnd << pVar->name() << ".UP = " << d2s(Op<T>::u(pVar->range())) << ";" << std::endl;
      break;
      
    case PolVar<T>::VARINT:
    case PolVar<T>::AUXINT:
      if( Op<T>::l(pVar->range()) > -1. && Op<T>::u(pVar->range()) < 2. )
        _BVarDec << (_BVarDec.tellp()>0?", ":" ") << pVar->name();
      else{
        _IVarDec << (_IVarDec.tellp()>0?", ":" ") << pVar->name();
        if( !isequal( Op<T>::l(pVar->range()), 0. ) )
          _VarBnd << pVar->name() << ".LO = " << d2s(Op<T>::l(pVar->range())) << ";" << std::endl;
        if( !isequal( Op<T>::u(pVar->range()), 100. ) )
          _VarBnd << pVar->name() << ".UP = " << d2s(Op<T>::u(pVar->range())) << ";" << std::endl;
      }
      break;

    default:
      throw std::runtime_error("GAMSWRITER - Error: Unsupported variable type");
  }

  _GAMSpolvar.insert( pVar );
}

template <typename T, typename... ExtOps>
inline std::string
GAMSWRITER<T,ExtOps...>::_lhs_cut
( PolCut<T> const* pCut )
{
  std::stringstream lhs;

  // Add linear terms to lhs
  for( unsigned k=0; k<pCut->nvar(); k++ ){
    if( pCut->coef()[k] == 0. ) continue;
    // Append term to lhs
    if( pCut->coef()[k] < 0. )
      lhs << " - ";
    else if( lhs.tellp() > 0 && pCut->coef()[k] > 0. )
      lhs << " + ";
    if( std::fabs(pCut->coef()[k]) != 1. )
      lhs << d2s(std::fabs(pCut->coef()[k])) << "*";
    lhs << pCut->var()[k].name();
  }

  // Add quadratic terms to lhs
  for( unsigned k=0; k<pCut->nqvar(); k++ ){
    if( pCut->qcoef()[k] == 0. ) continue;
    // Append term to lhs
    if( pCut->qcoef()[k] < 0. )
      lhs << " - ";
    else if( lhs.tellp() > 0 && pCut->qcoef()[k] > 0. )
      lhs << " + ";
    if( std::fabs(pCut->qcoef()[k]) != 1. )
      lhs << d2s(std::fabs(pCut->qcoef()[k])) << "*";
    if( pCut->qvar1()[k].name() == pCut->qvar2()[k].name() )
      lhs << "SQR(" << pCut->qvar1()[k].name() << ")";
    else
      lhs << pCut->qvar1()[k].name() << "*" << pCut->qvar2()[k].name();
    if( _type < QUAD ) _type = QUAD;
  }
  return lhs.str();
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::_add_cut
( PolCut<T> const* pCut )
{
  // Check valid cut
  if( !pCut->nvar() && !pCut->nqvar() ){
    std::cout << *pCut << std::endl;
    throw std::runtime_error("GAMSWRITER - Error: Invalid cut without participating variables");
  }

  // Declare equation and initiate definition
  _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
  _EqnDef << "E" << _EqnCnt << " .. ";
  
  // Add participating variables to GAMS model
  for( unsigned k=0; k<pCut->nvar(); k++ ){
    if( _GAMSpolvar.find( pCut->var()+k ) == _GAMSpolvar.end() )
      _add_var( pCut->var()+k );
  }
  for( unsigned k=0; k<pCut->nqvar(); k++ ){
    if( _GAMSpolvar.find( pCut->qvar1()+k ) == _GAMSpolvar.end() )
      _add_var( pCut->qvar1()+k );
    if( _GAMSpolvar.find( pCut->qvar2()+k ) == _GAMSpolvar.end() )
      _add_var( pCut->qvar2()+k );
  }

  // Add contraint to MIP model
  switch( pCut->type() ){

    case PolCut<T>::EQ:
      _EqnDef << _lhs_cut( pCut ) << " =E= " << d2s(pCut->rhs()) << ";" << std::endl;
      break;

    case PolCut<T>::LE:
      _EqnDef << _lhs_cut( pCut ) << " =L= " << d2s(pCut->rhs()) << ";" << std::endl;
      break;

    case PolCut<T>::GE:
      _EqnDef << _lhs_cut( pCut ) << " =G= " << d2s(pCut->rhs()) << ";" << std::endl;
      break;

    case PolCut<T>::SOS1:
    case PolCut<T>::SOS2:
      throw std::runtime_error("GAMSWRITER - Error: SOS variable not supported");
        
    case PolCut<T>::NLIN:
      if( pCut->nvar() < 2 || pCut->nvar() > 3 )
        throw std::runtime_error("GAMSWRITER - Error: Incorrect number of variables in nonlinear cut");
      switch( pCut->op()->type ){

        case FFOp::IPOW:
          if( pCut->op()->varin[1]->num().n < 0 )
            _EqnDef << pCut->var()[0].name() << " * POWER(" << pCut->var()[1].name()
                    << "," << -pCut->op()->varin[1]->num().n << ") =E= 1;" << std::endl;
          else if( pCut->op()->varin[1]->num().n == 0 )
            _EqnDef << pCut->var()[0].name() << " =E= 1;" << std::endl;
          else
            _EqnDef << pCut->var()[0].name() << " - POWER(" << pCut->var()[1].name()
                    << "," << pCut->op()->varin[1]->num().n << ") =E= 0;" << std::endl;
          break;

        case FFOp::DPOW:{
          _EqnDef << pCut->var()[0].name() << " - RPOWER(" << pCut->var()[1].name()
                  << "," << d2s(pCut->op()->varin[1]->num().x) << " =E= 0;" << std::endl;
          break;

        case FFOp::CHEB:{
          unsigned const ncoef = pCut->op()->varin[1]->num().n+1;
          std::vector<double>&& coef = chebcoef( ncoef-1 );
          _EqnDef << pCut->var()[0].name() << " - POLY(" << pCut->var()[1].name();
          for( int k=ncoef; k>0; ) _EqnDef << "," << coef[--k];
          for( int k=ncoef; k<3; ++k ) _EqnDef << ",0"; // at least quadratic
          _EqnDef << ") =E= 0;" << std::endl;
          break;}

        case FFOp::SQR:
          _EqnDef << pCut->var()[0].name() << " - SQR(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::SQRT:
          _EqnDef << pCut->var()[0].name() << " - SQRT(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::EXP:
          _EqnDef << pCut->var()[0].name() << " - EXP(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::LOG:
          _EqnDef << pCut->var()[0].name() << " - LOG(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::XLOG:
          _EqnDef << pCut->var()[0].name() << " + ENTROPY(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::COS:
          _EqnDef << pCut->var()[0].name() << " - COS(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::SIN:
          _EqnDef << pCut->var()[0].name() << " - SIN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::TAN:
          _EqnDef << pCut->var()[0].name() << " - TAN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ACOS:
          _EqnDef << pCut->var()[0].name() << " - ARCCOS(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ASIN:
          _EqnDef << pCut->var()[0].name() << " - ARCSIN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ATAN:
          _EqnDef << pCut->var()[0].name() << " - ARCTAN(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::COSH:
          _EqnDef << pCut->var()[0].name() << " - COSH(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::SINH:
          _EqnDef << pCut->var()[0].name() << " - SINH(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::TANH:
          _EqnDef << pCut->var()[0].name() << " - TANH(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::ERF:
          _EqnDef << pCut->var()[0].name() << " - ERRORF(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::FABS:
          _EqnDef << pCut->var()[0].name() << " - ABS(" << pCut->var()[1].name() << ") =E= 0;" << std::endl;
          break;

        case FFOp::MINF:
          _EqnDef << pCut->var()[0].name() << " - MIN(" << pCut->var()[1].name();
          if( pCut->nvar() == 2 )
            _EqnDef << "," << pCut->rhs();
          else
            for( unsigned k=2; k<pCut->nvar(); ++k )
              _EqnDef << "," << pCut->var()[k].name();
          _EqnDef << ") =E= 0;" << std::endl;
          break;

        case FFOp::MAXF:
          _EqnDef << pCut->var()[0].name() << " - MAX(" << pCut->var()[1].name();
          if( pCut->nvar() == 2 )
            _EqnDef << "," << pCut->rhs();
          else
            for( unsigned k=2; k<pCut->nvar(); ++k )
              _EqnDef << "," << pCut->var()[k].name();
          _EqnDef << ") =E= 0;" << std::endl;
          break;

        default:
          throw std::runtime_error("GAMSWRITER - Error: Nonlinear operand not supported");
      }
    }
    if( _type < NLIN ) _type = NLIN;
  }
}

template <typename T, typename... ExtOps>
inline bool
GAMSWRITER<T,ExtOps...>::set_variable
( PolVar<T> const& polVar, double const* l )
{
  auto itv = _GAMSpolvar.find( const_cast<PolVar<T>*>(&polVar) );
  if( itv == _GAMSpolvar.end() ) return false;
  if( l ) _VarIni << polVar.name() << ".L = " << d2s(*l) << ";" << std::endl;
  return true;
}

template <typename T, typename... ExtOps>
inline void
GAMSWRITER<T,ExtOps...>::set_objective
( PolVar<T> const& polObj, t_OBJ const& tObj )
{
  _DirObj = tObj;
  if( _GAMSpolvar.find( &polObj ) == _GAMSpolvar.end() ) _add_var( &polObj );
  _EqnDec << (_EqnDec.tellp()>0?", ":" ") << "E" << ++_EqnCnt;
  _EqnDef << "E" << _EqnCnt << " .. " << _VarObj << " =E= " << polObj.name() << ";" << std::endl;
}

} // end namespace mc

#endif

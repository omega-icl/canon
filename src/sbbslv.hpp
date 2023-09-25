// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__SBBSLV_HPP
#define MC__SBBSLV_HPP

#include <utility>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <chrono>

#include "mcop.hpp"
#include "base_opt.hpp"

#undef DEBUG__SBBSLV_PSEUDOCOST
#undef DEBUG__SBBSLV_STRONGBRANCHING
#undef DEBUG__SBBSLV_BRANCHINGVAR
#undef USE__SBBSLV_PSEUDOCOST_FILTER
#undef SAVE__SBBSLV_NODES_TO_FILE

// TO DO:
// - DOCUMENTATION

namespace mc
{

template <typename T> class SBBNode;
template <typename T> struct lt_SBBNode;

//! @brief Pure virtual base class for spatial branch-and-bound search
////////////////////////////////////////////////////////////////////////
//! mc::SBBSLV<T> is a pure virutal base C++ class implementing the
//! spatial branch-and-bound (a.k.a. branch-and-reduce) algorithm
//! for nonconvex, continuous optimization problems.
//!
//! REFERENCES:
//! - Horst, R., and H. Tuy, <A href="http://books.google.com/books?id=usFjGFvuBDEC&lpg=PA39&ots=DmNqBtTNOQ&dq=global%20optimization%20horst%20tuy&pg=PP1#v=onepage&q&f=false"><i>Global Optimization</i></A>, Springer-Verlag, Berlin, 1996.
//! - Ryoo, H.S., and N.V. Sahinidis, <A href="http://dx.doi.org/10.1016/0098-1354(94)00097-8">Global optimization of nonconvex NLPs and MINLPs with application in process design</A>, <i>Computers & Chemical Engineering</i>, <b>19</b>(5):551--566, 1995.
//! - Tawarmalani, M., and N. V. Sahinidis, <A HREF="http://books.google.com/books?id=MjueCVdGZfoC&lpg=PP1&ots=bgaQ0uqGU_&dq=tawarmalani%20sahinidis&pg=PP1#v=onepage&q&f=false">Convexification and Global Optimization in Continuous and Mixed-Integer Nonlinear Programming: Theory, Algorithms, Software, and Applications</A>, Kluwer Academic Publishers, Dordrecht, Vol. 65 in ``Nonconvex Optimization And Its Applications'' series, 2002.
//! .
////////////////////////////////////////////////////////////////////////
template <typename T>
class SBBSLV
: public virtual BASE_OPT
////////////////////////////////////////////////////////////////////////
{
  template <typename U> friend class SBBNode;

public:  
  typedef std::multiset< SBBNode<T>*, lt_SBBNode<T> > t_Nodes;
  typedef typename t_Nodes::iterator it_Nodes;
  template <typename U> friend std::ostream& operator<<
    ( std::ostream&, const SBBNode<U>& );

  //! @brief Status of subproblems
  enum STATUS{
    NORMAL=0,	//!< Normal termination
    INFEASIBLE,	//!< Termination w/ indication of infeasibility
    FAILURE,	//!< Termination after failure
    FATAL	//!< Termination after fatal error
  };

  //! @brief Task for subproblems
  enum TASK{
    LOWERBD=0,	//!< Solve lower bounding subproblem
    POSTPROC,	//!< Postprocess node (e.g. domain reduction)
    PREPROC,	//!< Preprocess node (e.g. constraint propagation)
    UPPERBD,	//!< Solve upper bounding subproblem
    FEASTEST	//!< Test feasibility of upper bounding subproblem
  };

  //! @brief Prototype for user-supplied function
  virtual STATUS subproblems
    ( TASK const,		// task
      SBBNode<T>*,		// pointer to node
      std::vector<double>&,	// optimal solution point - initial guess on entry
      double&, 			// optimal solution value
      double const&,		// current incumbent
      std::ostream& os          // output stream
    )
    = 0;

  //! @brief Public constructor
  SBBSLV()
    : options(), _f_inc(INF), _np(0)
    {}

  //! @brief Class destructor
  virtual ~SBBSLV()
    { _clean_stack(); }

  //! @brief SBBSLV options
  struct Options
  {
    //! @brief Constructor
    Options():
      STOPPING_ABSTOL(1e-3), STOPPING_RELTOL(1e-3), TREE_UPDATE(true),
      BRANCHING_STRATEGY(OMEGA), BRANCHING_BOUND_THRESHOLD(5e-2), 
      BRANCHING_VARIABLE_CRITERION(RGREL), BRANCHING_USERFUNCTION(0),
      BACKOFF_BRANCHING_INTEGER( 0.1 ),
      STRONG_BRANCHING_MAXDEPTH(0), STRONG_BRANCHING_WEIGHT(1./6.),
      SCORE_BRANCHING_USE(false), SCORE_BRANCHING_RELTOL(1e-1),
      SCORE_BRANCHING_ABSTOL(1e-2), SCORE_BRANCHING_MAXSIZE(0),
      DISPLAY_LEVEL(2), MAX_WALLTIME(6e2), MAX_NODES(0)
      {}
    //! @brief Display
    void display
      ( std::ostream&out ) const;
    //! @brief Branching strategy
    enum STRATEGY{
      MIDPOINT=0,	//!< Bisection at mid-point
      OMEGA		//!< Bisection at (i) incumbent location if inside current bounds; otherwise (ii) at lower-bounding-problem solution if not at bound; otherwise (iii) at mid-point
    };
    //! @brief Branching variable criterion
    enum CRITERION{
      RGREL=0,	//!< Relative variable range diameter
      RGABS	//!< Absolute variable range diameter
    };
    //! @brief Branching selection user-function
    typedef std::set<unsigned> (*SELECTION)( const SBBNode<T>* );
    //! @brief Absolute stopping tolerance
    double STOPPING_ABSTOL;
    //! @brief Relative stopping tolerance
    double STOPPING_RELTOL;
    //! @brief Whether the entire B&B tree is to be updated after every incumbent update
    bool TREE_UPDATE;
    //! @brief Branching strategy
    int BRANCHING_STRATEGY;
    //! @brief Relative tolerance within which a variable is considered to be at one of its bounds, i.e. excluded from branching variable selection
    double BRANCHING_BOUND_THRESHOLD;
    //! @brief Variable selection criterion
    int BRANCHING_VARIABLE_CRITERION;
    //! @brief Branching variable selection user-function
    SELECTION BRANCHING_USERFUNCTION;
    //! @brief Backoff applied to bisection point of integer variables when it matches an integer
    double BACKOFF_BRANCHING_INTEGER;
    //! @brief Maximum depth for strong branching interruption
    unsigned STRONG_BRANCHING_MAXDEPTH;
    //! @brief Weighting (between 0 and 1) used to account for the left and right nodes in strong branching
    double STRONG_BRANCHING_WEIGHT;
    //! @brief Whether to base branching variable selection on scores
    bool SCORE_BRANCHING_USE;
    //! @brief Relative tolerance for branching variable selection based on scores
    double SCORE_BRANCHING_RELTOL;
    //! @brief Absolute tolerance for branching variable selection based on scores
    double SCORE_BRANCHING_ABSTOL;
    //! @brief Maximal size of branching variable selection based on scores
    unsigned SCORE_BRANCHING_MAXSIZE;
    //! @brief Display option
    int DISPLAY_LEVEL;
    //! @brief Maximum CPU time limit
    double MAX_WALLTIME;
    //! @brief Maximum number of nodes (0 for no limit)
    unsigned MAX_NODES;
  } options;

  //! @brief SBBSLV exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for SBBSLV exception handling
    enum TYPE{
      BRANCH=0,		//!< Error due to an empty set of branching variables 
      INTERN=-3,	//!< SBBSLV internal error
      UNDEF=-33	//!< Error due to calling a function/feature not yet implemented in SBBSLV
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr()
      { return _ierr; }
    std::string what()
      {
        switch( _ierr ){
        case BRANCH:
          return "SBBSLV Error: empty set of branching variables";
        case INTERN:
          return "SBBSLV Internal Error";
        case UNDEF: default:
          return "SBBSLV Error: calling a feature not yet implemented";
        }
      }
  private:
    TYPE _ierr;
  };

  //! @brief Structure holding solve statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime_all = walltime_ubd = walltime_lbd = std::chrono::microseconds(0);
        node_inc = node_tot = node_max = 0; }
    //! @brief Display statistics
    //void display
    //  ( std::ostream&os=std::cout )
    //  { os << std::fixed << std::setprecision(2) << std::right
    //       << std::endl
    //       << "# WALL-CLOCK TIMES" << std::endl
    //       << "# UBD:   " << std::setw(10) << to_time( walltime_ubd )  << " SEC" << std::endl
    //       << "# LBD:   " << std::setw(10) << to_time( walltime_lbd )  << " SEC" << std::endl
    //       << "# TOTAL: " << std::setw(10) << to_time( walltime_all )  << " SEC" << std::endl
    //       << std::endl; }
    //! @brief Node where current incumbent was found
    unsigned node_inc;
    //! @brief Total number of nodes explored
    unsigned node_tot;
    //! @brief Maximal number of nodes in stack
    unsigned node_max;
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime_all;
    //! @brief Cumulated wall-clock time used for upper bounding (in microseconds)
    std::chrono::microseconds walltime_ubd;
    //! @brief Cumulated wall-clock time used for lower bounding (in microseconds)
    std::chrono::microseconds walltime_lbd;
    //! @brief Get current time point
    std::chrono::time_point<std::chrono::system_clock> now
      ()
      const
      { return std::chrono::system_clock::now(); }
    //! @brief Get current time lapse with respect to start time point
    std::chrono::microseconds walltime
      ( std::chrono::time_point<std::chrono::system_clock> const& start )
      const
      { return std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start ); }    
    //! @brief Convert microsecond ticks to time
    double to_time
      ( std::chrono::microseconds t )
      const
      { return t.count() * 1e-6; }
  } stats;

  //! @brief Apply branch-and-bound search
  STATUS solve
    ( t_OBJ const pb, unsigned const np, T const* P, double const* p0=nullptr, double const* f0=nullptr,
      const unsigned* ptyp=nullptr, std::set<unsigned> const& exclude=std::set<unsigned>(),
      std::ostream& os=std::cout );

  t_OBJ get_problem_type
    ()
    const
    { return _pb; }

  STATUS get_status
    ()
    const
    { return _status; }

  unsigned get_variable_size
    ()
    const
    { return _np; }

  unsigned const* get_variable_type
    ()
    const
    { return _P_type.data(); }
    
  
  std::pair<double, double const*> get_incumbent
    ()
    const
    { return std::make_pair( _f_inc, _p_inc.empty()? 0: _p_inc.data() ); }

protected:  
  //! @brief Problem type 
  t_OBJ _pb;
  //! @brief Exclusion set for branching variables (e.g. linear variables)
  std::set<unsigned> _exclude_vars;
  //! @brief SBBSLV status
  STATUS _status;
  //! @brief Variable values at current incumbent
  std::vector<double> _p_inc;
  //! @brief Current incumbent value
  double _f_inc;

  //! @brief Status of bounding problems
  enum NODESTAT{
    FATHOM=0,		//!< Node should be fathomed
    NOTUPDATED,		//!< Incumbent for node has NOT been updated
    UPDATED,		//!< Incumbent for node HAS been updated
    ABORT		//!< Execution should be aborted
  };
  //! @brief Node type
  enum NODETYPE{
    ROOT=0,	//!< Root node
    LEFT=-1,	//!< Left child node
    RIGHT=1	//!< Right child node
  };

  //! @brief Number of variables in optimization problem
  unsigned _np;
  //! @brief Variable types
  std::vector<unsigned> _P_type;
  //! @brief Variable bounds at root node
  std::vector<T> _P_root;
  //! @brief Default branching variable set
  std::set<unsigned> _branch_set;

  //! @brief Variable bounds (temporary)
  std::vector<T> _P_tmp;
  //! @brief Current node index
  unsigned _node_index;
  //! @brief Current node count
  unsigned _node_count;
  //! @brief Current branching variable
  unsigned _var_branch;
  //! @brief Set of nodes
  t_Nodes _Nodes;

  //! @brief Time point to enable TIMELIMIT option
  std::chrono::time_point<std::chrono::system_clock> _tstart;

  //! @brief Status after relaxed problem solution
  NODESTAT _REL_stat;
  //! @brief Status after original problem solution
  NODESTAT _ORI_stat;

  //! @brief maximum number of values displayed in a row
  static const unsigned _LDISP = 4;
  //! @brief reserved space for integer variable display
  static const unsigned _IPREC = 6;
  //! @brief reserved space for double variable display
  static const unsigned _DPREC = 6;
  //! @brief stringstream for displaying results
  std::ostringstream _odisp;
    
  //! @brief Set variables
  bool _variables
    ( unsigned const np, T const* P, unsigned const* ptyp, std::set<unsigned> const& exclude );
  //! @brief Erase stored nodes
  void _clean_stack
    ();
  //! @brief Reinitialize variables, incumbent, counters, time, etc.
  void _restart
    ( const double*p0, const double*f0 );
  //! @brief Termination test for piecewise-linear relaxation approach 
  bool _interrupted
    ()
    const;

  //! @brief Lower bound given node
  NODESTAT _lower_bound
    ( SBBNode<T>*pNode, const bool relaxed, const bool strongbranching, std::ostream& os );
  //! @brief Upper bound given node
  NODESTAT _upper_bound
    ( SBBNode<T>*pNode, const bool relaxed, const bool strongbranching, std::ostream& os );

  //! @brief Determines whether the relaxation solution point is feasible
  STATUS _feasible_relax
    ( SBBNode<T>*pNode, std::ostream& os );
  //! @brief Determines whether given node can be fathomed by value dominance
  bool _fathom_by_dominance
    ( SBBNode<T>*pNode );
  //! @brief Try and update incumbent
  bool _update_incumbent
    ( const double& fINC, const std::vector<double>&pINC );

  //! @brief Apply preprocessing to given node
  NODESTAT _preprocess
    ( SBBNode<T>*pNode, std::ostream& os );
  //! @brief Apply postprocessing to given node
  NODESTAT _postprocess
    ( SBBNode<T>*pNode, std::ostream& os );
  //! @brief Branch current node and create subdomains
  NODESTAT _branch_node
    ( SBBNode<T>*pNode, std::ostream& os );
  //! @brief Default set of branching variables
  void _branching_variable_set
    ( const SBBNode<T>*pNode );
  //! @brief Subset of branching variables based on scores
  void _branching_score_subset
    ( const SBBNode<T>*pNode );
  //! @brief Selects the branching variable for given node
  std::pair<unsigned, double> _select_branching_variable
    ( SBBNode<T>*pNode, std::ostream& os );
  //! @brief Apply strong branching for branching variable selection
  std::pair<unsigned, double> _strong_branching
    ( SBBNode<T>*pNode, const std::set<unsigned>&set_branch, std::ostream& os );
  //! @brief Score branching variable
  double _score_branching
    ( const double fLEFT, const double fRIGHT ) const;
  //! @brief Partition branching variable domain for given node
  std::pair<const T, const T> _partition_variable_domain
    ( SBBNode<T>*pNode, const unsigned ip ) const;

  //! @brief Test entire SBB tree for fathoming by value dominance
  void _update_tree
    ();
  //! @brief Apply postprocessing to entire SBB tree
  NODESTAT _postprocess_tree
    ( std::ostream& os );

  //! @brief Add information for display
  void _display_add
    ( const double dval );
  void _display_add
    ( const unsigned ival );
  void _display_add
    ( const std::string &sval );

  //! @brief Initialize display
  void _display_init
    ();
  //! @brief Final display
  void _display_final
    ( std::chrono::microseconds const& walltime );

  //! @brief Add walltime to display
  void _display_time
    ();
  //! @brief Add current time to buffer, display current buffer stream and reset it
  void _display_line
    ( std::ostream&os );
  //! @brief Display current buffer stream and reset it
  void _display
    ( std::ostream&os );
  //! @brief Display current nodes in tree
  void _display_tree( std::ostream&os );
  //! @brief Display current box (only 2 and 3 parameters)
  void _display_box
    ( const SBBNode<T>*pNode, std::ostream&os ) const;
};

//! @brief C++ base class for branch-and-bound nodes
////////////////////////////////////////////////////////////////////////
//! mc::SBBNode<T> is a C++ base class for defining nodes in the spatial
//! branch-and-bound algorithm.
////////////////////////////////////////////////////////////////////////
template <typename T>
class SBBNode
////////////////////////////////////////////////////////////////////////
{
  template <typename U> friend class SBBSLV;
  template <typename U> friend struct lt_SBBNode;
  template <typename U> friend std::ostream& operator<<
    ( std::ostream&, const SBBNode<U>& );

public:  
  //! @brief Const pointer to underlying branch-and-bound tree
  SBBSLV<T> const* sbb() const
    { return _pSBB; };
  //! @brief Retreive node strength (based on parent bound)
  double strength() const
    { return _strength; }
  //! @brief Retreive node index
  unsigned index() const
    { return _index; }
  //! @brief Retreive node depth
  unsigned depth() const
    { return _depth; }
  //! @brief Retreive node iteration
  unsigned iter() const
    { return _iter; }
  //! @brief Retreive/set pointer to user data
  void*& data()
    { return _data; }
  //! @brief Retreive/set node dependent variables
  std::set<unsigned>& depend()
    { return _depend; }
  //! @brief Retreive node dependent variables
  const std::set<unsigned>& depend() const
    { return _depend; }
  //! @brief Retreive/set node branching scores
  std::map<unsigned,double>& scores()
    { return _scores; }
  //! @brief Retreive node branching scores
  const std::map<unsigned,double>& scores() const
    { return _scores; }

  //! @brief Retreive current lower bound point
  double pLB
    ( const unsigned ip ) const
    { assert( ip < _pSBB->_np ); return _LB_var[ip]; }
  //! @brief Retreive current upper bound point
  const std::vector<double>& pLB() const
    { return _LB_var; }
  //! @brief Retreive current lower bound value
  double& fLB()
    { return _LB_obj; }
  double fLB() const
    { return _LB_obj; }
  //! @brief Retreive current upper bound point
  double pUB
    ( const unsigned ip ) const
    { assert( ip < _pSBB->_np ); return _UB_var[ip]; }
  //! @brief Retreive current upper bound point
  const std::vector<double>& pUB() const
    { return _UB_var; }
  //! @brief Retreive current upper bound value
  double& fUB()
    { return _UB_obj; }
  double fUB() const
    { return _UB_obj; }

  //! @brief Retreive current variable bounds
  const T& P
    ( const unsigned ip ) const
    { assert( ip < _pSBB->_np ); return _P[ip]; }
  //! @brief Retreive/set bound for variable <a>ip</a> 
  T& P
    ( const unsigned ip )
    { assert( ip < _pSBB->_np ); return _P[ip]; }
  //! @brief Retreive pointer to variable bounds
  const std::vector<T>& P() const
    { return _P; }
  //! @brief Retreive/set pointer to variable bounds
  std::vector<T>& P()
    { return _P; }

  //! @brief Retreive/set node type (ROOT/LEFT/RIGHT)
  typename SBBSLV<T>::NODETYPE& type()
    { return _type; }
  //! @brief Retreive/set parent branching variable and range
  std::pair<unsigned,T>& parent()
    { return _parent; }
  //! @brief Retreive/set strong branch status
  bool& strongbranch()
    { return _strongbranch; }

  //! @brief Public constructor (for root node)
  SBBNode
    ( SBBSLV<T>*pSBB, const std::vector<T>&P, const double*p0=nullptr,
      const unsigned index=1, const unsigned iter=0 );

  //! @brief Public constructor (for regular node)
  template <typename U> SBBNode
    ( SBBSLV<T>*pSBB, const std::vector<T>&P, const double strength,
      const unsigned index, const unsigned depth, const unsigned iter,
      const typename SBBSLV<T>::NODETYPE type, const std::pair<unsigned,T> parent,
      const std::set<unsigned>&depend, U*data );

  //! @brief Destructor
  ~SBBNode();

  //! @brief Lower bounding
  typename SBBSLV<T>::STATUS lower_bound
    ( const std::vector<double>&var, const double inc, std::ostream& os );

  //! @brief Preprocessing
  typename SBBSLV<T>::STATUS preprocess
    ( std::vector<double>&var, double&obj, const double inc, std::ostream& os );

  //! @brief Postprocessing
  typename SBBSLV<T>::STATUS postprocess
    ( std::vector<double>&var, double&obj, const double inc, std::ostream& os );

  //! @brief Upper bounding
  typename SBBSLV<T>::STATUS upper_bound
    ( const std::vector<double>&var, const double inc, std::ostream& os );

  //! @brief Feasibility test at <a>var</a>
  typename SBBSLV<T>::STATUS test_feasibility
    ( std::vector<double>&var, double&obj, const double inc, std::ostream& os );

  //! @brief Check if the point <a>val</a> is at a bound
  bool at_bound
    ( const double val, const unsigned ip, const double rtol ) const;

private:
  //! @brief Private default constructor
  SBBNode<T>(){};

  //! @brief Pointer to underlying branch-and-bound tree
  SBBSLV<T> *_pSBB;
  //! @breif Strength of parent node
  double _strength;
  //! @brief Depth in SBB tree
  unsigned _depth;
  //! @brief Index in SBB tree
  unsigned _index;
  //! @brief Iteration when created in SBB tree
  unsigned _iter;
  //! @brief Node type (ROOT/LEFT/RIGHT)
  typename SBBSLV<T>::NODETYPE _type;

  //! @brief Subset of dependent variables in node
  std::set<unsigned> _depend;
  //! @brief Map of variables with scores for branching variable selection
  std::map<unsigned,double> _scores;
  //! @brief Parent node branching variable and range
  std::pair<unsigned,T> _parent;
  //! @brief Whether or not strong branching has been applied
  bool _strongbranch;
  //! @brief Pointer to user data
  void* _data;

  //! @brief Variable bounds
  std::vector<T> _P;
  //! @brief Backup variable bounds (for domain reduction)
  std::vector<T> _P0;

  //! @brief Upper bound value
  double _UB_obj;
  //! @brief upper bound point
  std::vector<double> _UB_var;

  //! @brief Lower bound value
  double _LB_obj;
  //! @brief Lower bound point
  std::vector<double> _LB_var;
};

//! @brief C++ structure for comparing SBB Nodes
////////////////////////////////////////////////////////////////////////
//! mc::lt_SBBNode is a C++ structure for comparing nodes in branch-and-
//! bound tree based on their lower bound values.
////////////////////////////////////////////////////////////////////////
template <typename T>
struct lt_SBBNode
{
  bool operator()
    ( const SBBNode<T>*Node1, const SBBNode<T>*Node2 ) const
    { return( Node1->strength() < Node2->strength() ); }
};

///////////////////////////////   SBBSLV   ////////////////////////////////

template <typename T>
inline bool
SBBSLV<T>::_variables
( unsigned const np, T const* P, unsigned const* type, std::set<unsigned> const& exclude )
{
  if( !np || !P ){
    _np = 0;
    _P_root.clear();
    _P_type.clear();
    return false;
  }
  
  _np = np;
  _P_root.assign( P, P+_np );

  if( !type )
    _P_type.assign( _np, 0 );
  else
    _P_type.assign( type, type+_np );
  for( unsigned i=0; i<_np; ++i ){
    if( !_P_type[i] ) continue;
    if( std::ceil( Op<T>::l( _P_root[i] ) ) > std::floor( Op<T>::u( _P_root[i] ) ) ) return false;
    _P_root[i] = T( std::ceil( Op<T>::l( _P_root[i] ) ), std::floor( Op<T>::u( _P_root[i] ) ) );
  }

  if( &exclude == &_exclude_vars ) return true;
  _exclude_vars = exclude;

  return true;
}

template <typename T>
inline bool
SBBSLV<T>::_interrupted
()
const
{
  if( stats.to_time( stats.walltime_all + stats.walltime( _tstart ) ) > options.MAX_WALLTIME
   || ( options.MAX_NODES && _node_index > options.MAX_NODES ) )
    return true;
  return false;
}

template <typename T>
inline typename SBBSLV<T>::STATUS
SBBSLV<T>::solve
( BASE_OPT::t_OBJ const pb, unsigned const np, T const* P, double const* p0, double const* f0,
  unsigned const* ptyp, std::set<unsigned> const& exclude, std::ostream& os )
{
  // Create and add root node to set _Nodes
  if( !_variables( np, P, ptyp, exclude ) ) return INFEASIBLE;
  _pb = pb;
  _restart( p0, f0 );
  _display_init();
  _display( os );
  _Nodes.insert( new SBBNode<T>( this, _P_root, p0 ) );

#ifdef SAVE__SBBSLV_NODES_TO_FILE
  std::ofstream osbbtree( "sbb_tree.out", std::ios_base::out );
#endif
  
  // keep branch-and-bound going until set _Nodes is empty
  for( _node_index = 1; !_Nodes.empty() && !_interrupted(); ++_node_index ){

    // intermediate display
    _display_add( _node_index );
    _display_add( (unsigned)_Nodes.size() );
    _display_add( _f_inc );
    switch( _pb ){
    case MIN:
      _display_add( (*_Nodes.begin())->strength() ); break;
    case MAX:
      _display_add( -(*_Nodes.begin())->strength() ); break;
    }

    // get pointer to next node and remove it for set _Nodes
    _display_tree( os );
    SBBNode<T>* pNode = *_Nodes.begin();
    _Nodes.erase( _Nodes.begin() );
    _display_add( pNode->iter() );

#ifdef SAVE__SBBSLV_NODES_TO_FILE
    _display_box( pNode, osbbtree );
#endif

    // pre-processing
    typename SBBSLV<T>::NODESTAT PRE_stat = _preprocess( pNode, os );
    if( PRE_stat == FATHOM ){
      delete pNode;
      _display_add( "SKIPPED" );
      _display_add( "SKIPPED" );
      _display_add( "FATHOM" );
      _display_line( os );
      continue;
    }
    else if( PRE_stat == ABORT ){
      delete pNode;
      _display_add( "SKIPPED" );
      _display_add( "SKIPPED" );
      _display_add( "ABORT" );
      _display_line( os );
      break;
    }

    // relaxation
    switch( _pb ){
    case MIN:
      _REL_stat = _lower_bound( pNode, true, false, os ); break;
    case MAX:
      _REL_stat = _upper_bound( pNode, true, false, os ); break;
    }
    if( _REL_stat == FATHOM ){
      delete pNode;
      _display_add( "SKIPPED" );
      _display_add( "FATHOM" );
      _display_line( os );
      continue;
    }
    else if( _REL_stat == ABORT ){
      delete pNode;
      _display_add( "SKIPPED" );
      _display_add( "ABORT" );
      _display_line( os );
      break;
    }

    // tightening
    switch( _pb ){
    case MIN:
      _ORI_stat = _upper_bound( pNode, false, false, os ); break;
    case MAX:
      _ORI_stat = _lower_bound( pNode, false, false, os ); break;
    }
    if( _ORI_stat == FATHOM ){
      delete pNode;
      _display_add( "FATHOM" );
      _display_line( os );
      continue;
    }
    else if( _ORI_stat == ABORT ){
      delete pNode;
      _display_add( "ABORT" );
      _display_line( os );
      break;
    }

    // SBB tree reduction
    if( _REL_stat==UPDATED || _ORI_stat==UPDATED ){
      _update_tree();
      if( _postprocess_tree( os ) == ABORT ){
        delete pNode;
        _display_add( "ABORT" );
        _display_line( os );
        break;
      }
    }

    // post-processing
    typename SBBSLV<T>::NODESTAT POST_stat = _postprocess( pNode, os );
    if( POST_stat == FATHOM ){
      delete pNode;
      _display_add( "FATHOM" );
      _display_line( os );
      continue;
    }
    else if( POST_stat == ABORT ){
      delete pNode;
      _display_add( "ABORT" );
      _display_line( os );
      break;
    }

    // domain branching
    _REL_stat = _branch_node( pNode, os );
    delete pNode;
    if( _REL_stat == UPDATED ){
      std::ostringstream omsg;
      omsg << "BRANCH" << _var_branch;
      _display_add( omsg.str() );
    }
    else if( _REL_stat == ABORT )
      _display_add( "ABORT" );
    _display_line( os );

    // keep track of the maximum nodes in memory
    if( _Nodes.size() > stats.node_max ) stats.node_max = _Nodes.size();
  }

#ifdef SAVE__SBBSLV_NODES_TO_FILE
  osbbtree.close();
#endif

  stats.node_tot     = _node_index-1;
  stats.walltime_all = stats.walltime( _tstart );
  _display_final( stats.walltime_all );
  _display( os );
  _status = (!_Nodes.empty()? FAILURE: (_p_inc.empty()? INFEASIBLE: NORMAL));
  return _status;
}

template <typename T>
inline void
SBBSLV<T>::_restart
( double const* p0, double const* f0 )
{
  _clean_stack();
  _p_inc.clear();
  if( p0 && f0 ){
    _f_inc = *f0;
    _p_inc.assign( p0, p0+_np );
  }
  else{
    _f_inc = ( _pb==MIN? INF: -INF );
  }
  _node_index = _node_count = 0;
  _tstart = stats.now();
  stats.reset();
}

template <typename T>
inline void
SBBSLV<T>::_clean_stack
()
{
  it_Nodes it = _Nodes.begin();
  for( ; it != _Nodes.end(); it++ ) delete *it;
  _Nodes.clear();
}

template <typename T>
inline typename SBBSLV<T>::NODESTAT
SBBSLV<T>::_lower_bound
( SBBNode<T>* pNode, bool const relaxed, bool const strongbranching, std::ostream& os )
{
  auto tstart = stats.now();
  NODESTAT stat;
  switch( pNode->lower_bound( pNode->pUB(), _f_inc, os ) ){

    case INFEASIBLE:
      if( !strongbranching ) _display_add( "INFEASIBLE" );
      if( relaxed ) pNode->fLB() = INF;
      stat = FATHOM; break;
      
    case NORMAL:{
      if( !strongbranching ) _display_add( pNode->fLB() );
      if( relaxed ){
        // fathom by value dominance?
        if( _fathom_by_dominance( pNode ) )
          { stat = FATHOM; break; }
        // try and update incumbent w/ relaxation solution point if applicable
        if( _feasible_relax( pNode, os ) == NORMAL
         && _update_incumbent( pNode->fUB(), pNode->pLB() ) )
          { stat = UPDATED; break; }
      }
      else{
        // try and update incumbent
        if( _update_incumbent( pNode->fLB(), pNode->pLB() ) )
          { stat = UPDATED; break; }
      }
      stat = NOTUPDATED; break;
    }

    case FAILURE:
      if( !strongbranching ) _display_add( "FAILURE" );
      // retain node (to be sure...)
      pNode->fLB() = -INF;
      stat = NOTUPDATED; break;

    case FATAL: default:
      if( !strongbranching ) _display_add( "FATAL" );
      stat = ABORT; break;
  }
  stats.walltime_lbd += stats.walltime( tstart );
  return stat;
}

template <typename T>
inline typename SBBSLV<T>::NODESTAT
SBBSLV<T>::_upper_bound
( SBBNode<T>* pNode, bool const relaxed, bool const strongbranching, std::ostream& os )
{
  auto tstart = stats.now();
  NODESTAT stat;
  switch( pNode->upper_bound( pNode->pLB(), _f_inc, os ) ){

    case INFEASIBLE:
      if( !strongbranching ) _display_add( "INFEASIBLE" );
      if( relaxed ) pNode->fUB() = -INF;
      stat = FATHOM; break;
      
    case NORMAL:{
      if( !strongbranching ) _display_add( pNode->fUB() );
      if( relaxed ){
        // fathom by value dominance?
        if( _fathom_by_dominance( pNode ) )
          { stat = FATHOM; break; }
        // try and update incumbent w/ relaxation solution point if applicable
        if( _feasible_relax( pNode, os ) == NORMAL
         && _update_incumbent( pNode->fLB(), pNode->pUB() ) )
          { stat = UPDATED; break; }
      }
      else{
        // try and update incumbent
        if( _update_incumbent( pNode->fUB(), pNode->pUB() ) ) stat = UPDATED;
      }
      stat = NOTUPDATED; break;
    }

    case FAILURE:
      if( !strongbranching ) _display_add( "FAILURE" );
      // retain node (to be sure...)
      pNode->fUB() = INF;
      stat = NOTUPDATED; break;

    case FATAL: default:
      if( !strongbranching ) _display_add( "FATAL" );
      stat = ABORT; break;
  }
  stats.walltime_ubd += stats.walltime( tstart );
  return stat;
}

template <typename T>
inline typename SBBSLV<T>::STATUS
SBBSLV<T>::_feasible_relax
( SBBNode<T>* pNode, std::ostream& os )
{
  switch( _pb ){
  case MIN:
    return pNode->test_feasibility( pNode->_LB_var, pNode->_UB_obj, _f_inc, os );
  case MAX: default:
    return pNode->test_feasibility( pNode->_UB_var, pNode->_LB_obj, _f_inc, os );
  }
}

template <typename T>
inline bool
SBBSLV<T>::_fathom_by_dominance
( SBBNode<T>* pNode )
{
  double f_eff = _f_inc;
  switch( _pb ){

  case MIN:
    f_eff -= std::max( options.STOPPING_ABSTOL, std::fabs(_f_inc)*options.STOPPING_RELTOL );
    // try and fathom current node by value dominance
    return( pNode->fLB() > f_eff ? true: false );
    
  case MAX: default:
    f_eff += std::max( options.STOPPING_ABSTOL, std::fabs(_f_inc)*options.STOPPING_RELTOL );
    // try and fathom current node by value dominance
    return( pNode->fUB() < f_eff ? true: false );
  }
}

template <typename T>
inline bool
SBBSLV<T>::_update_incumbent
( double const& fINC, std::vector<double> const& pINC )
{
  // test incumbent w.r.t current solution point
  switch( _pb ){
   case MIN: if( fINC >= _f_inc ) return false; break;
   case MAX: if( fINC <= _f_inc ) return false; break;
  }

  _f_inc = fINC;
  _p_inc = pINC;
  stats.node_inc = _node_index;
  return true;
}

template <typename T>
inline void
SBBSLV<T>::_branching_variable_set
( SBBNode<T> const* pNode )
{
  _branch_set.clear();
  typename Options::SELECTION psel = options.BRANCHING_USERFUNCTION;
  std::set<unsigned> ssel;
  if( psel ) ssel = psel( pNode );
  for( unsigned ip=0; ip<_np; ip++ ){
    // Allow branching on var #ip if part of the user selection (if any)
    if( psel && ssel.find(ip) == ssel.end() ) continue;
    // Allow branching on var #ip if not excluded (_exclude_vars)
    // or not a dependent in current node (_pNode->depend())
    if( _exclude_vars.find(ip) != _exclude_vars.end()
     || pNode->depend().find(ip) != pNode->depend().end() )
      continue;
    // Add variable index to branching set
    _branch_set.insert( ip );
  }

  // Preselect variables based on scores
  _branching_score_subset( pNode );

  // Interrupt if branch set is empty - internal error...
  if( _branch_set.empty() ) throw Exceptions( Exceptions::BRANCH );
}

template <typename T>
inline void
SBBSLV<T>::_branching_score_subset
( SBBNode<T> const* pNode )
{
  if( !options.SCORE_BRANCHING_USE || pNode->scores().empty() ) return;

  // Create map of scores
  struct lt_scores{
    bool operator()
      ( const std::pair<unsigned,double>&el1, const std::pair<unsigned,double>el2 )
      const
      { return el1.second < el2.second; }
  };
  std::multiset< std::pair<unsigned,double>, lt_scores > allscores;
  for( auto it = _branch_set.begin(); it != _branch_set.end(); ++it ){
    auto its = pNode->scores().find( *it );
    if( its == pNode->scores().end() ) continue;
    allscores.insert( *its );
#ifdef DEBUG__SBBSLV_SCOREBRANCHING
    std::cout << "Score variable #" << its->first << ": " << its->second << std::endl;
    //{ int dum; std::cout << "PAUSED"; std::cin >> dum; }
#endif
  }

  // Keep best candidates based on map of scores
  for( auto it=allscores.begin(); it!=allscores.end(); ++it ){
    if( it->second >= allscores.rbegin()->second*(1.-options.SCORE_BRANCHING_RELTOL)
                     -options.SCORE_BRANCHING_ABSTOL
     && ( !options.SCORE_BRANCHING_MAXSIZE
       || _branch_set.size() < options.SCORE_BRANCHING_MAXSIZE ) ) break;
    _branch_set.erase( it->first );
  }
#ifdef MC__SBBSLV_SHOW__SCOREBRANCHING
  std::cout << "Branching variable score subset: {";
  for( auto it = _branch_set.begin(); it != _branch_set.end(); ++it )
    std::cout << " " << *it << " ";
  std:: cout << "}\n";
  { int dum; std::cout << "PAUSED"; std::cin >> dum; }
#endif
}

template <typename T>
inline std::pair<unsigned, double>
SBBSLV<T>::_select_branching_variable
( SBBNode<T>* pNode, std::ostream& os )
{
  // Strong branching strategy
  if( pNode->depth() < options.STRONG_BRANCHING_MAXDEPTH && _branch_set.size() > 1 )
   return _strong_branching( pNode, _branch_set, os );

  std::pair<unsigned, double> branchsel( _np, -1. ); // <- Can be any negative number
  switch( options.BRANCHING_VARIABLE_CRITERION ){

   // Branching based on relative range diameter
   case Options::RGREL:
    for( auto it = _branch_set.begin(); it != _branch_set.end(); ++it ){
      double score = Op<T>::diam( pNode->P(*it) ) / Op<T>::diam( _P_root[*it] );
      if( score > branchsel.second ) branchsel = std::make_pair( *it, score );
    }
    break;

   // Branching based on absolute range diameter
   case Options::RGABS:
    for( auto it = _branch_set.begin(); it != _branch_set.end(); ++it ){
      double score = Op<T>::diam( pNode->P(*it) );
      if( score > branchsel.second ) branchsel = std::make_pair( *it, score );
    }
    break;
  }

  return branchsel;
}

template <typename T>
inline std::pair<const T, const T>
SBBSLV<T>::_partition_variable_domain
( SBBNode<T>* pNode, unsigned const var_branch )
const
{
  std::pair<T,T> partition;

  switch( options.BRANCHING_STRATEGY ){
  case Options::OMEGA:
    // Branch at incumbent if interior to current variable range
    if( !_p_inc.empty()
     && !pNode->at_bound(_p_inc[var_branch], var_branch, options.BRANCHING_BOUND_THRESHOLD) ){
      partition.first = mc::Op<T>::l(pNode->P(var_branch)) + Op<T>::zeroone()
        *( _p_inc[var_branch] - mc::Op<T>::l(pNode->P(var_branch)) );
      partition.second = mc::Op<T>::u(pNode->P(var_branch)) + Op<T>::zeroone()
        *( _p_inc[var_branch] - mc::Op<T>::u(pNode->P(var_branch)) );
      break;
    }
    // Otherwise branch at relaxation solution point if available +
    // interior to current variable range
    if( _pb == MIN && pNode->fLB() > -INF
     && !pNode->at_bound(pNode->pLB(var_branch), var_branch, options.BRANCHING_BOUND_THRESHOLD) ){ 
      partition.first = mc::Op<T>::l(pNode->P(var_branch)) + Op<T>::zeroone()
        *( pNode->pLB(var_branch) - mc::Op<T>::l(pNode->P(var_branch)) );
      partition.second = mc::Op<T>::u(pNode->P(var_branch)) + Op<T>::zeroone()
        *( pNode->pLB(var_branch) - mc::Op<T>::u(pNode->P(var_branch)) );
      break;
    }
    else if( _pb == MAX && pNode->fUB() < INF
     && !pNode->at_bound(pNode->pUB(var_branch), var_branch, options.BRANCHING_BOUND_THRESHOLD) ){ 
      partition.first = mc::Op<T>::l(pNode->P(var_branch)) + Op<T>::zeroone()
        *( pNode->pUB(var_branch) - mc::Op<T>::l(pNode->P(var_branch)) );
      partition.second = mc::Op<T>::u(pNode->P(var_branch)) + Op<T>::zeroone()
        *( pNode->pUB(var_branch) - mc::Op<T>::u(pNode->P(var_branch)) );
      break;
    }
    
  case Options::MIDPOINT: default:
    // Branch at mid-point of current variable range
    partition.first = mc::Op<T>::l(pNode->P(var_branch)) + Op<T>::zeroone()
      *Op<T>::diam(pNode->P(var_branch))/2.;
    partition.second = mc::Op<T>::u(pNode->P(var_branch)) - Op<T>::zeroone()
      *Op<T>::diam(pNode->P(var_branch))/2.;
    break;
  }

  // Ensure integer variables are not duplicated in subpartition
  if( _P_type[var_branch] ){
    // shift bisection point if integer
    if( Op<T>::u(partition.first) == std::round( Op<T>::u(partition.first) ) ){
      partition.first  = Op<T>::min( partition.first,  Op<T>::u(partition.first)-options.BACKOFF_BRANCHING_INTEGER );
      partition.second = Op<T>::hull( partition.second, Op<T>::l(partition.second)-options.BACKOFF_BRANCHING_INTEGER );
    }
    if( Op<T>::l( partition.first ) > std::floor( Op<T>::u(partition.first) )
     || std::ceil( Op<T>::l( partition.second ) ) > Op<T>::u(partition.second) )
      throw Exceptions( Exceptions::BRANCH );
    partition.first  = T( Op<T>::l( partition.first ), std::floor( Op<T>::u(partition.first) ) );
    partition.second = T( std::ceil( Op<T>::l( partition.second ) ), Op<T>::u(partition.second) );
  }

  return partition;
}

template <typename T>
inline typename SBBSLV<T>::NODESTAT
SBBSLV<T>::_branch_node
( SBBNode<T>*pNode, std::ostream& os )
{
  // Branching set update
  _branching_variable_set( pNode );

  // Branching variable selection
  _var_branch = _select_branching_variable( pNode, os ).first;
  if( _REL_stat == ABORT ) return ABORT;

  // Partitionning
  std::pair<const T, const T> partition = _partition_variable_domain( pNode, _var_branch );
  _P_tmp = pNode->P();
  _P_tmp[_var_branch] = partition.first;
  switch( _pb ){
  case MIN:
    _Nodes.insert( new SBBNode<T>( this, _P_tmp, pNode->fLB(), ++_node_count,
      pNode->depth()+1, _node_index, LEFT, std::make_pair(_var_branch,
      pNode->P(_var_branch)), pNode->depend(), pNode->data() ) ); break;
  case MAX:
    _Nodes.insert( new SBBNode<T>( this, _P_tmp, -pNode->fUB(), ++_node_count,
      pNode->depth()+1, _node_index, LEFT, std::make_pair(_var_branch,
      pNode->P(_var_branch)), pNode->depend(), pNode->data() ) ); break;
  }
  _P_tmp[_var_branch] = partition.second;
  switch( _pb ){
  case MIN:
    _Nodes.insert( new SBBNode<T>( this, _P_tmp, pNode->fLB(), ++_node_count,
      pNode->depth()+1, _node_index, RIGHT, std::make_pair(_var_branch,
      pNode->P(_var_branch)), pNode->depend(), pNode->data() ) ); break;
  case MAX:
    _Nodes.insert( new SBBNode<T>( this, _P_tmp, -pNode->fUB(), ++_node_count,
      pNode->depth()+1, _node_index, RIGHT, std::make_pair(_var_branch,
      pNode->P(_var_branch)), pNode->depend(), pNode->data() ) ); break;
  }

  return UPDATED;
}

template <typename T>
inline double
SBBSLV<T>::_score_branching
( const double fLEFT, const double fRIGHT )
const
{
  return (1-options.STRONG_BRANCHING_WEIGHT) * fmin(fLEFT,fRIGHT)
          + options.STRONG_BRANCHING_WEIGHT  * fmax(fLEFT,fRIGHT);
}

template <typename T>
inline std::pair<unsigned, double>
SBBSLV<T>::_strong_branching
( SBBNode<T>*pNode, const std::set<unsigned>&set_branch, std::ostream& os )
{
  std::pair<unsigned, double> branchsel( _np, -INF );

  // Create child subnode
  //for( unsigned ip=0; ip<_np; ip++ ) _P_tmp[ip] = pNode->P(ip);
  SBBNode<T>* pSubnode = 0;
  switch( _pb ){
  case MIN:
    //if( pNode->fLB() == -INF ) return branchsel; // WHY???
    pSubnode = new SBBNode<T>( this, pNode->P(), pNode->fLB(), pNode->index(),
      pNode->depth()+1, _node_index, LEFT, std::make_pair(0,pNode->P(0)),
      pNode->depend(), pNode->data() ); break;
  case MAX:
    //if( pNode->fUB() == INF ) return branchsel;
    pSubnode = new SBBNode<T>( this, pNode->P(), -pNode->fUB(), pNode->index(),
      pNode->depth()+1, _node_index, LEFT, std::make_pair(0,pNode->P(0)),
      pNode->depend(), pNode->data() ); break;
  }
  
  // Repeat for all variables in branching set
#ifdef MC__SBBSLV_STRONGBRANCHING_SHOW
  std::cout << "*** SCORES";
#endif
  double fLEFT, fRIGHT, fSCORE;
  for( auto it = set_branch.begin(); it != set_branch.end(); ++it ){

    // Simulate partitionning
    pSubnode->strongbranch() = true;
    pSubnode->parent() = std::make_pair( *it, pNode->P(*it) );
    const std::pair<const T, const T> partition = _partition_variable_domain( pNode, *it );

    // Relax left partition
    pSubnode->P(*it) = partition.first;
    pSubnode->type() = LEFT;
    switch( _pb ){
    case MIN:
      _REL_stat = _lower_bound( pSubnode, true, true, os );
      fLEFT = pSubnode->fLB(); break;
    case MAX: default:
      _REL_stat = _upper_bound( pSubnode, true, true, os );
      fLEFT = -pSubnode->fUB(); break;
    }
    if( _REL_stat == ABORT ) break;
    // Left partition failed
    //if( fLEFT == -INF ) continue;
 
    // Relax right partition
    pSubnode->P(*it) = partition.second;
    pSubnode->type() = RIGHT;
    switch( _pb ){
    case MIN:
      _REL_stat = _lower_bound( pSubnode, true, true, os );
      fRIGHT = pSubnode->fLB(); break;
    case MAX: default:
      _REL_stat = _upper_bound( pSubnode, true, true, os );
      fRIGHT = -pSubnode->fUB(); break;
    }
    if( _REL_stat == ABORT ) break;
    // Right partition failed
    //if( fRIGHT == -INF ) continue;

    // Score current variable and compare
    fSCORE = _score_branching( fLEFT, fRIGHT );
#ifdef MC__SBBSLV_STRONGBRANCHING_SHOW
    std::cout << "   " << *it << ": " << fSCORE
              << " (" << fLEFT << "," << fRIGHT << ")";
#endif
    if( fSCORE > branchsel.second )
      branchsel = std::make_pair( *it, fSCORE );

    // Reset variable bounds
    pSubnode->P() = pNode->P();
  }
#ifdef MC__SBBSLV_STRONGBRANCHING_SHOW
  std::cout << std::endl;
#endif

  return branchsel;
}

template <typename T>
inline typename SBBSLV<T>::NODESTAT
SBBSLV<T>::_preprocess
( SBBNode<T>*pNode, std::ostream& os )
{
  switch( _pb==MIN?
          pNode->preprocess( pNode->_LB_var, pNode->_LB_obj, _f_inc, os ):
          pNode->preprocess( pNode->_UB_var, pNode->_UB_obj, _f_inc, os ) ){
  case INFEASIBLE:
    return FATHOM;
  case NORMAL: case FAILURE: default:
    return NOTUPDATED;
  }
}

template <typename T>
inline typename SBBSLV<T>::NODESTAT
SBBSLV<T>::_postprocess
( SBBNode<T>*pNode, std::ostream& os )
{
  if( _fathom_by_dominance( pNode ) ) return FATHOM;
  
  switch( _pb==MIN?
          pNode->postprocess( pNode->_LB_var, pNode->_LB_obj, _f_inc, os ):
          pNode->postprocess( pNode->_UB_var, pNode->_UB_obj, _f_inc, os ) ){
  case INFEASIBLE:
    return FATHOM;
  case NORMAL: case FAILURE:
    return NOTUPDATED;
  case FATAL: default:
    return ABORT;
  }
}

template <typename T>
inline void
SBBSLV<T>::_update_tree
()
{
  // try and reduce all nodes in B&B tree
  it_Nodes it = _Nodes.begin();
  for( ; it != _Nodes.end(); ++it )
    if( _fathom_by_dominance( *it ) ){ delete *it; _Nodes.erase( it ); }
  return;
}

template <typename T>
inline typename SBBSLV<T>::NODESTAT
SBBSLV<T>::_postprocess_tree
( std::ostream& os )
{
  if( !options.TREE_UPDATE ) return NOTUPDATED;

  // try and reduce all nodes in B&B tree
  it_Nodes it = _Nodes.begin();
  for( ; it != _Nodes.end(); ++it ){
    typename SBBSLV<T>::NODESTAT POST_stat = _postprocess( *it, os );
    if( POST_stat == ABORT ) return ABORT;
    else if( POST_stat == FATHOM ){ delete *it; _Nodes.erase( it ); }
  }
  return UPDATED;
}

template <typename T>
inline void
SBBSLV<T>::_display_tree
( std::ostream& os )
{
  if( options.DISPLAY_LEVEL <= 2 ) return;

  // Show current node
  if( options.DISPLAY_LEVEL == 3 ){
    os << *(*_Nodes.begin());
    return;
  }

  // Show all nodes in stack
  it_Nodes it = _Nodes.begin();
  for( unsigned id=1; it != _Nodes.end(); ++it, id++ ){
    os << "Node " << id << ":" << std::scientific << std::setprecision(_DPREC)
       << std::setw(_DPREC+8) << (*it)->strength() << *(*it);
  }
  return;
}

template <typename T>
inline void
SBBSLV<T>::_display_init()
{
  _odisp.str("");
  if( options.DISPLAY_LEVEL <= 1 ) return;
  _odisp << std::right
  	 << std::setw(_IPREC) << "INDEX"
  	 << std::setw(_IPREC) << "STACK"
  	 << std::setw(_DPREC+8) << "PRIMAL   "
  	 << std::setw(_DPREC+8) << "DUAL   "
  	 << std::setw(_IPREC) << "PARENT"
  	 << std::setw(_DPREC+8) << ( _pb==MIN? "LBD   ": "UBD   " )
  	 << std::setw(_DPREC+8) << ( _pb==MIN? "UBD   ": "LBD   " )
  	 << std::setw(_DPREC+8) << "ACTION"
  	 << std::setw(10) << "WALLTIME";
}

template <typename T>
inline void
SBBSLV<T>::_display_add
( const double dval )
{
  if( options.DISPLAY_LEVEL <= 1 ) return;
  _odisp << std::right << std::scientific << std::setprecision(_DPREC)
         << std::setw(_DPREC+8) << dval;
}

template <typename T>
inline void
SBBSLV<T>::_display_add
( const unsigned ival )
{
  if( options.DISPLAY_LEVEL <= 1 ) return;
  _odisp << std::right << std::setw(_IPREC) << ival;
}

template <typename T>
inline void
SBBSLV<T>::_display_add
( const std::string &sval )
{
  if( options.DISPLAY_LEVEL <= 1 ) return;
  _odisp << std::right << std::setw(_DPREC+8) << sval;
}

template <typename T>
inline void
SBBSLV<T>::_display_time
()
{
  if( options.DISPLAY_LEVEL <= 1 ) return;
  _odisp << std::right << std::fixed << std::setprecision(1) << std::setw(9) 
         << stats.to_time( stats.walltime( _tstart ) ) << "s";
}

template <typename T>
inline void
SBBSLV<T>::_display_final
( std::chrono::microseconds const& walltime )
{
  if( options.DISPLAY_LEVEL <= 0 ) return;
  // Solution found within allowed time?
  if( !_interrupted() )
    _odisp << std::endl << "#  NORMAL TERMINATION: ";
  else
    _odisp << std::endl << "#  EXECUTION STOPPED: ";
  _odisp << std::fixed << std::setprecision(3) << walltime.count()*1e-6 << " SEC" << std::endl;

  // No feasible solution found
  if( _p_inc.empty() )
    _odisp << "#  NO FEASIBLE SOLUTION FOUND" << std::endl;

  // Feasible solution found
  else{
    // Incumbent
    _odisp << "#  INCUMBENT VALUE:" << std::scientific
           << std::setprecision(_DPREC) << std::setw(_DPREC+8) << _f_inc
           << std::endl;
    _odisp << "#  INCUMBENT POINT:";
    for( unsigned ip=0, id=0; ip<_np; ip++, id++ ){
      if( id == _LDISP ){
        _odisp << std::endl << std::left << std::setw(19) << "#";
        id = 0;
      }
      _odisp << std::right << std::setw(_DPREC+8) << _p_inc[ip];
    }
    _odisp << std::endl
           << "#  INCUMBENT FOUND AT NODE: " << stats.node_inc << std::endl;
  }

  _odisp << "#  TOTAL NUMBER OF NODES:   " << stats.node_tot << std::endl
         << "#  MAXIMUM NODES IN STACK:  " << stats.node_max << std::endl;
}

template <typename T>
inline void
SBBSLV<T>::_display
( std::ostream &os )
{
  if( _odisp.str() == "" ) return;
  //if( options.DISPLAY_LEVEL > 0 ){
  os << _odisp.str() << std::endl;
  //}
  _odisp.str("");
  return;
}

template <typename T>
inline void
SBBSLV<T>::_display_line
( std::ostream &os )
{
  _display_time();
  if( _odisp.str() == "" ) return;
  //if( options.DISPLAY_LEVEL > 0 ){
  os << _odisp.str() << std::endl;
  //}
  _odisp.str("");
  return;
}

template <typename T>
inline void
SBBSLV<T>::Options::display
( std::ostream&out )
const
{
  // Display SBBSLV Options
  out << std::setw(60) << "  ABSOLUTE CONVERGENCE TOLERANCE"
      << std::scientific << std::setprecision(1)
      << STOPPING_ABSTOL << std::endl;
  out << std::setw(60) << "  RELATIVE CONVERGENCE TOLERANCE"
      << std::scientific << std::setprecision(1)
      << STOPPING_RELTOL << std::endl;
  out << std::setw(60) << "  BRANCHING STRATEGY FOR NODE PARTITIONING";
  switch( BRANCHING_STRATEGY ){
  case MIDPOINT: out << "MIDPOINT\n"; break;
  case OMEGA:    out << "OMEGA\n";    break;
  }
  out << std::setw(60) << "  BRANCHING TOLERANCE FOR VARIABLE AT BOUND"
      << std::scientific << std::setprecision(2)
      << BRANCHING_BOUND_THRESHOLD << std::endl;
  out << std::setw(60) << "  BRANCHING STRATEGY FOR VARIABLE SELECTION";
  switch( BRANCHING_VARIABLE_CRITERION ){
  case RGREL:  out << "RGREL\n";  break;
  case RGABS:  out << "RGABS\n";  break;
  }
  out << std::setw(60) << "  USE SCORE BRANCHING";
  switch( SCORE_BRANCHING_USE ){
   case 0:  out << "N\n"; break;
   default: out << "Y\n";
            out << std::setw(60) << "  RELATIVE TOLERANCE FOR SCORE BRANCHING"
                << std::scientific << std::setprecision(1)
                << SCORE_BRANCHING_RELTOL << std::endl;
            out << std::setw(60) << "  ABSOLUTE TOLERANCE FOR SCORE BRANCHING"
                << std::scientific << std::setprecision(1)
                << SCORE_BRANCHING_ABSTOL << std::endl; break;
  }
  out << std::setw(60) << "  MAXIMUM DEPTH FOR STRONG BRANCHING";
  switch( STRONG_BRANCHING_MAXDEPTH ){
   case 0:  out << "-\n"; break;
   default: out << STRONG_BRANCHING_MAXDEPTH << std::endl;
            out << std::setw(60) << "  CHILDREN NODE WEIGHT FOR STRONG BRANCHING"
                << std::scientific << std::setprecision(1)
                << STRONG_BRANCHING_WEIGHT << std::endl; break;
  }
  out << std::setw(60) << "  UPDATE ENTIRE TREE AFTER AN INCUMBENT IMPROVEMENT?"
      << (TREE_UPDATE?"Y\n":"N\n");
  out << std::setw(60) << "  MAXIMUM ITERATION COUNT";
  switch( MAX_NODES ){
   case 0:  out << "NO LIMIT\n"; break;
   default: out << MAX_NODES << std::endl; break;
  }
  out << std::setw(60) << "  MAXIMUM CPU TIME (SEC)"
      << std::scientific << std::setprecision(1)
      << MAX_WALLTIME << std::endl;
  out << std::setw(60) << "  DISPLAY LEVEL"
      << DISPLAY_LEVEL << std::endl;
}

template <typename T>
inline void
SBBSLV<T>::_display_box
( const SBBNode<T>*pNode, std::ostream&os ) const
{
  switch( _np ){
  case 2:
    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1)) << std::endl
       << std::endl;
    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1)) << std::endl
       << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1)) << std::endl
       << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1)) << std::endl
       << std::endl;
    break;

  case 3:
    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << std::endl << std::endl;

    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;

    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::l(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::l(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    os << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::l(pNode->P(2)) << std::endl
       << Op<T>::u(pNode->P(0)) << "  " << Op<T>::u(pNode->P(1))
       << "  " << Op<T>::u(pNode->P(2)) << std::endl
       << std::endl << std::endl;
    break;

  default:
    break;
  }
}

/////////////////////////////// SBBNode ///////////////////////////////

template <typename T>
inline
SBBNode<T>::SBBNode
( SBBSLV<T>*pSBB, const std::vector<T>&P, const double*p0,
  const unsigned index, const unsigned iter )
: _pSBB( pSBB ), _strength( -_pSBB->INF ), _depth( 0 ),
  _index( index ), _iter( iter ), _type( SBBSLV<T>::ROOT ),
  _parent( 0, T(0.) ), _strongbranch( false ), _data( 0 ),
  _P( P ), _UB_obj( _pSBB->INF ), _UB_var( _pSBB->_np ),
  _LB_obj( -_pSBB->INF ), _LB_var( _pSBB->_np ) 
{
  // Default initial points
  for( unsigned i=0; i<_pSBB->_np; i++ )
    _LB_var[i] = _UB_var[i] = ( p0? p0[i]: mc::Op<T>::mid( P[i] ) );
}

template <typename T> template <typename U>
inline
SBBNode<T>::SBBNode
( SBBSLV<T>*pSBB, const std::vector<T>&P, const double strength,
  const unsigned index, const unsigned depth, const unsigned iter,
  const typename SBBSLV<T>::NODETYPE type, const std::pair<unsigned,T> parent,
  const std::set<unsigned>&depend, U*data )
: _pSBB( pSBB ), _strength( strength ), _depth( depth ), _index( index ),
  _iter( iter ), _type( type ), _parent( parent ), _strongbranch( false ),
  _data( data ), _P( P ), _UB_obj( _pSBB->INF ), _UB_var( _pSBB->_np ),
  _LB_obj( -_pSBB->INF ), _LB_var( _pSBB->_np )
{
  // Default initial points
  for( unsigned i=0; i<_pSBB->_np; i++ )
    _LB_var[i] = _UB_var[i] = Op<T>::mid( P[i] );
}

template <typename T>
inline
SBBNode<T>::~SBBNode()
{}

template <typename T>
inline typename SBBSLV<T>::STATUS
SBBNode<T>::lower_bound
( const std::vector<double>&var, const double inc, std::ostream& os )
{
  _P0 = _P;
  if( var.size() >= _pSBB->_np ) _LB_var = var; // initial guess
  return _pSBB->subproblems( SBBSLV<T>::LOWERBD, this, _LB_var, _LB_obj, inc, os );
}

template <typename T>
inline typename SBBSLV<T>::STATUS
SBBNode<T>::preprocess
( std::vector<double>&var, double&obj, const double inc, std::ostream& os )
{
  _P0 = _P;
  return _pSBB->subproblems( SBBSLV<T>::PREPROC, this, var, obj, inc, os );
}

template <typename T>
inline typename SBBSLV<T>::STATUS
SBBNode<T>::postprocess
( std::vector<double>&var, double&obj, const double inc, std::ostream& os )
{
  _P0 = _P;
  return _pSBB->subproblems( SBBSLV<T>::POSTPROC, this, var, obj, inc, os );
}

template <typename T>
inline typename SBBSLV<T>::STATUS
SBBNode<T>::upper_bound
( const std::vector<double>&var, const double inc, std::ostream& os )
{
  _P0 = _P;
  if( var.size() >= _pSBB->_np ) _UB_var = var; // initial guess
  return _pSBB->subproblems( SBBSLV<T>::UPPERBD, this, _UB_var, _UB_obj, inc, os );
}

template <typename T>
inline typename SBBSLV<T>::STATUS
SBBNode<T>::test_feasibility
( std::vector<double>&var, double&obj, const double inc, std::ostream& os )
{
  return _pSBB->subproblems( SBBSLV<T>::FEASTEST, this, var, obj, inc, os );
}

template <typename T>
inline bool
SBBNode<T>::at_bound
( const double val, const unsigned ip, const double rtol )
const
{
  assert( ip < _pSBB->_np );
  const double margin = Op<T>::diam(_P[ip]) * rtol;
  return( val < Op<T>::l(_P[ip]) + margin
       || val > Op<T>::u(_P[ip]) - margin );
}

template <typename T> inline std::ostream&
operator <<
( std::ostream&out, const SBBNode<T>&CV )
{
  //std::set<unsigned>::iterator ivar = CV._pSBB->_branch_set.begin();
  //for( unsigned i=0; ivar != CV._pSBB->_branch_set.end(); ++ivar, i++ )
  //  out << "  " << CV._P[i];
  for( unsigned ip=0; ip < CV._pSBB->_np; ip++ )
    out << "  " << CV._P[ip];
  out << std::endl;
  return out;
}

} // namespace mc

#endif

// Copyright (C) 2015 Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MC__BASE_OPT_HPP
#define MC__BASE_OPT_HPP

#include <vector>
#include <cmath>

namespace mc
{
//! @brief C++ base class for definition of optimization problems
////////////////////////////////////////////////////////////////////////
//! mc::BASE_OPT is a C++ base class for definition of the objective 
//! and constraint functions participating in optimization problems.
////////////////////////////////////////////////////////////////////////
class BASE_OPT
{
public:

  //! @brief Infinity
  static double INF;

  //! @brief Enumeration type for objective function
  enum t_OBJ{
    MIN=0,	//!< Minimization
    MAX		//!< Maximization
  };

  //! @brief Enumeration type for constraints
  enum t_CTR{
    EQ=0,	//!< Equality constraint
    LE,		//!< Inequality constraint
    GE		//!< Inequality constraint
  };

  //! @brief Class constructor
  BASE_OPT()
    {}

  //! @brief Class destructor
  virtual ~BASE_OPT()
    {}

protected:
  //! @brief Private methods to block default compiler methods
  BASE_OPT(const BASE_OPT&);
  BASE_OPT& operator=(const BASE_OPT&);
};

inline double BASE_OPT::INF = 1E30;
}

#include <iostream>
#include <iomanip>

namespace mc
{
//! @brief C++ structure for holding the solution of optimization models
////////////////////////////////////////////////////////////////////////
//! mc::SOLUTION_OPT is a C++ structure for holding the solution of 
//! optimization models, including variables, cost and constraint
//! functions and multiplers.
////////////////////////////////////////////////////////////////////////
struct SOLUTION_OPT
{
  //! @brief Default constructor
  SOLUTION_OPT
    ( int stat_ = -999 )
    : stat( stat_ )
    {}
  //! @brief Destructor
  ~SOLUTION_OPT
    ()
    {}
  //! @brief Copy constructor
  SOLUTION_OPT
    ( const SOLUTION_OPT &sol )
    : stat( sol.stat ), x( sol.x ), ux( sol.ux ), f( sol.f ), uf( sol.uf )
    {}

  SOLUTION_OPT& operator=
    ( SOLUTION_OPT const& sol )
    { stat = sol.stat; x = sol.x; ux = sol.ux; f = sol.f; uf = sol.uf; 
      return *this; }

  //! @brief Resets the solution fields
  void reset
    ( int const stat_ = -999 )
    { stat = stat_; x.clear(); ux.clear(); f.clear(); uf.clear(); }

  //! @brief Solver status
  int stat;
  //! @brief Variable values
  std::vector<double> x;
  //! @brief Variable bound multipliers
  std::vector<double> ux;
  //! @brief Function values  
  std::vector<double> f;
  //! @brief Function multipliers
  std::vector<double> uf;
};

std::ostream&
operator<<
( std::ostream & out, SOLUTION_OPT const& sol )
{
  std::cout << "STATUS: " << sol.stat << std::endl;
  std::cout << std::scientific << std::setprecision(6) << std::right;
  for( unsigned int i=0; i<sol.x.size(); i++ )
    std::cout << "X[" << i << "]:  LEVEL = " << std::setw(13) << sol.x[i]
              << "  MARGINAL = " << std::setw(13) << sol.ux[i] << std::endl;
  for( unsigned int j=0; j<sol.f.size(); j++ )
    std::cout << "F[" << j << "]:  LEVEL = " << std::setw(13) << sol.f[j]
              << "  MARGINAL = " << std::setw(13) << sol.uf[j] << std::endl;
  return out;
}

} // end namescape mc

#endif


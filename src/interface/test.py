import pymc
import cronos
import canon
import numpy as np

def nlp_test():

  # Define DAG
  DAG = pymc.FFGraph()
  X1 = pymc.FFVar(DAG,"X1")
  X2 = pymc.FFVar(DAG,"X2")

  # Define NLP
  NLP = canon.NLPSLV()
  NLP.set_dag( DAG )
  NLP.add_variable( [X1,X2], [0.,0.], [6.,4.] )
  NLP.set_objective( NLP.MAX, X1+X2 )
  NLP.add_constraint( NLP.LE, X1*X2-4. )

  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = NLP.options.BSYM;

  NLP.setup()
  NLP.options.DISPLEVEL = 1;
  NLP.solve( [0.,0.] )

  print( "status:", NLP.status )
  print( "solution point:", NLP.solution.x )
  print( "solution value:", NLP.solution.f[0] )
#  print( NLP.solution )

  NLP.options.DISPLEVEL = 0;
  NLP.solve( 8 )

  print( "status:", NLP.status )
  print( "solution point:", NLP.solution.x )
  print( "solution value:", NLP.solution.f[0] )
#  print( NLP.solution )


def ode_define( NS ):

  # Define DAG
  DAG = pymc.FFGraph()
  X = [pymc.FFVar(DAG,"X")]
  Q = [pymc.FFVar(DAG,"Q")]
  U    = []
  RHS  = []
  QUAD = []
  F    = []
  for i in range(NS):
    U.append( pymc.FFVar(DAG,"U"+str(i)) )
    RHS.append( [ U[i] - X[0] ] )
    QUAD.append( [ 0.5 * pymc.sqr( U[i] ) ] )
    F.append( [ Q[0], pymc.FFVar(0.) ] )
  F[NS-1][1] = X[0]

  # Define IVP
  ODE = cronos.ODESLV()
  ODE.set_dag( DAG )
  ODE.set_time( np.arange(0., 1.01, 1./NS).tolist() )
#  print( ODE.val_stage, ODE.var_time )
  ODE.set_state( X )
  ODE.set_parameter( U )
  ODE.set_differential( RHS )
#  print( ODE.eqn_differential )
  ODE.set_initial( [ pymc.FFVar(1.) ] )
  ODE.set_quadrature( QUAD, Q )
  ODE.set_function( F )
#  print( ODE.eqn_function )

  ODE.options.DISPLEVEL = 0
  ODE.options.INTMETH   = ODE.options.MSBDF
  ODE.options.NLINSOL   = ODE.options.NEWTON #FIXEDPOINT
  ODE.options.LINSOL    = ODE.options.DENSE  #DIAG
  ODE.setup()

#  ODE.options.DISPLEVEL = 1
#  stat = ODE.solve_state( [-1e0]*NS )

  return ODE


def do_test( ODE ):

  OpODE = cronos.FFODE()
  DAG = cronos.FFGraphExt()
  NU = len( ODE.var_parameter )
  U = []
  for i in range( NU ):
    U.append( pymc.FFVar(DAG,"U"+str(i)) )
  F = OpODE( U, ODE );

#  print( "F @(-1): ", DAG.eval( F, U, [-1e0]*NU ) )
#  SGF = DAG.subgraph( F )
#  DAG.output( SGF )
#  DAG.dot_script( F, "F.dot" )

  # Define NLP
  NLP = canon.DOSLV()
  NLP.set_dag( DAG )
  NLP.add_variable( U, -1e1, 1e1 )
  NLP.set_objective( NLP.MIN, F[0] )
  NLP.add_constraint( NLP.EQ, F[1] )

  NLP.options.FEASTOL   = 1e-6;
  NLP.options.OPTIMTOL  = 1e-6;
  NLP.options.MAXITER   = 20;
  NLP.options.GRADMETH  = NLP.options.FSYM;
  NLP.options.GRADCHECK = 0;

  NLP.setup()
  NLP.options.DISPLEVEL = 1;
  NLP.solve( [-1e0]*NU )

  print( "status:", NLP.status )
  print( "solution point:", NLP.solution.x )
  print( "solution value:", NLP.solution.f[0] )
#  print( NLP.solution )

  ODE.options.DISPLEVEL = 1
  ODE.solve_state( NLP.solution.x )
  print( "solution trajectory: ", ODE.val_state )
  

nlp_test()

ODE = ode_define( 20 )
do_test( ODE )


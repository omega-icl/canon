import pymc
import cronos
import canon

def nlp_test():

  # Define DAG
  DAG = pymc.FFGraph()
  X1 = pymc.FFVar(DAG,"X1")
  X2 = pymc.FFVar(DAG,"X2")
  C  = pymc.FFVar(DAG,"C")

  # Define NLP
  NLP = canon.NLPSLV()
  NLP.set_dag( DAG )
  NLP.add_parameter( [C] )
  NLP.add_decision( [X1,X2], [0.,0.], [6.,4.] )
  NLP.set_objective( NLP.MAX, X1+X2 )
  NLP.add_constraint( NLP.LE, X1*X2-C )

  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = NLP.options.BSYM;

  NLP.setup()
  NLP.options.DISPLEVEL = 1;
  NLP.solve( [0.,0.], [3.] )

  print( "status:", NLP.status )
  print( "solution point:", NLP.solution.x )
  print( "solution value:", NLP.solution.f[0] )
#  print( NLP.solution )

  NLP.options.DISPLEVEL = 1;
  NLP.solve( 8, [3.] )

  print( "status:", NLP.status )
  print( "solution point:", NLP.solution.x )
  print( "solution value:", NLP.solution.f[0] )
#  print( NLP.solution )


def gams_test():

  # Define NLP
  NLP = canon.NLPSLV()
  NLP.read( "ex6_1_4.gms", False )
  NLP.options.FEASTOL   = 1e-8;
  NLP.options.OPTIMTOL  = 1e-8;
  NLP.options.GRADMETH  = NLP.options.BSYM;

  NLP.setup()
  NLP.options.DISPLEVEL = 1;
  NLP.solve()

  print( "status:", NLP.status )
  print( "solution point:", NLP.solution.x )
  print( "solution value:", NLP.solution.f[0] )  


def do_test( NS ):

  import numpy as np

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

  OpODE = cronos.FFODE()
#  F = OpODE( U, ODE );

#  print( "F @(-1): ", DAG.eval( F, U, [-1e0]*NU ) )
#  SGF = DAG.subgraph( F )
#  DAG.output( SGF )
#  DAG.dot_script( F, "F.dot" )

  # Define NLP
  NLP = canon.NLPSLV()
  NLP.set_dag( DAG )
  NLP.add_decision( U, -1e1, 1e1 )
  NLP.set_objective( NLP.MIN, OpODE( 0, U, ODE ) ) #F[0] )
  NLP.add_constraint( NLP.EQ, OpODE( 1, U, ODE ) ) #F[1] )

  NLP.options.FEASTOL   = 1e-6;
  NLP.options.OPTIMTOL  = 1e-6;
  NLP.options.MAXITER   = 20;
  NLP.options.GRADMETH  = NLP.options.FSYM;
  NLP.options.GRADCHECK = 0;

  NLP.setup()
  NLP.options.DISPLEVEL = 1;
  NLP.solve( [-1e0]*NS )

  print( "status:", NLP.status )
  print( "solution point:", NLP.solution.x )
  print( "solution value:", NLP.solution.f[0] )
#  print( NLP.solution )

  ODE.options.DISPLEVEL = 1
  ODE.solve_state( NLP.solution.x )
  print( "solution trajectory: ", ODE.val_state )
  

nlp_test()

#gams_test()

#do_test( 20 )


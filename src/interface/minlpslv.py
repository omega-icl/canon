#!/usr/bin/env python
# coding: utf-8

# # Mixed-Integer Nonlinear Optimization in CANON
# 
# The module `MINLPSLV` in `CANON` provides an implementation of local search methods for mixed-integer nonlinear optimization (MINLP), using either outer-approximation (OA) or branch-and-bound (BB). The local NLP solves are conducted using `NLPSLV`, in combination with the commercial solver [GUROBI](https://www.gurobi.com/) for MIP solves in the OA variant. `CANON` leverages the DAG evaluation and automatic differentiation capability in `pyMC` to generates all the necessary function evaluations and derivatives internally. One can also leverage capability in `CRONOS` to embed ordinary differential equations (ODEs) into a DAG to enable mixed-integer dynamic optimization (MIDO).

# ## Defining and Solving a Mixed-Integer Nonlinear Program (MINLP)

# Suppose we want to solve the following [MINLP](https://www.minlplib.org/st_e15.html):
# $$\begin{align}
# \min_{{\bf x}, {\bf y}}\ & 2x_1 + 3x_2 + 1.5y_1 + 2y_2 - 0.5y_3 \\
# \text{s.t.}\ \ & (x_1)^2 + y_1 = 1.25 \\
# & (x_2)^{1.5} + 1.5 y_2 = 3\\
# & x_1 + y_1 \leq 1.6\\
# & 1.333 x_2 + y_2 \leq 3\\
# & y_1 + y_2 \geq y_3\\
# & ({\bf x},{\bf y}) \in [0,10]^2\times \{0,1\}^3\,.
# \end{align}$$
# 

# We start by importing both the `PyMC`, `CRONOS` and `CANON` modules:

# In[1]:


import pymc
import cronos
import canon


# An environment `NLPSLV` is created and populated with the decision variables, cost and constraint expressions in the model:

# In[2]:


# Define DAG
DAG = pymc.FFGraph()
X = DAG.add_vars( 2, "X")
Y = DAG.add_vars( 3, "Y" )


# In[3]:


# Define MINLP
MINLP = canon.MINLPSLV()
MINLP.set_dag( DAG )
MINLP.add_decision( X, 0, 10 )
MINLP.add_decision( Y, 0, 1, 1 )
MINLP.set_objective( MINLP.MIN, 2*X[0] + 3*X[1] + 1.5*Y[0] + 2*Y[1] - 0.5*Y[2] )
MINLP.add_constraint( MINLP.EQ, X[0]**2 + Y[0] - 1.25 )
MINLP.add_constraint( MINLP.EQ, X[1]**1.5 + 1.5*Y[1] - 3 )
MINLP.add_constraint( MINLP.LE, X[0] + Y[0] - 1.6 )
MINLP.add_constraint( MINLP.LE, 1.333*X[1] + Y[1] - 3 )
MINLP.add_constraint( MINLP.GE, Y[0] + Y[1] - Y[2] )


# Options can be modified as follows - these options can vary depending on the solver used:

# In[4]:


MINLP.options.FEASTOL   = 1e-7;
MINLP.options.CVATOL    = 1e-5;
MINLP.options.SEARCHALG = MINLP.options.OA;
MINLP.options.LINMETH   = MINLP.options.CVX;
MINLP.options.DISPLEVEL = 1;
  
#help( MINLP.options )


# After setup, the NLP model can be solved to local optimality by passing an initial guess for the decision variables and the parameters to the method `solve`:

# In[5]:


MINLP.setup()
#MINLP.solve()


# In[6]:


MINLP.solve() 
print( "status:", MINLP.status )
print( "solution point:", MINLP.solution.x )
print( "solution value:", MINLP.solution.f[0] )
#print( NLP.solution )


# Similarly, the same model can be read from a GAMS file and solved:

# In[7]:


# Read MINLP from GAMS file
MINLP2 = canon.MINLPSLV()
MINLP2.read( "st_e15.gms" )
MINLP2.options = MINLP.options


# In[8]:


MINLP2.setup()
MINLP2.solve()


# In[ ]:





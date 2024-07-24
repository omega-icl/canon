# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

#PATH_MC = $(shell cd $(HOME)/Programs/bitbucket/mcpp30 ; pwd)
#include $(PATH_MC)/src/makeoptions.mk

PATH_CRONOS    = $(shell cd $(HOME)/Programs/bitbucket/cronos; pwd)
include $(PATH_CRONOS)/src/makeoptions.mk

PATH_CANON    = $(shell cd $(HOME)/Programs/bitbucket/canon30; pwd)

PATH_CLI =
LIB_CLI  = -lboost_program_options
INC_CLI  =
FLAG_CLI =

PATH_SOBOL  =
LIB_SOBOL   = -lboost_random
INC_SOBOL   =
FLAG_SOBOL  = -DMC__USE_SOBOL

PATH_IPOPT   = $(IPOPT_HOME)
#LIB_NLP      = -L$(PATH_IPOPT)/lib -lipopt -lm -ldl -lcoinhsl -lmetis -llapack -lblas -lgfortran -lm -lquadmath -fopenmp
#INC_NLP      = -I$(PATH_IPOPT)/include
#FLAG_NLP   = -DMC__USE_IPOPT

PATH_SNOPT = $(SNOPT_HOME)
LIB_NLP    = -L$(PATH_SNOPT)/lib -lsnopt7_cpp
INC_NLP    = -I$(PATH_SNOPT)/include
FLAG_NLP   = -DMC__USE_SNOPT

PATH_CPLEX   = /opt/ibm/ILOG/CPLEX_Studio1210/cplex
PATH_CONCERT = /opt/ibm/ILOG/CPLEX_Studio1210/concert
#LIB_MIP      = -L$(PATH_CPLEX)/lib/x86-64_linux/static_pic -lilocplex -lcplex \
#               -L$(PATH_CONCERT)/lib/x86-64_linux/static_pic -lconcert \
#               -lm -pthread
#INC_MIP      = -I$(PATH_CPLEX)/include -I$(PATH_CONCERT)/include
#FLAG_MIP    = -DMC__USE_CPLEX -m64 -fPIC -fexceptions -DIL_STD -Wno-ignored-attributes

PATH_GUROBI = $(GUROBI_HOME)
LIB_MIP     = -L$(PATH_GUROBI)/lib -lgurobi_c++ -lgurobi110 -pthread
INC_MIP     = -I$(PATH_GUROBI)/include
FLAG_MIP    = -DMC__USE_GUROBI

PATH_GAMS = /opt/gams/gams47.3_linux_x64_64_sfx
LIB_GAMS  =
INC_GAMS  = -I$(PATH_GAMS)/apifiles/C/api
FLAG_GAMS = -DMC__WITH_GAMS=\"$(PATH_GAMS)\"

# COMPILATION <<-- CHANGE AS APPROPRIATE -->>

PROF = #-pg
OPTIM = -O2
DEBUG = #-g
WARN  = -Wall -Wno-misleading-indentation -Wno-unknown-pragmas -Wno-parentheses -Wno-unused-result
CPP17 = -std=c++17
CC    = gcc-13
CPP   = g++-13
# CPP   = icpc

# <<-- NO CHANGE BEYOND THIS POINT -->>

FLAG_CPP  = $(DEBUG) $(OPTIM) $(CPP17) $(WARN) $(PROF)
LINK      = $(CPP)
FLAG_LINK = $(PROF)

FLAG_CANON = $(FLAG_CRONOS) $(FLAG_CLI) $(FLAG_SOBOL) $(FLAG_NLP) $(FLAG_MIP) $(FLAG_GAMS)
LIB_CANON  = $(LIB_CRONOS) $(LIB_CLI) $(LIB_SOBOL) $(LIB_NLP) $(LIB_MIP) $(LIB_GAMS)
INC_CANON  = -I$(PATH_CANON)/src $(INC_CRONOS) $(INC_CLI) $(INC_SOBOL) $(INC_NLP) $(INC_MIP) $(INC_GAMS)


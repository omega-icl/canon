# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_MC = $(shell cd $(HOME)/Programs/mcpp ; pwd)
#echo mcpp path is $(PATH_MC);
include $(PATH_MC)/src/makeoptions.mk

PATH_CLI =
LIB_CLI  = -lboost_program_options
INC_CLI  =
FLAG_CLI =

PATH_SOBOL  =
LIB_SOBOL   = -lboost_random
INC_SOBOL   =
FLAG_SOBOL  = -DMC__USE_SOBOL

PATH_IPOPT   = /opt/CoinIpopt
PATH_COINHSL = /opt/coinhsl
#LIB_NLP      = -L$(PATH_IPOPT)/lib -L$(PATH_COINHSL)/lib -lipopt -lcoinhsl -llapack -lblas -lgfortran -lm -lquadmath -lblas -lgfortran -lm -lquadmath -lm -ldl -lcoinmumps -lblas -lgfortran -lm -lquadmath -lgfortran -lm -lquadmath -lcoinmetis
#INC_NLP      = -I$(PATH_IPOPT)/include
#FLAG_NLP   = -DMC__USE_IPOPT

PATH_SNOPT = $(SNOPT_HOME)
LIB_NLP    = -L$(PATH_SNOPT)/lib -lsnopt7_cpp
INC_NLP    = -I$(PATH_SNOPT)/include
FLAG_NLP   = -DMC__USE_SNOPT

PATH_CPLEX   = /opt/ibm/ILOG/CPLEX_Studio128/cplex
PATH_CONCERT = /opt/ibm/ILOG/CPLEX_Studio128/concert
#LIB_MIP      = -L$(PATH_CPLEX)/lib/x86-64_linux/static_pic -lilocplex -lcplex \
#               -L$(PATH_CONCERT)/lib/x86-64_linux/static_pic -lconcert \
#               -lm -pthread
#INC_MIP      = -I$(PATH_CPLEX)/include -I$(PATH_CONCERT)/include
#FLAG_MIP    = -DMC__USE_CPLEX -m64 -fPIC -fexceptions -DIL_STD -Wno-ignored-attributes

PATH_GUROBI = $(GUROBI_HOME)
LIB_MIP     = -L$(PATH_GUROBI)/lib -lgurobi_g++5.2 -lgurobi95 -pthread
INC_MIP     = -I$(PATH_GUROBI)/include
FLAG_MIP    = -DMC__USE_GUROBI

PATH_GAMS = /opt/gams/gams35.1_linux_x64_64_sfx
LIB_GAMS  =
INC_GAMS  = -I$(PATH_GAMS)/apifiles/C/api
FLAG_GAMS = -DMC__WITH_GAMS=\"$(PATH_GAMS)\"

FLAG_DEP = -fPIC $(FLAG_MC) $(FLAG_CLI) $(FLAG_SOBOL) $(FLAG_NLP) $(FLAG_MIP) $(FLAG_GAMS)
LIB_DEP  = $(LIB_MC) $(LIB_CLI) $(LIB_SOBOL) $(LIB_NLP) $(LIB_MIP) $(LIB_GAMS)
INC_DEP  = $(INC_MC) $(INC_CLI) $(INC_SOBOL) $(INC_NLP) $(INC_MIP) $(INC_GAMS)

# COMPILATION <<-- CHANGE AS APPROPRIATE -->>

DEBUG =
#PROF = -pg
OPTIM = -Ofast
WARN  = -Wall -Wno-misleading-indentation -Wno-unknown-pragmas -Wno-unused-result
CPP17 = -std=c++17

CC  = gcc-10
CPP = g++-10
# CPP = icpc
FLAG_CPP = $(DEBUG) $(PROF) $(OPTIM) $(CPP17) $(WARN) $(FLAG_DEP) 

LINK = $(CPP)
FLAG_LINK = 

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)


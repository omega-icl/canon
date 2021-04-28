# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_MC = $(HOME)/Programs/mcpp
LIB_MC  = -llapack -lblas -lmc13 -lmc21 -lmc33 -lgfortran
INC_MC  = -I$(PATH_MC)/src/mc -I$(PATH_MC)/src/3rdparty/fadbad++ -I$(PATH_MC)/src/3rdparty/cpplapack-2015.05.11-1/include
FLAG_MC = -Wno-misleading-indentation -Wno-unknown-pragmas -DMC__USE_HSL -DMC__USE_PROFIL

PATH_PROFIL = /opt/Profil-2.0.8
LIB_PROFIL  = -L$(PATH_PROFIL)/lib -lProfilPackages -lProfil -lBias -llr
INC_PROFIL  = -I$(PATH_PROFIL)/include

PATH_BOOST = 
LIB_BOOST  = -lboost_program_options
INC_BOOST  = 

PATH_FILIB  = /opt/filib++
LIB_FILIB   = -L$(PATH_FILIB)/lib -lprim
INC_FILIB   = -I$(PATH_FILIB)/include -I$(PATH_FILIB)/include/interval
#FLAG_FILIB = -frounding-math -ffloat-store
FLAG_FILIB = -frounding-math

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
LIB_MIP     = -L$(PATH_GUROBI)/lib -lgurobi_g++5.2 -lgurobi91 -pthread
INC_MIP     = -I$(PATH_GUROBI)/include
FLAG_MIP    = -DMC__USE_GUROBI

PATH_GAMS = /opt/gams/gams32.2_linux_x64_64_sfx
LIB_GAMS  = 
INC_GAMS  = -I$(PATH_GAMS)/apifiles/C/api
FLAG_GAMS = -DMC__WITH_GAMS=\"$(PATH_GAMS)\"

FLAG_DEP = -fPIC $(FLAG_MC) $(FLAG_FILIB) $(FLAG_BOOST) $(FLAG_MIP) $(FLAG_NLP) $(FLAG_SOBOL) $(FLAG_GAMS)
LIB_DEP  = $(LIB_MC) $(LIB_PROFIL) $(LIB_FILIB) $(LIB_BOOST) $(LIB_MIP) $(LIB_NLP) $(LIB_SOBOL) $(LIB_GAMS)
INC_DEP  = $(INC_MC) $(INC_PROFIL) $(INC_FILIB) $(INC_BOOST) $(INC_MIP) $(INC_NLP) $(INC_SOBOL) $(INC_GAMS)

# COMPILATION <<-- CHANGE AS APPROPRIATE -->>

DEBUG = -g
#PROF = -pg
#OPTIM = -Ofast
WARN  = -Wall
CPP17 = -std=c++17

CC  = gcc-9
CPP = g++-9
# CPP = icpc
FLAG_CPP = $(DEBUG) $(PROF) $(OPTIM) $(CPP17) $(WARN) $(FLAG_DEP) 

LINK = $(CPP)
FLAG_LINK = 

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)


# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
LIB_CANON  = -larmadillo #-L$(PATH_CANON)/lib -Wl,-rpath=$(PATH_CANON)/lib -lcanon
INC_CANON  = -I$(PATH_CANON)/include

include $(PATH_CANON)/src/makeoptions.mk

PATH_SUNDIALS = /opt/sundials-7.0.0
LIB_SUNDIALS = -L$(PATH_SUNDIALS)/lib -lsundials_cvodes -lsundials_nvecserial -lsundials_core -llapack -lblas
INC_SUNDIALS = -I$(PATH_SUNDIALS)/include

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)

#####

test% : test%.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $< $(LIB_CANON) $(LIB_SUNDIALS) $(LIB_DEP) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_SUNDIALS) $(INC_CRONOS) $(INC_DEP) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_DEP) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

#####

clean:
	rm -f test? test?b
	rm -f *.o *.out *.dat *.eps *.dot *.svg *~


# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
LIB_CANON = #-L$(PATH_CANON)/lib -Wl,-rpath=$(PATH_CANON)/lib -lcanon
INC_CANON = -I$(PATH_CANON)/include

include $(PATH_CANON)/src/makeoptions.mk

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)

#####

doxydoc : doxydoc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

test% : test%.o
	$(CPP) $< $(LIB_CANON) $(LIB_DEP) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_DEP) $< -o $@

#####

clean:
	rm -f doxydoc test? test?b *.o *.out *.dat *.eps *.dot *.svg *~


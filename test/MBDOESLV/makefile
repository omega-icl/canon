# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
include $(PATH_CANON)/src/makeoptions.mk

#####

all : test0 test1 test2 test3

test% : test%.o
	$(CPP) $< $(LIB_CANON) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

#####

clean:
	rm -f test? test?b
	rm -f *.o *.out *.dat *.eps *.dot *.svg *.log *~


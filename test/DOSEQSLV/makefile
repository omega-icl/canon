# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
include $(PATH_CANON)/src/makeoptions.mk

#####

test% : test%.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $< $(LIB_CANON) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

#####

clean:
	rm -f test? test?b
	rm -f *.o *.out *.dat *.eps *.dot *.svg *.log *~


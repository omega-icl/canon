# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
include $(PATH_CANON)/src/makeoptions.mk

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)

#####

all : doxydoc main ReLUANN

doxydoc : doxydoc.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

main : main.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

ReLUANN : ReLUANN.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

#####

clean:
	rm -f doxydoc main ReLUANN
	rm -f *.o *.lp *.lst *.out *.dat *.eps *.dot *.svg *.ilp *.log *~


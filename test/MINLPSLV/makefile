# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
LIB_CANON = #-L$(PATH_CANON)/lib -Wl,-rpath=$(PATH_CANON)/lib -lcanon
INC_CANON = -I$(PATH_CANON)/include

include $(PATH_CANON)/src/makeoptions.mk

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)

#####

main : main.o gmomcc.o gevmcc.o gcmt.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@ $(LDFLAGS)

doxydoc : doxydoc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_DEP) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(INC_CRONOS) $(INC_DEP) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

#####

clean:
	rm -f doxydoc main *.o *.lp *.lst *.out *.dat *.eps *.dot *.svg *~


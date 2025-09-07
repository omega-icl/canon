# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
include $(PATH_CANON)/src/makeoptions.mk

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)

#####

all : doxydoc test_MINLPREF test_MINLPBND test_ANN test_EXTERN CS_Xmin CS_DB

doxydoc : doxydoc.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

test_MINLPREF : test_MINLPREF.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

test_MINLPBND : test_MINLPBND.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

test_EXTERN : test_EXTERN.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

test_ANN : test_ANN.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

CS_DB : CS_DB.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

CS_Xmin : CS_Xmin.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

UNIFAC_relu : UNIFAC_relu.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $(PROF) $^ $(LIB_CANON) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

#####

clean:
	rm -f doxydoc test_MINLPREF test_MINLPBND test_ANN test_EXTERN CS_DB CS_Xmin UNIFAC_relu
	rm -f *.o *.lp *.lst *.out *.dat *.eps *.dot *.svg *.ilp *.log *~


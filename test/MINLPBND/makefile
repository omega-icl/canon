# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_CANON = $(shell cd ../.. ; pwd)
LIB_CANON = #-L$(PATH_CANON)/lib -Wl,-rpath=$(PATH_CANON)/lib -lcanon
INC_CANON = -I$(PATH_CANON)/include

include $(PATH_CANON)/src/makeoptions.mk

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)

#####

all : doxydoc test_ANN test_MINLPREF test_MINLPBND CS_Xmin CS_DB

doxydoc : doxydoc.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

test_MINLPREF : test_MINLPREF.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

test_MINLPBND : test_MINLPBND.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

test_ANN : test_ANN.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

CS_DB : CS_DB.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

CS_Xmin : CS_Xmin.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

UNIFAC_relu : UNIFAC_relu.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_DEP) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_DEP) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

#####

clean:
	rm -f doxydoc test_MINLPREF test_MINLPBND test_ANN CS_DB CS_Xmin UNIFAC_relu
	rm -f *.o *.lp *.lst *.out *.dat *.eps *.dot *.svg *.ilp *~


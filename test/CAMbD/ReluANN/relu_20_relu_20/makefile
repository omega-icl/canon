PATH_CANON = $(shell cd ../../../.. ; pwd)
LIB_CANON = 
INC_CANON = -I$(PATH_CANON)/include

include $(PATH_CANON)/src/makeoptions.mk

LDFLAGS = -ldl -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(PATH_GAMS)

#####

relu_relu_model_20_20 : relu_relu_model_20_20.o gmomcc.o gevmcc.o optcc.o palmcc.o
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_DEP) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(INC_CANON) $(INC_DEP) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

#####

clean:
	rm -f relu_relu_model_20_20 *.o *.lp *.lst *.out *.dat *.eps *.dot *.svg *~


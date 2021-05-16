# This makefile compiles a chared library of CANON and it creates symbolic links
# to the header files in $(incpath), binaries in $(binpath), and libraries in $(libpath)

include $(srcpath)/makeoptions.mk

#####

incobjs = base_ae.hpp base_opt.hpp base_nlp.hpp \
          GamsNLinstr.h gamsio.hpp \
          nlpslv_ipopt.hpp nlpslv_snopt.hpp mipslv_gurobi.hpp \
          minlpslv.hpp minlpbnd.hpp minlgo.hpp

binobjs = minlgo.o gmomcc.o gevmcc.o optcc.o palmcc.o

libobjs = 

binname = canon

#libname = libcanon.so

#####

install: dispBuild canon canon_lib dispInstall
	@if test ! -e $(binpath)/$(binname); then \
		echo creating symolic link to executable $(binname); \
		cd $(binpath) ; ln -s $(srcpath)/$(binname) $(binname); \
	fi
#	@if test ! -e $(libpath)/$(libname); then \
#		echo creating symolic link to shared library $(libname); \
#		cd $(libpath) ; ln -s $(srcpath)/$(libname) $(libname); \
#	fi
	@for INC in $(incobjs); do \
		if test ! -e $(incpath)/$$INC; then \
			echo creating symbolic link to header file $$INC; \
			cd $(incpath); ln -s $(srcpath)/$$INC $$INC; \
		fi; \
	done
	@echo

canon: $(binobjs)
	$(CPP) $^ $(LIB_CANON) $(LIB_DEP) -o $@ $(LDFLAGS)

canon_lib: $(libobjs)
#	$(CPP) -shared -o $(libname) $(libobjs)

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(INC_DEP) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(INC_DEP) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

dispBuild:
	@echo
	@(echo '***Compiling CANON library (ver.' $(version)')***')
	@echo

dispInstall:
	@echo
	@(echo '***Installing CANON library (ver.' $(version)')***')
	@echo

#####

clean: dispClean
	rm -fi $(libobjs) $(binobjs) $(binname) $(libname)

dispClean:
	@echo
	@(echo '***Cleaning CANON directory (ver.' $(version)')***')
	@echo

#####

cleandist: dispCleanInstall
	rm -f $(libobjs) $(binname) $(libname)
	-(cd $(incpath) ; rm -f $(incobjs))
	-(cd $(binpath) ; rm -f $(binname))
#	-(cd $(libpath) ; rm -f $(libname))
	
dispCleanInstall:
	@echo
	@(echo '***Uninstalling CANON library (ver.' $(version)')***')
	@echo

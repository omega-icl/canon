# This makefile compiles a chared library of CRONOS and it creates symbolic links
# to the header files in $(incpath), binaries in $(binpath), and libraries in $(libpath)

include $(srcpath)/makeoptions.mk

#####

incobjs = base_ae.hpp base_opt.hpp base_nlp.hpp \
          aebnd.hpp \
           \
           \
          nlpslv_ipopt.hpp \
          nlpslv_snopt.hpp mipslv_gurobi.hpp nlpbnd.hpp \
          nlgo.hpp \
          gamscronos.hpp

#incobjs = base_ae.hpp base_opt.hpp base_nlp.hpp base_de.hpp base_do.hpp base_rk.hpp \
#          base_expand.hpp base_sundials.hpp aebnd.hpp odeslv_base.hpp odeslv_sundials.hpp \
#          odeslvs_base.hpp odeslvs_sundials.hpp odebnd_base.hpp odebnd_sundials.hpp \
#          odebnd_val.hpp odebnd_expand.hpp iodebnd_base.hpp iodebnd_sundials.hpp \
#          odebnds_base.hpp odebnds_sundials.hpp nlpslv_ipopt.hpp doseqslv_ipopt.hpp \
#          nlpslv_snopt.hpp mipslv_gurobi.hpp nlpbnd.hpp sbb.hpp sbp.hpp lprelax_base.hpp \
#          csearch_base.hpp nlgo.hpp nlcp.hpp doseqgo.hpp gamscronos.hpp

binobjs = nlgo.o \
          gmomcc.o gevmcc.o optcc.o palmcc.o

libobjs = 
#libobjs = odeslvs_sundials.o odeslv_sundials.o
#libobjs = base_de.o base_sundials.o odeslv_base.o odeslv_sundials.o odeslvs_base.o \
#          odeslvs_sundials.o

binname = cronos

#libname = libcronos.so

#####

install: dispBuild cronos cronos_lib dispInstall
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

cronos: $(binobjs)
	$(CPP) $^ $(LIB_CRONOS) $(LIB_DEP) -o $@ $(LDFLAGS)

#cronos_lib: $(libobjs)
#	$(CPP) -shared -o $(libname) $(libobjs)

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(INC_DEP) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(INC_DEP) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

dispBuild:
	@echo
	@(echo '***Compiling CRONOS library (ver.' $(version)')***')
	@echo

dispInstall:
	@echo
	@(echo '***Installing CRONOS library (ver.' $(version)')***')
	@echo

#####

clean: dispClean
	rm -fi $(libobjs) $(binobjs) $(binname) $(libname)

dispClean:
	@echo
	@(echo '***Cleaning CRONOS directory (ver.' $(version)')***')
	@echo

#####

cleandist: dispCleanInstall
	rm -f $(libobjs) $(binname) $(libname)
	-(cd $(incpath) ; rm -f $(incobjs))
	-(cd $(binpath) ; rm -f $(binname))
	-(cd $(libpath) ; rm -f $(libname))
	
dispCleanInstall:
	@echo
	@(echo '***Uninstalling CRONOS library (ver.' $(version)')***')
	@echo

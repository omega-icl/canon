# This makefile compiles the python interface and creates a symbolic link
# to the library in $(libpath)

include $(srcpath)/makeoptions.mk

#####

libobjs   = base.o mipslv.o nlpslv.o minlpslv.o canon.o \
            gmomcc.o gevmcc.o optcc.o palmcc.o
libname   = canon.so
libdep    = pymc.so cronos.so

#####

install: dispBuild $(libname) dispInstall
	@if test ! -e $(libpath)/$(libname); then \
		echo creating symbolic link to shared library $(libname); \
		cd $(libpath); ln -s $(interfacepath)/$(libname) $(libname); \
	fi
	@for DEP in $(libdep); do \
		echo dependent library $$DEP; \
		if test ! -e $$DEP; then \
			ln -s $(PATH_CRONOS)/lib/$$DEP; \
		fi; \
		if test ! -e $(libpath)/$$DEP; then \
			echo creating symbolic link to shared library $$DEP; \
			cd $(libpath); ln -s $(PATH_CRONOS)/lib/$$DEP; \
		fi; \
	done
	@echo

$(libname): $(libobjs)
	$(CPP) -shared -Wl,--export-dynamic $(libobjs) $(LIB_CANON) -o $(libname)

%.o: %.cpp
	$(CPP) $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $(INC_PYBIND11) -c $< -o $@

%.o: %.c
	$(CPP) -c $(FLAG_CPP) $(FLAG_CANON) $(INC_CANON) $< -o $@

%.c: $(PATH_GAMS)/apifiles/C/api/%.c
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
	rm -fi $(libobjs) $(libname) $(libdep)

dispClean:
	@echo
	@(echo '***Cleaning CANON directory (ver.' $(version)')***')
	@echo

#####

uninstall: dispUninstall
	rm -f $(libobjs) $(libname) $(libdep)
	-(cd $(libpath); rm -f $(libname) $(libdep))

dispUninstall:
	@echo
	@(echo '***Uninstalling CANON library (ver.' $(version)')***')
	@echo

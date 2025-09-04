# This is the main makefile for CANON
# Run `make help' for a list of available targets

export version = 4.0

export rootpath       = $(shell cd .. ; pwd)
export srcpath        = $(rootpath)/src
export interfacepath  = $(rootpath)/src/interface
export incpath        = $(rootpath)/include
export binpath        = $(rootpath)/bin
export libpath        = $(rootpath)/lib

#####

install: dirs_create interface_install canon_install

dirs_create :
	@if test ! -e $(incpath)/; then \
		echo creating directory for header files; \
		mkdir $(incpath); \
	fi
	@if test ! -e $(binpath)/; then \
		echo creating directory for executables; \
		mkdir $(binpath); \
	fi
	@if test ! -e $(libpath)/; then \
		echo creating directory for compiled libraries; \
		mkdir $(libpath); \
	fi

canon_install:
	-(cd $(srcpath); make -f canon.mk install)

interface_install:
	-(cd $(interfacepath); make -f canon.mk install)

#####

clean: canon_clean interface_clean

canon_clean:
	-(cd $(srcpath); make -f canon.mk clean)

interface_clean:
	-(cd $(interfacepath); make -f canon.mk clean)

#####

uninstall: canon_uninstall interface_uninstall

canon_uninstall:
	-(cd $(srcpath); make -f canon.mk uninstall)

interface_uninstall:
	-(cd $(interfacepath); make -f canon.mk uninstall)

#####

help:
	@echo
	@echo "CANON Makefile (ver. $(version))"
	@echo "Available targets:"
	@echo "  install     -  Create 'include', 'lib' and 'bin' directories, compile a shared library,"
	@echo "                 create symbolic links to header files and compiled binaries and libraries"
	@echo "  clean       -  Remove object files"
	@echo "  uninstall   -  Remove binaries, libraries, object files, and symbolic links"
	@echo "  help        -  Show this help message"
	@echo

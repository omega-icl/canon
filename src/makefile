# This is the main makefile for CANON
# - 'make install' creates the directories 'include', 'lib' and 'bin' (if not already
#   existing), compiles a shared library, and creates symbolic links to header files
#   and the compiled binaries and libraries
# - `make cleaninstall' removes the binaries, libraries, object files, and links

export version = 3.0

export rootpath       = $(shell cd .. ; pwd)
export srcpath        = $(rootpath)/src
export interfacepath  = $(rootpath)/src/interface
export incpath        = $(rootpath)/include
export binpath        = $(rootpath)/bin
export libpath        = $(rootpath)/lib

#####

install: dirs_create canon_install interface_install

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

cleaninstall: canon_cleandist interface_cleandist

canon_cleandist:
	-(cd $(srcpath); make -f canon.mk cleandist)

interface_cleandist:
	-(cd $(interfacepath); make -f canon.mk cleandist)


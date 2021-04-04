# This is the main makefile for CANON
# - 'make install' creates the directories 'include', 'lib' and 'bin' (if not already
#   existing), compiles a shared library, and creates symbolic links to header files
#   and the compiled binaries and libraries
# - `make cleaninstall' removes the binaries, libraries, object files, and links

export version = 1.0

export rootpath = $(shell cd .. ; pwd)
export srcpath  = $(rootpath)/src
export incpath  = $(rootpath)/include
export binpath  = $(rootpath)/bin
export libpath  = $(rootpath)/lib

#####

install: dirs canon_inst

dirs :
	@if test ! -e $(incpath)/; then \
		echo creating directory for header files; \
		mkdir $(incpath); \
	fi
	@if test ! -e $(binpath)/; then \
		echo creating directory for executables; \
		mkdir $(binpath); \
	fi
#	@if test ! -e $(libpath)/; then \
#		echo creating directory for compiled libraries; \
#		mkdir $(libpath); \
#	fi

canon_inst:
	-(cd $(srcpath); make -f canon.mk install)

#####

clean: canon_clean

canon_clean:
	-(cd $(srcpath); make -f canon.mk clean)

#####

cleaninstall: canon_cleandist

canon_cleandist:
	-(cd $(srcpath); make -f canon.mk cleandist)


#------------------------------------------------------------------------------
# AMD Makefile
#------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

demos: all

# Compile all C code.  Do not compile the FORTRAN versions.
all:
	( cd Lib    ; $(MAKE) )
	( cd Demo   ; $(MAKE) )

# compile just the C-callable libraries (not Demos)
library:
	( cd Lib    ; $(MAKE) )

# compile the static libraries only
static:
	( cd Lib    ; $(MAKE) static )

# compile the FORTRAN libraries and demo programs (not compiled by "make all")
fortran:
	( cd Lib    ; $(MAKE) fortran )
	( cd Demo   ; $(MAKE) fortran )

# compile a FORTRAN demo program that calls the C version of AMD
# (not compiled by "make all")
cross:
	( cd Demo   ; $(MAKE) cross )

# remove object files, but keep the compiled programs and library archives
clean:
	( cd Lib    ; $(MAKE) clean )
	( cd Demo   ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) )
	( cd Doc    ; $(MAKE) clean )

# clean, and then remove compiled programs and library archives
purge:
	( cd Lib    ; $(MAKE) purge )
	( cd Demo   ; $(MAKE) purge )
	( cd MATLAB ; $(RM) $(CLEAN) ; $(RM) *.mex* )
	( cd Doc    ; $(MAKE) purge )

distclean: purge

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# get ready for distribution
dist: purge
	( cd Demo   ; $(MAKE) dist )
	( cd Doc    ; $(MAKE) )

ccode: library

lib: library

# install AMD
install:
	( cd Lib  ; $(MAKE) install )

# uninstall AMD
uninstall:
	( cd Lib  ; $(MAKE) uninstall )


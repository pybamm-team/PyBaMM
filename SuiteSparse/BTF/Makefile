#------------------------------------------------------------------------------
# BTF Makefile
#------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

default: library

include ../SuiteSparse_config/SuiteSparse_config.mk

library:
	( cd Lib ; $(MAKE) )

# compile the static libraries only
static:
	( cd Lib    ; $(MAKE) static )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) )

distclean:
	( cd Lib ; $(MAKE) distclean )
	( cd MATLAB ; $(RM) $(CLEAN) ; $(RM) *.mex* )

purge: distclean

# install BTF
install:
	( cd Lib ; $(MAKE) install )

# uninstall BTF
uninstall:
	( cd Lib ; $(MAKE) uninstall )

#------------------------------------------------------------------------------
# CXSparse Makefile
#------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

default: C

include ../SuiteSparse_config/SuiteSparse_config.mk

C:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

all: C cov

library:
	( cd Lib ; $(MAKE) )

# compile the static libraries only
static:
	( cd Lib    ; $(MAKE) static )

cov:
	( cd Tcov ; $(MAKE) )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd MATLAB/CSparse ; $(RM) *.o cs_cl_*.c )
	( cd MATLAB/Test    ; $(RM) *.o cs_cl_*.c )

purge:
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd MATLAB/CSparse ; $(RM) *.o cs_cl_*.c *.mex* )
	( cd MATLAB/Test    ; $(RM) *.o cs_cl_*.c *.mex* )

distclean: purge

# install CXSparse
install:
	( cd Lib ; $(MAKE) install )

# uninstall CXSparse
uninstall:
	( cd Lib ; $(MAKE) uninstall )

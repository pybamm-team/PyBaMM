#-------------------------------------------------------------------------------
# SuiteSparseQR Makefile
#-------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

all:
	( cd Lib ; $(MAKE) --jobs=$(JOBS) )
	( cd Demo ; $(MAKE) )

library:
	( cd Lib ; $(MAKE) --jobs=$(JOBS) )

# compile the static libraries only
static:
	( cd Lib    ; $(MAKE) static --jobs=$(JOBS) )

demos:
	( cd Demo ; $(MAKE) )

gpu:
	( cd Demo ; $(MAKE) gpu )

cov:
	( cd Tcov ; $(MAKE) )

purge:
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd Doc ; $(MAKE) purge )
	( cd MATLAB ; $(RM) $(CLEAN) rename.h *.mex* )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) rename.h )

distclean: purge

ccode: demos

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# install SPQR
install:
	( cd Lib ; $(MAKE) install )

# uninstall SPQR
uninstall:
	( cd Lib ; $(MAKE) uninstall )


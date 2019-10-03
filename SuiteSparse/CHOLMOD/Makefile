#-------------------------------------------------------------------------------
# CHOLMOD Makefile
#-------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

# Compile the C-callable libraries and the Demo programs.
all:
	( cd Lib ; $(MAKE) --jobs=$(JOBS) )
	( cd Demo ; $(MAKE) )

# Compile the C-callable libraries only.
library:
	( cd Lib ; $(MAKE) --jobs=$(JOBS) )

# compile the static libraries only
static:
	( cd Lib    ; $(MAKE) static --jobs=$(JOBS) )

# Remove all files not in the original distribution
purge:
	( cd Tcov ; $(MAKE) purge )
	( cd Lib ; $(MAKE) purge )
	( cd Valgrind ; $(MAKE) dopurge )
	( cd Demo ; $(MAKE) purge )
	( cd Doc ; $(MAKE) purge )
	( cd MATLAB ; $(RM) $(CLEAN) rename.h *.mex* GKlib_util.c GKlib_graph.c )

# Remove all files not in the original distribution, except keep the 
# compiled libraries.
clean:
	( cd Tcov ; $(MAKE) clean )
	( cd Lib ; $(MAKE) clean )
	( cd Valgrind ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) )

distclean: purge

ccode: all

# Run the test coverage suite.  Takes about 40 minutes on a 3.2GHz Pentium.
# Requires Linux (gcc, gcov).
cov:
	( cd Tcov ; $(MAKE) )

# Run the test coverage suite using Valgrind.  This takes a *** long *** time.
valgrind:
	( cd Valgrind ; $(MAKE) )

# Compile the C-callable libraries and the Demo programs.
demos:
	( cd Demo ; $(MAKE) )

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# install CHOLMOD
install:
	( cd Lib ; $(MAKE) install )

# uninstall CHOLMOD
uninstall:
	( cd Lib ; $(MAKE) uninstall )

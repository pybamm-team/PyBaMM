#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime Makefile
#-------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

# Compile the C-callable libraries only (there is no Demo for this package)
all: library

# Compile the C-callable libraries only.
library:
	( cd Lib ; $(MAKE) )

# compile the static libraries only
static:
	( cd Lib    ; $(MAKE) static )

# Remove all files not in the original distribution
purge:
	( cd Lib ; $(MAKE) purge )

# Remove all files not in the original distribution, except keep the 
# compiled libraries.
clean:
	( cd Lib ; $(MAKE) clean )

distclean: purge

ccode: all

install:
	( cd Lib ; $(MAKE) install )

uninstall:
	( cd Lib ; $(MAKE) uninstall )


#-------------------------------------------------------------------------------
# GPUQREngine Makefile
#-------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

# default is 'library' since there is no demo program
default: library

include ../SuiteSparse_config/SuiteSparse_config.mk

# Compile the C-callable libraries
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
	( cd Demo ; $(MAKE) purge )

# Remove all files not in the original distribution, except keep the 
# compiled libraries.
clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )

distclean: purge

ccode: all

# install GPUQREngine.  Note that the include files are not installed.
install:
	( cd Lib ; $(MAKE) install )

uninstall:
	( cd Lib ; $(MAKE) uninstall )

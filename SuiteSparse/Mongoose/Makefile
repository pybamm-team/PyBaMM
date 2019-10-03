#-------------------------------------------------------------------------------
# Mongoose/Makefile
#-------------------------------------------------------------------------------

# Mongoose, Timothy A. Davis, Scott P. Kolodziej, William W. Hager,
# S. Nuri Yeralan, (c) 2017-2018, All Rights Reserved.
# http://suitesparse.com   See Mongoose/Doc/License.txt for license.

#-------------------------------------------------------------------------------

# simple Makefile for Mongoose, relies on cmake to do the actual build.  Use
# the CMAKE_OPTIONS argument to this Makefile to pass options to cmake.

JOBS ?= 1

# build the Mongoose library (static and dynamic) and run a quick test
default:
	( cd build ; CC=$(CC) CXX=$(CXX) cmake $(CMAKE_OPTIONS) .. ; $(MAKE) --jobs=$(JOBS) ; ./bin/demo )

# just build the static and dynamic libraries; do not run the demo
library:
	( cd build ; CC=$(CC) CXX=$(CXX) cmake $(CMAKE_OPTIONS) .. ; $(MAKE) --jobs=$(JOBS) )

# the same as "make library"
static: library

# installs Mongoose to the install location defined by cmake, usually
# /usr/local/lib and /usr/local/include
install:
	( cd build ; CC=$(CC) CXX=$(CXX) cmake $(CMAKE_OPTIONS) .. ; $(MAKE) --jobs=$(JOBS) ; $(MAKE) install )

# create the Doc/Mongoose_UserGuide.pdf
docs:
	( cd build ; cmake $(CMAKE_OPTIONS) .. ; $(MAKE) userguide )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# run the extensive tests
test:
	( cd build ; CC=$(CC) CXX=$(CXX) cmake $(CMAKE_OPTIONS) .. ; $(MAKE) test )

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	rm -rf build/* Lib/* MATLAB/*.mex*
	rm -rf SuiteSparse_config/*.o SuiteSparse_config/*.a


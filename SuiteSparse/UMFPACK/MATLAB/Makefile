#-------------------------------------------------------------------------------
# UMFPACK Makefile just for cleanup (use umfpack_make.m to compile, in MATLAB)
#-------------------------------------------------------------------------------

default:
	- echo "Use umfpack_make.m in MATLAB to compile the UMFPACK mexFunction"

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

distclean: purge

purge: clean
	- $(RM) luflop.mex* umfpack.mex* rename.h GKlib_util.c GKlib_graph.c
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

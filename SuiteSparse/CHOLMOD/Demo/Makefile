#===============================================================================
# CHOLMOD/Demo/Makefile
#===============================================================================

# If you compile CHOLMOD with -DNPARTITION, then you do not need METIS or
# CCOLAMD. 

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------

LIB2 = $(LDFLAGS) -L../../lib -lcholmod -lamd -lcolamd -lsuitesparseconfig \
	$(LIB_WITH_PARTITION) $(LAPACK) $(BLAS) $(LDLIBS)

#-------------------------------------------------------------------------------
# With the CUDA BLAS:
ifneq ($(GPU_CONFIG),)
LIB2 += $(CUDART_LIB) $(CUBLAS_LIB)
endif
#-------------------------------------------------------------------------------

C = $(CC) $(CF) $(CHOLMOD_CONFIG) $(CONFIG_PARTITION)

code: library cholmod_demo cholmod_l_demo cholmod_simple

fortran: readhb readhb2 reade 

all: code
	./cholmod_demo < Matrix/bcsstk01.tri
	./cholmod_l_demo < Matrix/bcsstk01.tri
	./cholmod_demo < Matrix/lp_afiro.tri
	./cholmod_l_demo < Matrix/lp_afiro.tri
	./cholmod_demo < Matrix/can___24.mtx
	./cholmod_l_demo < Matrix/can___24.mtx
	./cholmod_demo < Matrix/c.tri
	./cholmod_l_demo < Matrix/c.tri
	./cholmod_simple < Matrix/c.tri
	./cholmod_simple < Matrix/can___24.mtx
	./cholmod_simple < Matrix/bcsstk01.tri

# run on a big matrix.  This exercises the GPU, if compiled to exploit it
big: code
	tcsh gpu.sh

distclean: purge

purge: clean
	- $(RM) cholmod_demo cholmod_l_demo readhb readhb2 reade
	- $(RM) cholmod_simple
	- $(RM) timelog.m
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

#-------------------------------------------------------------------------------
library: metis
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
ifneq (,$(findstring -DNCAMD, $(CONFIG_PARTITION)))
else
	( cd ../../CCOLAMD ; $(MAKE) library )
	( cd ../../CAMD ; $(MAKE) library )
endif

metis: ../../include/metis.h

../../include/metis.h:
	- ( cd ../.. && $(MAKE) metis )

#-------------------------------------------------------------------------------

I = -I../../include $(CUDA_INC)

cholmod_demo: cholmod_demo.c cholmod_demo.h
	$(C) -o cholmod_demo $(I) cholmod_demo.c $(LIB2)

cholmod_simple: cholmod_simple.c
	$(C) -o cholmod_simple $(I) cholmod_simple.c $(LIB2)

cholmod_l_demo: cholmod_l_demo.c cholmod_demo.h
	$(C) -o cholmod_l_demo $(I) cholmod_l_demo.c $(LIB2)

readhb: readhb.f
	$(F77) $(FFLAGS) -o readhb readhb.f

readhb2: readhb2.f
	$(F77) $(FFLAGS) -O -o readhb2 readhb2.f

reade: reade.f
	$(F77) $(FFLAGS) -O -o reade reade.f

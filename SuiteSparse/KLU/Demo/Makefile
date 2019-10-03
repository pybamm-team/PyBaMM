# KLU Demo Makefile

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

CLIB = $(LDFLAGS) -L../../lib -lklu -lbtf -lamd -lcolamd -lsuitesparseconfig \
	$(LIB_WITH_PARTITION) $(LDLIBS)

CHOLMOD_LIB = -lcholmod
CHOLMOD = $(CHOLMOD_LIB) $(LAPACK) $(BLAS)

ifneq ($(GPU_CONFIG),)
CHOLMOD += $(CUDART_LIB) $(CUBLAS_LIB) 
endif

I = -I../../include

all: library klu_simple kludemo kluldemo
	- ./klu_simple
	- ./kludemo < ../Matrix/1c.mtx
	- ./kludemo < ../Matrix/arrowc.mtx
	- ./kludemo < ../Matrix/arrow.mtx
	- ./kludemo < ../Matrix/impcol_a.mtx
	- ./kludemo < ../Matrix/w156.mtx
	- ./kludemo < ../Matrix/ctina.mtx
	- ./kluldemo < ../Matrix/1c.mtx
	- ./kluldemo < ../Matrix/arrowc.mtx
	- ./kluldemo < ../Matrix/arrow.mtx
	- ./kluldemo < ../Matrix/impcol_a.mtx
	- ./kluldemo < ../Matrix/w156.mtx
	- ./kluldemo < ../Matrix/ctina.mtx

library:
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
	( cd ../../BTF ; $(MAKE) library )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
	( cd ../../CHOLMOD ; $(MAKE) library )
	- ( cd ../../CAMD ; $(MAKE) )
	- ( cd ../../CCOLAMD ; $(MAKE) )
	- ( cd ../.. ; $(MAKE) metis )

purge: distclean

distclean: clean
	- $(RM) kludemo kluldemo klu_simple
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

kludemo: kludemo.c Makefile
	$(CC) $(CF) $(I) kludemo.c -o kludemo $(CLIB) $(CHOLMOD)

kluldemo: kludemo.c Makefile
	$(CC) $(CF) $(I) kluldemo.c -o kluldemo $(CLIB) $(CHOLMOD)

klu_simple: klu_simple.c Makefile
	$(CC) $(CF) $(I) klu_simple.c -o klu_simple $(CLIB)
	- ./klu_simple


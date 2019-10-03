#-------------------------------------------------------------------------------
# GPUQREngine/Demo Makefile
#-------------------------------------------------------------------------------

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

LIBS = $(LDFLAGS) -L../../lib $(CUDART_LIB) -lGPUQREngine \
	-lSuiteSparse_GPURuntime -lcholmod -lsuitesparseconfig \
	-lamd -lcolamd -lcamd -lccolamd $(LIB_WITH_PARTIION) \
	-lmetis $(CUDART_LIB) $(CUBLAS_LIB) $(BLAS) $(LAPACK) -lgomp \
	-liomp5 $(LDLIBS)

INC  = -I../../SuiteSparse_GPURuntime/Include -I../Include -I../../include \
	$(CUDA_INC)

#-------------------------------------------------------------------------------

all: library demo

clean:
	$(RM) -r $(CLEAN)

purge: clean
	$(RM) -r gpuqrengine_demo troll.m $(PURGE)

library:
	( cd ../Lib ; $(MAKE) )
	( cd ../../CHOLMOD ; $(MAKE) library )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
	( cd ../../CAMD ; $(MAKE) library )
	( cd ../../CCOLAMD ; $(MAKE) library )
	( cd ../.. ; $(MAKE) metis )
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../../SuiteSparse_GPURuntime ; $(MAKE) library )

demo: gpuqrengine_demo
	./gpuqrengine_demo

gpuqrengine_demo: library gpuqrengine_demo.cpp
	$(CC) $(CF) $(INC) -o gpuqrengine_demo gpuqrengine_demo.cpp $(LIBS)



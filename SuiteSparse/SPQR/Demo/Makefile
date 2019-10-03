#===============================================================================
# SuiteSparseQR/Demo/Makefile
#===============================================================================

default: all

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# the optional SPQRGPU module requires CUDA, SuiteSparse_GPURuntime, GPUQREngine
ifneq ($(GPU_CONFIG),)
    LIB_WITH_SPQRGPU = $(CUDART_LIB) $(CUBLAS_LIB) \
        -lSuiteSparse_GPURuntime -lGPUQREngine
    I_WITH_SPQRGPU = -I../../SuiteSparse_GPURuntime/Include \
        -I../../GPUQREngine/Include $(CUDA_INC)
else
    LIB_WITH_SPQRGPU = 
    I_WITH_SPQRGPU = 
endif

#-------------------------------------------------------------------------------

CLIB = $(LDFLAGS) -L../../lib -lspqr -lsuitesparseconfig -lcholmod -lamd \
        -lcolamd $(LIB_WITH_PARTITION) $(LIB_WITH_SPQRGPU) $(LDLIBS)

# use the BLAS and LAPACK defined by SuiteSparse_config.mk; do not use valgrind 
FLIB = $(LAPACK) $(BLAS)
V =

# To use Valgrind and the plain BLAS and plain LAPACK (non-optimized):
# FLIB = -lgfortran -llapack_plain -lblas_plain -lg2c
# V = valgrind --quiet

all: library qrdemo qrsimple qrsimplec gpu
	- $(V) ./qrsimple < ../Matrix/ash219.mtx
	- $(V) ./qrsimplec < ../Matrix/ash219.mtx
	- $(V) ./qrsimple < ../Matrix/west0067.mtx
	- $(V) ./qrsimplec < ../Matrix/west0067.mtx
	- $(V) ./qrdemo < ../Matrix/a2.mtx
	- $(V) ./qrdemo < ../Matrix/r2.mtx
	- $(V) ./qrdemo < ../Matrix/a04.mtx
	- $(V) ./qrdemo < ../Matrix/a2.mtx
	- $(V) ./qrdemo < ../Matrix/west0067.mtx
	- $(V) ./qrdemo < ../Matrix/c2.mtx
	- $(V) ./qrdemo < ../Matrix/a0.mtx
	- $(V) ./qrdemo < ../Matrix/lfat5b.mtx
	- $(V) ./qrdemo < ../Matrix/bfwa62.mtx
	- $(V) ./qrdemo < ../Matrix/LFAT5.mtx
	- $(V) ./qrdemo < ../Matrix/b1_ss.mtx
	- $(V) ./qrdemo < ../Matrix/bcspwr01.mtx
	- $(V) ./qrdemo < ../Matrix/lpi_galenet.mtx
	- $(V) ./qrdemo < ../Matrix/lpi_itest6.mtx
	- $(V) ./qrdemo < ../Matrix/ash219.mtx
	- $(V) ./qrdemo < ../Matrix/a4.mtx
	- $(V) ./qrdemo < ../Matrix/s32.mtx
	- $(V) ./qrdemo < ../Matrix/c32.mtx
	- $(V) ./qrdemo < ../Matrix/lp_share1b.mtx
	- $(V) ./qrdemo < ../Matrix/a1.mtx
	- $(V) ./qrdemo < ../Matrix/GD06_theory.mtx
	- $(V) ./qrdemo < ../Matrix/GD01_b.mtx
	- $(V) ./qrdemo < ../Matrix/Tina_AskCal_perm.mtx
	- $(V) ./qrdemo < ../Matrix/Tina_AskCal.mtx
	- $(V) ./qrdemo < ../Matrix/GD98_a.mtx
	- $(V) ./qrdemo < ../Matrix/Ragusa16.mtx
	- $(V) ./qrdemo < ../Matrix/young1c.mtx
	- $(V) ./qrdemo < ../Matrix/lp_e226_transposed.mtx

cdemo: qrdemoc
	- $(V) ./qrdemoc < ../Matrix/a2.mtx
	- $(V) ./qrdemoc < ../Matrix/r2.mtx
	- $(V) ./qrdemoc < ../Matrix/a04.mtx
	- $(V) ./qrdemoc < ../Matrix/a2.mtx
	- $(V) ./qrdemoc < ../Matrix/west0067.mtx
	- $(V) ./qrdemoc < ../Matrix/c2.mtx
	- $(V) ./qrdemoc < ../Matrix/a0.mtx
	- $(V) ./qrdemoc < ../Matrix/lfat5b.mtx
	- $(V) ./qrdemoc < ../Matrix/bfwa62.mtx
	- $(V) ./qrdemoc < ../Matrix/LFAT5.mtx
	- $(V) ./qrdemoc < ../Matrix/b1_ss.mtx
	- $(V) ./qrdemoc < ../Matrix/bcspwr01.mtx
	- $(V) ./qrdemoc < ../Matrix/lpi_galenet.mtx
	- $(V) ./qrdemoc < ../Matrix/lpi_itest6.mtx
	- $(V) ./qrdemoc < ../Matrix/ash219.mtx
	- $(V) ./qrdemoc < ../Matrix/a4.mtx
	- $(V) ./qrdemoc < ../Matrix/s32.mtx
	- $(V) ./qrdemoc < ../Matrix/c32.mtx
	- $(V) ./qrdemoc < ../Matrix/lp_share1b.mtx
	- $(V) ./qrdemoc < ../Matrix/a1.mtx
	- $(V) ./qrdemoc < ../Matrix/GD06_theory.mtx
	- $(V) ./qrdemoc < ../Matrix/GD01_b.mtx
	- $(V) ./qrdemoc < ../Matrix/Tina_AskCal_perm.mtx
	- $(V) ./qrdemoc < ../Matrix/Tina_AskCal.mtx
	- $(V) ./qrdemoc < ../Matrix/GD98_a.mtx
	- $(V) ./qrdemoc < ../Matrix/Ragusa16.mtx
	- $(V) ./qrdemoc < ../Matrix/young1c.mtx
	- $(V) ./qrdemoc < ../Matrix/lp_e226_transposed.mtx

ifneq ($(GPU_CONFIG),)
GPU_DEMOS = gpu2 gpu3 gpu4 gpu1
else
GPU_DEMOS =
endif

gpu: $(GPU_DEMOS)

gpu1: qrdemo_gpu
	- $(V) ./qrdemo_gpu  ../Matrix/west0067.mtx 2
	- $(V) ./qrdemo_gpu  ../Matrix/lp_e226_transposed.mtx 2
	- $(V) ./qrdemo_gpu  ../Matrix/lp_e226_transposed.mtx 6

gpu2: qrdemo_gpu2
	- $(V) ./qrdemo_gpu2 ../Matrix/west0067.mtx 2
	- $(V) ./qrdemo_gpu2 ../Matrix/lp_e226_transposed.mtx 2
	- $(V) ./qrdemo_gpu2 ../Matrix/lp_e226_transposed.mtx 6

gpu3: qrdemo_gpu3
	- $(V) ./qrdemo_gpu3 ../Matrix/west0067.mtx 2
	- $(V) ./qrdemo_gpu3 ../Matrix/lp_e226_transposed.mtx 2
	- $(V) ./qrdemo_gpu3 ../Matrix/lp_e226_transposed.mtx 6

gpu4: qrdemo_gpu
	- $(V) ./qrdemo_gpu  ../Matrix/Groebner_id2003_aug.mtx 6
	- $(V) ./qrdemo_gpu  ../Matrix/Franz6_id1959_aug.mtx 6

library: metis
	( cd ../Lib ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../SuiteSparse_config ; $(MAKE) library )
	- ( cd ../../CHOLMOD && $(MAKE) library )
	- ( cd ../../COLAMD && $(MAKE) library )
	- ( cd ../../CCOLAMD && $(MAKE) library )
	- ( cd ../../CAMD && $(MAKE) library )
ifneq ($(GPU_CONFIG),)
	- ( cd ../../SuiteSparse_GPURuntime && $(MAKE) library )
	- ( cd ../../GPUQREngine && $(MAKE) library )
endif

metis: ../../include/metis.h

../../include/metis.h:
	- ( cd ../.. && $(MAKE) metis )

purge: distclean

distclean: clean
	- $(RM) qrdemo qrdemo_gpu qrdemoc qrsimple qrsimplec X.mtx
	- $(RM) R.mtx C.mtx E.txt gpu_results.txt qrdemo_gpu2 qrdemo_gpu3
	- $(RM) *.dot pfile tfile
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp Makefile

I = -I../../include $(I_WITH_SPQRGPU)

C = $(CXX) $(CF) $(SPQR_CONFIG) $(CONFIG_PARTITION) $(CONFIG_GPU) $(I) \
	$(CHOLMOD_CONFIG)

LIBS = $(CLIB) $(FLIB) $(TBB) $(GPULIB)

# With the CUDA BLAS
ifneq ($(GPU_CONFIG),)
LIBS += $(CUDART_LIB) $(CUBLAS_LIB)
endif

qrsimple: qrsimple.cpp $(INC)
	$(C) qrsimple.cpp -o qrsimple $(LIBS)

qrdemo: qrdemo.cpp $(INC)
	$(C) qrdemo.cpp -o qrdemo $(LIBS)

qrdemo_gpu: qrdemo_gpu.cpp $(INC)
ifneq ($(GPU_CONFIG),)
	$(C) qrdemo_gpu.cpp -o qrdemo_gpu $(LIBS)
else
	echo
endif

qrdemo_gpu2: qrdemo_gpu2.cpp $(INC)
ifneq ($(GPU_CONFIG),)
	$(C) qrdemo_gpu2.cpp -o qrdemo_gpu2 $(LIBS)
else
	echo
endif

qrdemo_gpu3: qrdemo_gpu3.cpp $(INC)
ifneq ($(GPU_CONFIG),)
	$(C) qrdemo_gpu3.cpp -o qrdemo_gpu3 $(LIBS)
else
	echo
endif

# compile the C code with gcc, but link with g++ to use SuiteSparseQR:

qrdemoc.o: qrdemoc.c $(INC)
	$(CC) $(CF) $(SPQR_CONFIG) $(I) -c $<

qrdemoc: qrdemoc.o $(INC)
	$(C) -o qrdemoc qrdemoc.o $(LIBS)

qrsimplec.o: qrsimplec.c $(INC)
	$(CC) $(CF) $(SPQR_CONFIG) $(CHOLMOD_CONFIG) $(I) -c $<

qrsimplec: qrsimplec.o $(INC)
	$(C) -o qrsimplec qrsimplec.o $(LIBS)


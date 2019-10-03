#===============================================================================
# SuiteSparseQR/Tcov/Makefile
#===============================================================================

# This test requires METIS, and it only works on Linux.

# run statement-coverage test without Valgrind
default: go

# to run with Valgrind as well
valgrind: vgo

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

# this test requires gcc
CC = gcc
CXX= g++
BLAS = -lrefblas
LAPACK = -llapack

# for statement coverage (with gcov; see go) and picky compiler warnings
CF = -pg \
	-O0 -g -fprofile-arcs -ftest-coverage \
	-Wall -W -Wshadow -Winline -Wno-unused-parameter \
	-Wredundant-decls -Wdisabled-optimization \
	-fexceptions -fopenmp

# Using an optimized BLAS can cause problems in Valgrind
# FLIB = -llapack_plain -lblas_plain -lgfortran
# FLIB = -llapack_plain -lblas_plain -lgfortran -lg2c
  FLIB = $(LAPACK) $(BLAS)

#-------------------------------------------------------------------------------

CUDA_LIB = $(CUDART_LIB) $(CUBLAS_LIB)

NVCC = /usr/local/cuda/bin/nvcc -g --profile --generate-line-info $(NV20) \
	-Xcompiler -fprofile-arcs -Xcompiler -ftest-coverage

#-------------------------------------------------------------------------------

CLIB = -L../../lib -lcholmod -lamd -lcolamd -lccolamd -lcamd -lmetis -lsuitesparseconfig

all: qrtest gpudemo qrdemo_gpu

library: qrtest gpudemo

purge: distclean

distclean: clean
	- $(RM) qrtest qrtest_out.txt pfile tfile cov.out qrtest_out1.txt
	- $(RM) gpuqrengine_demo troll.m qrdemo_gpu gpu_results.txt X.mtx
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp
# Makefile

OBJ = \
    spqr_rmap.o                              \
    SuiteSparseQR_C.o                        \
    SuiteSparseQR_expert.o                   \
    spqr_parallel.o                          \
    spqr_kernel.o                            \
    spqr_analyze.o                           \
    spqr_assemble.o                          \
    spqr_cpack.o                             \
    spqr_csize.o                             \
    spqr_fcsize.o                            \
    spqr_debug.o                             \
    spqr_front.o                             \
    spqr_factorize.o                         \
    spqr_freenum.o                           \
    spqr_freesym.o                           \
    spqr_freefac.o                           \
    spqr_fsize.o                             \
    spqr_maxcolnorm.o                        \
    spqr_rconvert.o                          \
    spqr_rcount.o                            \
    spqr_rhpack.o                            \
    spqr_rsolve.o                            \
    spqr_stranspose1.o                       \
    spqr_stranspose2.o                       \
    spqr_hpinv.o                             \
    spqr_1fixed.o                            \
    spqr_1colamd.o                           \
    SuiteSparseQR.o                          \
    spqr_1factor.o                           \
    spqr_cumsum.o                            \
    spqr_shift.o                             \
    spqr_happly.o                            \
    spqr_panel.o                             \
    spqr_happly_work.o                       \
    SuiteSparseQR_qmult.o                    \
    spqr_trapezoidal.o                       \
    spqr_larftb.o                            \
    spqr_append.o                            \
    spqr_type.o                              \
    spqr_tol.o                               \
    qrtestc.o

ifneq ($(GPU_CONFIG),)
OBJ += \
    spqrgpu_kernel.o                         \
    spqrgpu_buildAssemblyMaps.o              \
    spqrgpu_computeFrontStaging.o            \
    SuiteSparseGPU_Workspace.o               \
    SuiteSparseGPU_Workspace_cpuAllocators.o \
    SuiteSparseGPU_Workspace_gpuAllocators.o \
    SuiteSparseGPU_Workspace_transfer.o      \
    GPUQREngine_GraphVizHelper.o             \
    GPUQREngine_UberKernel.o                 \
    GPUQREngine_ExpertSparse.o               \
    GPUQREngine_Internal.o                   \
    GPUQREngine_ExpertDense.o                \
    BucketList.o                 \
    BucketList_AdvanceBundles.o  \
    BucketList_CreateBundles.o   \
    BucketList_FillWorkQueue.o   \
    BucketList_GrowBundles.o     \
    BucketList_Manage.o          \
    BucketList_PostProcessing.o  \
    LLBundle.o                               \
    LLBundle_AddTiles.o                      \
    LLBundle_Advance.o                       \
    LLBundle_GPUPack.o                       \
    LLBundle_PipelinedRearrange.o            \
    LLBundle_UpdateSecondMinIndex.o          \
    Scheduler.o                              \
    Scheduler_FillWorkQueue.o                \
    Scheduler_Front.o                        \
    Scheduler_LaunchKernel.o                 \
    Scheduler_PostProcess.o                  \
    Scheduler_Render.o                       \
    Scheduler_TransferData.o                 \
    ssgpu_maxQueueSize.o                     \
    TaskDescriptor_flops.o
endif

$(OBJ): $(INC)

I = -I../../CHOLMOD/Include -I../../SuiteSparse_config -I../Include \

ifneq ($(GPU_CONFIG),)
    I += -I../../SuiteSparse_GPURuntime/Include \
        -I../../GPUQREngine/Include $(CUDA_INC)
endif

C = $(CXX) $(CF) $(I) $(SPQR_CONFIG)

LIBS = $(CLIB) $(FLIB) $(LDLIBS) $(CUDA_LIB)

qrtestc.o: qrtestc.c $(INC)
	$(CC) $(CF) $(I) -c $<

qrtest: libraries qrtest.cpp $(INC) $(OBJ)
	$(C) qrtest.cpp -o qrtest $(OBJ) $(LIBS) -lm

ifneq ($(GPU_CONFIG),)
gpu: gpuqrengine_demo qrdemo_gpu
	- ./gpuqrengine_demo
	- ./qrdemo_gpu ../Matrix/west0067.mtx 2
	- ./qrdemo_gpu ../Matrix/lp_e226_transposed.mtx 2
	- ./qrdemo_gpu ../Matrix/lp_e226_transposed.mtx 6
	- ./qrdemo_gpu ../Matrix/Groebner_id2003_aug.mtx 6
	- ./qrdemo_gpu ../Matrix/Franz6_id1959_aug.mtx 6
else
gpu:
endif

gpuqrengine_demo: libraries \
        ../../GPUQREngine/Demo/gpuqrengine_demo.cpp $(INC) $(OBJ)
	$(C) $(GPUQRDEMO)/gpuqrengine_demo.cpp -o gpuqrengine_demo \
                $(OBJ) $(LIBS) -lm

qrdemo_gpu: ../Demo/qrdemo_gpu.cpp $(INC) $(OBJ)
ifneq ($(GPU_CONFIG),)
	$(C) ../Demo/qrdemo_gpu.cpp -o qrdemo_gpu $(OBJ) $(LIBS)
endif

go: qrtest gpu qrdemo_gpu
	- ./qrtest matrixlist.txt > qrtest_out.txt
	- ./cov

go1: qrtest
	- ./qrtest matrix1.txt > qrtest_out1.txt
	- ./cov

vgo1: qrtest
	- valgrind ./qrtest matrix1.txt > qrtest_out1.txt
	# - valgrind --leak-check=full --show-reachable=yes ./qrtest matrix1.txt > qrtest_out1.txt
	- ./cov

vgo: qrtest
	- valgrind --leak-check=full --show-reachable=yes ./qrtest matrixlist.txt > qrtest_out.txt
	- ./cov

spqr_1colamd.o: ../Source/spqr_1colamd.cpp
	$(C) -c $<

spqr_1factor.o: ../Source/spqr_1factor.cpp
	$(C) -c $<

spqr_1fixed.o: ../Source/spqr_1fixed.cpp
	$(C) -c $<

spqr_analyze.o: ../Source/spqr_analyze.cpp
	$(C) -c $<

spqr_parallel.o: ../Source/spqr_parallel.cpp
	$(C) -c $<

spqr_kernel.o: ../Source/spqr_kernel.cpp
	$(C) -c $<

spqr_append.o: ../Source/spqr_append.cpp
	$(C) -c $<

spqr_assemble.o: ../Source/spqr_assemble.cpp
	$(C) -c $<

spqr_cpack.o: ../Source/spqr_cpack.cpp
	$(C) -c $<

spqr_csize.o: ../Source/spqr_csize.cpp
	$(C) -c $<

spqr_cumsum.o: ../Source/spqr_cumsum.cpp
	$(C) -c $<

spqr_debug.o: ../Source/spqr_debug.cpp
	$(C) -c $<

spqr_factorize.o: ../Source/spqr_factorize.cpp
	$(C) -c $<

spqr_fcsize.o: ../Source/spqr_fcsize.cpp
	$(C) -c $<

spqr_freefac.o: ../Source/spqr_freefac.cpp
	$(C) -c $<

spqr_freenum.o: ../Source/spqr_freenum.cpp
	$(C) -c $<

spqr_freesym.o: ../Source/spqr_freesym.cpp
	$(C) -c $<

spqr_fsize.o: ../Source/spqr_fsize.cpp
	$(C) -c $<

spqr_happly.o: ../Source/spqr_happly.cpp
	$(C) -c $<

spqr_panel.o: ../Source/spqr_panel.cpp
	$(C) -c $<

spqr_happly_work.o: ../Source/spqr_happly_work.cpp
	$(C) -c $<

spqr_hpinv.o: ../Source/spqr_hpinv.cpp
	$(C) -c $<

spqr_larftb.o: ../Source/spqr_larftb.cpp
	$(C) -c $<

spqr_rconvert.o: ../Source/spqr_rconvert.cpp
	$(C) -c $<

spqr_rcount.o: ../Source/spqr_rcount.cpp
	$(C) -c $<

spqr_rhpack.o: ../Source/spqr_rhpack.cpp
	$(C) -c $<

spqr_rsolve.o: ../Source/spqr_rsolve.cpp
	$(C) -c $<

spqr_shift.o: ../Source/spqr_shift.cpp
	$(C) -c $<

spqr_stranspose1.o: ../Source/spqr_stranspose1.cpp
	$(C) -c $<

spqr_stranspose2.o: ../Source/spqr_stranspose2.cpp
	$(C) -c $<

spqr_trapezoidal.o: ../Source/spqr_trapezoidal.cpp
	$(C) -c $<

spqr_type.o: ../Source/spqr_type.cpp
	$(C) -c $<

spqr_front.o: ../Source/spqr_front.cpp
	$(C) -c $<

SuiteSparseQR_expert.o: ../Source/SuiteSparseQR_expert.cpp
	$(C) -c $<

spqr_maxcolnorm.o: ../Source/spqr_maxcolnorm.cpp
	$(C) -c $<

SuiteSparseQR_qmult.o: ../Source/SuiteSparseQR_qmult.cpp
	$(C) -c $<

SuiteSparseQR.o: ../Source/SuiteSparseQR.cpp
	$(C) -c $<

spqr_tol.o: ../Source/spqr_tol.cpp
	$(C) -c $<

SuiteSparseQR_C.o: ../Source/SuiteSparseQR_C.cpp
	$(C) -c $<

spqr_rmap.o: ../Source/spqr_rmap.cpp
	$(C) -c $<

spqrgpu_kernel.o: ../SPQRGPU/spqrgpu_kernel.cpp
	$(C) -c $<

spqrgpu_buildAssemblyMaps.o: ../SPQRGPU/spqrgpu_buildAssemblyMaps.cpp
	$(C) -c $<

spqrgpu_computeFrontStaging.o: ../SPQRGPU/spqrgpu_computeFrontStaging.cpp
	$(C) -c $<

#-------------------------------------------------------------------------------
# libraries compiled without test coverage
#-------------------------------------------------------------------------------

libraries:
	( cd ../.. && $(MAKE) metis )
	( cd ../../SuiteSparse_config ; $(MAKE) library TCOV=yes )
	( cd ../../AMD && $(MAKE) library TCOV=yes )
	( cd ../../COLAMD && $(MAKE) library TCOV=yes )
	( cd ../../CCOLAMD && $(MAKE) library TCOV=yes )
	( cd ../../CAMD && $(MAKE) library TCOV=yes )
	( cd ../../CHOLMOD && $(MAKE) library TCOV=yes )

#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime
#-------------------------------------------------------------------------------

GPURUNTIME = ../../SuiteSparse_GPURuntime
GPURUNSRC = $(GPURUNTIME)/Source
GPURUNINC = -I$(GPURUNTIME)/Include -I../../SuiteSparse_config

RUNH = \
        $(GPURUNTIME)/Include/SuiteSparseGPU_Workspace.hpp \
        $(GPURUNTIME)/Include/SuiteSparseGPU_debug.hpp \
        $(GPURUNTIME)/Include/SuiteSparseGPU_macros.hpp \
        $(GPURUNTIME)/Include/SuiteSparseGPU_workspace_macros.hpp \
        $(GPURUNTIME)/Include/SuiteSparseGPU_Runtime.hpp
#        Makefile

SuiteSparseGPU_Workspace.o: $(GPURUNSRC)/SuiteSparseGPU_Workspace.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_cpuAllocators.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_cpuAllocators.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_gpuAllocators.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_gpuAllocators.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_memset.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_memset.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_transfer.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_transfer.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

#-------------------------------------------------------------------------------
# GPUQREngine
#-------------------------------------------------------------------------------

GPUQR = ../../GPUQREngine
GPUQRSRC = $(GPUQR)/Source
GPUQRDEMO = $(GPUQR)/Demo
GPUQRINC = $(GPURUNINC) -I$(GPUQR)/Include

KERNELH = \
    $(GPUQR)/Include/GPUQREngine_Common.hpp \
    $(GPUQR)/Include/GPUQREngine_BucketList.hpp \
    $(GPUQR)/Include/GPUQREngine_Front.hpp \
    $(GPUQR)/Include/GPUQREngine_FrontState.hpp \
    $(GPUQR)/Include/GPUQREngine.hpp \
    $(GPUQR)/Include/GPUQREngine_Internal.hpp \
    $(GPUQR)/Include/GPUQREngine_GraphVizHelper.hpp \
    $(GPUQR)/Include/Kernel/Apply/block_apply_1_by_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_2_by_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_2.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_3_by_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_3.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_chunk.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply.cu \
    $(GPUQR)/Include/Kernel/Apply/cevta_tile.cu \
    $(GPUQR)/Include/Kernel/Apply/pipelined_rearrange.cu \
    $(GPUQR)/Include/Kernel/Assemble/packAssemble.cu \
    $(GPUQR)/Include/Kernel/Assemble/sAssemble.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_3_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_1_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_1_by_1_edge.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_2_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_2_by_1_edge.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_3_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_3_by_1_edge.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt.cu \
    $(GPUQR)/Include/Kernel/qrKernel.cu \
    $(GPUQR)/Include/Kernel/sharedMemory.hpp \
    $(GPUQR)/Include/Kernel/uberKernel.cu \
    $(GPUQR)/Include/GPUQREngine_LLBundle.hpp \
    $(GPUQR)/Include/GPUQREngine_Stats.hpp \
    $(GPUQR)/Include/GPUQREngine_Scheduler.hpp \
    $(GPUQR)/Include/GPUQREngine_SEntry.hpp \
    $(GPUQR)/Include/GPUQREngine_SparseMeta.hpp \
    $(GPUQR)/Include/GPUQREngine_TaskDescriptor.hpp \
    $(GPUQR)/Include/GPUQREngine_Timing.hpp
#   Makefile

GPUQREngine_GraphVizHelper.o: \
        $(GPUQRSRC)/GPUQREngine_GraphVizHelper.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_UberKernel.o: $(GPUQRSRC)/GPUQREngine_UberKernel.cu $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_ExpertDense.o: $(GPUQRSRC)/GPUQREngine_ExpertDense.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_Internal.o: $(GPUQRSRC)/GPUQREngine_Internal.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_ExpertSparse.o: $(GPUQRSRC)/GPUQREngine_ExpertSparse.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

BucketList.o: $(GPUQRSRC)/BucketList/BucketList.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

BucketList_AdvanceBundles.o: $(GPUQRSRC)/BucketList/BucketList_AdvanceBundles.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

BucketList_CreateBundles.o: $(GPUQRSRC)/BucketList/BucketList_CreateBundles.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

BucketList_FillWorkQueue.o: $(GPUQRSRC)/BucketList/BucketList_FillWorkQueue.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

BucketList_GrowBundles.o: $(GPUQRSRC)/BucketList/BucketList_GrowBundles.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

BucketList_Manage.o: $(GPUQRSRC)/BucketList/BucketList_Manage.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

BucketList_PostProcessing.o: $(GPUQRSRC)/BucketList/BucketList_PostProcessing.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

LLBundle.o: $(GPUQRSRC)/LLBundle/LLBundle.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

LLBundle_AddTiles.o: $(GPUQRSRC)/LLBundle/LLBundle_AddTiles.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

LLBundle_Advance.o: $(GPUQRSRC)/LLBundle/LLBundle_Advance.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

LLBundle_GPUPack.o: $(GPUQRSRC)/LLBundle/LLBundle_GPUPack.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

LLBundle_PipelinedRearrange.o: $(GPUQRSRC)/LLBundle/LLBundle_PipelinedRearrange.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

LLBundle_UpdateSecondMinIndex.o: $(GPUQRSRC)/LLBundle/LLBundle_UpdateSecondMinIndex.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

Scheduler.o: $(GPUQRSRC)/Scheduler/Scheduler.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

Scheduler_FillWorkQueue.o: $(GPUQRSRC)/Scheduler/Scheduler_FillWorkQueue.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

Scheduler_Front.o: $(GPUQRSRC)/Scheduler/Scheduler_Front.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

Scheduler_LaunchKernel.o: $(GPUQRSRC)/Scheduler/Scheduler_LaunchKernel.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

Scheduler_PostProcess.o: $(GPUQRSRC)/Scheduler/Scheduler_PostProcess.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

Scheduler_Render.o: $(GPUQRSRC)/Scheduler/Scheduler_Render.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

Scheduler_TransferData.o: $(GPUQRSRC)/Scheduler/Scheduler_TransferData.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

ssgpu_maxQueueSize.o: $(GPUQRSRC)/Scheduler/ssgpu_maxQueueSize.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

TaskDescriptor_flops.o: $(GPUQRSRC)/TaskDescriptor/TaskDescriptor_flops.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<


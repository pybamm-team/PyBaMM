#-------------------------------------------------------------------------------
# GPUQREngine/Lib/Makefile: for compiling the GPUQREngine library
#-------------------------------------------------------------------------------

LIBRARY = libGPUQREngine
VERSION = 1.0.5
SO_VERSION = 1

default: library

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

# GPUQREngine depends on SuiteSparse_config, CUDA, and SuiteSparse_GPURuntime
LDLIBS += -lSuiteSparse_GPURuntime -lsuitesparseconfig $(CUDART_LIB)

# compile and install in SuiteSparse/lib
library:
	$(MAKE) install INSTALL=$(SUITESPARSE)

OBJS = GPUQREngine_GraphVizHelper.o                        \
       GPUQREngine_Internal.o                              \
       GPUQREngine_ExpertDense.o                           \
       GPUQREngine_ExpertSparse.o                          \
       BucketList.o                                        \
       BucketList_AdvanceBundles.o                         \
       BucketList_CreateBundles.o                          \
       BucketList_FillWorkQueue.o                          \
       BucketList_GrowBundles.o                            \
       BucketList_Manage.o                                 \
       BucketList_PostProcessing.o                         \
       LLBundle.o                                          \
       LLBundle_AddTiles.o                                 \
       LLBundle_Advance.o                                  \
       LLBundle_GPUPack.o                                  \
       LLBundle_PipelinedRearrange.o                       \
       LLBundle_UpdateSecondMinIndex.o                     \
       Scheduler.o                                         \
       Scheduler_FillWorkQueue.o                           \
       Scheduler_Front.o                                   \
       Scheduler_LaunchKernel.o                            \
       Scheduler_PostProcess.o                             \
       Scheduler_Render.o                                  \
       Scheduler_TransferData.o                            \
       ssgpu_maxQueueSize.o                                \
       TaskDescriptor_flops.o                              \
       GPUQREngine_UberKernel.o

INC  = -I../../SuiteSparse_GPURuntime/Include -I../Include \
    -I../../SuiteSparse_config

# general include file dependencies
H =   ../Include/GPUQREngine_Common.hpp                       \
      ../Include/GPUQREngine_BucketList.hpp                   \
      ../Include/GPUQREngine_Front.hpp                        \
      ../Include/GPUQREngine_FrontState.hpp                   \
      ../Include/GPUQREngine.hpp                              \
      ../Include/GPUQREngine_Internal.hpp                     \
      ../Include/GPUQREngine_GraphVizHelper.hpp               \
      ../Include/GPUQREngine_LLBundle.hpp                     \
      ../Include/GPUQREngine_Stats.hpp                        \
      ../Include/GPUQREngine_Scheduler.hpp                    \
      ../Include/GPUQREngine_SEntry.hpp                       \
      ../Include/GPUQREngine_SparseMeta.hpp                   \
      ../Include/GPUQREngine_TaskDescriptor.hpp               \
      ../Include/GPUQREngine_Timing.hpp

# dependencies only for the CUDA kernel:
H2 = \
      ../Include/GPUQREngine_Common.hpp                       \
      ../Include/GPUQREngine_SEntry.hpp                       \
      ../Include/GPUQREngine_TaskDescriptor.hpp               \
      ../Include/Kernel/Apply/block_apply_1_by_1.cu           \
      ../Include/Kernel/Apply/block_apply_1.cu                \
      ../Include/Kernel/Apply/block_apply_2_by_1.cu           \
      ../Include/Kernel/Apply/block_apply_2.cu                \
      ../Include/Kernel/Apply/block_apply_3_by_1.cu           \
      ../Include/Kernel/Apply/block_apply_3.cu                \
      ../Include/Kernel/Apply/block_apply_chunk.cu            \
      ../Include/Kernel/Apply/block_apply.cu                  \
      ../Include/Kernel/Apply/cevta_tile.cu                   \
      ../Include/Kernel/Apply/params_apply.hpp                \
      ../Include/Kernel/Apply/pipelined_rearrange.cu          \
      ../Include/Kernel/Assemble/packAssemble.cu              \
      ../Include/Kernel/Assemble/sAssemble.cu                 \
      ../Include/Kernel/Factorize/factorize_3_by_1.cu         \
      ../Include/Kernel/Factorize/factorize_vt_1_by_1.cu      \
      ../Include/Kernel/Factorize/factorize_vt_1_by_1_edge.cu \
      ../Include/Kernel/Factorize/factorize_vt_2_by_1.cu      \
      ../Include/Kernel/Factorize/factorize_vt_2_by_1_edge.cu \
      ../Include/Kernel/Factorize/factorize_vt_3_by_1.cu      \
      ../Include/Kernel/Factorize/factorize_vt_3_by_1_edge.cu \
      ../Include/Kernel/Factorize/factorize_vt.cu             \
      ../Include/Kernel/qrKernel.cu                           \
      ../Include/Kernel/sharedMemory.hpp                      \
      ../Include/Kernel/uberKernel.cu

H3 = \
    ../../SuiteSparse_GPURuntime/Include/SuiteSparseGPU_debug.hpp \
    ../../SuiteSparse_GPURuntime/Include/SuiteSparseGPU_macros.hpp \
    ../../SuiteSparse_GPURuntime/Include/SuiteSparseGPU_Runtime.hpp \
    ../../SuiteSparse_GPURuntime/Include/SuiteSparseGPU_Workspace.hpp \
    ../../SuiteSparse_GPURuntime/Include/SuiteSparseGPU_workspace_macros.hpp

H  += $(H3)
H2 += $(H3)

#-------------------------------------------------------------------------------

C = $(NVCC) $(NVCCFLAGS) $(INC)

all: library

clean:
	$(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	$(RM) -r $(PURGE)

static: $(AR_TARGET)

$(AR_TARGET): $(OBJS)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

#-------------------------------------------------------------------------------

# install GPUQREngine.  Note that the include files are not installed.
install: $(AR_TARGET) $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(OBJS)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CXX) $(SO_OPTS) $^ -o $@ $(LDLIBS)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../README.txt $(INSTALL_DOC)/GPUQRENGINE_README.txt
	chmod 755 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_DOC)/GPUQRENGINE_README.txt

# uninstall GPUQREngine
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_DOC)/GPUQRENGINE_README.txt

#-------------------------------------------------------------------------------

GPUQREngine_GraphVizHelper.o: ../Source/GPUQREngine_GraphVizHelper.cpp $(H)
	$(C) -c $<

GPUQREngine_Internal.o: ../Source/GPUQREngine_Internal.cpp $(H)
	$(C) -c $<

GPUQREngine_ExpertDense.o: ../Source/GPUQREngine_ExpertDense.cpp $(H)
	$(C) -c $<

GPUQREngine_ExpertSparse.o: ../Source/GPUQREngine_ExpertSparse.cpp $(H)
	$(C) -c $<

BucketList.o: ../Source/BucketList/BucketList.cpp $(H)
	$(C) -c $<

BucketList_AdvanceBundles.o: ../Source/BucketList/BucketList_AdvanceBundles.cpp $(H)
	$(C) -c $<

BucketList_CreateBundles.o: ../Source/BucketList/BucketList_CreateBundles.cpp $(H)
	$(C) -c $<

BucketList_FillWorkQueue.o: ../Source/BucketList/BucketList_FillWorkQueue.cpp $(H)
	$(C) -c $<

BucketList_GrowBundles.o: ../Source/BucketList/BucketList_GrowBundles.cpp $(H)
	$(C) -c $<

BucketList_Manage.o: ../Source/BucketList/BucketList_Manage.cpp $(H)
	$(C) -c $<

BucketList_PostProcessing.o: ../Source/BucketList/BucketList_PostProcessing.cpp $(H)
	$(C) -c $<

LLBundle.o: ../Source/LLBundle/LLBundle.cpp $(H)
	$(C) -c $<

LLBundle_AddTiles.o: ../Source/LLBundle/LLBundle_AddTiles.cpp $(H)
	$(C) -c $<

LLBundle_Advance.o: ../Source/LLBundle/LLBundle_Advance.cpp $(H)
	$(C) -c $<

LLBundle_GPUPack.o: ../Source/LLBundle/LLBundle_GPUPack.cpp $(H)
	$(C) -c $<

LLBundle_PipelinedRearrange.o: ../Source/LLBundle/LLBundle_PipelinedRearrange.cpp $(H)
	$(C) -c $<

LLBundle_UpdateSecondMinIndex.o: ../Source/LLBundle/LLBundle_UpdateSecondMinIndex.cpp $(H)
	$(C) -c $<

Scheduler.o: ../Source/Scheduler/Scheduler.cpp $(H)
	$(C) -c $<

Scheduler_FillWorkQueue.o: ../Source/Scheduler/Scheduler_FillWorkQueue.cpp $(H)
	$(C) -c $<

Scheduler_Front.o: ../Source/Scheduler/Scheduler_Front.cpp $(H)
	$(C) -c $<

Scheduler_LaunchKernel.o: ../Source/Scheduler/Scheduler_LaunchKernel.cpp $(H)
	$(C) -c $<

Scheduler_PostProcess.o: ../Source/Scheduler/Scheduler_PostProcess.cpp $(H)
	$(C) -c $<

Scheduler_Render.o: ../Source/Scheduler/Scheduler_Render.cpp $(H)
	$(C) -c $<

Scheduler_TransferData.o: ../Source/Scheduler/Scheduler_TransferData.cpp $(H)
	$(C) -c $<

ssgpu_maxQueueSize.o: ../Source/Scheduler/ssgpu_maxQueueSize.cpp $(H)
	$(C) -c $<

TaskDescriptor_flops.o: ../Source/TaskDescriptor/TaskDescriptor_flops.cpp $(H)
	$(C) -c $<
	
GPUQREngine_UberKernel.o: ../Source/GPUQREngine_UberKernel.cu $(H2)
	$(C) -c $<


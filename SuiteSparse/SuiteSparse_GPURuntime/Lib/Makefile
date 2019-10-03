#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime/Lib/Makfile
#-------------------------------------------------------------------------------

LIBRARY = libSuiteSparse_GPURuntime
VERSION = 1.0.5
SO_VERSION = 1

default: library

include ../../SuiteSparse_config/SuiteSparse_config.mk

# SuiteSparse_GPURuntime depends on SuiteSparse_config and CUDA
LDLIBS += -lsuitesparseconfig $(CUDART_LIB)

# compile and install in SuiteSparse/lib
library:
	$(MAKE) install INSTALL=$(SUITESPARSE)

OBJS = SuiteSparseGPU_Workspace.o                     \
       SuiteSparseGPU_Workspace_cpuAllocators.o       \
       SuiteSparseGPU_Workspace_gpuAllocators.o       \
       SuiteSparseGPU_Workspace_transfer.o

INC = -I../Include -I../../SuiteSparse_config

H = \
        ../Include/SuiteSparseGPU_Workspace.hpp \
        ../Include/SuiteSparseGPU_debug.hpp \
        ../Include/SuiteSparseGPU_macros.hpp \
        ../Include/SuiteSparseGPU_workspace_macros.hpp \
        ../Include/SuiteSparseGPU_Runtime.hpp

#-------------------------------------------------------------------------------

all: library

runtime: library

lib: library

clean:
	$(RM) -r $(CLEAN)

purge: clean
	$(RM) -r $(PURGE)

distclean: purge

static: $(AR_TARGET)

$(AR_TARGET): $(OBJS)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

#-------------------------------------------------------------------------------

SuiteSparseGPU_Workspace.o: ../Source/SuiteSparseGPU_Workspace.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

SuiteSparseGPU_Workspace_cpuAllocators.o: ../Source/SuiteSparseGPU_Workspace_cpuAllocators.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

SuiteSparseGPU_Workspace_gpuAllocators.o: ../Source/SuiteSparseGPU_Workspace_gpuAllocators.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

SuiteSparseGPU_Workspace_transfer.o: ../Source/SuiteSparseGPU_Workspace_transfer.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

#-------------------------------------------------------------------------------

# install SuiteSparse_GPURuntime (just the library, not the include files)
install: $(AR_TARGET) $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(OBJS)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CXX) $(SO_OPTS) $^ -o $@ $(LDLIBS)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../README.txt $(INSTALL_DOC)/GPURUNTIME_README.txt
	chmod 755 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_DOC)/GPURUNTIME_README.txt

# uninstall SuiteSparse_GPURuntime
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_DOC)/GPURUNTIME_README.txt


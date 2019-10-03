#===============================================================================
# SuiteSparseQR/Lib/Makefile
#===============================================================================

LIBRARY = libspqr
VERSION = 2.0.9
SO_VERSION = 2

default: library

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

# SPQR depends on CHOLMOD, AMD, COLAMD, LAPACK, the BLAS and SuiteSparse_config
LDLIBS += -lamd -lcolamd -lcholmod -lsuitesparseconfig $(LAPACK) $(BLAS) $(TBB)

# compile and install in SuiteSparse/lib
library:
	$(MAKE) install INSTALL=$(SUITESPARSE)

#-------------------------------------------------------------------------------

all: library

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp

OBJ = \
    spqr_rmap.o \
    SuiteSparseQR_C.o \
    SuiteSparseQR_expert.o \
    spqr_parallel.o \
    spqr_kernel.o \
    spqr_analyze.o \
    spqr_assemble.o \
    spqr_cpack.o \
    spqr_csize.o \
    spqr_fcsize.o \
    spqr_debug.o \
    spqr_front.o \
    spqr_factorize.o \
    spqr_freenum.o \
    spqr_freesym.o \
    spqr_freefac.o \
    spqr_fsize.o \
    spqr_maxcolnorm.o \
    spqr_rconvert.o \
    spqr_rcount.o \
    spqr_rhpack.o \
    spqr_rsolve.o \
    spqr_stranspose1.o \
    spqr_stranspose2.o \
    spqr_hpinv.o \
    spqr_1fixed.o \
    spqr_1colamd.o \
    SuiteSparseQR.o \
    spqr_1factor.o \
    spqr_cumsum.o \
    spqr_shift.o \
    spqr_happly.o \
    spqr_panel.o \
    spqr_happly_work.o \
    SuiteSparseQR_qmult.o \
    spqr_trapezoidal.o \
    spqr_larftb.o \
    spqr_append.o \
    spqr_type.o \
    spqr_tol.o

ifneq ($(GPU_CONFIG),)
OBJ += \
    spqrgpu_kernel.o \
    spqrgpu_buildAssemblyMaps.o \
    spqrgpu_computeFrontStaging.o
endif

$(OBJ): $(INC)

I = -I../../CHOLMOD/Include -I../../SuiteSparse_config -I../Include

ifneq ($(GPU_CONFIG),)
    I += -I../../SuiteSparse_GPURuntime/Include \
         -I../../GPUQREngine/Include $(CUDA_INC)
endif

C = $(CXX) $(CF) $(SPQR_CONFIG) $(CHOLMOD_CONFIG) $(CONFIG_PARTITION) $(I)

ifneq ($(GPU_CONFIG),)
    LDLIBS += $(CUDART_LIB) $(CUBLAS_LIB) \
        -lSuiteSparse_GPURuntime -lGPUQREngine
endif

#-------------------------------------------------------------------------------

static: $(AR_TARGET)

$(AR_TARGET): $(OBJ)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

#-------------------------------------------------------------------------------

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

# install SPQR
install: $(AR_TARGET) $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(OBJ)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CXX) $(SO_OPTS) $^ -o $@ $(LDLIBS)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../Include/SuiteSparseQR.hpp $(INSTALL_INCLUDE)
	$(CP) ../Include/SuiteSparseQR_C.h $(INSTALL_INCLUDE)
	$(CP) ../Include/SuiteSparseQR_definitions.h $(INSTALL_INCLUDE)
	$(CP) ../Include/spqr.hpp $(INSTALL_INCLUDE)
	$(CP) ../Doc/spqr_user_guide.pdf $(INSTALL_DOC)
	$(CP) ../README.txt $(INSTALL_DOC)/SPQR_README.txt
	chmod 755 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_INCLUDE)/SuiteSparseQR.hpp
	chmod 644 $(INSTALL_INCLUDE)/SuiteSparseQR_C.h
	chmod 644 $(INSTALL_INCLUDE)/SuiteSparseQR_definitions.h
	chmod 644 $(INSTALL_INCLUDE)/spqr.hpp
	chmod 644 $(INSTALL_DOC)/spqr_user_guide.pdf
	chmod 644 $(INSTALL_DOC)/SPQR_README.txt

# uninstall SPQR
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR.hpp
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR_C.h
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR_definitions.h
	$(RM) $(INSTALL_INCLUDE)/spqr.hpp
	$(RM) $(INSTALL_DOC)/spqr_user_guide.pdf
	$(RM) $(INSTALL_DOC)/SPQR_README.txt


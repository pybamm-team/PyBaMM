# If the libraries (AMD, COLAMD, CAMD, CCOLAMD, metis, and CHOLMOD) are not
# yet built, use "make libs" first.  Then "make" to compile and run all tests.
#
# This test requires BTF, METIS, CHOLMOD, AMD, COLAMD, CAMD, and CCOLAMD.

TCOV=yes

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

BLAS = -lrefblas
LAPACK = -llapack

# CF = -Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
# 	-Wredundant-decls -Wnested-externs -Wdisabled-optimization \
# 	-pedantic -ansi -O3 -pg

# for statement coverage, picky tests
CF = -Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
   	-Wredundant-decls -Wnested-externs -Wdisabled-optimization \
 	-ansi -g -ftest-coverage -fprofile-arcs -fexceptions \
	-fopenmp

C = $(CC) $(CF)

LDLIBS = -L../../lib -lamd -lcolamd -lcholmod -lcamd -lccolamd \
    -lmetis -lsuitesparseconfig \
    $(LAPACK) $(BLAS) $(CUDART_LIB) $(CUBLAS_LIB) -lm -lrt \
    -Wl,-rpath=$(SUITESPARSE)/lib -lstdc++

I = -I../../include -I../Include -I../User -I../../BTF/Include

all: purge libs klutest klultest
	- ./klultests > klultests.out
	- ./klutests > klutests.out
	- ./coverage

valgrind: purge klutest klultest
	- ./vklutests > klutests.out
	- ./vklultests > klultests.out
	- ./coverage

libs:
	( cd ../../SuiteSparse_config && $(MAKE) TCOV=yes)
	( cd ../../AMD && $(MAKE) library TCOV=yes)
	( cd ../../COLAMD && $(MAKE) library TCOV=yes)
	( cd ../../CAMD && $(MAKE) library TCOV=yes)
	( cd ../../CCOLAMD && $(MAKE) library TCOV=yes)
	( cd ../.. && $(MAKE) metis )
	( cd ../../CHOLMOD && $(MAKE) library TCOV=yes)

purge: distclean

distclean: clean
	- $(RM) klutest klultest *.out cov_*.c
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC =  \
	../Include/klu.h \
	../Include/klu_internal.h \
	../Include/klu_version.h

BTFOBJ = \
	cov_btf_order.o \
	cov_btf_maxtrans.o \
	cov_btf_strongcomp.o \

BTFLOBJ = \
	cov_btf_l_order.o \
	cov_btf_l_maxtrans.o \
	cov_btf_l_strongcomp.o

KLUOBJ = \
	cov_klu_analyze.o \
	cov_klu_analyze_given.o \
	cov_klu_defaults.o \
	cov_klu_free_symbolic.o \
	cov_klu_memory.o \
	cov_klu_d.o \
	cov_klu_d_diagnostics.o \
	cov_klu_d_dump.o \
	cov_klu_d_factor.o \
	cov_klu_d_free_numeric.o \
	cov_klu_d_kernel.o \
	cov_klu_d_extract.o \
	cov_klu_d_refactor.o \
	cov_klu_d_scale.o \
	cov_klu_d_solve.o \
	cov_klu_d_tsolve.o \
	cov_klu_z.o \
	cov_klu_z_diagnostics.o \
	cov_klu_z_dump.o \
	cov_klu_z_factor.o \
	cov_klu_z_free_numeric.o \
	cov_klu_z_kernel.o \
	cov_klu_z_extract.o \
	cov_klu_z_refactor.o \
	cov_klu_z_scale.o \
	cov_klu_z_solve.o \
	cov_klu_z_tsolve.o

KLULOBJ = \
	cov_klu_l_analyze.o \
	cov_klu_l_analyze_given.o \
	cov_klu_l_defaults.o \
	cov_klu_l_free_symbolic.o \
	cov_klu_l_memory.o \
	cov_klu_l.o \
	cov_klu_l_diagnostics.o \
	cov_klu_l_dump.o \
	cov_klu_l_factor.o \
	cov_klu_l_free_numeric.o \
	cov_klu_l_kernel.o \
	cov_klu_l_extract.o \
	cov_klu_l_refactor.o \
	cov_klu_l_scale.o \
	cov_klu_l_solve.o \
	cov_klu_l_tsolve.o \
	cov_klu_zl.o \
	cov_klu_zl_diagnostics.o \
	cov_klu_zl_dump.o \
	cov_klu_zl_factor.o \
	cov_klu_zl_free_numeric.o \
	cov_klu_zl_kernel.o \
	cov_klu_zl_extract.o \
	cov_klu_zl_refactor.o \
	cov_klu_zl_scale.o \
	cov_klu_zl_solve.o \
	cov_klu_zl_tsolve.o

KLUCHOLMODOBJ = cov_klu_cholmod.o

KLUCHOLMODLOBJ = cov_klu_l_cholmod.o

OBJ = $(BTFOBJ) $(KLUOBJ) $(KLUCHOLMODOBJ)

LOBJ = $(BTFLOBJ) $(KLULOBJ) $(KLUCHOLMODLOBJ)

$(OBJ): $(INC)

$(LOBJ): $(INC)

klutest: $(OBJ) klutest.c
	$(C) $(I) klutest.c -o klutest $(OBJ) $(LDLIBS)

klultest: $(LOBJ) klutest.c
	$(C) -DDLONG $(I) klutest.c -o klultest $(LOBJ) $(LDLIBS)

.c.o:
	$(C) -c $(I) $*.c

#-------------------------------------------------------------------------------

cov_klu_d.o: ../Source/klu.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d.c
	$(C) -c $(I) cov_klu_d.c

cov_klu_z.o: ../Source/klu.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z.c
	$(C) -c $(I) cov_klu_z.c

cov_klu_d_kernel.o: ../Source/klu_kernel.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_kernel.c
	$(C) -c $(I) cov_klu_d_kernel.c

cov_klu_z_kernel.o: ../Source/klu_kernel.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_kernel.c
	$(C) -c $(I) cov_klu_z_kernel.c

cov_klu_d_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_diagnostics.c
	$(C) -c $(I) cov_klu_d_diagnostics.c

cov_klu_z_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_diagnostics.c
	$(C) -c $(I) cov_klu_z_diagnostics.c

cov_klu_d_dump.o: ../Source/klu_dump.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_dump.c
	$(C) -c $(I) cov_klu_d_dump.c

cov_klu_z_dump.o: ../Source/klu_dump.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_dump.c
	$(C) -c $(I) cov_klu_z_dump.c

cov_klu_d_factor.o: ../Source/klu_factor.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_factor.c
	$(C) -c $(I) cov_klu_d_factor.c

cov_klu_z_factor.o: ../Source/klu_factor.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_factor.c
	$(C) -c $(I) cov_klu_z_factor.c

cov_klu_d_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_free_numeric.c
	$(C) -c $(I) cov_klu_d_free_numeric.c

cov_klu_z_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_free_numeric.c
	$(C) -c $(I) cov_klu_z_free_numeric.c

cov_klu_d_extract.o: ../Source/klu_extract.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_extract.c
	$(C) -c $(I) cov_klu_d_extract.c

cov_klu_z_extract.o: ../Source/klu_extract.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_extract.c
	$(C) -c $(I) cov_klu_z_extract.c

cov_klu_d_refactor.o: ../Source/klu_refactor.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_refactor.c
	$(C) -c $(I) cov_klu_d_refactor.c

cov_klu_z_refactor.o: ../Source/klu_refactor.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_refactor.c
	$(C) -c $(I) cov_klu_z_refactor.c

cov_klu_d_scale.o: ../Source/klu_scale.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_scale.c
	$(C) -c $(I) cov_klu_d_scale.c

cov_klu_z_scale.o: ../Source/klu_scale.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_scale.c
	$(C) -c $(I) cov_klu_z_scale.c

cov_klu_d_solve.o: ../Source/klu_solve.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_solve.c
	$(C) -c $(I) cov_klu_d_solve.c

cov_klu_z_solve.o: ../Source/klu_solve.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_solve.c
	$(C) -c $(I) cov_klu_z_solve.c

cov_klu_d_tsolve.o: ../Source/klu_tsolve.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_tsolve.c
	$(C) -c $(I) cov_klu_d_tsolve.c

cov_klu_z_tsolve.o: ../Source/klu_tsolve.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_tsolve.c
	$(C) -c $(I) cov_klu_z_tsolve.c

#-------------------------------------------------------------------------------

cov_klu_analyze.o: ../Source/klu_analyze.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_analyze.c
	$(C) -c $(I) cov_klu_analyze.c

cov_klu_analyze_given.o: ../Source/klu_analyze_given.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_analyze_given.c
	$(C) -c $(I) cov_klu_analyze_given.c

cov_klu_defaults.o: ../Source/klu_defaults.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_defaults.c
	$(C) -c $(I) cov_klu_defaults.c

cov_klu_free_symbolic.o: ../Source/klu_free_symbolic.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_free_symbolic.c
	$(C) -c $(I) cov_klu_free_symbolic.c

cov_klu_memory.o: ../Source/klu_memory.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_memory.c
	$(C) -c $(I) cov_klu_memory.c

#-------------------------------------------------------------------------------

cov_btf_order.o: ../../BTF/Source/btf_order.c
	$(C) -E $(I) $< | $(PRETTY) > cov_btf_order.c
	$(C) -c $(I) cov_btf_order.c

cov_btf_maxtrans.o: ../../BTF/Source/btf_maxtrans.c
	$(C) -E $(I) $< | $(PRETTY) > cov_btf_maxtrans.c
	$(C) -c $(I) cov_btf_maxtrans.c

cov_btf_strongcomp.o: ../../BTF/Source/btf_strongcomp.c
	$(C) -E $(I) $< | $(PRETTY) > cov_btf_strongcomp.c
	$(C) -c $(I) cov_btf_strongcomp.c

#-------------------------------------------------------------------------------

cov_klu_cholmod.o: ../User/klu_cholmod.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_cholmod.c
	$(C) -c $(I) cov_klu_cholmod.c







#-------------------------------------------------------------------------------

cov_klu_l.o: ../Source/klu.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l.c
	$(C) -c $(I) cov_klu_l.c

cov_klu_zl.o: ../Source/klu.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl.c
	$(C) -c $(I) cov_klu_zl.c

cov_klu_l_kernel.o: ../Source/klu_kernel.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_kernel.c
	$(C) -c $(I) cov_klu_l_kernel.c

cov_klu_zl_kernel.o: ../Source/klu_kernel.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_kernel.c
	$(C) -c $(I) cov_klu_zl_kernel.c

cov_klu_l_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_diagnostics.c
	$(C) -c $(I) cov_klu_l_diagnostics.c

cov_klu_zl_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_diagnostics.c
	$(C) -c $(I) cov_klu_zl_diagnostics.c

cov_klu_l_dump.o: ../Source/klu_dump.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_dump.c
	$(C) -c $(I) cov_klu_l_dump.c

cov_klu_zl_dump.o: ../Source/klu_dump.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_dump.c
	$(C) -c $(I) cov_klu_zl_dump.c

cov_klu_l_factor.o: ../Source/klu_factor.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_factor.c
	$(C) -c $(I) cov_klu_l_factor.c

cov_klu_zl_factor.o: ../Source/klu_factor.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_factor.c
	$(C) -c $(I) cov_klu_zl_factor.c

cov_klu_l_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_free_numeric.c
	$(C) -c $(I) cov_klu_l_free_numeric.c

cov_klu_zl_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_free_numeric.c
	$(C) -c $(I) cov_klu_zl_free_numeric.c

cov_klu_l_extract.o: ../Source/klu_extract.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_extract.c
	$(C) -c $(I) cov_klu_l_extract.c

cov_klu_zl_extract.o: ../Source/klu_extract.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_extract.c
	$(C) -c $(I) cov_klu_zl_extract.c

cov_klu_l_refactor.o: ../Source/klu_refactor.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_refactor.c
	$(C) -c $(I) cov_klu_l_refactor.c

cov_klu_zl_refactor.o: ../Source/klu_refactor.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_refactor.c
	$(C) -c $(I) cov_klu_zl_refactor.c

cov_klu_l_scale.o: ../Source/klu_scale.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_scale.c
	$(C) -c $(I) cov_klu_l_scale.c

cov_klu_zl_scale.o: ../Source/klu_scale.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_scale.c
	$(C) -c $(I) cov_klu_zl_scale.c

cov_klu_l_solve.o: ../Source/klu_solve.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_solve.c
	$(C) -c $(I) cov_klu_l_solve.c

cov_klu_zl_solve.o: ../Source/klu_solve.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_solve.c
	$(C) -c $(I) cov_klu_zl_solve.c

cov_klu_l_tsolve.o: ../Source/klu_tsolve.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_tsolve.c
	$(C) -c $(I) cov_klu_l_tsolve.c

cov_klu_zl_tsolve.o: ../Source/klu_tsolve.c
	$(C) -E $(I) -DDLONG -DCOMPLEX $< | $(PRETTY) > cov_klu_zl_tsolve.c
	$(C) -c $(I) cov_klu_zl_tsolve.c

#-------------------------------------------------------------------------------

cov_klu_l_analyze.o: ../Source/klu_analyze.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_analyze.c
	$(C) -c $(I) cov_klu_l_analyze.c

cov_klu_l_analyze_given.o: ../Source/klu_analyze_given.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_analyze_given.c
	$(C) -c $(I) cov_klu_l_analyze_given.c

cov_klu_l_defaults.o: ../Source/klu_defaults.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_defaults.c
	$(C) -c $(I) cov_klu_l_defaults.c

cov_klu_l_free_symbolic.o: ../Source/klu_free_symbolic.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_free_symbolic.c
	$(C) -c $(I) cov_klu_l_free_symbolic.c

cov_klu_l_memory.o: ../Source/klu_memory.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_memory.c
	$(C) -c $(I) cov_klu_l_memory.c

#-------------------------------------------------------------------------------

cov_btf_l_order.o: ../../BTF/Source/btf_order.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_btf_l_order.c
	$(C) -c $(I) cov_btf_l_order.c

cov_btf_l_maxtrans.o: ../../BTF/Source/btf_maxtrans.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_btf_l_maxtrans.c
	$(C) -c $(I) cov_btf_l_maxtrans.c

cov_btf_l_strongcomp.o: ../../BTF/Source/btf_strongcomp.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_btf_l_strongcomp.c
	$(C) -c $(I) cov_btf_l_strongcomp.c

#-------------------------------------------------------------------------------

cov_klu_l_cholmod.o: ../User/klu_l_cholmod.c
	$(C) -E $(I) -DDLONG $< | $(PRETTY) > cov_klu_l_cholmod.c
	$(C) -c $(I) cov_klu_l_cholmod.c


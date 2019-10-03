
# To run with valgrind:
V =
# V = valgrind -q

# Linux test coverage
CC = gcc
CFLAGS = -O -g -fprofile-arcs -ftest-coverage \
    	-Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
    	-Wredundant-decls -Wnested-externs -Wdisabled-optimization -ansi \
	-Wno-unused-parameter -I../Include \
        -I../Demo -I../../SuiteSparse_config

C = $(CC) $(CFLAGS)

run: all run_convert run_di run_dl run_ci run_cl
	./covall

all:	cs_demo1_di cs_demo2_di cs_demo3_di cstcov_test_di \
	cs_demo1_dl cs_demo2_dl cs_demo3_dl cstcov_test_dl \
	cs_demo1_ci cs_demo2_ci cs_demo3_ci cstcov_test_ci \
	cs_demo1_cl cs_demo2_cl cs_demo3_cl cstcov_test_cl \
	cs_idemo cs_ldemo

CS_DI = cs_add_di.o cs_amd_di.o cs_chol_di.o cs_cholsol_di.o cs_counts_di.o \
	cs_cumsum_di.o cs_droptol_di.o cs_dropzeros_di.o cs_dupl_di.o \
	cs_entry_di.o cs_etree_di.o cs_fkeep_di.o cs_gaxpy_di.o cs_happly_di.o \
	cs_house_di.o cs_ipvec_di.o cs_lsolve_di.o cs_ltsolve_di.o cs_lu_di.o \
	cs_lusol_di.o cs_util_di.o cs_multiply_di.o cs_permute_di.o \
	cs_pinv_di.o cs_post_di.o cs_pvec_di.o cs_qr_di.o cs_qrsol_di.o \
	cs_scatter_di.o cs_schol_di.o cs_sqr_di.o cs_symperm_di.o cs_tdfs_di.o \
	cs_transpose_di.o cs_compress_di.o cs_usolve_di.o cs_scc_di.o \
	cs_maxtrans_di.o cs_dmperm_di.o cs_updown_di.o cs_print_di.o \
	cs_norm_di.o cs_load_di.o cs_dfs_di.o cstcov_malloc_test_di.o \
	cs_utsolve_di.o cs_reach_di.o cs_spsolve_di.o \
	cs_leaf_di.o cs_ereach_di.o cs_randperm_di.o

CS_DL = cs_add_dl.o cs_amd_dl.o cs_chol_dl.o cs_cholsol_dl.o cs_counts_dl.o \
	cs_cumsum_dl.o cs_droptol_dl.o cs_dropzeros_dl.o cs_dupl_dl.o \
	cs_entry_dl.o cs_etree_dl.o cs_fkeep_dl.o cs_gaxpy_dl.o cs_happly_dl.o \
	cs_house_dl.o cs_ipvec_dl.o cs_lsolve_dl.o cs_ltsolve_dl.o cs_lu_dl.o \
	cs_lusol_dl.o cs_util_dl.o cs_multiply_dl.o cs_permute_dl.o \
	cs_pinv_dl.o cs_post_dl.o cs_pvec_dl.o cs_qr_dl.o cs_qrsol_dl.o \
	cs_scatter_dl.o cs_schol_dl.o cs_sqr_dl.o cs_symperm_dl.o cs_tdfs_dl.o \
	cs_transpose_dl.o cs_compress_dl.o cs_usolve_dl.o cs_scc_dl.o \
	cs_maxtrans_dl.o cs_dmperm_dl.o cs_updown_dl.o cs_print_dl.o \
	cs_norm_dl.o cs_load_dl.o cs_dfs_dl.o cstcov_malloc_test_dl.o \
	cs_utsolve_dl.o cs_reach_dl.o cs_spsolve_dl.o \
	cs_leaf_dl.o cs_ereach_dl.o cs_randperm_dl.o

CS_CI = cs_add_ci.o cs_amd_ci.o cs_chol_ci.o cs_cholsol_ci.o cs_counts_ci.o \
	cs_cumsum_ci.o cs_droptol_ci.o cs_dropzeros_ci.o cs_dupl_ci.o \
	cs_entry_ci.o cs_etree_ci.o cs_fkeep_ci.o cs_gaxpy_ci.o cs_happly_ci.o \
	cs_house_ci.o cs_ipvec_ci.o cs_lsolve_ci.o cs_ltsolve_ci.o cs_lu_ci.o \
	cs_lusol_ci.o cs_util_ci.o cs_multiply_ci.o cs_permute_ci.o \
	cs_pinv_ci.o cs_post_ci.o cs_pvec_ci.o cs_qr_ci.o cs_qrsol_ci.o \
	cs_scatter_ci.o cs_schol_ci.o cs_sqr_ci.o cs_symperm_ci.o cs_tdfs_ci.o \
	cs_transpose_ci.o cs_compress_ci.o cs_usolve_ci.o cs_scc_ci.o \
	cs_maxtrans_ci.o cs_dmperm_ci.o cs_updown_ci.o cs_print_ci.o \
	cs_norm_ci.o cs_load_ci.o cs_dfs_ci.o cstcov_malloc_test_ci.o \
	cs_utsolve_ci.o cs_reach_ci.o cs_spsolve_ci.o \
	cs_leaf_ci.o cs_ereach_ci.o cs_randperm_ci.o

CS_CL = cs_add_cl.o cs_amd_cl.o cs_chol_cl.o cs_cholsol_cl.o cs_counts_cl.o \
	cs_cumsum_cl.o cs_droptol_cl.o cs_dropzeros_cl.o cs_dupl_cl.o \
	cs_entry_cl.o cs_etree_cl.o cs_fkeep_cl.o cs_gaxpy_cl.o cs_happly_cl.o \
	cs_house_cl.o cs_ipvec_cl.o cs_lsolve_cl.o cs_ltsolve_cl.o cs_lu_cl.o \
	cs_lusol_cl.o cs_util_cl.o cs_multiply_cl.o cs_permute_cl.o \
	cs_pinv_cl.o cs_post_cl.o cs_pvec_cl.o cs_qr_cl.o cs_qrsol_cl.o \
	cs_scatter_cl.o cs_schol_cl.o cs_sqr_cl.o cs_symperm_cl.o cs_tdfs_cl.o \
	cs_transpose_cl.o cs_compress_cl.o cs_usolve_cl.o cs_scc_cl.o \
	cs_maxtrans_cl.o cs_dmperm_cl.o cs_updown_cl.o cs_print_cl.o \
	cs_norm_cl.o cs_load_cl.o cs_dfs_cl.o cstcov_malloc_test_cl.o \
	cs_utsolve_cl.o cs_reach_cl.o cs_spsolve_cl.o \
	cs_leaf_cl.o cs_ereach_cl.o cs_randperm_cl.o

OBJ = $(CS_DI) $(CS_DL) $(CS_CI) $(CS_CL) cs_convert.o

$(OBJ): ../Include/cs.h cstcov_malloc_test.h Makefile


.PRECIOUS: %demo.c %demo1.c %demo2.c %demo3.c cs_%.c cs_%_ci.c cs_%_cl.c cs_%_di.c cs_%_dl.c cstcov_%.c


%demo.c:
	- ln -s ../Demo/$*demo.c

%demo1.c:
	- ln -s ../Demo/$*demo1.c

%demo2.c:
	- ln -s ../Demo/$*demo2.c

%demo3.c:
	- ln -s ../Demo/$*demo3.c


cstcov_%.c:
	- ln -s cstcov_malloc_test.c cstcov_$*.c

cs_convert.c:
	- ln -s ../Source/cs_convert.c

cs_%_ci.c:
	- ln -s ../Source/cs_$*.c cs_$*_ci.c

cs_%_di.c:
	- ln -s ../Source/cs_$*.c cs_$*_di.c

cs_%_dl.c:
	- ln -s ../Source/cs_$*.c cs_$*_dl.c

cs_%_cl.c:
	- ln -s ../Source/cs_$*.c cs_$*_cl.c


%_di.o: %_di.c
	$(C) -c $<

%_dl.o: %_dl.c
	$(C) -DCS_LONG -c $<

%_ci.o: %_ci.c
	$(C) -DCS_COMPLEX -c $<

%_cl.o: %_cl.c
	$(C) -DCS_LONG -DCS_COMPLEX -c $<


cs_idemo: $(OBJ) cs_idemo.c
	$(C) -o cs_idemo cs_idemo.c $(OBJ) -lm

cs_ldemo: $(OBJ) cs_ldemo.c
	$(C) -o cs_ldemo cs_ldemo.c $(OBJ) -lm


cs_demo1_di: $(CS_DI) cs_di_demo1.c
	$(C) -o cs_demo1_di cs_di_demo1.c $(CS_DI) -lm

cs_demo2_di: $(CS_DI) cs_di_demo2.c cs_di_demo.c
	$(C) -o cs_demo2_di cs_di_demo2.c cs_di_demo.c $(CS_DI) -lm

cs_demo3_di: $(CS_DI) cs_di_demo3.c cs_di_demo.c
	$(C) -o cs_demo3_di cs_di_demo3.c cs_di_demo.c $(CS_DI) -lm

cstcov_test_di: $(CS_DI) cstcov_test.c cs_di_demo.c
	$(C) -o cstcov_test_di cstcov_test.c cs_di_demo.c $(CS_DI) -lm


cs_demo1_dl: $(CS_DL) cs_dl_demo1.c
	$(C) -DCS_LONG -o cs_demo1_dl cs_dl_demo1.c $(CS_DL) -lm

cs_demo2_dl: $(CS_DL) cs_dl_demo2.c cs_dl_demo.c
	$(C) -DCS_LONG -o cs_demo2_dl cs_dl_demo2.c cs_dl_demo.c $(CS_DL) -lm

cs_demo3_dl: $(CS_DL) cs_dl_demo3.c cs_dl_demo.c
	$(C) -DCS_LONG -o cs_demo3_dl cs_dl_demo3.c cs_dl_demo.c $(CS_DL) -lm

cstcov_test_dl: $(CS_DL) cstcov_test.c cs_dl_demo.c
	$(C) -DCS_LONG -o cstcov_test_dl cstcov_test.c cs_dl_demo.c $(CS_DL) -lm


cs_demo1_ci: $(CS_CI) cs_ci_demo1.c
	$(C) -DCS_COMPLEX -o cs_demo1_ci cs_ci_demo1.c $(CS_CI) -lm

cs_demo2_ci: $(CS_CI) cs_ci_demo2.c cs_ci_demo.c
	$(C) -DCS_COMPLEX -o cs_demo2_ci cs_ci_demo2.c cs_ci_demo.c $(CS_CI) -lm

cs_demo3_ci: $(CS_CI) cs_ci_demo3.c cs_ci_demo.c
	$(C) -DCS_COMPLEX -o cs_demo3_ci cs_ci_demo3.c cs_ci_demo.c $(CS_CI) -lm

cstcov_test_ci: $(CS_CI) cstcov_test.c cs_ci_demo.c
	$(C) -DCS_COMPLEX -o cstcov_test_ci cstcov_test.c cs_ci_demo.c $(CS_CI) -lm


cs_demo1_cl: $(CS_CL) cs_cl_demo1.c
	$(C) -DCS_LONG -DCS_COMPLEX -o cs_demo1_cl cs_cl_demo1.c $(CS_CL) -lm

cs_demo2_cl: $(CS_CL) cs_cl_demo2.c cs_cl_demo.c
	$(C) -DCS_LONG -DCS_COMPLEX -o cs_demo2_cl cs_cl_demo2.c cs_cl_demo.c $(CS_CL) -lm

cs_demo3_cl: $(CS_CL) cs_cl_demo3.c cs_cl_demo.c
	$(C) -DCS_LONG -DCS_COMPLEX -o cs_demo3_cl cs_cl_demo3.c cs_cl_demo.c $(CS_CL) -lm

cstcov_test_cl: $(CS_CL) cstcov_test.c cs_cl_demo.c
	$(C) -DCS_LONG -DCS_COMPLEX -o cstcov_test_cl cstcov_test.c cs_cl_demo.c $(CS_CL) -lm


run_convert: cs_idemo cs_ldemo
	- $(V) ./cs_idemo < ../Matrix/t2
	- $(V) ./cs_ldemo < ../Matrix/t2

run_di: cs_demo1_di cs_demo2_di cs_demo3_di cstcov_test_di
	- $(V) ./cs_demo1_di < ../Matrix/t1
	- $(V) ./cs_demo1_di < nil
	- $(V) ./cs_demo1_di < zero
	- $(V) ./cs_demo2_di < nil
	- $(V) ./cs_demo2_di < zero
	- $(V) ./cs_demo2_di < ../Matrix/t1
	- $(V) ./cs_demo2_di < ../Matrix/bcsstk01
	- $(V) ./cs_demo2_di < ../Matrix/fs_183_1
	- $(V) ./cs_demo2_di < ../Matrix/west0067
	- $(V) ./cs_demo2_di < ../Matrix/lp_afiro
	- $(V) ./cs_demo2_di < ../Matrix/ash219
	- $(V) ./cs_demo2_di < ../Matrix/mbeacxc
	- $(V) ./cs_demo2_di < ../Matrix/bcsstk16
	- $(V) ./cs_demo2_di < ../Matrix/ibm32a
	- $(V) ./cs_demo2_di < ../Matrix/ibm32b
	- $(V) ./cs_demo3_di < nil
	- $(V) ./cs_demo3_di < ../Matrix/bcsstk01
	- $(V) ./cs_demo3_di < ../Matrix/bcsstk16
	- $(V) ./cstcov_test_di nil > test_di_nil.out
	- $(V) ./cstcov_test_di zero > test_di_zero.out
	- $(V) ./cstcov_test_di ../Matrix/t1 > test_di_t1.out
	- $(V) ./cstcov_test_di ../Matrix/bcsstk01 > test_di_k1.out
	- $(V) ./cstcov_test_di ../Matrix/fs_183_1 > test_di_fs.out
	- $(V) ./cstcov_test_di ../Matrix/west0067 > test_di_we.out
	- $(V) ./cstcov_test_di ../Matrix/ash219 > test_di_ash.out
	- $(V) ./cstcov_test_di ../Matrix/lp_afiro > test_di_afiro.out

run_dl: cs_demo1_dl cs_demo2_dl cs_demo3_dl cstcov_test_dl
	- $(V) ./cs_demo1_dl < ../Matrix/t1
	- $(V) ./cs_demo1_dl < nil
	- $(V) ./cs_demo1_dl < zero
	- $(V) ./cs_demo2_dl < nil
	- $(V) ./cs_demo2_dl < zero
	- $(V) ./cs_demo2_dl < ../Matrix/t1
	- $(V) ./cs_demo2_dl < ../Matrix/bcsstk01
	- $(V) ./cs_demo2_dl < ../Matrix/fs_183_1
	- $(V) ./cs_demo2_dl < ../Matrix/west0067
	- $(V) ./cs_demo2_dl < ../Matrix/lp_afiro
	- $(V) ./cs_demo2_dl < ../Matrix/ash219
	- $(V) ./cs_demo2_dl < ../Matrix/mbeacxc
	- $(V) ./cs_demo2_dl < ../Matrix/bcsstk16
	- $(V) ./cs_demo2_dl < ../Matrix/ibm32a
	- $(V) ./cs_demo2_dl < ../Matrix/ibm32b
	- $(V) ./cs_demo3_dl < nil
	- $(V) ./cs_demo3_dl < ../Matrix/bcsstk01
	- $(V) ./cs_demo3_dl < ../Matrix/bcsstk16
	- $(V) ./cstcov_test_dl nil > test_dl_nil.out
	- $(V) ./cstcov_test_dl zero > test_dl_zero.out
	- $(V) ./cstcov_test_dl ../Matrix/t1 > test_dl_t1.out
	- $(V) ./cstcov_test_dl ../Matrix/bcsstk01 > test_dl_k1.out
	- $(V) ./cstcov_test_dl ../Matrix/fs_183_1 > test_dl_fs.out
	- $(V) ./cstcov_test_dl ../Matrix/west0067 > test_dl_we.out
	- $(V) ./cstcov_test_dl ../Matrix/ash219 > test_dl_ash.out
	- $(V) ./cstcov_test_dl ../Matrix/lp_afiro > test_dl_afiro.out

run_ci: cs_demo1_ci cs_demo2_ci cs_demo3_ci cstcov_test_ci
	- $(V) ./cs_demo1_ci < ../Matrix/t2
	- $(V) ./cs_demo2_ci < ../Matrix/t2
	- $(V) ./cs_demo1_ci < czero
	- $(V) ./cs_demo2_ci < czero
	- $(V) ./cs_demo1_ci < ../Matrix/t3
	- $(V) ./cs_demo2_ci < ../Matrix/t3
	- $(V) ./cs_demo1_ci < ../Matrix/t4
	- $(V) ./cs_demo2_ci < ../Matrix/t4
	- $(V) ./cs_demo2_ci < ../Matrix/c_west0067
	- $(V) ./cs_demo2_ci < ../Matrix/c_mbeacxc
	- $(V) ./cs_demo2_ci < ../Matrix/c_ibm32a
	- $(V) ./cs_demo2_ci < ../Matrix/c_ibm32b
	- $(V) ./cs_demo2_ci < ../Matrix/young1c
	- $(V) ./cs_demo2_ci < ../Matrix/qc324
	- $(V) ./cs_demo2_ci < ../Matrix/neumann
	- $(V) ./cs_demo2_ci < ../Matrix/c4
	- $(V) ./cs_demo3_ci < ../Matrix/c4
	- $(V) ./cs_demo2_ci < ../Matrix/mhd1280b
	- $(V) ./cs_demo3_ci < ../Matrix/mhd1280b
	- $(V) ./cstcov_test_ci ../Matrix/t2 > test_ci_t2.out
	- $(V) ./cstcov_test_ci ../Matrix/young1c > test_ci_young1c.out
	- $(V) ./cstcov_test_ci ../Matrix/qc324 > test_ci_qc324.out
	- $(V) ./cstcov_test_ci ../Matrix/neumann > test_ci_neumann.out
	- $(V) ./cstcov_test_ci ../Matrix/mhd1280b > test_ci_mhd1280b.out

run_cl: cs_demo1_cl cs_demo2_cl cs_demo3_cl cstcov_test_cl
	- $(V) ./cs_demo1_cl < ../Matrix/t2
	- $(V) ./cs_demo2_cl < ../Matrix/t2
	- $(V) ./cs_demo1_cl < czero
	- $(V) ./cs_demo2_cl < czero
	- $(V) ./cs_demo1_cl < ../Matrix/t3
	- $(V) ./cs_demo2_cl < ../Matrix/t3
	- $(V) ./cs_demo1_cl < ../Matrix/t4
	- $(V) ./cs_demo2_cl < ../Matrix/t4
	- $(V) ./cs_demo2_cl < ../Matrix/c_west0067
	- $(V) ./cs_demo2_cl < ../Matrix/c_mbeacxc
	- $(V) ./cs_demo2_cl < ../Matrix/c_ibm32a
	- $(V) ./cs_demo2_cl < ../Matrix/c_ibm32b
	- $(V) ./cs_demo2_cl < ../Matrix/young1c
	- $(V) ./cs_demo2_cl < ../Matrix/qc324
	- $(V) ./cs_demo2_cl < ../Matrix/neumann
	- $(V) ./cs_demo2_cl < ../Matrix/c4
	- $(V) ./cs_demo3_cl < ../Matrix/c4
	- $(V) ./cs_demo2_cl < ../Matrix/mhd1280b
	- $(V) ./cs_demo3_cl < ../Matrix/mhd1280b
	- $(V) ./cstcov_test_cl ../Matrix/t2 > test_cl_t2.out
	- $(V) ./cstcov_test_cl ../Matrix/young1c > test_cl_young1c.out
	- $(V) ./cstcov_test_cl ../Matrix/qc324 > test_cl_qc324.out
	- $(V) ./cstcov_test_cl ../Matrix/neumann > test_cl_neumann.out
	- $(V) ./cstcov_test_cl ../Matrix/mhd1280b > test_cl_mhd1280b.out


readhb: readhb.f
	f77 -o readhb readhb.f

readhb.f:
	- ln -s ../Demo/readhb.f

clean:
	- $(RM) *.o *.bbg *.da *.gcov *.gcda *.gcno

purge: distclean

distclean: clean
	- $(RM) readhb *.out *.a cov.sort
	- $(RM) cs_demo1_di cs_demo2_di cs_demo3_di cstcov_test_di
	- $(RM) cs_demo1_dl cs_demo2_dl cs_demo3_dl cstcov_test_dl
	- $(RM) cs_demo1_ci cs_demo2_ci cs_demo3_ci cstcov_test_ci
	- $(RM) cs_demo1_cl cs_demo2_cl cs_demo3_cl cstcov_test_cl
	- $(RM) cs_idemo cs_ldemo
	- $(RM) cs_*.c
	- $(RM) cs*_di.c cs*_dl.c cs*_ci.c cs*_cl.c
	- $(RM) -r *.dSYM

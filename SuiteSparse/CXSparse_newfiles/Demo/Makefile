# CC = cc
# CF = -O

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../../include

CS = $(LDFLAGS) -L../../lib -lcxsparse $(LDLIBS)

all: library cs_demo1 cs_demo2 cs_demo3 \
	cs_di_demo1 cs_di_demo2 cs_di_demo3 \
	cs_dl_demo1 cs_dl_demo2 cs_dl_demo3 \
	cs_ci_demo1 cs_ci_demo2 cs_ci_demo3 \
	cs_cl_demo1 cs_cl_demo2 cs_cl_demo3  \
	tests cs_idemo

library:
	( cd ../Lib ; $(MAKE) )
	( cd ../../SuiteSparse_config ; $(MAKE) )

tests: library test_convert test test_di test_dl test_ci test_cl

test: cs_demo1 cs_demo2 cs_demo3
	- ./cs_demo1 < ../Matrix/t1
	- ./cs_demo2 < ../Matrix/t1
	- ./cs_demo2 < ../Matrix/fs_183_1
	- ./cs_demo2 < ../Matrix/west0067
	- ./cs_demo2 < ../Matrix/lp_afiro
	- ./cs_demo2 < ../Matrix/ash219
	- ./cs_demo2 < ../Matrix/mbeacxc
	- ./cs_demo2 < ../Matrix/bcsstk01
	- ./cs_demo3 < ../Matrix/bcsstk01
	- ./cs_demo2 < ../Matrix/bcsstk16
	- ./cs_demo3 < ../Matrix/bcsstk16

test_di: cs_di_demo1 cs_di_demo2 cs_di_demo3
	- ./cs_di_demo1 < ../Matrix/t1
	- ./cs_di_demo2 < ../Matrix/t1
	- ./cs_di_demo2 < ../Matrix/fs_183_1
	- ./cs_di_demo2 < ../Matrix/west0067
	- ./cs_di_demo2 < ../Matrix/lp_afiro
	- ./cs_di_demo2 < ../Matrix/ash219
	- ./cs_di_demo2 < ../Matrix/mbeacxc
	- ./cs_di_demo2 < ../Matrix/bcsstk01
	- ./cs_di_demo3 < ../Matrix/bcsstk01
	- ./cs_di_demo2 < ../Matrix/bcsstk16
	- ./cs_di_demo3 < ../Matrix/bcsstk16

test_dl: cs_dl_demo1 cs_dl_demo2 cs_dl_demo3
	- ./cs_dl_demo1 < ../Matrix/t1
	- ./cs_dl_demo2 < ../Matrix/t1
	- ./cs_dl_demo2 < ../Matrix/fs_183_1
	- ./cs_dl_demo2 < ../Matrix/west0067
	- ./cs_dl_demo2 < ../Matrix/lp_afiro
	- ./cs_dl_demo2 < ../Matrix/ash219
	- ./cs_dl_demo2 < ../Matrix/mbeacxc
	- ./cs_dl_demo2 < ../Matrix/bcsstk01
	- ./cs_dl_demo3 < ../Matrix/bcsstk01
	- ./cs_dl_demo2 < ../Matrix/bcsstk16
	- ./cs_dl_demo3 < ../Matrix/bcsstk16

test_ci: cs_ci_demo1 cs_ci_demo2 cs_ci_demo3
	- ./cs_ci_demo1 < ../Matrix/t2
	- ./cs_ci_demo2 < ../Matrix/t2
	- ./cs_ci_demo2 < ../Matrix/t3
	- ./cs_ci_demo2 < ../Matrix/t4
	- ./cs_ci_demo2 < ../Matrix/c_west0067
	- ./cs_ci_demo2 < ../Matrix/c_mbeacxc
	- ./cs_ci_demo2 < ../Matrix/young1c
	- ./cs_ci_demo2 < ../Matrix/qc324
	- ./cs_ci_demo2 < ../Matrix/neumann
	- ./cs_ci_demo2 < ../Matrix/c4
	- ./cs_ci_demo3 < ../Matrix/c4
	- ./cs_ci_demo2 < ../Matrix/mhd1280b
	- ./cs_ci_demo3 < ../Matrix/mhd1280b

test_cl: cs_cl_demo1 cs_cl_demo2 cs_cl_demo3
	- ./cs_cl_demo1 < ../Matrix/t2
	- ./cs_cl_demo2 < ../Matrix/t2
	- ./cs_cl_demo2 < ../Matrix/t3
	- ./cs_cl_demo2 < ../Matrix/t4
	- ./cs_cl_demo2 < ../Matrix/c_west0067
	- ./cs_cl_demo2 < ../Matrix/c_mbeacxc
	- ./cs_cl_demo2 < ../Matrix/young1c
	- ./cs_cl_demo2 < ../Matrix/qc324
	- ./cs_cl_demo2 < ../Matrix/neumann
	- ./cs_cl_demo2 < ../Matrix/c4
	- ./cs_cl_demo3 < ../Matrix/c4
	- ./cs_cl_demo2 < ../Matrix/mhd1280b
	- ./cs_cl_demo3 < ../Matrix/mhd1280b

test_convert: cs_idemo cs_ldemo
	- ./cs_idemo < ../Matrix/t2
	- ./cs_ldemo < ../Matrix/t2

cs_demo1: cs_demo1.c Makefile
	$(CC) $(CF) $(I) -o cs_demo1 cs_demo1.c $(CS)

cs_demo2: cs_demo2.c cs_demo.c cs_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_demo2 cs_demo2.c cs_demo.c $(CS)

cs_demo3: cs_demo3.c cs_demo.c cs_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_demo3 cs_demo3.c cs_demo.c $(CS)




cs_di_demo1: cs_di_demo1.c Makefile
	$(CC) $(CF) $(I) -o cs_di_demo1 cs_di_demo1.c $(CS)

cs_di_demo2: cs_di_demo2.c cs_di_demo.c cs_di_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_di_demo2 cs_di_demo2.c cs_di_demo.c $(CS)

cs_di_demo3: cs_di_demo3.c cs_di_demo.c cs_di_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_di_demo3 cs_di_demo3.c cs_di_demo.c $(CS)




cs_ci_demo1: cs_ci_demo1.c Makefile
	$(CC) $(CF) $(I) -o cs_ci_demo1 cs_ci_demo1.c $(CS)

cs_ci_demo2: cs_ci_demo2.c cs_ci_demo.c cs_ci_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_ci_demo2 cs_ci_demo2.c cs_ci_demo.c $(CS)

cs_ci_demo3: cs_ci_demo3.c cs_ci_demo.c cs_ci_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_ci_demo3 cs_ci_demo3.c cs_ci_demo.c $(CS)



cs_dl_demo1: cs_dl_demo1.c Makefile
	$(CC) $(CF) $(I) -o cs_dl_demo1 cs_dl_demo1.c $(CS)

cs_dl_demo2: cs_dl_demo2.c cs_dl_demo.c cs_dl_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_dl_demo2 cs_dl_demo2.c cs_dl_demo.c $(CS)

cs_dl_demo3: cs_dl_demo3.c cs_dl_demo.c cs_dl_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_dl_demo3 cs_dl_demo3.c cs_dl_demo.c $(CS)


cs_cl_demo1: cs_cl_demo1.c Makefile
	$(CC) $(CF) $(I) -o cs_cl_demo1 cs_cl_demo1.c $(CS)

cs_cl_demo2: cs_cl_demo2.c cs_cl_demo.c cs_cl_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_cl_demo2 cs_cl_demo2.c cs_cl_demo.c $(CS)

cs_cl_demo3: cs_cl_demo3.c cs_cl_demo.c cs_cl_demo.h Makefile
	$(CC) $(CF) $(I) -o cs_cl_demo3 cs_cl_demo3.c cs_cl_demo.c $(CS)




cs_idemo: cs_idemo.c Makefile
	$(CC) $(CF) $(I) -o cs_idemo cs_idemo.c $(CS)

cs_ldemo: cs_ldemo.c Makefile
	$(CC) $(CF) $(I) -DCS_LONG -o cs_ldemo cs_ldemo.c $(CS)


clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) cs_demo1 cs_demo2 cs_demo3 *.a
	- $(RM) cs_di_demo1 cs_di_demo2 cs_di_demo3
	- $(RM) cs_dl_demo1 cs_dl_demo2 cs_dl_demo3
	- $(RM) cs_ci_demo1 cs_ci_demo2 cs_ci_demo3
	- $(RM) cs_cl_demo1 cs_cl_demo2 cs_cl_demo3
	- $(RM) cs_idemo cs_ldemo
	- $(RM) -r $(PURGE)


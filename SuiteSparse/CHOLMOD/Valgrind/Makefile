
doall: links go

links:
	touch links
	ln -s ../Tcov/amdtest.c
	ln -s ../Tcov/aug.c
	ln -s ../Tcov/camdtest.c
	ln -s ../Tcov/cctest.c
	ln -s ../Tcov/cm.c
	ln -s ../Tcov/cm.h
	ln -s ../Tcov/cmread.c
	ln -s ../Tcov/ctest.c
	ln -s ../Tcov/huge.c
	ln -s ../Tcov/leak.c
	ln -s ../Tcov/lpdemo.c
	ln -s ../Tcov/memory.c
	ln -s ../Tcov/null2.c
	ln -s ../Tcov/null.c
	ln -s ../Tcov/raw_factor.c
	ln -s ../Tcov/solve.c
	ln -s ../Tcov/test_ops.c
	ln -s ../Tcov/unpack.c
	ln -s ../Tcov/comments.txt
	ln -s ../Tcov/Matrix

include ../Tcov/Makefile

dopurge: distclean
	- $(RM) amdtest.c aug.c camdtest.c cctest.c cm.c cm.h cmread.c \
	    ctest.c huge.c leak.c lpdemo.c memory.c null2.c null.c \
	    raw_factor.c solve.c test_ops.c unpack.c comments.txt \
	    links Matrix


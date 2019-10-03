default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

all: libklu_cholmod.a

I = -I../../CHOLMOD/Include -I../../SuiteSparse_config \
    -I../Include -I../../AMD/Include \
    -I../../BTF/Include -I../../COLAMD/Include

libklu_cholmod.a:  library klu_cholmod.c klu_cholmod.h
	$(CC) $(CF) $(I) -c klu_cholmod.c
	$(ARCHIVE)  libklu_cholmod.a klu_cholmod.o
	- $(RANLIB) libklu_cholmod.a

distclean: purge

purge: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

library:
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
	( cd ../../CHOLMOD/Lib ; $(MAKE) )
	( cd ../../SuiteSparse_config ; $(MAKE) )

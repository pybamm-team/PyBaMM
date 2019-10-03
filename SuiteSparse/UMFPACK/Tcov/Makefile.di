SUITESPARSE = $(CURDIR)
export SUITESPARSE

KIND = -DDINT

all: go

include SuiteSparse_config/SuiteSparse_config.mk

go: run
	- ( cd UMFPACK/Source ; ./ucov.di )
	- ( cd AMD/Source     ; ./acov.di )

run: prog
	- ./ut > ut.out
	- tail ut.out

prog:
	( cd SuiteSparse_config ; $(MAKE) )
	( cd AMD ; $(MAKE) library )
	( cd UMFPACK ; $(MAKE) library )
	$(CC) $(KIND) $(CF) $(UMFPACK_CONFIG) -IUMFPACK/Source -IAMD/Include \
		-Iinclude -o ut ut.c UMFPACK/Lib/*.o \
		SuiteSparse_config/*.o AMD/Lib/*.o \
		-Llib -lcholmod -lcolamd -lmetis -lccolamd -lcamd -lsuitesparseconfig \
		$(LDLIBS)

utcov:
	- ( cd UMFPACK/Source ; ./ucov.di )
	- ( cd AMD/Source     ; ./acov.di )


purge:
	( cd UMFPACK ; $(MAKE) purge )
	( cd AMD ; $(MAKE) purge )

#-------------------------------------------------------------------------------
# compile the UMFPACK demos
#-------------------------------------------------------------------------------

# UMFPACK Copyright (c) Timothy A. Davis.
# All Rights Reserved.  See ../Doc/License.txt for License.

default: libs run

all: libs run hb

all32: libs run hb fortran

all64: libs run hb fortran64

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# UMFPACK optionally uses the CHOLMOD Partition module
LIB_WITH_CHOLMOD =
ifeq (,$(findstring -DNCHOLMOD, $(UMFPACK_CONFIG)))
    LIB_WITH_CHOLMOD = $(LIB_WITH_PARTITION) $(CUBLAS_LIB) $(CUDART_LIB)
endif

#-------------------------------------------------------------------------------

C = $(CC) $(CF) $(UMFPACK_CONFIG) $(CONFIG_PARTITION) \
    -I../../include

LIBS = $(LDLIBS) -L../../lib -lumfpack -lamd -lsuitesparseconfig \
	$(LIB_WITH_CHOLMOD) $(LAPACK) $(BLAS)

libs: metis
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../Lib ; $(MAKE) )
	- ( cd ../../CHOLMOD && $(MAKE) library )
	- ( cd ../../COLAMD && $(MAKE) library )
	- ( cd ../../CCOLAMD ; $(MAKE) library )
	- ( cd ../../CAMD ; $(MAKE) library )

metis: ../../include/metis.h

../../include/metis.h:
	- ( cd ../.. && $(MAKE) metis )

#-------------------------------------------------------------------------------
# Create the demo programs, run them, and compare the output
#-------------------------------------------------------------------------------

dist: umfpack_di_demo.c umfpack_dl_demo.c umfpack_zi_demo.c umfpack_zl_demo.c

umfpack_simple: umfpack_simple.c
	$(C) -o umfpack_simple umfpack_simple.c $(LIBS)

# these rules are simpler:
# umfpack_%_demo.c: umfpack_xx_demo.c umfpack_%_demo.sed
# 	- sed -f umfpack_$*_demo.sed < umfpack_xx_demo.c > umfpack_$*_demo.c
# 
# umfpack_%_demo: umfpack_%_demo.c $(UMFPACK)
# 	$(C) -o umfpack_$*_demo umfpack_$*_demo.c $(LIBS)
# 	./umfpack_$*_demo > my_umfpack_$*_demo.out

# but do this via brute-force, so we can use just a single Makefile:

# double-precision, int verion:
umfpack_di_demo.c: umfpack_xx_demo.c umfpack_di_demo.sed
	- sed -f umfpack_di_demo.sed < umfpack_xx_demo.c > umfpack_di_demo.c

umfpack_di_demo: umfpack_di_demo.c
	$(C) -o umfpack_di_demo umfpack_di_demo.c $(LIBS)

# double-precision, SuiteSparse_long verion:
umfpack_dl_demo.c: umfpack_xx_demo.c umfpack_dl_demo.sed
	- sed -f umfpack_dl_demo.sed < umfpack_xx_demo.c > umfpack_dl_demo.c

umfpack_dl_demo: umfpack_dl_demo.c
	$(C) -o umfpack_dl_demo umfpack_dl_demo.c $(LIBS)

# complex, int verion:
umfpack_zi_demo.c: umfpack_xx_demo.c umfpack_zi_demo.sed
	- sed -f umfpack_zi_demo.sed < umfpack_xx_demo.c > umfpack_zi_demo.c

umfpack_zi_demo: umfpack_zi_demo.c
	$(C) -o umfpack_zi_demo umfpack_zi_demo.c $(LIBS)

# complex, SuiteSparse_long verion:
umfpack_zl_demo.c: umfpack_xx_demo.c umfpack_zl_demo.sed
	- sed -f umfpack_zl_demo.sed < umfpack_xx_demo.c > umfpack_zl_demo.c

umfpack_zl_demo: umfpack_zl_demo.c
	$(C) -o umfpack_zl_demo umfpack_zl_demo.c $(LIBS)

run: umfpack_di_demo umfpack_zi_demo umfpack_dl_demo umfpack_zl_demo umfpack_simple
	./umfpack_simple
	./umfpack_di_demo > my_umfpack_di_demo.out
	- diff umfpack_di_demo.out my_umfpack_di_demo.out
	./umfpack_dl_demo > my_umfpack_dl_demo.out
	- diff umfpack_dl_demo.out my_umfpack_dl_demo.out
	./umfpack_zi_demo > my_umfpack_zi_demo.out
	- diff umfpack_zi_demo.out my_umfpack_zi_demo.out
	./umfpack_zl_demo > my_umfpack_zl_demo.out
	- diff umfpack_zl_demo.out my_umfpack_zl_demo.out

#-------------------------------------------------------------------------------
# create a demo program that reads in Harwell/Boeing matrices, and run it
#-------------------------------------------------------------------------------

# the output of "make hb" is in the file umf4.out
hb: $(UMFPACK) umf4 readhb readhb_nozeros readhb_size
	- ./readhb_nozeros < HB/can_24.psa > tmp/A
	- ./readhb_size    < HB/can_24.psa > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/west0067.rua > tmp/A
	- ./readhb_size    < HB/west0067.rua > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/fs_183_6.rua > tmp/A
	- ./readhb_size    < HB/fs_183_6.rua > tmp/Asize
	- ./umf4
	- ./readhb         < HB/fs_183_6.rua > tmp/A
	- ./readhb_size    < HB/fs_183_6.rua > tmp/Asize
	- ./umf4
	- ./readhb         < HB/arc130.rua > tmp/A
	- ./readhb_size    < HB/arc130.rua > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/arc130.rua > tmp/A
	- ./readhb_size    < HB/arc130.rua > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/arc130.rua > tmp/A
	- ./readhb_size    < HB/arc130.rua > tmp/Asize
	- ./umf4 a 1e-6

umf4: umf4.c
	$(C) -o umf4 umf4.c $(LIBS)

readhb:  readhb.f
	$(F77) $(F77FLAGS) -o readhb readhb.f $(F77LIB)

readhb_size:  readhb_size.f
	$(F77) $(F77FLAGS) -o readhb_size readhb_size.f $(F77LIB)

readhb_nozeros:  readhb_nozeros.f
	$(F77) $(F77FLAGS) -o readhb_nozeros readhb_nozeros.f $(F77LIB)

#-------------------------------------------------------------------------------
# compile the FORTRAN interface and demo
#-------------------------------------------------------------------------------

# This doesn't seem to work on the Mac (Snow Leopard, OS X 10.6.1)
fortran: $(UMFPACK) umf4hb.f umf4_f77wrapper.o umf4zhb.f umf4_f77zwrapper.o $(UMFPACK)
	$(F77) $(F77FLAGS) -o umf4hb umf4hb.f umf4_f77wrapper.o $(LIBS)
	- ./umf4hb < HB/west0067.rua > my_umf4hb.out
	- diff my_umf4hb.out umf4hb.out
	$(F77) $(F77FLAGS) -o umf4zhb umf4zhb.f umf4_f77zwrapper.o $(LIBS)
	- ./umf4zhb < HB/qc324.cua > my_umf4zhb.out
	- diff my_umf4zhb.out umf4zhb.out

fortran64: $(UMFPACK) umf4hb64.f umf4_f77wrapper64.o umf4_f77zwrapper64.o $(UMFPACK)
	$(F77) $(F77FLAGS) -o umf4hb64 umf4hb64.f umf4_f77wrapper64.o $(LIBS)
	- ./umf4hb64 < HB/west0067.rua > my_umf4hb64.out
	- diff my_umf4hb64.out umf4hb64.out

umf4_f77wrapper.o: umf4_f77wrapper.c
	$(C) -c umf4_f77wrapper.c -o umf4_f77wrapper.o

umf4_f77zwrapper.o: umf4_f77zwrapper.c
	$(C) -c umf4_f77zwrapper.c -o umf4_f77zwrapper.o

umf4_f77wrapper64.o: umf4_f77wrapper.c
	$(C) -DDLONG -c umf4_f77wrapper.c -o umf4_f77wrapper64.o

umf4_f77zwrapper64.o: umf4_f77zwrapper.c
	$(C) -DZLONG -c umf4_f77zwrapper.c -o umf4_f77zwrapper64.o

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

purge: clean
	- $(RM) umfpack_simple a.out
	- $(RM) umfpack_di_demo my_umfpack_di_demo.out
	- $(RM) umfpack_dl_demo my_umfpack_dl_demo.out
	- $(RM) umfpack_zi_demo my_umfpack_zi_demo.out
	- $(RM) umfpack_zl_demo my_umfpack_zl_demo.out
	- $(RM) umf4hb umf4zhb *.umf my_umf4hb.out
	- $(RM) umf4hb64 my_umf4hb64.out my_umf4zhb.out
	- $(RM) umf4 readhb readhb_nozeros readhb_size tmp/*
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

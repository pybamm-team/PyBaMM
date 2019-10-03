#-------------------------------------------------------------------------------
# LDL Demo Makefile
#-------------------------------------------------------------------------------

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../../include

C = $(CC) $(CF) $(I)

all: library ldlsimple ldllsimple ldlmain ldllmain ldlamd ldllamd

LIB2 = $(LDFLAGS) -L../../lib -lldl -lamd -lsuitesparseconfig $(LDLIBS)

library:
	( cd ../Lib ; $(MAKE) )
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) )

#-------------------------------------------------------------------------------
# stand-alone C programs:
#-------------------------------------------------------------------------------

ldlmain:  ldlmain.c
	$(C) ldlmain.c -o ldlmain $(LIB2)
	- ./ldlmain > my_ldlmain.out
	- diff ldlmain.out my_ldlmain.out

ldllmain:  ldlmain.c
	$(C) -DLDL_LONG ldlmain.c -o ldllmain $(LIB2)
	- ./ldllmain > my_ldllmain.out
	- diff ldlmain.out my_ldllmain.out

ldlsimple:  ldlsimple.c
	$(C) ldlsimple.c -o ldlsimple $(LIB2)
	- ./ldlsimple > my_ldlsimple.out
	- diff ldlsimple.out my_ldlsimple.out

ldllsimple:  ldlsimple.c library
	$(C) $(I) -DLDL_LONG ldlsimple.c -o ldllsimple $(LIB2)
	- ./ldllsimple > my_ldllsimple.out
	- diff ldlsimple.out my_ldllsimple.out

ldlamd:  ldlmain.c library
	- $(C) -I../../AMD/Include -DUSE_AMD ldlmain.c -o ldlamd $(LIB2)
	- ./ldlamd > my_ldlamd.out
	- diff ldlamd.out my_ldlamd.out

ldllamd:  ldlmain.c library
	- $(C) -DLDL_LONG $(I) -I../../AMD/Include -DUSE_AMD \
		ldlmain.c -o ldllamd $(LIB2)
	- ./ldllamd > my_ldllamd.out
	- diff ldllamd.out my_ldllamd.out

run:
	- ./ldlsimple > my_ldlsimple.out
	- diff ldlsimple.out my_ldlsimple.out
	- ./ldllsimple > my_ldllsimple.out
	- diff ldlsimple.out my_ldllsimple.out
	- ./ldlmain > my_ldlmain.out
	- diff ldlmain.out my_ldlmain.out
	- ./ldllmain > my_ldllmain.out
	- diff ldlmain.out my_ldllmain.out
	- ./ldlamd > my_ldlamd.out
	- diff ldlamd.out my_ldlamd.out
	- ./ldllamd > my_ldllamd.out
	- diff ldllamd.out my_ldllamd.out


#-------------------------------------------------------------------------------
# clean-up:
#-------------------------------------------------------------------------------

distclean: purge

purge: clean
	- $(RM) ldlmain ldllmain ldlsimple ldllsimple ldlamd ldllamd
	- $(RM) my_ldlmain.out my_ldlamd.out my_ldlsimple.out my_ldllamd.out
	- $(RM) my_ldllsimple.out my_ldllmain.out
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

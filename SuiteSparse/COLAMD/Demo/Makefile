#-----------------------------------------------------------------------------
# compile the COLAMD demo
#-----------------------------------------------------------------------------

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../../include

C = $(CC) $(CF) $(I)

LIB2 = $(LDFLAGS) -L../../lib -lcolamd -lsuitesparseconfig $(LDLIBS)

all: library colamd_example colamd_l_example

library:
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )

#------------------------------------------------------------------------------
# Create the demo program, run it, and compare the output
#------------------------------------------------------------------------------

dist:

colamd_example: colamd_example.c
	$(C) -o colamd_example colamd_example.c $(LIB2)
	- ./colamd_example > my_colamd_example.out
	- diff colamd_example.out my_colamd_example.out

colamd_l_example: colamd_l_example.c
	$(C) -o colamd_l_example colamd_l_example.c $(LIB2)
	- ./colamd_l_example > my_colamd_l_example.out
	- diff colamd_l_example.out my_colamd_l_example.out

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) colamd_example colamd_l_example
	- $(RM) my_colamd_example.out my_colamd_l_example.out
	- $(RM) -r $(PURGE)

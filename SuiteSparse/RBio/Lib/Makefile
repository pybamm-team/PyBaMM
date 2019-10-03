#===============================================================================
# RBio/Lib/Makefile: for compiling the RBio library
#===============================================================================

LIBRARY = librbio
VERSION = 2.2.6
SO_VERSION = 2

default: library

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

# RBio depends on SuiteSparse_config
LDLIBS += -lsuitesparseconfig

# compile and install in SuiteSparse/lib
library:
	$(MAKE) install INSTALL=$(SUITESPARSE)

C = $(CC) $(CF)

I = -I../Include -I../../SuiteSparse_config

all: library

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC = ../Include/RBio.h

DL = RBio.o
DI = RBio_i.o
OBJ = $(DI) $(DL)

$(OBJ): $(INC)

static: $(AR_TARGET)

$(AR_TARGET): $(OBJ)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

#-------------------------------------------------------------------------------

# 64-bit long version (default)
RBio.o: ../Source/RBio.c
	$(C) -c $(I) $<

# 32-bit int version (not the default)
RBio_i.o: ../Source/RBio.c
	$(C) -c -DINT $(I) $< -o $@

#-------------------------------------------------------------------------------

# install RBio
install: $(AR_TARGET) $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(OBJ)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CC) $(SO_OPTS) $^ -o $@ $(LDLIBS)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../Include/RBio.h $(INSTALL_INCLUDE)
	$(CP) ../README.txt $(INSTALL_DOC)/RBIO_README.txt
	chmod 755 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_INCLUDE)/RBio.h
	chmod 644 $(INSTALL_DOC)/RBIO_README.txt

# uninstall RBio
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_INCLUDE)/RBio.h
	$(RM) $(INSTALL_DOC)/RBIO_README.txt

#-------------------------------------------------------------------------------
# Makefile for the LDL documentation
#-------------------------------------------------------------------------------

default: ldl_userguide.pdf

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# clean-up:
#-------------------------------------------------------------------------------

distclean: purge

purge: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

#-------------------------------------------------------------------------------
# user guide:
#-------------------------------------------------------------------------------

ldl_userguide.pdf: ldl_userguide.tex ldl.bib
	latex ldl_userguide
	- bibtex ldl_userguide
	latex ldl_userguide
	latex ldl_userguide
	dvips ldl_userguide -o ldl_userguide.ps
	pdflatex ldl_userguide
	pdflatex ldl_userguide
	- $(RM) *.dvi *.aux *.log *.bak *.bbl *.blg *.ps

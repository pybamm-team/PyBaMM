#------------------------------------------------------------------------------
# CAMD Makefile for compiling on Unix systems
#------------------------------------------------------------------------------

default: dist

include ../../SuiteSparse_config/SuiteSparse_config.mk

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) -r camd_temp $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

#------------------------------------------------------------------------------
# Create the User Guide and Quick Start Guide
#------------------------------------------------------------------------------

CAMD_UserGuide.pdf: CAMD_UserGuide.tex CAMD_UserGuide.bib ../Include/camd.h
	echo '\\begin{verbatim}' > camd_h.tex
	expand -8 ../Include/camd.h >> camd_h.tex
	echo '\\end{verbatim}' >> camd_h.tex
	pdflatex CAMD_UserGuide
	bibtex CAMD_UserGuide
	pdflatex CAMD_UserGuide
	pdflatex CAMD_UserGuide

dist:  CAMD_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc camd_h.tex
	- $(RM) camd_temp


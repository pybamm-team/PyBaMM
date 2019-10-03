#------------------------------------------------------------------------------
# AMD Makefile for compiling on Unix systems
#------------------------------------------------------------------------------

default: dist

include ../../SuiteSparse_config/SuiteSparse_config.mk

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

#------------------------------------------------------------------------------
# Create the User Guide and Quick Start Guide
#------------------------------------------------------------------------------

AMD_UserGuide.pdf: AMD_UserGuide.tex AMD_UserGuide.bib ../Include/amd.h
	echo '\\begin{verbatim}' > amd_h.tex
	expand -8 ../Include/amd.h >> amd_h.tex
	echo '\\end{verbatim}' >> amd_h.tex
	pdflatex AMD_UserGuide
	bibtex AMD_UserGuide
	pdflatex AMD_UserGuide
	pdflatex AMD_UserGuide

dist:  AMD_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc amd_h.tex


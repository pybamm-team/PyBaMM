#------------------------------------------------------------------------------
# KLU Makefile for creating the user guide
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

KLU_UserGuide.pdf: KLU_UserGuide.tex KLU_UserGuide.bib \
    ../Include/klu.h ../../BTF/Include/btf.h Makefile
	echo '\\begin{verbatim}' > klu_h.tex
	expand -8 ../Include/klu.h >> klu_h.tex
	echo '\\end{verbatim}' >> klu_h.tex
	echo '\\begin{verbatim}' > btf_h.tex
	expand -8 ../../BTF/Include/btf.h >> btf_h.tex
	echo '\\end{verbatim}' >> btf_h.tex
	echo '\\begin{verbatim}' > klu_simple_c.tex
	expand -8 ../Demo/klu_simple.c >> klu_simple_c.tex
	echo '\\end{verbatim}' >> klu_simple_c.tex
	pdflatex KLU_UserGuide
	bibtex KLU_UserGuide
	pdflatex KLU_UserGuide
	pdflatex KLU_UserGuide

dist:  KLU_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc
	- $(RM) klu_simple_c.tex klu_h.tex btf_h.tex


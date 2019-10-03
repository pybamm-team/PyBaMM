
default: spqr_user_guide.pdf

include ../../SuiteSparse_config/SuiteSparse_config.mk

spqr_user_guide.pdf: spqr_user_guide.tex spqr_user_guide.bib Makefile
	pdflatex spqr_user_guide.tex
	bibtex spqr_user_guide
	pdflatex spqr_user_guide.tex
	pdflatex spqr_user_guide.tex
	- $(RM) -r $(CLEAN) *.out

distclean: purge

clean:
	- $(RM) -r $(CLEAN) *.out

purge: clean
	- $(RM) -r $(PURGE)

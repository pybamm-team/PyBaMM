/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

GLOBAL Int UMF_analyze
(
    Int n_row,		/* A is n_row-by-n_col */
    Int n_col,
    Int Ai [ ],		/* Ai [Ap [0]..Ap[n_row]-1]: column indices */
    Int Ap [ ],		/* of size MAX (n_row, n_col) + 1 */
    Int Up [ ],		/* workspace of size n_col, and output column perm. */
    Int fixQ,
    /* temporary workspaces: */
    Int W [ ],		/* W [0..n_col-1] */
    Int Link [ ],	/* Link [0..n_col-1] */
    /* output: information about each frontal matrix: */
    Int Front_ncols [ ],	/* size n_col */
    Int Front_nrows [ ],	/* of size n_col */
    Int Front_npivcol [ ],	/* of size n_col */
    Int Front_parent [ ],	/* of size n_col */
    Int *nfr_out,
    Int *p_ncompactions		/* number of compactions in UMF_analyze */
) ;

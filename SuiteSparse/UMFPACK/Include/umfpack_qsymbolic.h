/* ========================================================================== */
/* === umfpack_qsymbolic ==================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

int umfpack_di_qsymbolic
(
    int n_row,
    int n_col,
    const int Ap [ ],
    const int Ai [ ],
    const double Ax [ ],
    const int Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

SuiteSparse_long umfpack_dl_qsymbolic
(
    SuiteSparse_long n_row,
    SuiteSparse_long n_col,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ],
    const double Ax [ ],
    const SuiteSparse_long Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_qsymbolic
(
    int n_row,
    int n_col,
    const int Ap [ ],
    const int Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

SuiteSparse_long umfpack_zl_qsymbolic
(
    SuiteSparse_long n_row,
    SuiteSparse_long n_col,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ],
    const double Ax [ ], const double Az [ ],
    const SuiteSparse_long Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_di_fsymbolic
(
    int n_row,
    int n_col,
    const int Ap [ ],
    const int Ai [ ],
    const double Ax [ ],
    int (*user_ordering) ( int, int, int, int *, int *, int *, void *, double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

SuiteSparse_long umfpack_dl_fsymbolic
(
    SuiteSparse_long n_row,
    SuiteSparse_long n_col,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ],
    const double Ax [ ],
    int (*user_ordering) (SuiteSparse_long, SuiteSparse_long, SuiteSparse_long,
        SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *, void *,
        double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_fsymbolic
(
    int n_row,
    int n_col,
    const int Ap [ ],
    const int Ai [ ],
    const double Ax [ ], const double Az [ ],
    int (*user_ordering) (int, int, int, int *, int *, int *, void *, double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

SuiteSparse_long umfpack_zl_fsymbolic
(
    SuiteSparse_long n_row,
    SuiteSparse_long n_col,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ],
    const double Ax [ ], const double Az [ ],
    int (*user_ordering) (SuiteSparse_long, SuiteSparse_long, SuiteSparse_long,
        SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *, void *,
        double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

/*
double int Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int n_row, n_col, *Ap, *Ai, *Qinit, status ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax ;
    status = umfpack_di_qsymbolic (n_row, n_col, Ap, Ai, Ax, Qinit,
	&Symbolic, Control, Info) ;

double SuiteSparse_long Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    SuiteSparse_long n_row, n_col, *Ap, *Ai, *Qinit, status ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax ;
    status = umfpack_dl_qsymbolic (n_row, n_col, Ap, Ai, Ax, Qinit,
	&Symbolic, Control, Info) ;

complex int Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int n_row, n_col, *Ap, *Ai, *Qinit, status ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax, *Az ;
    status = umfpack_zi_qsymbolic (n_row, n_col, Ap, Ai, Ax, Az, Qinit,
	&Symbolic, Control, Info) ;

complex SuiteSparse_long Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    SuiteSparse_long n_row, n_col, *Ap, *Ai, *Qinit, status ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax, *Az ;
    status = umfpack_zl_qsymbolic (n_row, n_col, Ap, Ai, Ax, Az, Qinit,
	&Symbolic, Control, Info) ;

packed complex Syntax:

    Same as above, except Az is NULL.

Purpose:

    Given the nonzero pattern of a sparse matrix A in column-oriented form, and
    a sparsity preserving column pre-ordering Qinit, umfpack_*_qsymbolic
    performs the symbolic factorization of A*Qinit (or A (:,Qinit) in MATLAB
    notation).  This is identical to umfpack_*_symbolic, except that neither
    COLAMD nor AMD are called and the user input column order Qinit is used
    instead.  Note that in general, the Qinit passed to umfpack_*_qsymbolic
    can differ from the final Q found in umfpack_*_numeric.  The unsymmetric
    strategy will perform a column etree postordering done in
    umfpack_*_qsymbolic and sparsity-preserving modifications are made within
    each frontal matrix during umfpack_*_numeric.  The symmetric
    strategy will preserve Qinit, unless the matrix is structurally singular.

    See umfpack_*_symbolic for more information.  Note that Ax and Ax are
    optional.  The may be NULL.

    *** WARNING ***  A poor choice of Qinit can easily cause umfpack_*_numeric
    to use a huge amount of memory and do a lot of work.  The "default" symbolic
    analysis method is umfpack_*_symbolic, not this routine.  If you use this
    routine, the performance of UMFPACK is your responsibility;  UMFPACK will
    not try to second-guess a poor choice of Qinit.

Returns:

    The value of Info [UMFPACK_STATUS]; see umfpack_*_symbolic.
    Also returns UMFPACK_ERROR_invalid_permuation if Qinit is not a valid
    permutation vector.

Arguments:

    All arguments are the same as umfpack_*_symbolic, except for the following:

    Int Qinit [n_col] ;		Input argument, not modified.

	The user's fill-reducing initial column pre-ordering.  This must be a
	permutation of 0..n_col-1.  If Qinit [k] = j, then column j is the kth
	column of the matrix A (:,Qinit) to be factorized.  If Qinit is an
	(Int *) NULL pointer, then COLAMD or AMD are called instead.

    double Control [UMFPACK_CONTROL] ;	Input argument, not modified.

	If Qinit is not NULL, then only two strategies are recognized:
	the unsymmetric strategy and the symmetric strategy.
	If Control [UMFPACK_STRATEGY] is UMFPACK_STRATEGY_SYMMETRIC,
	then the symmetric strategy is used.  Otherwise the unsymmetric
	strategy is used.

The umfpack_*_fsymbolic functions are identical to their umfpack_*_qsymbolic
functions, except that Qinit is replaced with a pointer to an ordering
function (user_ordering), and a pointer to an extra input that is passed
to the user_ordering (user_params).  The arguments have the following syntax:


    int (*user_ordering)    // TRUE if OK, FALSE otherwise
    (
        // inputs, not modified on output
        int,            // nrow
        int,            // ncol
        int,            // sym: if TRUE and nrow==ncol do A+A', else do A'A
        int *,          // Ap, size ncol+1
        int *,          // Ai, size nz
        // output
        int *,          // size ncol, fill-reducing permutation
        // input/output
        void *,         // user_params (ignored by UMFPACK)
        double *        // user_info[0..2], optional output for symmetric case.
                        // user_info[0]: max column count for L=chol(A+A')
                        // user_info[1]: nnz (L)
                        // user_info[2]: flop count for chol(A+A'), if A real
    ),
    void *user_params,  // passed to user_ordering function

*/

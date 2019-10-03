/* ========================================================================== */
/* === UMFPACK mexFunction ================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    MATLAB interface for umfpack.

    Factor or solve a sparse linear system, returning either the solution
    x to Ax=b or A'x'=b', or the factorization LU=P(R\A)Q or LU=PAQ.  A must be
    sparse, with nonzero dimensions, but it may be complex, singular, and/or
    rectangular.  b must be a dense n-by-1 vector (real or complex).
    L is unit lower triangular, U is upper triangular, and R is diagonal.
    P and Q are permutation matrices (permutations of an identity matrix).

    The matrix A is scaled, by default.  Each row i is divided by r (i), where
    r (i) is the sum of the absolute values of the entries in that row.  The
    scaled matrix has an infinity norm of 1.  The scale factors r (i) are
    returned in a diagonal sparse matrix.  If the factorization is:

        [L, U, P, Q, R] = umfpack (A) ;

    then the factorization is

        L*U = P * (R \ A) * Q

    This is safer than returning a matrix R such that L*U = P*R*A*Q, because
    it avoids the division by small entries.  If r(i) is subnormal, multiplying
    by 1/r(i) would result in an IEEE Infinity, but dividing by r(i) is safe.

    The factorization

        [L, U, P, Q] = umfpack (A) ;

    returns LU factors such that L*U = P*A*Q, with no scaling.

    See umfpack.m, umfpack_details.m, and umfpack.h for details.

    Note that this mexFunction accesses only the user-callable UMFPACK routines.
    Thus, is also provides another example of how user C code can access
    UMFPACK.

    Unlike MATLAB, x=b/A is solved by factorizing A, and then solving via the
    transposed L and U matrices.  The solution is still x = (A.'\b.').', except
    that A is factorized instead of A.'.

    v5.1: port to 64-bit MATLAB
*/

#include "umfpack.h"
#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <math.h>
#include <float.h>

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MATCH(s1,s2) (strcmp ((s1), (s2)) == 0)
#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif
#define Long SuiteSparse_long

/* ========================================================================== */
/* === error ================================================================ */
/* ========================================================================== */

/* Return an error message */

static void error
(
    char *s,
    Long A_is_complex,
    int nargout,
    mxArray *pargout [ ],
    double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO],
    Long status
)
{
    Long i ;
    double *Out_Info ;
    if (A_is_complex)
    {
	umfpack_zl_report_status (Control, status) ;
	umfpack_zl_report_info (Control, Info) ;
    }
    else
    {
	umfpack_dl_report_status (Control, status) ;
	umfpack_dl_report_info (Control, Info) ;
    }

    mexErrMsgTxt (s) ;
}


/* ========================================================================== */
/* === get_option =========================================================== */
/* ========================================================================== */

/* get a single string or numeric option from the MATLAB options struct */

static int get_option
(
    /* inputs: */
    const mxArray *mxopts,      /* the MATLAB struct */
    const char *field,          /* the field to get from the MATLAB struct */

    /* outputs: */
    double *x,                  /* double value of the field, if present */
    Long *x_present,            /* true if double x is present */
    char **s                    /* char value of the field, if present; */
                                /* must be mxFree'd by caller when done */
)
{
    Long f ;
    mxArray *p ;

    /* find the field number */
    if (mxopts == NULL || mxIsEmpty (mxopts) || !mxIsStruct (mxopts))
    {
        /* mxopts is not present, or [ ], or not a struct */
        f = -1 ;
    }
    else
    {
        /* f will be -1 if the field is not present */
        f = mxGetFieldNumber (mxopts, field) ;
    }

    /* get the field, or NULL if not present */
    if (f == -1)
    {
        p = NULL ;
    }
    else
    {
        p = mxGetFieldByNumber (mxopts, 0, f) ;
    }

    *x_present = FALSE ;
    if (s != NULL)
    {
        *s = NULL ;
    }

    if (p == NULL)
    {
        /* option not present */
        return (TRUE) ;
    }
    if (mxIsNumeric (p))
    {
        /* option is numeric */
        if (x == NULL)
        {
            mexPrintf ("opts.%s field must be a string\n", field) ;
            mexErrMsgIdAndTxt ("UMFPACK:invalidInput", "invalid option") ;
        }
        *x = mxGetScalar (p) ;
        *x_present = TRUE ;
    }
    else if (mxIsChar (p))
    {
        /* option is a MATLAB string; convert it to a C-style string */
        if (s == NULL)
        {
            mexPrintf ("opts.%s field must be a numeric value\n", field) ;
            mexErrMsgIdAndTxt ("UMFPACK:invalidInput", "invalid option") ;
        }
        *s = mxArrayToString (p) ;
    }
    return (TRUE) ;
}


/* ========================================================================== */
/* === get_all_options ====================================================== */
/* ========================================================================== */

/* get all the options from the MATLAB struct.

    opts.prl        >= 0, default 1 (errors only)
    opts.strategy   'auto', 'unsymmetric', 'symmetric', default auto
    opts.ordering   'amd'       AMD for A+A', COLAMD for A'A
                    'default'   use CHOLMOD (AMD then METIS; take best fount)
                    'metis'     use METIS
                    'none'      no fill-reducing ordering
                    'given'     use Qinit (this is default if Qinit present)
                    'best'      try AMD/COLAMD, METIS, and NESDIS; take best
    opts.tol        default 0.1
    opts.symtol     default 0.001
    opts.scale      row scaling: 'none', 'sum', 'max'
    opts.irstep     max # of steps of iterative refinement, default 2
    opts.singletons 'yes','no' default 'yes'
 */

int get_all_options
(
    const mxArray *mxopts,
    double *Control
)
{
    double x ;
    char *s ;
    Long x_present, i, info_details ;

    /* ---------------------------------------------------------------------- */
    /* prl: an integer, default 1 */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "prl", &x, &x_present, NULL) ;
    Control [UMFPACK_PRL] = x_present ? ((Long) x) : UMFPACK_DEFAULT_PRL ;
    if (mxIsNaN (Control [UMFPACK_PRL]))
    {
	Control [UMFPACK_PRL] = UMFPACK_DEFAULT_PRL ;
    }

    /* ---------------------------------------------------------------------- */
    /* strategy: a string */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "strategy", NULL, &x_present, &s) ;
    if (s != NULL)
    {
        if (MATCH (s, "auto"))
        {
            Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO ;
        }
        else if (MATCH (s, "unsymmetric"))
        {
            Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC ;
        }
        else if (MATCH (s, "symmetric"))
        {
            Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC ;
        }
        else if (MATCH (s, "default"))
        {
            Control [UMFPACK_STRATEGY] = UMFPACK_DEFAULT_STRATEGY ;
        }
        else
        {
            mexErrMsgIdAndTxt ("UMFPACK:invalidInput","invalid strategy") ;
        }
        mxFree (s) ;
    }

    /* ---------------------------------------------------------------------- */
    /* ordering: a string */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "ordering", NULL, &x_present, &s) ;
    if (s != NULL)
    {
        if (MATCH (s, "amd") || MATCH (s, "colamd") || MATCH (s,"bestamd"))
        {
            Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD ;
        }
#ifndef NCHOLMOD
        else if (MATCH (s, "fixed") || MATCH (s, "none") || MATCH (s,"natural"))
        {
            Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_NONE ;
        }
        else if (MATCH (s, "metis"))
        {
            Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS ;
        }
        else if (MATCH (s, "cholmod"))
        {
            Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD ;
        }
        else if (MATCH (s, "best"))
        {
            Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_BEST ;
        }
#endif
        else if (MATCH (s, "given"))
        {
            Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_GIVEN ;
        }
        else if (MATCH (s, "default"))
        {
            Control [UMFPACK_ORDERING] = UMFPACK_DEFAULT_ORDERING ;
        }
        else
        {
            mexErrMsgIdAndTxt ("UMFPACK:invalidInput","invalid ordering") ;
        }
        mxFree (s) ;
    }

    /* ---------------------------------------------------------------------- */
    /* scale: a string */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "scale", NULL, &x_present, &s) ;
    if (s != NULL)
    {
        if (MATCH (s, "none"))
        {
            Control [UMFPACK_SCALE] = UMFPACK_SCALE_NONE ;
        }
        else if (MATCH (s, "sum"))
        {
            Control [UMFPACK_SCALE] = UMFPACK_SCALE_SUM ;
        }
        else if (MATCH (s, "max"))
        {
            Control [UMFPACK_SCALE] = UMFPACK_SCALE_MAX ;
        }
        else if (MATCH (s, "default"))
        {
            Control [UMFPACK_SCALE] = UMFPACK_DEFAULT_SCALE ;
        }
        else
        {
            mexErrMsgIdAndTxt ("UMFPACK:invalidInput","invalid scale") ;
        }
        mxFree (s) ;
    }

    /* ---------------------------------------------------------------------- */
    /* tol: a double */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "tol", &x, &x_present, NULL) ;
    Control [UMFPACK_PIVOT_TOLERANCE]
        = x_present ? x : UMFPACK_DEFAULT_PIVOT_TOLERANCE ;

    /* ---------------------------------------------------------------------- */
    /* symtol: a double */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "symtol", &x, &x_present, NULL) ;
    Control [UMFPACK_SYM_PIVOT_TOLERANCE]
        = x_present ? x : UMFPACK_DEFAULT_SYM_PIVOT_TOLERANCE ;

    /* ---------------------------------------------------------------------- */
    /* irstep: an integer */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "irstep", &x, &x_present, NULL) ;
    Control [UMFPACK_IRSTEP] = x_present ? x : UMFPACK_DEFAULT_IRSTEP ;

    /* ---------------------------------------------------------------------- */
    /* singletons: a string */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "singletons", NULL, &x_present, &s) ;
    if (s != NULL)
    {
        if (MATCH (s, "enable"))
        {
            Control [UMFPACK_SINGLETONS] = TRUE ;
        }
        else if (MATCH (s, "disable"))
        {
            Control [UMFPACK_SINGLETONS] = FALSE ;
        }
        else if (MATCH (s, "default"))
        {
            Control [UMFPACK_SINGLETONS] = UMFPACK_DEFAULT_SINGLETONS ;
        }
        mxFree (s) ;
    }

    /* ---------------------------------------------------------------------- */
    /* details: an int */
    /* ---------------------------------------------------------------------- */

    get_option (mxopts, "details", &x, &x_present, NULL) ;
    info_details = x_present ? x : 0 ;
    return (info_details) ;
}


/* ========================================================================== */
/* === umfpack_mx_info_details ============================================== */
/* ========================================================================== */

/* Return detailed info struct; useful for UMFPACK development only  */

#define XFIELD(x) mxSetFieldByNumber (info, 0, k++, mxCreateDoubleScalar (x))
#define SFIELD(s) mxSetFieldByNumber (info, 0, k++, mxCreateString (s))
#define YESNO(x) ((x) ? "yes" : "no")

mxArray *umfpack_mx_info_details    /* return a struct with info statistics */
(
    double *Control,
    double *Info
)
{
    Long k = 0 ;
    mxArray *info ;
    Long sizeof_unit = (Long) Info [UMFPACK_SIZE_OF_UNIT] ;

    const char *info_struct [ ] =
    {
        "control_prl",
        "control_dense_row",
        "control_dense_col",
        "control_tol",
        "control_block_size",
        "control_strategy",
        "control_alloc_init",
        "control_irstep",
        "umfpack_compiled_with_BLAS",
        "control_ordering",
        "control_singletons",
        "control_fixQ",
        "control_amd_dense",
        "control_symtol",
        "control_scale",
        "control_front_alloc",
        "control_droptol",
        "control_aggressive",

        "status",

        "nrow",
        "ncol",
        "anz",

        "sizeof_unit",
        "sizeof_int",
        "sizeof_long",
        "sizeof_pointer",
        "sizeof_entry",
        "number_of_dense_rows",
        "number_of_empty_rows",
        "number_of_dense_cols",
        "number_of_empty_cols",

        "number_of_memory_defragmentations_during_symbolic_analysis",
        "memory_usage_for_symbolic_analysis_in_bytes",
        "size_of_symbolic_factorization_in_bytes",
        "symbolic_time",
        "symbolic_walltime",
        "strategy_used",
        "ordering_used",
        "Qfixed",
        "diagonol_pivots_preferred",

        "number_of_column_singletons",
        "number_of_row_singletons",

        /* only computed if symmetric strategy used */
        "symmetric_strategy_S_size",
        "symmetric_strategy_S_symmetric",
        "symmetric_strategy_pattern_symmetry",
        "symmetric_strategy_nnz_A_plus_AT",
        "symmetric_strategy_nnz_diag",
        "symmetric_strategy_lunz",
        "symmetric_strategy_flops",
        "symmetric_strategy_ndense",
        "symmetric_strategy_dmax",

        "estimated_size_of_numeric_factorization_in_bytes",
        "estimated_peak_memory_in_bytes",
        "estimated_number_of_floating_point_operations",
        "estimated_number_of_entries_in_L",
        "estimated_number_of_entries_in_U",
        "estimated_variable_init_in_bytes",
        "estimated_variable_peak_in_bytes",
        "estimated_variable_final_in_bytes",
        "estimated_number_of_entries_in_largest_frontal_matrix",
        "estimated_largest_frontal_matrix_row_dimension",
        "estimated_largest_frontal_matrix_col_dimension",

        "size_of_numeric_factorization_in_bytes",
        "total_memory_usage_in_bytes",
        "number_of_floating_point_operations",
        "number_of_entries_in_L",
        "number_of_entries_in_U",
        "variable_init_in_bytes",
        "variable_peak_in_bytes",
        "variable_final_in_bytes",
        "number_of_entries_in_largest_frontal_matrix",
        "largest_frontal_matrix_row_dimension",
        "largest_frontal_matrix_col_dimension",

        "number_of_memory_defragmentations_during_numeric_factorization",
        "number_of_memory_reallocations_during_numeric_factorization",
        "number_of_costly_memory_reallocations_during_numeric_factorization",
        "number_of_integers_in_compressed_pattern_of_L_and_U",
        "number_of_entries_in_LU_data_structure",
        "numeric_time",
        "nnz_diagonal_of_U",
        "rcond_estimate",
        "scaling_used",
        "min_abs_row_sum_of_A",
        "max_abs_row_sum_of_A",
        "min_abs_diagonal_of_U",
        "max_abs_diagonal_of_U",
        "alloc_init_used",
        "number_of_forced_updates",
        "numeric_walltime",
        "symmetric_strategy_number_off_diagonal_pivots",

        "number_of_entries_in_L_including_dropped_entries",
        "number_of_entries_in_U_including_dropped_entries",
        "number_of_small_entries_dropped_from_L_and_U",

        "number_of_iterative_refinement_steps_taken",
        "number_of_iterative_refinement_steps_attempted",
        "omega1",
        "omega2",
        "solve_flops",
        "solve_time",
        "solve_walltime"
    } ;

    info = mxCreateStructMatrix (1, 1, 100, info_struct) ;

    XFIELD (Control [UMFPACK_PRL]) ;
    XFIELD (Control [UMFPACK_DENSE_ROW]) ;
    XFIELD (Control [UMFPACK_DENSE_COL]) ;
    XFIELD (Control [UMFPACK_PIVOT_TOLERANCE]) ;
    XFIELD (Control [UMFPACK_BLOCK_SIZE]) ;

    switch ((int) Control [UMFPACK_STRATEGY])
    {
        case UMFPACK_STRATEGY_UNSYMMETRIC: SFIELD ("unsymmetric") ;  break ;
        case UMFPACK_STRATEGY_SYMMETRIC:   SFIELD ("symmetric") ;    break ;
        default:
        case UMFPACK_DEFAULT_STRATEGY:     SFIELD ("auto") ;         break ;
    }

    XFIELD (Control [UMFPACK_ALLOC_INIT]) ;
    XFIELD (Control [UMFPACK_IRSTEP]) ;
    SFIELD (YESNO (Control [UMFPACK_COMPILED_WITH_BLAS])) ;

    switch ((int) Control [UMFPACK_ORDERING])
    {
        case UMFPACK_ORDERING_NONE:     SFIELD ("none") ;    break ;
        case UMFPACK_ORDERING_AMD:      SFIELD ("amd") ;     break ;
        case UMFPACK_ORDERING_METIS:    SFIELD ("metis") ;   break ;
        default:
        case UMFPACK_ORDERING_CHOLMOD:  SFIELD ("cholmod") ; break ;
        case UMFPACK_ORDERING_BEST:     SFIELD ("best") ;    break ;
        case UMFPACK_ORDERING_GIVEN:    SFIELD ("given") ;   break ;
    }

    SFIELD (YESNO (Control [UMFPACK_SINGLETONS])) ;

    if (Control [UMFPACK_FIXQ] > 0)
    {
        SFIELD ("forced true") ;
    }
    else if (Control [UMFPACK_FIXQ] < 0)
    {
        SFIELD ("forced false") ;
    }
    else
    {
        SFIELD ("auto") ;
    }

    XFIELD (Control [UMFPACK_AMD_DENSE]) ;
    XFIELD (Control [UMFPACK_SYM_PIVOT_TOLERANCE]) ;

    switch ((int) Control [UMFPACK_SCALE])
    {
        case UMFPACK_SCALE_NONE: SFIELD ("none") ;  break ;
        case UMFPACK_SCALE_MAX:  SFIELD ("max") ;   break ;
        default:
        case UMFPACK_SCALE_SUM:  SFIELD ("sum") ;   break ;
    }

    XFIELD (Control [UMFPACK_FRONT_ALLOC_INIT]) ;
    XFIELD (Control [UMFPACK_DROPTOL]) ;
    SFIELD (YESNO (Control [UMFPACK_AGGRESSIVE])) ;

    switch ((int) Info [UMFPACK_STATUS])
    {
        case UMFPACK_OK:
            SFIELD ("ok") ; break ;
        case UMFPACK_WARNING_singular_matrix:
            SFIELD ("singular matrix") ; break ;
        case UMFPACK_WARNING_determinant_underflow:
            SFIELD ("determinant underflow") ; break ;
        case UMFPACK_WARNING_determinant_overflow:
            SFIELD ("determinant overflow") ; break ;
        case UMFPACK_ERROR_out_of_memory:
            SFIELD ("out of memory") ; break ;
        case UMFPACK_ERROR_invalid_Numeric_object:
            SFIELD ("invalid numeric LU object") ; break ;
        case UMFPACK_ERROR_invalid_Symbolic_object:
            SFIELD ("invalid symbolic LU object") ; break ;
        case UMFPACK_ERROR_argument_missing:
            SFIELD ("argument missing") ; break ;
        case UMFPACK_ERROR_n_nonpositive:
            SFIELD ("n < 0") ; break ;
        case UMFPACK_ERROR_invalid_matrix:
            SFIELD ("invalid matrix") ; break ;
        case UMFPACK_ERROR_different_pattern:
            SFIELD ("pattern changed") ; break ;
        case UMFPACK_ERROR_invalid_system:
            SFIELD ("invalid system") ; break ;
        case UMFPACK_ERROR_invalid_permutation:
            SFIELD ("invalid permutation") ; break ;
        case UMFPACK_ERROR_internal_error:
            SFIELD ("internal error; contact DrTimothyAldenDavis@gmail.com") ;
            break ;
        case UMFPACK_ERROR_file_IO:
            SFIELD ("file I/O error") ; break ;
        case UMFPACK_ERROR_ordering_failed:
            SFIELD ("ordering failed") ; break ;
        default:
            if (Info [UMFPACK_STATUS] < 0)
            {
                SFIELD ("unknown error") ;
            }
            else
            {
                SFIELD ("unknown warning") ;
            }
        break ;
    }

    XFIELD (Info [UMFPACK_NROW]) ;
    XFIELD (Info [UMFPACK_NCOL]) ;
    XFIELD (Info [UMFPACK_NZ]) ;

    XFIELD (Info [UMFPACK_SIZE_OF_UNIT]) ;
    XFIELD (Info [UMFPACK_SIZE_OF_INT]) ;
    XFIELD (Info [UMFPACK_SIZE_OF_LONG]) ;
    XFIELD (Info [UMFPACK_SIZE_OF_POINTER]) ;
    XFIELD (Info [UMFPACK_SIZE_OF_ENTRY]) ;
    XFIELD (Info [UMFPACK_NDENSE_ROW]) ;
    XFIELD (Info [UMFPACK_NEMPTY_ROW]) ;
    XFIELD (Info [UMFPACK_NDENSE_COL]) ;
    XFIELD (Info [UMFPACK_NEMPTY_COL]) ;

    XFIELD (Info [UMFPACK_SYMBOLIC_DEFRAG]) ;
    XFIELD (Info [UMFPACK_SYMBOLIC_PEAK_MEMORY] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_SYMBOLIC_SIZE] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_SYMBOLIC_TIME]) ;
    XFIELD (Info [UMFPACK_SYMBOLIC_WALLTIME]) ;

    switch ((int) Info [UMFPACK_STRATEGY_USED])
    {
        default:
        case UMFPACK_STRATEGY_UNSYMMETRIC: SFIELD ("unsymmetric") ;  break ;
        case UMFPACK_STRATEGY_SYMMETRIC:   SFIELD ("symmetric") ;    break ;
    }

    switch ((int) Info [UMFPACK_ORDERING_USED])
    {
        case UMFPACK_ORDERING_AMD:      SFIELD ("amd") ;     break ;
        case UMFPACK_ORDERING_METIS:    SFIELD ("metis") ;   break ;
        case UMFPACK_ORDERING_GIVEN:    SFIELD ("given") ;   break ;
        default:                        SFIELD ("none") ;    break ;
    }

    SFIELD (YESNO (Info [UMFPACK_QFIXED])) ;
    SFIELD (YESNO (Info [UMFPACK_DIAG_PREFERRED])) ;

    XFIELD (Info [UMFPACK_COL_SINGLETONS]) ;
    XFIELD (Info [UMFPACK_ROW_SINGLETONS]) ;

    /* only computed if symmetric ordering is used */
    XFIELD (Info [UMFPACK_N2]) ;
    SFIELD (YESNO (Info [UMFPACK_S_SYMMETRIC])) ;
    XFIELD (Info [UMFPACK_PATTERN_SYMMETRY]) ;
    XFIELD (Info [UMFPACK_NZ_A_PLUS_AT]) ;
    XFIELD (Info [UMFPACK_NZDIAG]) ;
    XFIELD (Info [UMFPACK_SYMMETRIC_LUNZ]) ;
    XFIELD (Info [UMFPACK_SYMMETRIC_FLOPS]) ;
    XFIELD (Info [UMFPACK_SYMMETRIC_NDENSE]) ;
    XFIELD (Info [UMFPACK_SYMMETRIC_DMAX]) ;

    XFIELD (Info [UMFPACK_NUMERIC_SIZE_ESTIMATE] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_PEAK_MEMORY_ESTIMATE] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_FLOPS_ESTIMATE]) ;
    XFIELD (Info [UMFPACK_LNZ_ESTIMATE]) ;
    XFIELD (Info [UMFPACK_UNZ_ESTIMATE]) ;
    XFIELD (Info [UMFPACK_VARIABLE_INIT_ESTIMATE] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_VARIABLE_PEAK_ESTIMATE] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_VARIABLE_FINAL_ESTIMATE] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_MAX_FRONT_SIZE_ESTIMATE]) ;
    XFIELD (Info [UMFPACK_MAX_FRONT_NROWS_ESTIMATE]) ;
    XFIELD (Info [UMFPACK_MAX_FRONT_NCOLS_ESTIMATE]) ;

    XFIELD (Info [UMFPACK_NUMERIC_SIZE] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_PEAK_MEMORY] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_FLOPS]) ;
    XFIELD (Info [UMFPACK_LNZ]) ;
    XFIELD (Info [UMFPACK_UNZ]) ;
    XFIELD (Info [UMFPACK_VARIABLE_INIT] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_VARIABLE_PEAK] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_VARIABLE_FINAL] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_MAX_FRONT_SIZE]) ;
    XFIELD (Info [UMFPACK_MAX_FRONT_NROWS]) ;
    XFIELD (Info [UMFPACK_MAX_FRONT_NCOLS]) ;

    XFIELD (Info [UMFPACK_NUMERIC_DEFRAG]) ;
    XFIELD (Info [UMFPACK_NUMERIC_REALLOC]) ;
    XFIELD (Info [UMFPACK_NUMERIC_COSTLY_REALLOC]) ;
    XFIELD (Info [UMFPACK_COMPRESSED_PATTERN]) ;
    XFIELD (Info [UMFPACK_LU_ENTRIES]) ;
    XFIELD (Info [UMFPACK_NUMERIC_TIME]) ;
    XFIELD (Info [UMFPACK_UDIAG_NZ]) ;
    XFIELD (Info [UMFPACK_RCOND]) ;

    switch ((int) Info [UMFPACK_WAS_SCALED])
    {
        case UMFPACK_SCALE_NONE: SFIELD ("none") ;  break ;
        case UMFPACK_SCALE_MAX:  SFIELD ("max") ;   break ;
        default:
        case UMFPACK_SCALE_SUM:  SFIELD ("sum") ;   break ;
    }

    XFIELD (Info [UMFPACK_RSMIN]) ;
    XFIELD (Info [UMFPACK_RSMAX]) ;
    XFIELD (Info [UMFPACK_UMIN]) ;
    XFIELD (Info [UMFPACK_UMAX]) ;
    XFIELD (Info [UMFPACK_ALLOC_INIT_USED]) ;
    XFIELD (Info [UMFPACK_FORCED_UPDATES]) ;
    XFIELD (Info [UMFPACK_NUMERIC_WALLTIME]) ;
    XFIELD (Info [UMFPACK_NOFF_DIAG]) ;

    XFIELD (Info [UMFPACK_ALL_LNZ]) ;
    XFIELD (Info [UMFPACK_ALL_UNZ]) ;
    XFIELD (Info [UMFPACK_NZDROPPED]) ;

    XFIELD (Info [UMFPACK_IR_TAKEN]) ;
    XFIELD (Info [UMFPACK_IR_ATTEMPTED]) ;
    XFIELD (Info [UMFPACK_OMEGA1]) ;
    XFIELD (Info [UMFPACK_OMEGA2]) ;
    XFIELD (Info [UMFPACK_SOLVE_FLOPS]) ;
    XFIELD (Info [UMFPACK_SOLVE_TIME]) ;
    XFIELD (Info [UMFPACK_SOLVE_WALLTIME]) ;

    return (info) ;
}


/* ========================================================================== */
/* === umfpack_mx_info_user ================================================= */
/* ========================================================================== */

/* Return user-friendly info struct */

mxArray *umfpack_mx_info_user    /* return a struct with info statistics */
(
    double *Control,
    double *Info,
    Long do_solve
)
{
    Long k = 0 ;
    mxArray *info ;
    Long sizeof_unit = (Long) Info [UMFPACK_SIZE_OF_UNIT] ;

    const char *info_struct [ ] =
    {
        "analysis_time",
        "strategy_used",
        "ordering_used",
        "memory_usage_in_bytes",
        "factorization_flop_count",
        "nnz_in_L_plus_U",
        "rcond_estimate",
        "factorization_time",
        /* if solve */
        "iterative_refinement_steps",
        "solve_flop_count",
        "solve_time"
    } ;

    info = mxCreateStructMatrix (1, 1, do_solve ? 11 : 8, info_struct) ;

    XFIELD (Info [UMFPACK_SYMBOLIC_WALLTIME]) ;

    switch ((int) Info [UMFPACK_STRATEGY_USED])
    {
        default:
        case UMFPACK_STRATEGY_UNSYMMETRIC: SFIELD ("unsymmetric") ;  break ;
        case UMFPACK_STRATEGY_SYMMETRIC:   SFIELD ("symmetric") ;    break ;
    }

    switch ((int) Info [UMFPACK_ORDERING_USED])
    {
        case UMFPACK_ORDERING_AMD:      SFIELD ("amd") ;     break ;
        case UMFPACK_ORDERING_METIS:    SFIELD ("metis") ;   break ;
        case UMFPACK_ORDERING_GIVEN:    SFIELD ("given") ;   break ;
        default:                        SFIELD ("none") ;    break ;
    }

    XFIELD (Info [UMFPACK_PEAK_MEMORY] * sizeof_unit) ;
    XFIELD (Info [UMFPACK_FLOPS]) ;
    XFIELD (Info [UMFPACK_LNZ] + Info [UMFPACK_UNZ] - Info [UMFPACK_UDIAG_NZ]) ;
    XFIELD (Info [UMFPACK_RCOND]) ;
    XFIELD (Info [UMFPACK_NUMERIC_WALLTIME]) ;

    if (do_solve)
    {
        XFIELD (Info [UMFPACK_IR_TAKEN]) ;
        XFIELD (Info [UMFPACK_SOLVE_FLOPS]) ;
        XFIELD (Info [UMFPACK_SOLVE_WALLTIME]) ;
    }

    return (info) ;
}


/* ========================================================================== */
/* === umfpack_mx_defaults ================================================== */
/* ========================================================================== */

/* Return a struct with default Control settings (except opts.details). */

mxArray *umfpack_mx_defaults ( void )
{
    mxArray *opts ;
    const char *opts_struct [ ] =
    {
        "prl",
        "strategy",
        "ordering",
        "tol",
        "symtol",
        "scale",
        "irstep",
        "singletons"
    } ;
    opts = mxCreateStructMatrix (1, 1, 8, opts_struct) ;
    mxSetFieldByNumber (opts, 0, 0,
        mxCreateDoubleScalar (UMFPACK_DEFAULT_PRL)) ;
    mxSetFieldByNumber (opts, 0, 1, mxCreateString ("auto")) ;
    mxSetFieldByNumber (opts, 0, 2, mxCreateString ("default")) ;
    mxSetFieldByNumber (opts, 0, 3,
        mxCreateDoubleScalar (UMFPACK_DEFAULT_PIVOT_TOLERANCE)) ;
    mxSetFieldByNumber (opts, 0, 4,
        mxCreateDoubleScalar (UMFPACK_DEFAULT_SYM_PIVOT_TOLERANCE)) ;
    mxSetFieldByNumber (opts, 0, 5, mxCreateString ("sum")) ;
    mxSetFieldByNumber (opts, 0, 6,
        mxCreateDoubleScalar (UMFPACK_DEFAULT_IRSTEP)) ;
    mxSetFieldByNumber (opts, 0, 7, mxCreateString ("enable")) ;
    return (opts) ;
}


/* ========================================================================== */
/* === UMFPACK ============================================================== */
/* ========================================================================== */

void mexFunction
(
    int nargout,		/* number of outputs */
    mxArray *pargout [ ],	/* output arguments */
    int nargin,			/* number of inputs */
    const mxArray *pargin [ ]	/* input arguments */
)
{

    /* ---------------------------------------------------------------------- */
    /* local variables */
    /* ---------------------------------------------------------------------- */

    double Info [UMFPACK_INFO], Control [UMFPACK_CONTROL], dx, dz, dexp ;
    double *Lx, *Lz, *Ux, *Uz, *Ax, *Az, *Bx, *Bz, *Xx, *Xz, *User_Control,
	*p, *q, *Out_Info, *p1, *p2, *p3, *p4, *Ltx, *Ltz, *Rs, *Px, *Qx ;
    void *Symbolic, *Numeric ;
    Long *Lp, *Li, *Up, *Ui, *Ap, *Ai, *P, *Q, do_solve, lnz, unz, nn, i,
	transpose, size, do_info, do_numeric, *Front_npivcol, op, k, *Rp, *Ri,
	*Front_parent, *Chain_start, *Chain_maxrows, *Chain_maxcols, nz, status,
	nfronts, nchains, *Ltp, *Ltj, *Qinit, print_level, status2, no_scale,
	*Front_1strow, *Front_leftmostdesc, n_row, n_col, n_inner, sys,
	ignore1, ignore2, ignore3, A_is_complex, B_is_complex, X_is_complex,
	*Pp, *Pi, *Qp, *Qi, do_recip, do_det ;
    mxArray *Amatrix, *Bmatrix, *User_Control_struct, *User_Qinit ;
    char *operator, *operation ;
    mxComplexity Atype, Xtype ;
    char warning [200] ;
    int info_details ;

    /* ---------------------------------------------------------------------- */
    /* get inputs A, b, and the operation to perform */
    /* ---------------------------------------------------------------------- */

    if (nargin > 1 && mxIsStruct (pargin [nargin-1]))
    {
        User_Control_struct = (mxArray *) (pargin [nargin-1]) ;
    }
    else
    {
        User_Control_struct = (mxArray *) NULL ;
    }

    User_Qinit = (mxArray *) NULL ;

    do_info = 0 ;
    do_solve = FALSE ;
    do_numeric = TRUE ;
    transpose = FALSE ;
    no_scale = FALSE ;
    do_det = FALSE ;

    /* find the operator */
    op = 0 ;
    for (i = 0 ; i < nargin ; i++)
    {
	if (mxIsChar (pargin [i]))
	{
	    op = i ;
	    break ;
	}
    }

    if (op > 0)
    {
	operator = mxArrayToString (pargin [op]) ;

	if (MATCH (operator, "\\"))
	{

	    /* -------------------------------------------------------------- */
	    /* matrix left divide, x = A\b */
	    /* -------------------------------------------------------------- */

	    /*
		[x, Info] = umfpack (A, '\', b) ;
		[x, Info] = umfpack (A, '\', b, Control) ;
		[x, Info] = umfpack (A, Qinit, '\', b) ;
		[x, Info] = umfpack (A, Qinit, '\', b, Control) ;
	    */

	    operation = "x = A\\b" ;
	    do_solve = TRUE ;
	    Amatrix = (mxArray *) pargin [0] ;
	    Bmatrix = (mxArray *) pargin [op+1] ;

	    if (nargout == 2)
	    {
		do_info = 1 ;
	    }
	    if (op == 2)
	    {
		User_Qinit = (mxArray *) pargin [1] ;
	    }
	    if (nargin < 3 || nargin > 5 || nargout > 2)
	    {
		mexErrMsgTxt ("wrong number of arguments") ;
	    }

	}
	else if (MATCH (operator, "/"))
	{

	    /* -------------------------------------------------------------- */
	    /* matrix right divide, x = b/A */
	    /* -------------------------------------------------------------- */

	    /*
		[x, Info] = umfpack (b, '/', A) ;
		[x, Info] = umfpack (b, '/', A, Control) ;
		[x, Info] = umfpack (b, '/', A, Qinit) ;
		[x, Info] = umfpack (b, '/', A, Qinit, Control) ;
	    */

	    operation = "x = b/A" ;
	    do_solve = TRUE ;
	    transpose = TRUE ;
	    Amatrix = (mxArray *) pargin [2] ;
	    Bmatrix = (mxArray *) pargin [0] ;

	    if (nargout == 2)
	    {
		do_info = 1 ;
	    }
	    if (nargin >= 4 && mxIsDouble (pargin [3]))
	    {
		User_Qinit = (mxArray *) pargin [3] ;
	    }
	    if (nargin < 3 || nargin > 5 || nargout > 2)
	    {
		mexErrMsgTxt ("wrong number of arguments") ;
	    }

	}
	else if (MATCH (operator, "symbolic"))
	{

	    /* -------------------------------------------------------------- */
	    /* symbolic factorization only */
	    /* -------------------------------------------------------------- */

	    /*
	    [P Q Fr Ch Info] = umfpack (A, 'symbolic') ;
	    [P Q Fr Ch Info] = umfpack (A, 'symbolic', Control) ;
	    [P Q Fr Ch Info] = umfpack (A, Qinit, 'symbolic') ;
	    [P Q Fr Ch Info] = umfpack (A, Qinit, 'symbolic', Control) ;
	    */

	    operation = "symbolic factorization" ;
	    do_numeric = FALSE ;
	    Amatrix = (mxArray *) pargin [0] ;

	    if (nargout == 5)
	    {
		do_info = 4 ;
	    }
	    if (op == 2)
	    {
		User_Qinit = (mxArray *) pargin [1] ;
	    }
	    if (nargin < 2 || nargin > 4 || nargout > 5 || nargout < 4)
	    {
		mexErrMsgTxt ("wrong number of arguments") ;
	    }

	}
	else if (MATCH (operator, "det"))
	{

	    /* -------------------------------------------------------------- */
	    /* compute the determinant */
	    /* -------------------------------------------------------------- */

	    /*
	     * [det] = umfpack (A, 'det') ;
	     * [dmantissa dexp] = umfpack (A, 'det') ;
	     */

	    operation = "determinant" ;
	    do_det = TRUE ;
	    Amatrix = (mxArray *) pargin [0] ;
	    if (nargin > 2 || nargout > 2)
	    {
		mexErrMsgTxt ("wrong number of arguments") ;
	    }

	}
	else
	{
	    mexErrMsgTxt ("operator must be '/', '\\', or 'symbolic'") ;
	}
	mxFree (operator) ;

    }
    else if (nargin > 0)
    {

	/* ------------------------------------------------------------------ */
	/* LU factorization */
	/* ------------------------------------------------------------------ */

	/*
	    with scaling:
	    [L, U, P, Q, R, Info] = umfpack (A) ;
	    [L, U, P, Q, R, Info] = umfpack (A, Qinit) ;

	    scaling determined by Control settings:
	    [L, U, P, Q, R, Info] = umfpack (A, Control) ;
	    [L, U, P, Q, R, Info] = umfpack (A, Qinit, Control) ;

	    with no scaling:
	    [L, U, P, Q] = umfpack (A) ;
	    [L, U, P, Q] = umfpack (A, Control) ;
	    [L, U, P, Q] = umfpack (A, Qinit) ;
	    [L, U, P, Q] = umfpack (A, Qinit, Control) ;
	*/

	operation = "numeric factorization" ;
	Amatrix = (mxArray *) pargin [0] ;

	no_scale = (nargout <= 4) ;

	if (nargout == 6)
	{
	    do_info = 5 ;
	}
        if (nargin >= 2 && mxIsDouble (pargin [1]))
        {
            User_Qinit = (mxArray *) pargin [1] ;
        }
	if (nargin > 3 || nargout > 6 || nargout < 4)
	{
	    mexErrMsgTxt ("wrong number of arguments") ;
	}

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* return default control settings */
	/* ------------------------------------------------------------------ */

	/*
	    Control = umfpack ;
	    umfpack ;
	*/

	if (nargout > 1)
	{
	    mexErrMsgTxt ("wrong number of arguments") ;
	}

        /* return default opts struct */
	pargout [0] = umfpack_mx_defaults ( ) ;
	return ;
    }

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (mxGetNumberOfDimensions (Amatrix) != 2)
    {
	mexErrMsgTxt ("input matrix A must be 2-dimensional") ;
    }
    n_row = mxGetM (Amatrix) ;
    n_col = mxGetN (Amatrix) ;
    nn = MAX (n_row, n_col) ;
    n_inner = MIN (n_row, n_col) ;
    if (do_solve && n_row != n_col)
    {
	mexErrMsgTxt ("input matrix A must square for '\\' or '/'") ;
    }
    if (!mxIsSparse (Amatrix))
    {
	mexErrMsgTxt ("input matrix A must be sparse") ;
    }
    if (n_row == 0 || n_col == 0)
    {
	mexErrMsgTxt ("input matrix A cannot have zero rows or zero columns") ;
    }

    /* The real/complex status of A determines which version to use, */
    /* (umfpack_dl_* or umfpack_zl_*). */
    A_is_complex = mxIsComplex (Amatrix) ;
    Atype = A_is_complex ? mxCOMPLEX : mxREAL ;
    Ap = (Long *) mxGetJc (Amatrix) ;
    Ai = (Long *) mxGetIr (Amatrix) ;
    Ax = mxGetPr (Amatrix) ;
    Az = mxGetPi (Amatrix) ;

    if (do_solve)
    {

	if (n_row != n_col)
	{
	    mexErrMsgTxt ("A must be square for \\ or /") ;
	}
	if (transpose)
	{
	    if (mxGetM (Bmatrix) != 1 || mxGetN (Bmatrix) != nn)
	    {
		mexErrMsgTxt ("b has the wrong dimensions") ;
	    }
	}
	else
	{
	    if (mxGetM (Bmatrix) != nn || mxGetN (Bmatrix) != 1)
	    {
		mexErrMsgTxt ("b has the wrong dimensions") ;
	    }
	}
	if (mxGetNumberOfDimensions (Bmatrix) != 2)
	{
	    mexErrMsgTxt ("input matrix b must be 2-dimensional") ;
	}
	if (mxIsSparse (Bmatrix))
	{
	    mexErrMsgTxt ("input matrix b cannot be sparse") ;
	}
	if (mxGetClassID (Bmatrix) != mxDOUBLE_CLASS)
	{
	    mexErrMsgTxt ("input matrix b must double precision matrix") ;
	}

	B_is_complex = mxIsComplex (Bmatrix) ;
	Bx = mxGetPr (Bmatrix) ;
	Bz = mxGetPi (Bmatrix) ;

	X_is_complex = A_is_complex || B_is_complex ;
	Xtype = X_is_complex ? mxCOMPLEX : mxREAL ;
    }

    /* ---------------------------------------------------------------------- */
    /* set the Control parameters */
    /* ---------------------------------------------------------------------- */

    if (A_is_complex)
    {
	umfpack_zl_defaults (Control) ;
    }
    else
    {
	umfpack_dl_defaults (Control) ;
    }

    info_details = 0 ;
    if (User_Control_struct != NULL)
    {
        info_details = get_all_options (User_Control_struct, Control) ;
    }

    if (no_scale)
    {
	/* turn off scaling for [L, U, P, Q] = umfpack (A) ;
	 * ignoring the input value of Control (24) for the usage
	 * [L, U, P, Q] = umfpack (A, Control) ; */
	Control [UMFPACK_SCALE] = UMFPACK_SCALE_NONE ;
    }

    print_level = (Long) Control [UMFPACK_PRL] ;

    /* ---------------------------------------------------------------------- */
    /* get Qinit, if present */
    /* ---------------------------------------------------------------------- */

    if (User_Qinit)
    {
	if (mxGetM (User_Qinit) != 1 || mxGetN (User_Qinit) != n_col)
	{
	    mexErrMsgTxt ("Qinit must be 1-by-n_col") ;
	}
	if (mxGetNumberOfDimensions (User_Qinit) != 2)
	{
	    mexErrMsgTxt ("input Qinit must be 2-dimensional") ;
	}
	if (mxIsComplex (User_Qinit))
	{
	    mexErrMsgTxt ("input Qinit must not be complex") ;
	}
	if (mxGetClassID (User_Qinit) != mxDOUBLE_CLASS)
	{
	    mexErrMsgTxt ("input Qinit must be a double matrix") ;
	}
	if (mxIsSparse (User_Qinit))
	{
	    mexErrMsgTxt ("input Qinit must be dense") ;
	}
	Qinit = (Long *) mxMalloc (n_col * sizeof (Long)) ;
	p = mxGetPr (User_Qinit) ;
	for (k = 0 ; k < n_col ; k++)
	{
	    /* convert from 1-based to 0-based */
	    Qinit [k] = ((Long) (p [k])) - 1 ;
	}
        Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_GIVEN ;
    }
    else
    {
	/* umfpack_*_qsymbolic will call colamd to get Qinit. This is the */
	/* same as calling umfpack_*_symbolic with Qinit set to NULL*/
	Qinit = (Long *) NULL ;
    }

    /* ---------------------------------------------------------------------- */
    /* report the inputs A and Qinit */
    /* ---------------------------------------------------------------------- */

    if (print_level >= 2)
    {
	/* print the operation */
	mexPrintf ("\numfpack: %s\n", operation) ;
    }

    if (A_is_complex)
    {
	umfpack_zl_report_control (Control) ;
	if (print_level >= 3) mexPrintf ("\nA: ") ;
	(void) umfpack_zl_report_matrix (n_row, n_col, Ap, Ai, Ax, Az,
	    1, Control) ;
	if (Qinit)
	{
	    if (print_level >= 3) mexPrintf ("\nQinit: ") ;
	    (void) umfpack_zl_report_perm (n_col, Qinit, Control) ;
	}
    }
    else
    {
	umfpack_dl_report_control (Control) ;
	if (print_level >= 3) mexPrintf ("\nA: ") ;
	(void) umfpack_dl_report_matrix (n_row, n_col, Ap, Ai, Ax,
	    1, Control) ;
	if (Qinit)
	{
	    if (print_level >= 3) mexPrintf ("\nQinit: ") ;
	    (void) umfpack_dl_report_perm (n_col, Qinit, Control) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* perform the symbolic factorization */
    /* ---------------------------------------------------------------------- */

    if (A_is_complex)
    {
	status = umfpack_zl_qsymbolic (n_row, n_col, Ap, Ai, Ax, Az,
	    Qinit, &Symbolic, Control, Info) ;
    }
    else
    {
	status = umfpack_dl_qsymbolic (n_row, n_col, Ap, Ai, Ax,
	    Qinit, &Symbolic, Control, Info) ;
    }

    if (Qinit)
    {
	mxFree (Qinit) ;
    }

    if (status < 0)
    {
	error ("symbolic factorization failed", A_is_complex, nargout, pargout,
	    Control, Info, status) ;
	return ;
    }

    /* ---------------------------------------------------------------------- */
    /* report the Symbolic object */
    /* ---------------------------------------------------------------------- */

    if (A_is_complex)
    {
	(void) umfpack_zl_report_symbolic (Symbolic, Control) ;
    }
    else
    {
	(void) umfpack_dl_report_symbolic (Symbolic, Control) ;
    }

    /* ---------------------------------------------------------------------- */
    /* perform numeric factorization, or just return symbolic factorization */
    /* ---------------------------------------------------------------------- */

    if (do_numeric)
    {

	/* ------------------------------------------------------------------ */
	/* perform the numeric factorization */
	/* ------------------------------------------------------------------ */

	if (A_is_complex)
	{
	    status = umfpack_zl_numeric (Ap, Ai, Ax, Az, Symbolic, &Numeric,
		Control, Info) ;
	}
	else
	{
	    status = umfpack_dl_numeric (Ap, Ai, Ax, Symbolic, &Numeric,
		Control, Info) ;
	}

	/* ------------------------------------------------------------------ */
	/* free the symbolic factorization */
	/* ------------------------------------------------------------------ */

	if (A_is_complex)
	{
	    umfpack_zl_free_symbolic (&Symbolic) ;
	}
	else
	{
	    umfpack_dl_free_symbolic (&Symbolic) ;
	}

	/* ------------------------------------------------------------------ */
	/* report the Numeric object */
	/* ------------------------------------------------------------------ */

	if (status < 0)
	{
	    error ("numeric factorization failed", A_is_complex, nargout,
		pargout, Control, Info, status);
	    return ;
	}

	if (A_is_complex)
	{
	    (void) umfpack_zl_report_numeric (Numeric, Control) ;
	}
	else
	{
	    (void) umfpack_dl_report_numeric (Numeric, Control) ;
	}

	/* ------------------------------------------------------------------ */
	/* return the solution, determinant, or the factorization */
	/* ------------------------------------------------------------------ */

	if (do_solve)
	{
	    /* -------------------------------------------------------------- */
	    /* solve Ax=b or A'x'=b', and return just the solution x */
	    /* -------------------------------------------------------------- */

	    if (transpose)
	    {
		/* If A is real, A'x=b is the same as A.'x=b. */
		/* x and b are vectors, so x and b are the same as x' and b'. */
		/* If A is complex, then A.'x.'=b.' gives the same solution x */
		/* as the complex conjugate transpose.  If we used the A'x=b */
		/* option in umfpack_*_solve, we would have to form b' on */
		/* input and x' on output (negating the imaginary part). */
		/* We can save this work by just using the A.'x=b option in */
		/* umfpack_*_solve.  Then, forming x.' and b.' is implicit, */
		/* since x and b are just vectors anyway. */
		/* In both cases, the system to solve is A.'x=b */
		pargout [0] = mxCreateDoubleMatrix (1, nn, Xtype) ;
		sys = UMFPACK_Aat ;
	    }
	    else
	    {
		pargout [0] = mxCreateDoubleMatrix (nn, 1, Xtype) ;
		sys = UMFPACK_A ;
	    }

	    /* -------------------------------------------------------------- */
	    /* print the right-hand-side, B */
	    /* -------------------------------------------------------------- */

	    if (print_level >= 3) mexPrintf ("\nright-hand side, b: ") ;
	    if (B_is_complex)
	    {
		(void) umfpack_zl_report_vector (nn, Bx, Bz, Control) ;
	    }
	    else
	    {
		(void) umfpack_dl_report_vector (nn, Bx, Control) ;
	    }

	    /* -------------------------------------------------------------- */
	    /* solve the system */
	    /* -------------------------------------------------------------- */

	    Xx = mxGetPr (pargout [0]) ;
	    Xz = mxGetPi (pargout [0]) ;
	    status2 = UMFPACK_OK ;

	    if (A_is_complex)
	    {
		if (!B_is_complex)
		{
		    /* umfpack_zl_solve expects a complex B */
		    Bz = (double *) mxCalloc (nn, sizeof (double)) ;
		}
		status = umfpack_zl_solve (sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz,
		    Numeric, Control, Info) ;
		if (!B_is_complex)
		{
		    mxFree (Bz) ;
		}
	    }
	    else
	    {
		if (B_is_complex)
		{
		    /* Ax=b when b is complex and A is sparse can be split */
		    /* into two systems, A*xr=br and A*xi=bi, where r denotes */
		    /* the real part and i the imaginary part of x and b. */
		    status2 = umfpack_dl_solve (sys, Ap, Ai, Ax, Xz, Bz,
		    Numeric, Control, Info) ;
		}
		status = umfpack_dl_solve (sys, Ap, Ai, Ax, Xx, Bx,
		    Numeric, Control, Info) ;
	    }

	    /* -------------------------------------------------------------- */
	    /* free the Numeric object */
	    /* -------------------------------------------------------------- */

	    if (A_is_complex)
	    {
		umfpack_zl_free_numeric (&Numeric) ;
	    }
	    else
	    {
		umfpack_dl_free_numeric (&Numeric) ;
	    }

	    /* -------------------------------------------------------------- */
	    /* check error status */
	    /* -------------------------------------------------------------- */

	    if (status < 0 || status2 < 0)
	    {
		mxDestroyArray (pargout [0]) ;
		error ("solve failed", A_is_complex, nargout, pargout, Control,
			Info, status) ;
		return ;
	    }

	    /* -------------------------------------------------------------- */
	    /* print the solution, X */
	    /* -------------------------------------------------------------- */

	    if (print_level >= 3) mexPrintf ("\nsolution, x: ") ;
	    if (X_is_complex)
	    {
		(void) umfpack_zl_report_vector (nn, Xx, Xz, Control) ;
	    }
	    else
	    {
		(void) umfpack_dl_report_vector (nn, Xx, Control) ;
	    }

	    /* -------------------------------------------------------------- */
	    /* warn about singular or near-singular matrices */
	    /* -------------------------------------------------------------- */

	    /* no warning is given if Control (1) is zero */

	    if (print_level >= 1)
	    {
		if (status == UMFPACK_WARNING_singular_matrix)
		{
		    mexWarnMsgTxt (
                        "matrix is singular\n"
			"Try increasing opts.tol and opts.symtol.\n"
                        "Suppress this warning with opts.prl=0\n") ;
		}
		else if (Info [UMFPACK_RCOND] < DBL_EPSILON)
		{
		    sprintf (warning, "matrix is nearly singular, rcond = %g\n"
			"Try increasing opts.tol and opts.symtol.\n"
                        "Suppress this warning with opts.prl=0\n",
			Info [UMFPACK_RCOND]) ;
		    mexWarnMsgTxt (warning) ;
		}
	    }

	}
	else if (do_det)
	{

	    /* -------------------------------------------------------------- */
	    /* get the determinant */
	    /* -------------------------------------------------------------- */

	    if (nargout == 2)
	    {
		/* [det dexp] = umfpack (A, 'det') ;
		 * return determinant in the form det * 10^dexp */
		p = &dexp ;
	    }
	    else
	    {
		/* [det] = umfpack (A, 'det') ;
		 * return determinant as a single scalar (overflow or
		 * underflow is much more likely) */
		p = (double *) NULL ;
	    }
	    if (A_is_complex)
	    {
		status = umfpack_zl_get_determinant (&dx, &dz, p,
			Numeric, Info) ;
		umfpack_zl_free_numeric (&Numeric) ;
	    }
	    else
	    {
		status = umfpack_dl_get_determinant (&dx, p,
			Numeric, Info) ;
		umfpack_dl_free_numeric (&Numeric) ;
		dz = 0 ;
	    }
	    if (status < 0)
	    {
		error ("extracting LU factors failed", A_is_complex, nargout,
		    pargout, Control, Info, status) ;
	    }
	    if (A_is_complex)
	    {
		pargout [0] = mxCreateDoubleMatrix (1, 1, mxCOMPLEX) ;
		p = mxGetPr (pargout [0]) ;
		*p = dx ;
		p = mxGetPi (pargout [0]) ;
		*p = dz ;
	    }
	    else
	    {
		pargout [0] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
		p = mxGetPr (pargout [0]) ;
		*p = dx ;
	    }
	    if (nargout == 2)
	    {
		pargout [1] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
		p = mxGetPr (pargout [1]) ;
		*p = dexp ;
	    }

	}
	else
	{

	    /* -------------------------------------------------------------- */
	    /* get L, U, P, Q, and r */
	    /* -------------------------------------------------------------- */

	    if (A_is_complex)
	    {
	        status = umfpack_zl_get_lunz (&lnz, &unz, &ignore1, &ignore2,
		    &ignore3, Numeric) ;
	    }
	    else
	    {
	        status = umfpack_dl_get_lunz (&lnz, &unz, &ignore1, &ignore2,
		    &ignore3, Numeric) ;
	    }

	    if (status < 0)
	    {
		if (A_is_complex)
		{
		    umfpack_zl_free_numeric (&Numeric) ;
		}
		else
		{
		    umfpack_dl_free_numeric (&Numeric) ;
		}
		error ("extracting LU factors failed", A_is_complex, nargout,
		    pargout, Control, Info, status) ;
		return ;
	    }

	    /* avoid malloc of zero-sized arrays */
	    lnz = MAX (lnz, 1) ;
	    unz = MAX (unz, 1) ;

	    /* get temporary space, for the *** ROW *** form of L */
	    Ltp = (Long *) mxMalloc ((n_row+1) * sizeof (Long)) ;
	    Ltj = (Long *) mxMalloc (lnz * sizeof (Long)) ;
	    Ltx = (double *) mxMalloc (lnz * sizeof (double)) ;
	    if (A_is_complex)
	    {
	        Ltz = (double *) mxMalloc (lnz * sizeof (double)) ;
	    }
	    else
	    {
	        Ltz = (double *) NULL ;
	    }

	    /* create permanent copy of the output matrix U */
	    pargout [1] = mxCreateSparse (n_inner, n_col, unz, Atype) ;
	    Up = (Long *) mxGetJc (pargout [1]) ;
	    Ui = (Long *) mxGetIr (pargout [1]) ;
	    Ux = mxGetPr (pargout [1]) ;
	    Uz = mxGetPi (pargout [1]) ;

	    /* temporary space for the integer permutation vectors */
	    P = (Long *) mxMalloc (n_row * sizeof (Long)) ;
	    Q = (Long *) mxMalloc (n_col * sizeof (Long)) ;

	    /* get scale factors, if requested */
	    status2 = UMFPACK_OK ;
	    if (!no_scale)
	    {
		/* create a diagonal sparse matrix for the scale factors */
		pargout [4] = mxCreateSparse (n_row, n_row, n_row, mxREAL) ;
		Rp = (Long *) mxGetJc (pargout [4]) ;
		Ri = (Long *) mxGetIr (pargout [4]) ;
		for (i = 0 ; i < n_row ; i++)
		{
		    Rp [i] = i ;
		    Ri [i] = i ;
		}
		Rp [n_row] = n_row ;
		Rs = mxGetPr (pargout [4]) ;
	    }
	    else
	    {
		Rs = (double *) NULL ;
	    }

	    /* get Lt, U, P, Q, and Rs from the numeric object */
	    if (A_is_complex)
	    {
		status = umfpack_zl_get_numeric (Ltp, Ltj, Ltx, Ltz, Up, Ui, Ux,
		    Uz, P, Q, (double *) NULL, (double *) NULL,
		    &do_recip, Rs, Numeric) ;
		umfpack_zl_free_numeric (&Numeric) ;
	    }
	    else
	    {
		status = umfpack_dl_get_numeric (Ltp, Ltj, Ltx, Up, Ui,
		    Ux, P, Q, (double *) NULL,
		    &do_recip, Rs, Numeric) ;
		umfpack_dl_free_numeric (&Numeric) ;
	    }

	    /* for the mexFunction, -DNRECIPROCAL must be set,
	     * so do_recip must be FALSE */

	    if (status < 0 || status2 < 0 || do_recip)
	    {
		mxFree (Ltp) ;
		mxFree (Ltj) ;
		mxFree (Ltx) ;
		if (Ltz) mxFree (Ltz) ;
		mxFree (P) ;
		mxFree (Q) ;
		mxDestroyArray (pargout [1]) ;
		error ("extracting LU factors failed", A_is_complex, nargout,
		    pargout, Control, Info, status) ;
		return ;
	    }

	    /* create sparse permutation matrix for P */
	    pargout [2] = mxCreateSparse (n_row, n_row, n_row, mxREAL) ;
	    Pp = (Long *) mxGetJc (pargout [2]) ;
	    Pi = (Long *) mxGetIr (pargout [2]) ;
	    Px = mxGetPr (pargout [2]) ;
	    for (k = 0 ; k < n_row ; k++)
	    {
		Pp [k] = k ;
		Px [k] = 1 ;
		Pi [P [k]] = k ;
	    }
	    Pp [n_row] = n_row ;

	    /* create sparse permutation matrix for Q */
	    pargout [3] = mxCreateSparse (n_col, n_col, n_col, mxREAL) ;
	    Qp = (Long *) mxGetJc (pargout [3]) ;
	    Qi = (Long *) mxGetIr (pargout [3]) ;
	    Qx = mxGetPr (pargout [3]) ;
	    for (k = 0 ; k < n_col ; k++)
	    {
		Qp [k] = k ;
		Qx [k] = 1 ;
		Qi [k] = Q [k] ;
	    }
	    Qp [n_col] = n_col ;

	    /* permanent copy of L */
	    pargout [0] = mxCreateSparse (n_row, n_inner, lnz, Atype) ;
	    Lp = (Long *) mxGetJc (pargout [0]) ;
	    Li = (Long *) mxGetIr (pargout [0]) ;
	    Lx = mxGetPr (pargout [0]) ;
	    Lz = mxGetPi (pargout [0]) ;

	    /* convert L from row form to column form */
	    if (A_is_complex)
	    {
		/* non-conjugate array transpose */
	        status = umfpack_zl_transpose (n_inner, n_row, Ltp, Ltj, Ltx,
		    Ltz, (Long *) NULL, (Long *) NULL, Lp, Li, Lx, Lz,
		    FALSE) ;
	    }
	    else
	    {
	        status = umfpack_dl_transpose (n_inner, n_row, Ltp, Ltj, Ltx,
		    (Long *) NULL, (Long *) NULL, Lp, Li, Lx) ;
	    }

	    mxFree (Ltp) ;
	    mxFree (Ltj) ;
	    mxFree (Ltx) ;
	    if (Ltz) mxFree (Ltz) ;

	    if (status < 0)
	    {
		mxFree (P) ;
		mxFree (Q) ;
		mxDestroyArray (pargout [0]) ;
		mxDestroyArray (pargout [1]) ;
		mxDestroyArray (pargout [2]) ;
		mxDestroyArray (pargout [3]) ;
		error ("constructing L failed", A_is_complex, nargout, pargout,
		    Control, Info, status) ;
		return ;
	    }

	    /* -------------------------------------------------------------- */
	    /* print L, U, P, and Q */
	    /* -------------------------------------------------------------- */

	    if (A_is_complex)
	    {
		if (print_level >= 3) mexPrintf ("\nL: ") ;
	        (void) umfpack_zl_report_matrix (n_row, n_inner, Lp, Li,
		    Lx, Lz, 1, Control) ;
		if (print_level >= 3) mexPrintf ("\nU: ") ;
	        (void) umfpack_zl_report_matrix (n_inner, n_col,  Up, Ui,
		    Ux, Uz, 1, Control) ;
		if (print_level >= 3) mexPrintf ("\nP: ") ;
	        (void) umfpack_zl_report_perm (n_row, P, Control) ;
		if (print_level >= 3) mexPrintf ("\nQ: ") ;
	        (void) umfpack_zl_report_perm (n_col, Q, Control) ;
	    }
	    else
	    {
		if (print_level >= 3) mexPrintf ("\nL: ") ;
	        (void) umfpack_dl_report_matrix (n_row, n_inner, Lp, Li,
		    Lx, 1, Control) ;
		if (print_level >= 3) mexPrintf ("\nU: ") ;
	        (void) umfpack_dl_report_matrix (n_inner, n_col,  Up, Ui,
		    Ux, 1, Control) ;
		if (print_level >= 3) mexPrintf ("\nP: ") ;
	        (void) umfpack_dl_report_perm (n_row, P, Control) ;
		if (print_level >= 3) mexPrintf ("\nQ: ") ;
	        (void) umfpack_dl_report_perm (n_col, Q, Control) ;
	    }

	    mxFree (P) ;
	    mxFree (Q) ;

	}

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* return the symbolic factorization */
	/* ------------------------------------------------------------------ */

	Q = (Long *) mxMalloc (n_col * sizeof (Long)) ;
	P = (Long *) mxMalloc (n_row * sizeof (Long)) ;
	Front_npivcol = (Long *) mxMalloc ((nn+1) * sizeof (Long)) ;
	Front_parent = (Long *) mxMalloc ((nn+1) * sizeof (Long)) ;
	Front_1strow = (Long *) mxMalloc ((nn+1) * sizeof (Long)) ;
	Front_leftmostdesc = (Long *) mxMalloc ((nn+1) * sizeof (Long)) ;
	Chain_start = (Long *) mxMalloc ((nn+1) * sizeof (Long)) ;
	Chain_maxrows = (Long *) mxMalloc ((nn+1) * sizeof (Long)) ;
	Chain_maxcols = (Long *) mxMalloc ((nn+1) * sizeof (Long)) ;

	if (A_is_complex)
	{
	    status = umfpack_zl_get_symbolic (&ignore1, &ignore2, &ignore3,
	        &nz, &nfronts, &nchains, P, Q, Front_npivcol,
	        Front_parent, Front_1strow, Front_leftmostdesc,
	        Chain_start, Chain_maxrows, Chain_maxcols, Symbolic) ;
	    umfpack_zl_free_symbolic (&Symbolic) ;
	}
	else
	{
	    status = umfpack_dl_get_symbolic (&ignore1, &ignore2, &ignore3,
	        &nz, &nfronts, &nchains, P, Q, Front_npivcol,
	        Front_parent, Front_1strow, Front_leftmostdesc,
	        Chain_start, Chain_maxrows, Chain_maxcols, Symbolic) ;
	    umfpack_dl_free_symbolic (&Symbolic) ;
	}

	if (status < 0)
	{
	    mxFree (P) ;
	    mxFree (Q) ;
	    mxFree (Front_npivcol) ;
	    mxFree (Front_parent) ;
	    mxFree (Front_1strow) ;
	    mxFree (Front_leftmostdesc) ;
	    mxFree (Chain_start) ;
	    mxFree (Chain_maxrows) ;
	    mxFree (Chain_maxcols) ;
	    error ("extracting symbolic factors failed", A_is_complex, nargout,
		pargout, Control, Info, status) ;
	    return ;
	}

	/* create sparse permutation matrix for P */
	pargout [0] = mxCreateSparse (n_row, n_row, n_row, mxREAL) ;
	Pp = (Long *) mxGetJc (pargout [0]) ;
	Pi = (Long *) mxGetIr (pargout [0]) ;
	Px = mxGetPr (pargout [0]) ;
	for (k = 0 ; k < n_row ; k++)
	{
	    Pp [k] = k ;
	    Px [k] = 1 ;
	    Pi [P [k]] = k ;
	}
	Pp [n_row] = n_row ;

	/* create sparse permutation matrix for Q */
	pargout [1] = mxCreateSparse (n_col, n_col, n_col, mxREAL) ;
	Qp = (Long *) mxGetJc (pargout [1]) ;
	Qi = (Long *) mxGetIr (pargout [1]) ;
	Qx = mxGetPr (pargout [1]) ;
	for (k = 0 ; k < n_col ; k++)
	{
	    Qp [k] = k ;
	    Qx [k] = 1 ;
	    Qi [k] = Q [k] ;
	}
	Qp [n_col] = n_col ;

	/* create Fr */
	pargout [2] = mxCreateDoubleMatrix (nfronts+1, 4, mxREAL) ;

	p1 = mxGetPr (pargout [2]) ;
	p2 = p1 + nfronts + 1 ;
	p3 = p2 + nfronts + 1 ;
	p4 = p3 + nfronts + 1 ;
	for (i = 0 ; i <= nfronts ; i++)
	{
	    /* convert parent, 1strow, and leftmostdesc to 1-based */
	    p1 [i] = (double) (Front_npivcol [i]) ;
	    p2 [i] = (double) (Front_parent [i] + 1) ;
	    p3 [i] = (double) (Front_1strow [i] + 1) ;
	    p4 [i] = (double) (Front_leftmostdesc [i] + 1) ;
	}

	/* create Ch */
	pargout [3] = mxCreateDoubleMatrix (nchains+1, 3, mxREAL) ;
	p1 = mxGetPr (pargout [3]) ;
	p2 = p1 + nchains + 1 ;
	p3 = p2 + nchains + 1 ;
	for (i = 0 ; i < nchains ; i++)
	{
	    p1 [i] = (double) (Chain_start [i] + 1) ;	/* convert to 1-based */
	    p2 [i] = (double) (Chain_maxrows [i]) ;
	    p3 [i] = (double) (Chain_maxcols [i]) ;
	}
	p1 [nchains] = Chain_start [nchains] + 1 ;
	p2 [nchains] = 0 ;
	p3 [nchains] = 0 ;

	mxFree (P) ;
	mxFree (Q) ;
	mxFree (Front_npivcol) ;
	mxFree (Front_parent) ;
	mxFree (Front_1strow) ;
	mxFree (Front_leftmostdesc) ;
	mxFree (Chain_start) ;
	mxFree (Chain_maxrows) ;
	mxFree (Chain_maxcols) ;
    }

    /* ---------------------------------------------------------------------- */
    /* report Info */
    /* ---------------------------------------------------------------------- */

    if (A_is_complex)
    {
	umfpack_zl_report_info (Control, Info) ;
    }
    else
    {
	umfpack_dl_report_info (Control, Info) ;
    }

    if (do_info > 0)
    {
	/* return Info */
        if (info_details > 0)
        {
            pargout [do_info] = umfpack_mx_info_details (Control, Info) ;
        }
        else
        {
            pargout [do_info] = umfpack_mx_info_user (Control, Info, do_solve) ;
        }
    }
}

/* ========================================================================== */
/* === MATLAB/cholmod_matlab.h ============================================== */
/* ========================================================================== */

/* Shared prototypes and definitions for CHOLMOD mexFunctions */

#include "SuiteSparse_config.h"
#ifndef DLONG
#define DLONG
#endif
#define Long SuiteSparse_long
#define Long_max SuiteSparse_long_max

/* Ensure cholmod_read_* and cholmod_write_* work for large files.  This
 * requires MATLAB 7.0 or later.  If you are using MATLAB 6.5 or earlier,
 * you must delete the following line, or compile CHOLMOD with -DNLARGEFILE */
#include "cholmod_io64.h"

#ifndef NPARTITION
#include "metis.h"
#endif
#undef ASSERT

#include "cholmod.h"
#include <limits.h>
#include <string.h>
#include <ctype.h>
#include <float.h>
#include "mex.h"
#define EMPTY (-1)
#define TRUE 1
#define FALSE 0
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define LEN 16

#define MXFREE(a) { \
    void *ptr ; \
    ptr = (void *) (a) ; \
    if (ptr != NULL) mxFree (ptr) ; \
}

#define ERROR_TOO_SMALL 0
#define ERROR_HUGE 1
#define ERROR_NOT_INTEGER 2
#define ERROR_TOO_LARGE 3
#define ERROR_USAGE 4
#define ERROR_LENGTH 5
#define ERROR_INVALID_TYPE 6
#define ERROR_OUT_OF_MEMORY 7

/* getting spumoni at run-time takes way too much time */
#ifndef SPUMONI
#define SPUMONI 0
#endif

/* closed by sputil_error_handler if not NULL */
extern FILE *sputil_file ;

void sputil_error   /* reports an error */
(
    Long error,     /* kind of error */
    Long is_index   /* TRUE if a matrix index, FALSE if a matrix dimension */
) ;

Long sputil_double_to_int   /* returns integer value of x */
(
    double x,       /* double value to convert */
    Long is_index,  /* TRUE if a matrix index, FALSE if a matrix dimension */
    Long n          /* if a matrix index, x cannot exceed this dimension */
) ;

double sputil_get_double (const mxArray *arg) ; /* like mxGetScalar */

Long sputil_get_integer  /* returns the integer value of a MATLAB argument */
(
    const mxArray *arg,     /* MATLAB argument to convert */
    Long is_index,          /* TRUE if an index, FALSE if a matrix dimension */
    Long n                  /* maximum value, if an index */
) ;


Long sputil_copy_ij     /* returns the dimension, n */
(
    Long is_scalar,     /* TRUE if argument is a scalar, FALSE otherwise */
    Long scalar,        /* scalar value of the argument */
    void *vector,       /* vector value of the argument */
    mxClassID category, /* type of vector */
    Long nz,            /* length of output vector I */
    Long n,             /* maximum dimension, EMPTY if not yet known */
    Long *I             /* vector of length nz to copy into */
) ;

/* converts a triplet matrix to a compressed-column matrix */
cholmod_sparse *sputil_triplet_to_sparse
(
    Long nrow, Long ncol, Long nz, Long nzmax,
    Long i_is_scalar, Long i, void *i_vector, mxClassID i_class,
    Long j_is_scalar, Long j, void *j_vector, mxClassID j_class,
    Long s_is_scalar, double x, double z, void *x_vector, double *z_vector,
    mxClassID s_class, Long s_complex,
    cholmod_common *cm
) ;

mxArray *sputil_copy_sparse (const mxArray *A) ;    /* copy a sparse matrix */

Long sputil_nelements (const mxArray *arg) ; /* like mxGetNumberOfElements */

void sputil_sparse      /* top-level wrapper for "sparse" function */
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
) ;

void sputil_error_handler (int status, const char *file, int line,
    const char *message) ;

void sputil_config (Long spumoni, cholmod_common *cm) ;

mxArray *sputil_sparse_to_dense (const mxArray *S) ;

cholmod_sparse *sputil_get_sparse
(
    const mxArray *Amatlab, /* MATLAB version of the matrix */
    cholmod_sparse *A,      /* CHOLMOD version of the matrix */
    double *dummy,          /* a pointer to a valid scalar double */
    Long stype              /* -1: lower, 0: unsymmetric, 1: upper */
) ;

cholmod_dense *sputil_get_dense
(
    const mxArray *Amatlab, /* MATLAB version of the matrix */
    cholmod_dense *A,       /* CHOLMOD version of the matrix */
    double *dummy           /* a pointer to a valid scalar double */
) ;

mxArray *sputil_put_dense   /* returns the MATLAB version */
(
    cholmod_dense **Ahandle,    /* CHOLMOD version of the matrix */
    cholmod_common *cm
) ;

mxArray *sputil_put_sparse
(
    cholmod_sparse **Ahandle,   /* CHOLMOD version of the matrix */
    cholmod_common *cm
) ;

Long sputil_drop_zeros      /* drop numerical zeros from a CHOLMOD matrix */
(
    cholmod_sparse *S
) ;

mxArray *sputil_put_int     /* copy Long vector to mxArray */
(
    Long *P,            /* vector to convert */
    Long n,             /* length of P */
    Long one_based      /* 1 if convert from 0-based to 1-based, else 0 */
) ;

mxArray *sputil_dense_to_sparse (const mxArray *arg) ;

void sputil_check_ijvector (const mxArray *arg) ;

void sputil_trim
(
    cholmod_sparse *S,
    Long k,
    cholmod_common *cm
) ;

cholmod_sparse *sputil_get_sparse_pattern
(
    const mxArray *Amatlab,     /* MATLAB version of the matrix */
    cholmod_sparse *Ashallow,   /* shallow CHOLMOD version of the matrix */
    double *dummy,              /* a pointer to a valid scalar double */
    cholmod_common *cm
) ;

cholmod_sparse *sputil_extract_zeros
(
    cholmod_sparse *A,
    cholmod_common *cm
) ;

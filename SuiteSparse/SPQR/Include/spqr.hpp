// =============================================================================
// === spqr.hpp ================================================================
// =============================================================================

// Internal definitions and non-user-callable routines.  This should not be
// included in the user's code.

#ifndef SPQR_INTERNAL_H
#define SPQR_INTERNAL_H

// -----------------------------------------------------------------------------
// include files
// -----------------------------------------------------------------------------

#include "SuiteSparseQR.hpp"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cstring>

#include <complex>
typedef std::complex<double> Complex ;

// -----------------------------------------------------------------------------
// debugging and printing control
// -----------------------------------------------------------------------------

// force debugging off
#ifndef NDEBUG
#define NDEBUG
#endif

// force printing off
#ifndef NPRINT
#define NPRINT
#endif

// uncomment the following line to turn on debugging (SPQR will be slow!)
/*
#undef NDEBUG
*/

// uncomment the following line to turn on printing (LOTS of output!)
/*
#undef NPRINT
*/

// uncomment the following line to turn on expensive debugging (very slow!)
/*
#define DEBUG_EXPENSIVE
*/

// -----------------------------------------------------------------------------
// Long is defined at SuiteSparse_long, from SuiteSparse_config.h
// -----------------------------------------------------------------------------

#define Long SuiteSparse_long

// -----------------------------------------------------------------------------
// basic macros
// -----------------------------------------------------------------------------

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define EMPTY (-1)
#define TRUE 1
#define FALSE 0 
#define IMPLIES(p,q) (!(p) || (q))

// NULL should already be defined, but ensure it is here.
#ifndef NULL
#define NULL ((void *) 0)
#endif

// column-major indexing; A[i,j] is A (INDEX (i,j,lda))
#define INDEX(i,j,lda) ((i) + ((j)*(lda)))

// FLIP is a "negation about -1", and is used to mark an integer i that is
// normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
// is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
// for all integers i.  UNFLIP (i) is >= EMPTY.
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP (i) : (i))

// -----------------------------------------------------------------------------
// additional include files
// -----------------------------------------------------------------------------

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif

#define ITYPE CHOLMOD_LONG
#define DTYPE CHOLMOD_DOUBLE
#define ID SuiteSparse_long_id

// -----------------------------------------------------------------------------

#define ERROR(status,msg) \
    cholmod_l_error (status, __FILE__, __LINE__, msg, cc)

// Check a pointer and return if null.  Set status to invalid, unless the
// status is already "out of memory"
#define RETURN_IF_NULL(A,result) \
{ \
    if ((A) == NULL) \
    { \
	if (cc->status != CHOLMOD_OUT_OF_MEMORY) \
	{ \
	    ERROR (CHOLMOD_INVALID, NULL) ; \
	} \
	return (result) ; \
    } \
}

// Return if Common is NULL or invalid
#define RETURN_IF_NULL_COMMON(result) \
{ \
    if (cc == NULL) \
    { \
	return (result) ; \
    } \
    if (cc->itype != ITYPE || cc->dtype != DTYPE) \
    { \
	cc->status = CHOLMOD_INVALID ; \
	return (result) ; \
    } \
}

#define RETURN_IF_XTYPE_INVALID(A,result) \
{ \
    if (A->xtype != xtype) \
    { \
        ERROR (CHOLMOD_INVALID, "invalid xtype") ; \
        return (result) ; \
    } \
}

// -----------------------------------------------------------------------------
// debugging and printing macros
// -----------------------------------------------------------------------------

#ifndef NDEBUG

    #ifdef MATLAB_MEX_FILE

        // #define ASSERT(e) mxAssert (e, "error: ")

        extern char spqr_mx_debug_string [200] ;
        char *spqr_mx_id (int line) ;

        #define ASSERT(e) \
            ((e) ? (void) 0 : \
            mexErrMsgIdAndTxt (spqr_mx_id (__LINE__), \
            "assert: (" #e ") file:"  __FILE__ ))

    #else

        #include <assert.h>
        #define ASSERT(e) assert (e)

    #endif

    #define DEBUG(e) e
    #ifdef DEBUG_EXPENSIVE
        #define DEBUG2(e) e
        #define ASSERT2(e) ASSERT(e)
    #else
        #define DEBUG2(e)
        #define ASSERT2(e)
    #endif

#else

    #define ASSERT(e)
    #define ASSERT2(e)
    #define DEBUG(e)
    #define DEBUG2(e)

#endif

#ifndef NPRINT

    #ifdef MATLAB_MEX_FILE
        #define PR(e) mexPrintf e
    #else
        #define PR(e) printf e
    #endif

    #define PRVAL(e) spqrDebug_print (e)

#else

    #define PR(e)
    #define PRVAL(e)

#endif

// -----------------------------------------------------------------------------
// For counting flops; disabled if TBB is used
// -----------------------------------------------------------------------------

#define FLOP_COUNT(f) { if (cc->SPQR_grain <= 1) cc->SPQR_flopcount += (f) ; }

// =============================================================================
// === spqr_work ===============================================================
// =============================================================================

// workspace required for each stack in spqr_factorize and spqr_kernel
template <typename Entry> struct spqr_work
{
    Long *Stair1 ;          // size maxfn if H not kept
    Long *Cmap ;            // size maxfn
    Long *Fmap ;            // size n
    Entry *WTwork ;         // size (fchunk + (keepH ? 0:1)) * maxfn

    Entry *Stack_head ;     // head of Stack
    Entry *Stack_top ;      // top of Stack

    Long sumfrank ;         // sum of ranks of the fronts in this stack
    Long maxfrank ;         // largest rank of fronts in this stack

    // for computing the 2-norm of w, the vector of the dead column norms
    double wscale ;         // scale factor for norm (w (of this stack))
    double wssq ;           // sum-of-squares for norm (w (of this stack))
} ;


// =============================================================================
// === spqr_blob ===============================================================
// =============================================================================

// The spqr_blob is a collection of objects that the spqr_kernel requires.

template <typename Entry> struct spqr_blob
{
    double tol ;
    spqr_symbolic *QRsym ;
    spqr_numeric <Entry> *QRnum ;
    spqr_work <Entry> *Work ;
    Long *Cm ;
    Entry **Cblock ;
    Entry *Sx ;
    Long ntol ;
    Long fchunk ;
    cholmod_common *cc ;
} ;


// =============================================================================
// === SuiteSparseQR non-user-callable functions ===============================
// =============================================================================

spqr_symbolic *spqr_analyze
( 
    // inputs, not modified
    cholmod_sparse *A,
    int ordering,           // all ordering options available
    Long *Quser,            // user provided ordering, if given (may be NULL)

    int do_rank_detection,  // if TRUE, then rank deficient matrices may be
                            // considered during numerical factorization,
    // with tol >= 0 (tol < 0 is also allowed).  If FALSE, then the tol
    // parameter is ignored by the numerical factorization, and no rank
    // detection is performed.

    int keepH,                      // if nonzero, H is kept

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> spqr_numeric <Entry> *spqr_factorize
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    Long freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    Long ntol,                      // apply tol only to first ntol columns
    spqr_symbolic *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;

// returns tol (-1 if error)
template <typename Entry> double spqr_tol
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> double spqr_maxcolnorm
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> void spqr_kernel
(
    Long task,
    spqr_blob <Entry> *Blob
) ;

template <typename Entry> void spqr_parallel
(
    Long ntasks,
    int nthreads,
    spqr_blob <Entry> *Blob
) ;

void spqr_freesym
(
    spqr_symbolic **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> void spqr_freenum
(
    spqr_numeric <Entry> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> void spqr_freefac
(
    SuiteSparseQR_factorization <Entry> **QR_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

void spqr_stranspose1
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    Long *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j if the kth column of S is the jth
                        // column of A.  Identity permutation is used if
                        // Qfill is NULL.

    // output, contents not defined on input
    Long *Sp,           // size m+1, row pointers of S
    Long *Sj,           // size nz, column indices of S
    Long *PLinv,        // size m, inverse row permutation, PLinv [i] = k
    Long *Sleft,        // size n+2, Sleft [j] ... Sleft [j+1]-1 is the list of
                        // rows of S whose leftmost column index is j.  The list
                        // can be empty (that is, Sleft [j] == Sleft [j+1]).
                        // Sleft [n] is the number of non-empty rows of S, and
                        // Sleft [n+1] is always m.  That is, Sleft [n] ...
                        // Sleft [n+1]-1 gives the empty rows of S.

    // workspace, not defined on input or output
    Long *W             // size m
) ;


template <typename Entry> void spqr_stranspose2
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    Long *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j
                        // if the kth column of S is the jth column of A.
                        // Identity permutation is used if Qfill is NULL.

    Long *Sp,           // size m+1, row pointers of S
    Long *PLinv,        // size m, inverse row permutation, PLinv [i] = k

    // output, contents not defined on input
    Entry *Sx,          // size nz, numerical values of S

    // workspace, not defined on input or output
    Long *W             // size m
) ;


// =============================================================================

#ifndef NDEBUG

template <typename Entry> void spqrDebug_dumpdense
(
    Entry *A,
    Long m,
    Long n,
    Long lda,
    cholmod_common *cc
) ;

template <typename Entry> void spqrDebug_dumpsparse
(
    Long *Ap,
    Long *Ai,
    Entry *Ax,
    Long m,
    Long n,
    cholmod_common *cc
) ;

void spqrDebug_print (double x) ;
void spqrDebug_print (Complex x) ;

void spqrDebug_dump_Parent (Long n, Long *Parent, const char *filename) ;

Long spqrDebug_rhsize             // returns # of entries in R+H block
(
    // input, not modified
    Long m,                 // # of rows in F
    Long n,                 // # of columns in F
    Long npiv,              // number of pivotal columns in F
    Long *Stair,            // size n; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.
    cholmod_common *cc
) ;
#endif

#ifdef DEBUG_EXPENSIVE
Long spqrDebug_listcount
(
    Long x, Long *List, Long len, Long what,
    cholmod_common *cc
) ;
#endif

// =============================================================================

Long spqr_fsize     // returns # of rows of F
(
    // inputs, not modified
    Long f,
    Long *Super,            // size nf, from QRsym
    Long *Rp,               // size nf, from QRsym
    Long *Rj,               // size rjsize, from QRsym
    Long *Sleft,            // size n+2, from QRsym
    Long *Child,            // size nf, from QRsym
    Long *Childp,           // size nf+1, from QRsym
    Long *Cm,               // size nf, from QRwork

    // outputs, not defined on input
    Long *Fmap,             // size n, from QRwork
    Long *Stair             // size fn, from QRwork
) ;


template <typename Entry> void spqr_assemble
(
    // inputs, not modified
    Long f,                 // front to assemble F
    Long fm,                // number of rows of F
    int keepH,              // if TRUE, then construct row pattern of H
    Long *Super,
    Long *Rp,
    Long *Rj,
    Long *Sp,
    Long *Sj,
    Long *Sleft,
    Long *Child,
    Long *Childp,
    Entry *Sx,
    Long *Fmap,
    Long *Cm,
    Entry **Cblock,
#ifndef NDEBUG
    char *Rdead,
#endif
    Long *Hr,

    // input/output
    Long *Stair,
    Long *Hii,              // if keepH, construct list of row indices for F
    // input only
    Long *Hip,

    // outputs, not defined on input
    Entry *F,

    // workspace, not defined on input or output
    Long *Cmap
) ;

template <typename Entry> Long spqr_cpack     // returns # of rows in C
(
    // input, not modified
    Long m,                 // # of rows in F
    Long n,                 // # of columns in F
    Long npiv,              // number of pivotal columns in F
    Long g,                 // the C block starts at F (g,npiv)

    // input, not modified unless the pack occurs in-place
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
) ;

Long spqr_fcsize    // returns # of entries in C of current front F
(
    // input, not modified
    Long m,                 // # of rows in F
    Long n,                 // # of columns in F
    Long npiv,              // number of pivotal columns in F
    Long g                  // the C block starts at F (g,npiv)
) ;

Long spqr_csize     // returns # of entries in C of a child
(
    // input, not modified
    Long c,                 // child c
    Long *Rp,               // size nf+1, pointers for pattern of R
    Long *Cm,               // size nf, Cm [c] = # of rows in child C
    Long *Super             // size nf, pivotal columns in each front
) ;

template <typename Entry> void spqr_rcount
(
    // inputs, not modified
    spqr_symbolic *QRsym,
    spqr_numeric <Entry> *QRnum,

    Long n1rows,        // added to each row index of Ra and Rb
    Long econ,          // only get entries in rows n1rows to econ-1
    Long n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, count Rb' instead of Rb

    // input/output
    Long *Ra,           // size n2; Ra [j] += nnz (R (:,j)) if j < n2
    Long *Rb,           // If getT is false: size n-n2 and
                        // Rb [j-n2] += nnz (R (:,j)) if j >= n2.
                        // If getT is true: size econ, and
                        // Rb [i] += nnz (R (i, n2:n-1))
    Long *Hp,           // size rjsize+1.  Column pointers for H.
                        // Only computed if H was kept during factorization.
                        // Only Hp [0..nh] is used.
    Long *p_nh          // number of Householder vectors (nh <= rjsize)
) ;

template <typename Entry> void spqr_rconvert
(
    // inputs, not modified
    spqr_symbolic *QRsym,
    spqr_numeric <Entry> *QRnum,

    Long n1rows,        // added to each row index of Ra, Rb, and H
    Long econ,          // only get entries in rows n1rows to econ-1
    Long n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, get Rb' instead of Rb

    // input/output
    Long *Rap,          // size n2+1; on input, Rap [j] is the column pointer
                        // for Ra.  Incremented on output by the number of
                        // entries added to column j of Ra.

    // output, not defined on input
    Long *Rai,          // size rnz1 = nnz(Ra); row indices of Ra
    Entry *Rax,         // size rnz; numerical values of Ra

    // input/output
    Long *Rbp,          // if getT is false:
                        // size (n-n2)+1; on input, Rbp [j] is the column
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to column j of Rb.
                        // if getT is true:
                        // size econ+1; on input, Rbp [i] is the row
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to row i of Rb.

    // output, not defined on input
    Long *Rbi,          // size rnz2 = nnz(Rb); indices of Rb
    Entry *Rbx,         // size rnz2; numerical values of Rb

    // input
    Long *H2p,          // size nh+1; H2p [j] is the column pointer for H.
                        // H2p, H2i, and H2x are ignored if H was not kept
                        // during factorization.  nh computed by rcount

    // output, not defined on input
    Long *H2i,           // size hnz = nnz(H); indices of H
    Entry *H2x,         // size hnz; numerical values of H
    Entry *H2Tau        // size nh; Householder coefficients
) ;

template <typename Entry> Long spqr_rhpack    // returns # of entries in R+H
(
    // input, not modified
    int keepH,              // if true, then H is packed
    Long m,                 // # of rows in F
    Long n,                 // # of columns in F
    Long npiv,              // number of pivotal columns in F
    Long *Stair,            // size npiv; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.

    // input, not modified (unless the pack occurs in-place)
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *R,               // packed columns of R+H
    Long *p_rm              // number of rows in R block
) ;

template <typename Entry> void spqr_hpinv
(
    // input
    spqr_symbolic *QRsym,
    // input/output
    spqr_numeric <Entry> *QRnum,
    // workspace
    Long *W              // size QRnum->m
) ;

template <typename Entry> int spqr_1colamd
(
    // inputs, not modified
    int ordering,           // all available, except 0:fixed and 3:given
                            // treated as 1:natural
    double tol,             // only accept singletons above tol
    Long bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    Long **p_Q1fill,        // size n+bncols, fill-reducing
                            // or natural ordering

    Long **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    Long **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    Long *p_n1cols,         // number of column singletons found
    Long *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> int spqr_1fixed
(
    // inputs, not modified
    double tol,             // only accept singletons above tol
    Long bncols,            // number of columns of B
    cholmod_sparse *A,      // m-by-n sparse matrix

    // output arrays, neither allocated nor defined on input.

    Long **p_R1p,           // size n1rows+1, R1p [k] = # of nonzeros in kth
                            // row of R1.  NULL if n1cols == 0.
    Long **p_P1inv,         // size m, singleton row inverse permutation.
                            // If row i of A is the kth singleton row, then
                            // P1inv [i] = k.  NULL if n1cols is zero.

    cholmod_sparse **p_Y,   // on output, only the first n-n1cols+1 entries of
                            // Y->p are defined (if Y is not NULL), where
                            // Y = [A B] or Y = [A2 B2].  If B is empty and
                            // there are no column singletons, Y is NULL

    Long *p_n1cols,         // number of column singletons found
    Long *p_n1rows,         // number of corresponding rows found

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> SuiteSparseQR_factorization <Entry> *spqr_1factor
(
    // inputs, not modified
    int ordering,           // all ordering options available
    double tol,             // only accept singletons above tol
    Long bncols,            // number of columns of B
    int keepH,              // if TRUE, keep the Householder vectors
    cholmod_sparse *A,      // m-by-n sparse matrix
    Long ldb,               // leading dimension of B, if dense
    Long *Bp,               // size bncols+1, column pointers of B
    Long *Bi,               // size bnz = Bp [bncols], row indices of B
    Entry *Bx,              // size bnz, numerical values of B

    // workspace and parameters
    cholmod_common *cc
) ;

Long spqr_cumsum            // returns total sum
(
    // input, not modified
    Long n,

    // input/output
    Long *X                 // size n+1. X = cumsum ([0 X])
) ;

void spqr_shift
(
    // input, not modified
    Long n,

    // input/output
    Long *X                 // size n+1
) ;

template <typename Entry> void spqr_larftb
(
    // inputs, not modified (V is modified and then restored on output)
    int method,     // 0,1,2,3
    Long m,         // C is m-by-n
    Long n,
    Long k,         // V is v-by-k
                    // for methods 0 and 1, v = m,
                    // for methods 2 and 3, v = n
    Long ldc,       // leading dimension of C
    Long ldv,       // leading dimension of V
    Entry *V,       // V is v-by-k, unit lower triangular (diag not stored)
    Entry *Tau,     // size k, the k Householder coefficients

    // input/output
    Entry *C,       // C is m-by-n, with leading dimension ldc

    // workspace, not defined on input or output
    Entry *W,       // for methods 0,1: size k*k + n*k
                    // for methods 2,3: size k*k + m*k
    cholmod_common *cc
) ;

int spqr_happly_work
(
    // input
    int method,     // 0,1,2,3 

    Long m,         // X is m-by-n
    Long n,

    // FUTURE : make H cholmod_sparse:
    Long nh,        // number of Householder vectors
    Long *Hp,       // size nh+1, column pointers for H
    Long hchunk, 

    // outputs; sizes of workspaces needed
    Long *p_vmax, 
    Long *p_vsize, 
    Long *p_csize
) ;

template <typename Entry> void spqr_happly
(
    // input
    int method,     // 0,1,2,3 

    Long m,         // X is m-by-n
    Long n,

    Long nh,        // number of Householder vectors
    Long *Hp,       // size nh+1, column pointers for H
    Long *Hi,       // size hnz = Hp [nh], row indices of H
    Entry *Hx,      // size hnz, Householder values.  Note that the first
                    // entry in each column must be equal to 1.0

    Entry *Tau,     // size nh

    // input/output
    Entry *X,       // size m-by-n with leading dimension m

    // workspace
    Long vmax,
    Long hchunk,
    Long *Wi,       // size vmax
    Long *Wmap,     // size MAX(mh,1) where H is mh-by-nh
    Entry *C,       // size csize
    Entry *V,       // size vsize
    cholmod_common *cc
) ;

template <typename Entry> void spqr_panel
(
    // input
    int method,
    Long m,
    Long n,
    Long v,
    Long h,             // number of Householder vectors in the panel
    Long *Vi,           // Vi [0:v-1] defines the pattern of the panel
    Entry *V,           // v-by-h, panel of Householder vectors
    Entry *Tau,         // size h, Householder coefficients for the panel
    Long ldx,

    // input/output
    Entry *X,           // m-by-n with leading dimension ldx

    // workspace
    Entry *C,           // method 0,1: v-by-n;  method 2,3: m-by-v
    Entry *W,           // method 0,1: k*k+n*k; method 2,3: k*k+m*k

    cholmod_common *cc
) ;

template <typename Entry> int spqr_append       // TRUE if OK, FALSE otherwise
(
    // inputs, not modified
    Entry *X,       // size m-by-1
    Long *P,        // size m, or NULL; permutation to apply to X.
                    // P [k] = i if row k of A is row i of X

    // input/output
    cholmod_sparse *A,    // size m-by-n2 where n2 > n
    Long *p_n,       // number of columns of A; increased by one

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> Long spqr_trapezoidal       // rank of R, or EMPTY
(
    // inputs, not modified

    Long n,         // R is m-by-n (m is not needed here; can be economy R)
    Long *Rp,       // size n+1, column pointers of R
    Long *Ri,       // size rnz = Rp [n], row indices of R
    Entry *Rx,      // size rnz, numerical values of R

    Long bncols,    // number of columns of B

    Long *Qfill,    // size n+bncols, fill-reducing ordering.  Qfill [k] = j if
                    // the jth column of A is the kth column of R.  If Qfill is
                    // NULL, then it is assumed to be the identity
                    // permutation.

    int skip_if_trapezoidal,        // if R is already in trapezoidal form,
                                    // and skip_if_trapezoidal is TRUE, then
                                    // the matrix T is not created.

    // outputs, not allocated on input
    Long **p_Tp,    // size n+1, column pointers of T
    Long **p_Ti,    // size rnz, row indices of T
    Entry **p_Tx,   // size rnz, numerical values of T

    Long **p_Qtrap, // size n+bncols, modified Qfill

    // workspace and parameters
    cholmod_common *cc
) ;

template <typename Entry> int spqr_type (void) ;

template <typename Entry> void spqr_rsolve
(
    // inputs
    SuiteSparseQR_factorization <Entry> *QR,
    int use_Q1fill,

    Long nrhs,              // number of columns of B
    Long ldb,               // leading dimension of B
    Entry *B,               // size m-by-nrhs with leading dimesion ldb

    // output
    Entry *X,               // size n-by-nrhs with leading dimension n

    // workspace
    Entry **Rcolp,
    Long *Rlive,
    Entry *W,

    cholmod_common *cc
) ;

// returns rank of F, or 0 on error
template <typename Entry> Long spqr_front
(
    // input, not modified
    Long m,             // F is m-by-n with leading dimension m
    Long n,
    Long npiv,          // number of pivot columns
    double tol,         // a column is flagged as dead if its norm is <= tol
    Long ntol,          // apply tol only to first ntol pivot columns
    Long fchunk,        // block size for compact WY Householder reflections,
                        // treated as 1 if fchunk <= 1

    // input/output
    Entry *F,           // frontal matrix F of size m-by-n
    Long *Stair,        // size n, entries F (Stair[k]:m-1, k) are all zero,
                        // and remain zero on output.
    char *Rdead,        // size npiv; all zero on input.  If k is dead,
                        // Rdead [k] is set to 1

    // output, not defined on input
    Entry *Tau,         // size n, Householder coefficients

    // workspace, undefined on input and output
    Entry *W,           // size b*(n+b), where b = min (fchunk,n,m)

    // input/output
    double *wscale,
    double *wssq,

    cholmod_common *cc
) ;

template <typename Entry> int spqr_rmap
(
    SuiteSparseQR_factorization <Entry> *QR,
    cholmod_common *cc
) ;

// =============================================================================
// === spqrgpu features ========================================================
// =============================================================================

#ifdef GPU_BLAS
#include "spqrgpu.hpp"
#endif

// =============================================================================
// === spqr_conj ===============================================================
// =============================================================================

inline double spqr_conj (double x)
{
    return (x) ;
}

inline Complex spqr_conj (Complex x)
{
    return (std::conj (x)) ;
}


// =============================================================================
// === spqr_abs ================================================================
// =============================================================================

inline double spqr_abs (double x, cholmod_common *cc)       // cc is unused
{
    return (fabs (x)) ;
}

inline double spqr_abs (Complex x, cholmod_common *cc)
{
    return (SuiteSparse_config.hypot_func (x.real ( ), x.imag ( ))) ;
}


// =============================================================================
// === spqr_divide =============================================================
// =============================================================================

inline double spqr_divide (double a, double b, cholmod_common *cc)  // cc unused
{
    return (a/b) ;
}

inline Complex spqr_divide (Complex a, Complex b, cholmod_common *cc)
{
    double creal, cimag ;
    SuiteSparse_config.divcomplex_func
        (a.real(), a.imag(), b.real(), b.imag(), &creal, &cimag) ;
    return (Complex (creal, cimag)) ;
}


// =============================================================================
// === spqr_add ================================================================
// =============================================================================

// Add two non-negative Long's, and return the result.  Checks for Long overflow
// and sets ok to FALSE if it occurs.

inline Long spqr_add (Long a, Long b, int *ok)
{
    Long c = a + b ;
    if (c < 0)
    {
        (*ok) = FALSE ;
        return (EMPTY) ;
    }
    return (c) ;
}


// =============================================================================
// === spqr_mult ===============================================================
// =============================================================================

// Multiply two positive Long's, and return the result.  Checks for Long
// overflow and sets ok to FALSE if it occurs.

inline Long spqr_mult (Long a, Long b, int *ok)
{
    Long c = a * b ;
    if (((double) c) != ((double) a) * ((double) b))
    {
        (*ok) = FALSE ;
        return (EMPTY) ;
    }
    return (c) ;
}


// =============================================================================
// === BLAS interface ==========================================================
// =============================================================================

// To compile SuiteSparseQR with 64-bit BLAS, use -DBLAS64.  See also
// CHOLMOD/Include/cholmod_blas.h

extern "C" {
#include "cholmod_blas.h"
}

#undef CHECK_BLAS_INT
#undef EQ
#define CHECK_BLAS_INT (sizeof (BLAS_INT) < sizeof (Long))
#define EQ(K,k) (((BLAS_INT) K) == ((Long) k))

#ifdef SUN64

#define BLAS_DNRM2    dnrm2_64_
#define LAPACK_DLARF  dlarf_64_
#define LAPACK_DLARFG dlarfg_64_
#define LAPACK_DLARFT dlarft_64_
#define LAPACK_DLARFB dlarfb_64_

#define BLAS_DZNRM2   dznrm2_64_
#define LAPACK_ZLARF  zlarf_64_
#define LAPACK_ZLARFG zlarfg_64_
#define LAPACK_ZLARFT zlarft_64_
#define LAPACK_ZLARFB zlarfb_64_

#elif defined (BLAS_NO_UNDERSCORE)

#define BLAS_DNRM2    dnrm2
#define LAPACK_DLARF  dlarf
#define LAPACK_DLARFG dlarfg
#define LAPACK_DLARFT dlarft
#define LAPACK_DLARFB dlarfb

#define BLAS_DZNRM2   dznrm2
#define LAPACK_ZLARF  zlarf
#define LAPACK_ZLARFG zlarfg
#define LAPACK_ZLARFT zlarft
#define LAPACK_ZLARFB zlarfb

#else

#define BLAS_DNRM2    dnrm2_
#define LAPACK_DLARF  dlarf_
#define LAPACK_DLARFG dlarfg_
#define LAPACK_DLARFT dlarft_
#define LAPACK_DLARFB dlarfb_

#define BLAS_DZNRM2   dznrm2_
#define LAPACK_ZLARF  zlarf_
#define LAPACK_ZLARFG zlarfg_
#define LAPACK_ZLARFT zlarft_
#define LAPACK_ZLARFB zlarfb_

#endif

// =============================================================================
// === BLAS and LAPACK prototypes ==============================================
// =============================================================================

extern "C"
{

void LAPACK_DLARFT (char *direct, char *storev, BLAS_INT *n, BLAS_INT *k,
    double *V, BLAS_INT *ldv, double *Tau, double *T, BLAS_INT *ldt) ;

void LAPACK_ZLARFT (char *direct, char *storev, BLAS_INT *n, BLAS_INT *k,
    Complex *V, BLAS_INT *ldv, Complex *Tau, Complex *T, BLAS_INT *ldt) ;

void LAPACK_DLARFB (char *side, char *trans, char *direct, char *storev,
    BLAS_INT *m, BLAS_INT *n, BLAS_INT *k, double *V, BLAS_INT *ldv,
    double *T, BLAS_INT *ldt, double *C, BLAS_INT *ldc, double *Work,
    BLAS_INT *ldwork) ;

void LAPACK_ZLARFB (char *side, char *trans, char *direct, char *storev,
    BLAS_INT *m, BLAS_INT *n, BLAS_INT *k, Complex *V, BLAS_INT *ldv,
    Complex *T, BLAS_INT *ldt, Complex *C, BLAS_INT *ldc, Complex *Work,
    BLAS_INT *ldwork) ;

double BLAS_DNRM2 (BLAS_INT *n, double *X, BLAS_INT *incx) ;

double BLAS_DZNRM2 (BLAS_INT *n, Complex *X, BLAS_INT *incx) ;

void LAPACK_DLARFG (BLAS_INT *n, double *alpha, double *X, BLAS_INT *incx,
    double *tau) ;

void LAPACK_ZLARFG (BLAS_INT *n, Complex *alpha, Complex *X, BLAS_INT *incx,
    Complex *tau) ;

void LAPACK_DLARF (char *side, BLAS_INT *m, BLAS_INT *n, double *V,
    BLAS_INT *incv, double *tau, double *C, BLAS_INT *ldc, double *Work) ;

void LAPACK_ZLARF (char *side, BLAS_INT *m, BLAS_INT *n, Complex *V,
    BLAS_INT *incv, Complex *tau, Complex *C, BLAS_INT *ldc, Complex *Work) ;

}

#endif

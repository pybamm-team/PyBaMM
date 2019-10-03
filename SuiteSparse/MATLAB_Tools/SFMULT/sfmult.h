// =============================================================================
// === sfmult.h ================================================================
// =============================================================================

#ifndef _SFMULT_H
#define _SFMULT_H

#include "mex.h"

// Like UMFPACK, CHOLMOD, AMD, COLAMD, CSparse, MA57, and all other sane sparse
// matrix functions used internally in MATLAB, these functions will NOT work
// with mwIndex.

#define Int mwSignedIndex

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define MXFREE(a) { \
    double *ptr ; \
    ptr = (a) ; \
    if (ptr != NULL) mxFree (ptr) ; \
}

// -----------------------------------------------------------------------------
// primary sparse-times-full and full-times-sparse
// -----------------------------------------------------------------------------

mxArray *sfmult		// returns y = A*x or variants
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int at,		// if true: trans(A)  if false: A
    int ac,		// if true: conj(A)   if false: A. ignored if A real
    int xt,		// if true: trans(x)  if false: x
    int xc,		// if true: conj(x)   if false: x. ignored if x real
    int yt,		// if true: trans(y)  if false: y
    int yc		// if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *fsmult		// returns y = x*A or variants
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int at,		// if true: trans(A)  if false: A
    int ac,		// if true: conj(A)   if false: A. ignored if A real
    int xt,		// if true: trans(x)  if false: x
    int xc,		// if true: conj(x)   if false: x. ignored if x real
    int yt,		// if true: trans(y)  if false: y
    int yc		// if true: conj(y)   if false: y. ignored if y real
) ;

// -----------------------------------------------------------------------------
// transpose
// -----------------------------------------------------------------------------

mxArray *ssmult_transpose	// returns C = A' or A.'    (TO DO) rename
(
    // --- inputs, not modified:
    const mxArray *A,
    int conj			// compute A' if true, compute A.' if false
) ;

// -----------------------------------------------------------------------------
// 8 primary variants of op(op(A)*op(B)) in sfmult.c
// -----------------------------------------------------------------------------

mxArray *sfmult_AN_XN_YN    // returns y = A*x
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *sfmult_AN_XN_YT    // returns y = (A*x)'
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *sfmult_AN_XT_YN    // returns y = A*x'
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *sfmult_AN_XT_YT    // returns y = (A*x')'
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *sfmult_AT_XN_YN    // returns y = A'*x
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *sfmult_AT_XN_YT    // returns y = (A'*x)'
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *sfmult_AT_XT_YN    // returns y = A'*x'
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

mxArray *sfmult_AT_XT_YT    // returns y = (A'*x')'
(
    // --- inputs, not modified:
    const mxArray *A,
    const mxArray *X,
    int ac,		    // if true: conj(A)   if false: A. ignored if A real
    int xc,		    // if true: conj(x)   if false: x. ignored if x real
    int yc		    // if true: conj(y)   if false: y. ignored if y real
) ;

// -----------------------------------------------------------------------------
// kernels in sfmult_anxnyt_k.c
// -----------------------------------------------------------------------------

void sfmult_AN_XN_YT_2	// y = (A*x)'  where x is n-by-2, and y is 2-by-m
(
    // --- outputs, not initialized on input:
    double *Yx,		// 2-by-m
    double *Yz,		// 2-by-m if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// n-by-2
    const double *Xz,	// n-by-2 if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

void sfmult_AN_XN_YT_3	// y = (A*x)'	x is n-by-3, and y is 3-by-m (ldy = 4)
(
    // --- outputs, not initialized on input:
    double *Yx,		// 3-by-m
    double *Yz,		// 3-by-m if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// n-by-3
    const double *Xz,	// n-by-3 if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

void sfmult_AN_XN_YT_4	// y = (A*x)'	x is n-by-4, and y is 4-by-m
(
    // --- outputs, not initialized on input:
    double *Yx,		// 4-by-m
    double *Yz,		// 4-by-m if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// n-by-4
    const double *Xz,	// n-by-4 if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

// -----------------------------------------------------------------------------
// kernels in sfmult_anxtyt_k.c
// -----------------------------------------------------------------------------

void sfmult_AN_XT_YT_2	// y = (A*x')'	x is 2-by-n, and y is 2-by-m
(
    // --- outputs, not initialized on input:
    double *Yx,		// 2-by-m
    double *Yz,		// 2-by-m if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 2-by-n with leading dimension k
    const double *Xz,	// 2-by-n with leading dimension k if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// leading dimension of X
) ;

void sfmult_AN_XT_YT_3	// y = (A*x')'	x is 3-by-n, and y is 3-by-m (ldy = 4)
(
    // --- outputs, not initialized on input:
    double *Yx,		// 3-by-m
    double *Yz,		// 3-by-m if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 3-by-n with leading dimension k
    const double *Xz,	// 3-by-n with leading dimension k if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// leading dimension of X
) ;

void sfmult_AN_XT_YT_4 // y = (A*x')'	x is 4-by-n, and y is 4-by-m
(
    // --- outputs, not initialized on input:
    double *Yx,		// 4-by-m
    double *Yz,		// 4-by-m if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 4-by-n with leading dimension k
    const double *Xz,	// 4-by-n with leading dimension k if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// leading dimension of X
) ;

// -----------------------------------------------------------------------------
// kernels in sfmult_atxtyn_k.c
// -----------------------------------------------------------------------------

void sfmult_AT_XT_YN_2	// y = A'*x'	x is 2-by-m, and y is n-by-2
(
    // --- outputs, not initialized on input:
    double *Yx,		// n-by-2
    double *Yz,		// n-by-2 if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 2-by-m
    const double *Xz,	// 2-by-m if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

void sfmult_AT_XT_YN_3	// y = A'*x'	x is 3-by-m, and y is n-by-3 (ldx = 4)
(
    // --- outputs, not initialized on input:
    double *Yx,		// n-by-3
    double *Yz,		// n-by-3 if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 3-by-m
    const double *Xz,	// 3-by-m if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

void sfmult_AT_XT_YN_4	// y = A'*x'	x is 4-by-m, and y is n-by-4
(
    // --- outputs, not initialized on input:
    double *Yx,		// n-by-4
    double *Yz,		// n-by-4 if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 4-by-m
    const double *Xz,	// 4-by-m if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

// -----------------------------------------------------------------------------
// kernels in sfmult_atxtyt_k.c
// -----------------------------------------------------------------------------

void sfmult_AT_XT_YT_2	// y = (A'*x')'	x is 2-by-m, and y is 2-by-n
(
    // --- outputs, not initialized on input:
    double *Yx,		// 2-by-n with leading dimension k
    double *Yz,		// 2-by-n with leading dimension k if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 2-by-m
    const double *Xz,	// 2-by-m if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// leading dimension of Y
) ;

void sfmult_AT_XT_YT_3	// y = (A'*x')'	x is 3-by-m, and y is 3-by-n (ldx = 4)
(
    // --- outputs, not initialized on input:
    double *Yx,		// 3-by-n with leading dimension k
    double *Yz,		// 3-by-n with leading dimension k if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 3-by-m
    const double *Xz,	// 3-by-m if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// leading dimension of Y
) ;

void sfmult_AT_XT_YT_4	// y = (A'*x')'	x is 4-by-m, and y is 4-by-n
(
    // --- outputs, not initialized on input:
    double *Yx,		// 4-by-n with leading dimension k
    double *Yz,		// 4-by-n with leading dimension k if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 4-by-m
    const double *Xz,	// 4-by-m if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// leading dimension of Y
) ;

// -----------------------------------------------------------------------------
// kernel in sfmult_xA
// -----------------------------------------------------------------------------

void sfmult_xA		// y = (A'*x')' = x*A, x is k-by-m, and y is k-by-n
(
    // --- outputs, not initialized on input:
    double *Yx,		// k-by-n
    double *Yz,		// k-by-n if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// k-by-m
    const double *Xz,	// k-by-m if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k
) ;

// -----------------------------------------------------------------------------
// vector kernels in sfmult_vector_1.c
// -----------------------------------------------------------------------------

void sfmult_AN_x_1	// y = A*x	x is n-by-1 unit stride, y is m-by-1
(
    // --- outputs, not initialized on input:
    double *Yx,		// m-by-1
    double *Yz,		// m-by-1 if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// n-by-1
    const double *Xz,	// n-by-1 if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

void sfmult_AT_x_1	// y = A'*x	x is m-by-1, y is n-by-1
(
    // --- outputs, not initialized on input:
    double *Yx,		// n-by-1
    double *Yz,		// n-by-1 if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// m-by-1
    const double *Xz,	// m-by-1 if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
) ;

// -----------------------------------------------------------------------------
// vector kernels in sfmult_vector_k.c
// -----------------------------------------------------------------------------

void sfmult_AN_xk_1	// y = A*x	x is n-by-1 non-unit stride, y is m-by-1
(
    // --- outputs, not initialized on input:
    double *Yx,		// m-by-1
    double *Yz,		// m-by-1 if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// n-by-1
    const double *Xz,	// n-by-1 if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// stride of x
) ;

void sfmult_AT_xk_1	// y = A'*x	x is m-by-1, y is n-by-1 non-unit stride
(
    // --- outputs, not initialized on input:
    double *Yx,		// n-by-1
    double *Yz,		// n-by-1 if Y is complex

    // --- inputs, not modified:
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// m-by-1
    const double *Xz,	// m-by-1 if X complex
    int ac,		// true: use conj(A), otherwise use A
    int xc,		// true: use conj(X), otherwise use X
    int yc		// true: compute conj(Y), otherwise compute Y
    , Int k		// stride of y
) ;

// -----------------------------------------------------------------------------
// utilities
// -----------------------------------------------------------------------------

mxArray *sfmult_yalloc	    // allocate and return result y
(
    // --- inputs, not modified:
    Int m,
    Int n,
    int Ycomplex
) ;

mxArray *sfmult_yzero	    // set y to zero
(
    // --- must exist on input, set to zero on output:
    mxArray *Y
) ;

void sfmult_walloc	    // allocate workspace
(
    // --- inputs, not modified:
    Int k,
    Int m,
    // --- outputs, not initialized on input:
    double **Wx,	    // real part (first k*m doubles)
    double **Wz		    // imaginary part (next k*m doubles)
) ;

void sfmult_invalid (void) ;

#endif

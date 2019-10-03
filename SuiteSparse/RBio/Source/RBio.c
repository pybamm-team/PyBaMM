/* ========================================================================== */
/* === RBio/Source/RBio.c: C-callable RBio functions ======================== */
/* ========================================================================== */

/* Copyright 2009, Timothy A. Davis, All Rights Reserved.
   Refer to RBio/Doc/license.txt for the RBio license. */

/* This file contains functions for writing/reading a sparse matrix to/from a
   file in Rutherford-Boeing format.  User-callable functions are declared
   as PUBLIC.  PRIVATE functions are available only within this file.
*/

/* ========================================================================== */
/* === definitions ========================================================== */
/* ========================================================================== */

#include "RBio.h"

#ifdef INT
/* int version */
#define Int int
#define IDD "d"
#define RB(name) RB ## name ## _i
#else
/* Default: long (except for Windows, which is __int64) */
#define Int SuiteSparse_long
#define IDD SuiteSparse_long_idd
#define RB(name) RB ## name
#endif
#define ID "%" IDD

#define TRUE (1)
#define FALSE (0)
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(a)   (((a) > 0) ? (a) : -(a))
#define ISNAN(a) ((a) != (a))
#define PRIVATE static
#define PUBLIC

#define SLEN 4096
#define FREE_WORK   { SuiteSparse_free (w) ; \
                      SuiteSparse_free (cp) ; }

#define FREE_ALL    { FREE_WORK ; \
                      SuiteSparse_free (Ap) ; \
                      SuiteSparse_free (Ai) ; \
                      SuiteSparse_free (Ax) ; \
                      SuiteSparse_free (Az) ; \
                      SuiteSparse_free (Zp) ; \
                      SuiteSparse_free (Zi) ; }

#define FREE_RAW    { SuiteSparse_free (Ap) ; \
                      SuiteSparse_free (Ai) ; \
                      SuiteSparse_free (Ax) ; }


/* ========================================================================== */
/* === internal prototypes ================================================== */
/* ========================================================================== */

PRIVATE Int RB(format)  /* return format to use (index in F_, C_format) */
(
    /* input */
    Int nnz,            /* number of nonzeros */
    double *x,          /* of size nnz */
    Int is_int,         /* true if integer format is to be used */
    double xmin,        /* minimum value of x */
    double xmax,        /* maximum value of x */
    Int fmt,            /* initial format to use (index into F_format, ...) */

    /* output */
    char valfmt [21],   /* Fortran format to use */
    char valcfm [21],   /* C format to use */
    Int *valn           /* number of entries per line */
) ;

PRIVATE void RB(iformat)
(
    /* input */
    double xmin,            /* smallest integer to print */
    double xmax,            /* largest integer to print */

    /* output */
    char indfmt [21],       /* Fortran format to use */
    char indcfm [21],       /* C format to use */
    Int *indn               /* number of entries per line */
) ;

PRIVATE Int RB(cards)
(
    Int nitems,         /* number of items to print */
    Int nperline        /* number of items per line */
) ;

PRIVATE Int RB(iprint)        /* returns TRUE if OK, FALSE otherwise */
(
    /* input */
    FILE *file,             /* which file to write to */
    char *indcfm,           /* C format to use */
    Int i,                  /* value to write */
    Int indn,               /* number of entries to write per line */

    /* input/output */
    Int *nbuf               /* number of entries written to current line */
) ;

PRIVATE Int RB(xprint)        /* returns TRUE if OK, FALSE otherwise */
(
    /* input */
    FILE *file,             /* which file to write to */
    char *valcfm,           /* C format to use */
    double x,               /* value to write */
    Int valn,               /* number of entries to write per line */
    Int mkind,              /* 0:real, 1:pattern, 2:complex, 3:integer */

    /* input/output */
    Int *nbuf               /* number of entries written to current line */
) ;

PRIVATE void RB(fill)
(
    char *s,            /* string to fill */
    Int len,            /* length of s (including trailing '\0') */
    char c              /* character to fill s with */
) ;

PRIVATE Int RB(fix_mkind_in)      /* return revised mkind */
(
    Int mkind_in,       /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    double *Ax,
    double *Az
) ;

PRIVATE Int RB(writeTask)       /* returns TRUE if OK, FALSE on failure */
(
    /* input */
    FILE *file,     /* file to print to (already open) */
    Int task,       /* 0 to 3 (see above) */
    Int nrow,       /* A is nrow-by-ncol */
    Int ncol,
    Int mkind,      /* 0:real, 1:pattern, 2:complex, 3:integer */
    Int skind,      /* -1:rect, 0:unsym, 1:sym, 2:hermitian, 3:skew */
    Int *Ap,        /* size ncol+1, column pointers */
    Int *Ai,        /* size anz=Ap[ncol], row indices */
    double *Ax,     /* size anz, real values */
    double *Az,     /* size anz, imaginary part (may be NULL) */
    Int *Zp,        /* size ncol+1, column pointers for Z (may be NULL) */
    Int *Zi,        /* size Zp[ncol], row indices for Z */
    char *indcfm,   /* C format for indices */
    Int indn,       /* # of indices per line */
    char *valcfm,   /* C format for values */
    Int valn,       /* # of values per line */

    /* output */
    Int *nnz,           /* number of entries that will be printed to the file */

    /* workspace */
    Int *w,         /* size MAX(nrow,ncol)+1 */
    Int *cp         /* size MAX(nrow,ncol)+1 */
) ;

PRIVATE Int RB(read2)     /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    FILE *file,         /* must be already open for reading */
    Int nrow,           /* A is nrow-by-ncol */
    Int ncol,
    Int nnz,            /* number of entries in A, from the header */
    Int mkind,          /* R: 0, P: 1, C: 2, I: 3 */
    Int skind,          /* R: -1, U: 0, S: 1, H: 2, Z: 3 */
    Int build_upper,    /* if TRUE and skind>0, then create upper part.  Ai, Ax,
                           and Az must be twice the size as given below */

    /* output */
    Int *Ap,            /* size ncol+1, column pointers for A */
    Int *Ai,            /* size nnz, row indices for A */
    double *Ax,         /* size nnz or 2*nnz if complex and Az NULL */
    double *Az,         /* size nnz, or NULL, for complex matrices only */

    /* workspace */
    Int *w,             /* size MAX(nrow,ncol)+1 */
    Int *cp,            /* size ncol+1 */

    /* input/workspace */
    char *s,            /* first line of column pointers on input */
    Int slen
) ;

PRIVATE Int RB(zcount)    /* return number of explicit zeros in A */
(
    Int nnz,            /* number of entries to check */
    Int mkind,          /* R: 0, P: 1, C: 2, I: 3 */
    double *Ax,         /* NULL, size nnz, or 2*nnz */
    double *Az          /* NULL or size nnz */
) ;

PRIVATE Int RB(extract)   /* return # of explicit zero entries */
(
    /* input */
    Int ncol,
    Int mkind,          /* R: 0, P: 1, C: 2, I: 3 */

    /* input/output */
    Int *Ap,            /* size ncol+1, column pointers A A */
    Int *Ai,            /* size nnz=Ap[ncol], row indices of A */
    double *Ax,         /* NULL, size nnz, or 2*nnz */
    double *Az,         /* NULL or size nnz */

    /* output */
    Int *Zp,            /* size ncol+1, column pointers for Z */
    Int *Zi             /* size znz = Zp [ncol] = # of zeros in A on input */
) ;

PRIVATE Int RB(readline) /* return string length, or -1 on error or EOF */
(
    char *s,            /* buffer to read into */
    Int slen,           /* s [0..slen] is of length slen+1 */
    FILE *file          /* file to read from */
) ;

PRIVATE void RB(substring)
(
    /* input */
    char *s,            /* input string */
    Int len,            /* length of string s [0..len] */
    Int start,          /* extract t [0:len-1] from s [start:start+len-1] */
    Int tlen,           /* length of substring t, excluding null terminator */

    /* output */
    char *t             /* size tlen+1 */
) ;

PRIVATE Int RB(xtoken)   /* TRUE if token found, FALSE othewise */
(
    /* input/output */
    char *s,            /* parse the next token in s [k..len] and update k */
    Int len,            /* length of s (input only) */
    Int *k,             /* start parsing at s [k] */
    /* output */
    double *x           /* value of the token, or 0 if not found */
) ;

PRIVATE Int RB(itoken)
(
    /* input/output */
    char *s,            /* parse the next token in s [k..len] and update k */
    Int len,            /* length of s (input only) */
    Int *k,             /* start parsing at s [k] */
    /* output */
    Int *i              /* value of the token, or 0 if not found */
) ;

PRIVATE void RB(prune_space)
(
    /* input/output */
    char *s
) ;

PRIVATE Int RB(header)    /* 0: success, < 0: error, > 0: warning */
(
    /* input */
    FILE *file,         /* must be already open for reading */

    /* output */
    char title [73],    /* title, from first line of header */
    char key [9],       /* 8-character key, from first header line */
    char mtype [4],     /* RUA, RSA, PUA, PSA, RRA, etc */
    Int *nrow,          /* A is nrow-by-ncol */
    Int *ncol,
    Int *nnz,           /* number of entries in A (tril(A) if symmetric) */
    Int *nelnz,         /* number of finite-elements */
    char ptrfmt [21],   /* Fortran format for column pointers */
    char indfmt [21],   /* Fortran format for row indices */
    char valfmt [21],   /* Fortran format for numerical values */
    Int *mkind,         /* R__: 0, P__: 1, C__: 2, I__: 3 */
    Int *skind,         /* _R_: -1, _U_: 0, _S_: 1, _H_: 2, _Z_: 3 */
    Int *fem,           /* __A: false, __E: true */
    char *s,            /* first line of data after the header */
    Int slen            /* s is of length slen+1 */
) ;

PRIVATE Int RB(read1i)    /* TRUE if OK, false otherwise */
(
    FILE *file,         /* file to read from (must be already open) */
    char *s,            /* buffer to use */
    Int *len,           /* strlen(s) */
    Int slen,           /* size of s */
    Int *k,             /* read position in s */
    Int *x              /* value read from the file */
) ;

PRIVATE Int RB(read1x)    /* TRUE if OK, false otherwise */
(
    FILE *file,         /* file to read from (must be already open) */
    char *s,            /* buffer to use */
    Int *len,           /* strlen(s) */
    Int slen,           /* size of s */
    Int *k,             /* read position in s */
    double *x           /* value read from the file */
) ;

PRIVATE Int RB(iread)     /* TRUE if OK, false otherwise */
(
    /* input */
    FILE *file,         /* file to read from (must be already open) */
    Int n,              /* number of integers to read */
    Int offset,         /* offset to subtract from each integer */

    /* output */
    Int *A,             /* read integers from file into A [0..n-1] */

    /* input/workspace */
    char *s,            /* first line of input may be present in s */
    Int slen            /* s is of size slen+1 */
) ;

PRIVATE Int RB(xread)     /* TRUE if OK, false otherwise */
(
    /* input */
    FILE *file,         /* file to read from (must be already open) */
    Int n,              /* number of values to read (2*n if complex) */
    Int mkind,          /* R: 0, P: 1, C: 2, I: 3 */

    /* output */
    double *Ax,         /* read reals from file into Ax [0..n-1] */
    double *Az,         /* read reals from file into Az [0..n-1], may be NULL */

    /* input/workspace */
    char *s,            /* first line of input may be present in s */
    Int slen            /* s is of size slen+1 */
) ;

PRIVATE void RB(skipheader)
(
    char *s,
    Int slen,
    FILE *file
) ;


/* ========================================================================== */
/* === functions ============================================================ */
/* ========================================================================== */


/* -------------------------------------------------------------------------- */
/* RBget_entry: get numerical entry in the matrix at position p */
/* -------------------------------------------------------------------------- */

PUBLIC void RB(get_entry)
(
    Int mkind,          /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    double *Ax,         /* real part, or both if merged-complex */
    double *Az,         /* imaginary part if split-complex */
    Int p,              /* index of the entry */
    double *xr,         /* real part */
    double *xz          /* imaginary part */
)
{
    if (mkind == 0 || mkind == 3)
    {
        /* A is real or integer */
        *xr = Ax ? Ax [p] : 1 ;
        *xz = 0 ;
    }
    else if (mkind == 2)
    {
        /* A is split-complex */
        *xr = Ax ? Ax [p] : 1 ;
        *xz = Ax ? Az [p] : 0 ;
    }
    else if (mkind == 4)
    {
        /* A is merged-complex */
        *xr = Ax ? Ax [2*p  ] : 1 ;
        *xz = Ax ? Ax [2*p+1] : 0 ;
    }
    else /* if (mkind == 1) */
    {
        /* A is pattern-only */
        *xr = 1 ;
        *xz = 0 ;
    }
}


/* -------------------------------------------------------------------------- */
/* RBput_entry: put numerical entry in the matrix in position p */
/* -------------------------------------------------------------------------- */

PUBLIC void RB(put_entry)
(
    Int mkind,          /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    double *Ax,         /* real part, or both if merged-complex */
    double *Az,         /* imaginary part if split-complex */
    Int p,              /* index of the entry */
    double xr,          /* real part */
    double xz           /* imaginary part */
)
{
    if (mkind == 0 || mkind == 3)
    {
        /* A is real or integer; ignore xz */
        if (Ax) Ax [p] = xr ;
        if (Az) Az [p] = 0 ;
    }
    else if (mkind == 2)
    {
        /* A is split-complex */
        if (Ax) Ax [p] = xr ;
        if (Az) Az [p] = xz ;
    }
    else if (mkind == 4)
    {
        /* A is merged-complex */
        if (Ax) Ax [2*p  ] = xr ;
        if (Ax) Ax [2*p+1] = xz ;
    }
    else /* if (mkind == 1) */
    {
        /* A is pattern-only; ignore xr and xz */
        if (Ax) Ax [p] = 1 ;
        if (Az) Az [p] = 0 ;
    }
}


/* -------------------------------------------------------------------------- */
/* RBskipheader: skip past the header */
/* -------------------------------------------------------------------------- */

PRIVATE void RB(skipheader)
(
    char *s,            /* size slen+1 */
    Int slen,
    FILE *file          /* file to read from */
)
{
    s [0] = '\0' ;
    RB(readline) (s, slen, file) ;
    RB(readline) (s, slen, file) ;
    RB(readline) (s, slen, file) ;
    RB(readline) (s, slen, file) ;
    RB(readline) (s, slen, file) ;
    if ((s [0] == 'F') || (s [0] == 'f') || (s [0] == 'M') || (s[0] == 'm'))
    {
        RB(readline) (s, slen, file) ;
    }
    /* s now contains the first non-header line */
}


/* -------------------------------------------------------------------------- */
/* RBread2: read all but the header and construct the matrix */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(read2)   /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    FILE *file,         /* must be already open for reading */
    Int nrow,           /* A is nrow-by-ncol */
    Int ncol,
    Int nnz,            /* number of entries in A, from the header */
    Int mkind,          /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    Int skind,          /* R: -1, U: 0, S: 1, H: 2, Z: 3 */
    Int build_upper,    /* if TRUE and skind>0, then create upper part.  Ai, Ax,
                           and Az must be twice the size as given below */

    /* output */
    Int *Ap,            /* size ncol+1, column pointers for A */
    Int *Ai,            /* size nnz, row indices for A */
    double *Ax,         /* size nnz or 2*nnz if complex and Az NULL */
    double *Az,         /* size nnz, or NULL, for complex matrices only */

    /* workspace */
    Int *w,             /* size MAX(nrow,ncol)+1 */
    Int *cp,            /* size ncol+1 */

    /* input/workspace */
    char *s,            /* first line of column pointers on input */
    Int slen
)
{
    double xr = 0, xz = 0 ;
    Int p, i, j, k, ilast, alen, llen, psrc, pdst ;

    /* ---------------------------------------------------------------------- */
    /* skip past the header, if reading from a file */
    /* ---------------------------------------------------------------------- */

    if (file)
    {
        RB(skipheader) (s, slen, file) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the column pointers and check them */
    /* ---------------------------------------------------------------------- */

    if (!RB(iread) (file, ncol+1, 1, Ap, s, slen))
    {
        return (RBIO_CP_IOERROR) ;      /* I/O error reading column pointers */
    }
    if (Ap [0] != 0 || Ap [ncol] != nnz)
    {
        return (RBIO_CP_INVALID) ;      /* column pointers invalid */
    }
    for (j = 1 ; j <= ncol ; j++)
    {
        if (Ap [j] < Ap [j-1])
        {
            return (RBIO_CP_INVALID) ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* read the row indices and check them */
    /* ---------------------------------------------------------------------- */

    if (!RB(iread) (file, nnz, 1, Ai, s, slen))
    {
        return (RBIO_ROW_IOERROR) ;       /* I/O error reading row indices */
    }

    for (i = 0 ; i < nrow ; i++)
    {
        w [i] = -1 ;
    }

    for (j = 0 ; j < ncol ; j++)
    {
        ilast = -1 ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            if (i < 0 || i >= nrow)
            {
                return (RBIO_ROW_INVALID) ; /* row index out of range */
            }
            if (w [i] == j)
            {
                return (RBIO_DUPLICATE) ;   /* duplicate entry in matrix */
            }
            w [i]= j ;
            if (i < ilast)
            {
                return (RBIO_JUMBLED) ;     /* row indices unsorted */
            }
            ilast = i ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* read the values */
    /* ---------------------------------------------------------------------- */

    if (!RB(xread) (file, nnz, mkind, Ax, Az, s, slen))
    {
        return (RBIO_VALUE_IOERROR) ;     /* I/O error reading values */
    }

    /* ---------------------------------------------------------------------- */
    /* construct or check the upper triangular part for symmetric matrices */
    /* ---------------------------------------------------------------------- */

    if (skind > 0)
    {

        /* ------------------------------------------------------------------ */
        /* check the matrix and compute new column counts */
        /* ------------------------------------------------------------------ */

        for (j = 0 ; j <= ncol ; j++)
        {
            w [j] = 0 ;
        }

        for (j = 0 ; j < ncol ; j++)
        {
            for (p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                i = Ai [p] ;
                if (i == j)
                {
                    /* diagonal entry, only appears as A(j,j) */
                    w [j]++ ;
                }
                else if (i > j)
                {
                    /* entry in lower triangular part, A(i,j) will be */
                    /* duplicated as A(j,i), so count it in column i and j  */
                    w [i]++ ;
                    w [j]++ ;
                }
                else
                {
                    /* error: entry in upper triangular part */
                    return (RBIO_EXTRANEOUS) ;
                }
            }
        }

        /* ------------------------------------------------------------------ */
        /* construct the upper trianglar part, if requested */
        /* ------------------------------------------------------------------ */

        if (build_upper)
        {

            /* -------------------------------------------------------------- */
            /* compute the new column pointers */
            /* -------------------------------------------------------------- */

            cp [0] = 0 ;
            for (j = 1 ; j <= ncol ; j++)
            {
                cp [j] = cp [j-1] + w [j-1] ;
            }

            /* -------------------------------------------------------------- */
            /* shift the matrix by adding gaps to the top of each column */
            /* -------------------------------------------------------------- */

            for (j = ncol-1 ; j >= 0 ; j--)
            {
                /* number of entries in lower tri. part (including diagonal) */
                llen = Ap [j+1] - Ap [j] ;

                /* number of entries in entire column */
                alen = cp [j+1] - cp [j] ;

                /* move the column from Ai [Ap [j] ... Ap [j+1]-1] down to
                   Ai [cp [j+1]-llen ... cp[j+1]-1], leaving a gap
                   at Ai [Ap [j] ... cp [j+1]-llen] */

                for (k = 1 ; k <= llen ; k++)
                {
                    psrc = Ap [j+1] - k ;
                    pdst = cp [j+1] - k ;
                    Ai [pdst] = Ai [psrc] ;
                    RB(get_entry) (mkind, Ax, Az, psrc, &xr, &xz) ;
                    RB(put_entry) (mkind, Ax, Az, pdst, xr, xz) ;
                }
            }

            /* -------------------------------------------------------------- */
            /* populate the upper triangular part */
            /* -------------------------------------------------------------- */

            /* create temporary column pointers to point to the gaps */
            for (j = 0 ; j < ncol ; j++)
            {
                w [j] = cp [j] ;
            }

            for (j = 0 ; j < ncol ; j++)
            {
                /* scan the entries in the lower tri. part, in
                    Ai [cp[j+1]-llen ... cp[j+1]-1] */
                llen = Ap [j+1] - Ap [j] ;

                for (k = 1 ; k <= llen ; k++)
                {
                    /* get the A(i,j) entry in strictly lower triangular part */
                    psrc = cp [j+1] - k ;
                    i = Ai [psrc] ;
                    if (i != j)
                    {
                        /* get the numerical value of the A(i,j) entry */
                        RB(get_entry) (mkind, Ax, Az, psrc, &xr, &xz) ;

                        /* negate for Hermitian and skew-symmetric */
                        if (skind == 2)
                        {
                            xz = -xz ;      /* complex Hermitian */
                        }
                        else if (skind == 3)
                        {
                            xr = -xr ;      /* skew symmetric */
                            xz = -xz ;
                        }

                        /* add A(j,i) as the next entry in column i */
                        pdst = w [i]++ ;
                        Ai [pdst] = j ;
                        RB(put_entry) (mkind, Ax, Az, pdst, xr, xz) ;
                    }
                }
            }

            /* -------------------------------------------------------------- */
            /* finalize the column pointers */
            /* -------------------------------------------------------------- */

            for (j = 0 ; j <= ncol ; j++)
            {
                Ap [j] = cp [j] ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* matrix is valid */
    /* ---------------------------------------------------------------------- */

    return (RBIO_OK) ;
}


/* -------------------------------------------------------------------------- */
/* RBzcount: count the number of explicit zeros */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(zcount)    /* return number of explicit zeros in A */
(
    Int nnz,            /* number of entries in A */
    Int mkind,          /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    double *Ax,         /* NULL, size nnz, or 2*nnz */
    double *Az          /* NULL or size nnz */
)
{
    double xr, xz ;
    Int p, znz = 0 ;
    for (p = 0 ; p < nnz ; p++)
    {
        RB(get_entry) (mkind, Ax, Az, p, &xr, &xz) ;
        if (xr == 0 && xz == 0)
        {
            znz++ ;
        }
    }
    return (znz) ;
}


/* -------------------------------------------------------------------------- */
/* RBextract: extract explicit zeros */
/* -------------------------------------------------------------------------- */

/* The matrix is A is split into A (the nonzeros) and Z (the pattern of
   explicit zero entries in A).  On input, A may have explicit zeros.  On
   output they are removed from A.   If Zp is NULL on input, these explicit
   zeros are removed from A and simply discarded. */

PRIVATE Int RB(extract) /* return # of explicit zero entries */
(
    /* input */
    Int ncol,
    Int mkind,          /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */

    /* input/output */
    Int *Ap,            /* size ncol+1, column pointers A A */
    Int *Ai,            /* size nnz=Ap[ncol], row indices of A */
    double *Ax,         /* NULL, size nnz, or 2*nnz */
    double *Az,         /* NULL or size nnz */

    /* output */
    Int *Zp,            /* size ncol+1, column pointers for Z */
    Int *Zi             /* size znz = Zp [ncol] = # of zeros in A on input */
)
{
    double xr, xz ;
    Int p, i, j, pa, pz ;

    pa = 0 ;
    pz = 0 ;

    for (j = 0 ; j < ncol ; j++)
    {
        /* save the new start of column j of A */
        p = Ap [j] ;
        Ap [j] = pa ;
        if (Zp) Zp [j] = pz ;

        /* split column j of A */
        for ( ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            RB(get_entry) (mkind, Ax, Az, p, &xr, &xz) ;
            if (xr == 0 && xz == 0)
            {
                /* copy into Z, if Z exists */
                if (Zp) Zi [pz++] = i ;
            }
            else
            {
                /* copy into A */
                Ai [pa] = i ;
                RB(put_entry) (mkind, Ax, Az, pa++, xr, xz) ;
            }
        }
    }
    Ap [ncol] = pa ;
    if (Zp) Zp [ncol] = pz ;
    return (pz) ;
}


/* -------------------------------------------------------------------------- */
/* RBreadline: read a line from the file */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(readline)  /* return length of string, or -1 on error or EOF */
(
    char *s,            /* buffer to read into */
    Int slen,           /* s [0..slen] is of length slen+1 */
    FILE *file          /* file to read from */
)
{
    Int len, ok ;
    if (!file) file = stdin ;                   /* file defaults to stdin */
    ok = (fgets (s, slen, file) != NULL) ;      /* read line in s [0..slen-1] */
    s [slen] = '\0' ;                           /* ensure line is terminated */
    len = ok ? ((Int) strlen (s)) : (-1) ;      /* get the length */
    return ((len < slen) ? len : (-1)) ;        /* return len, or -1 on error */
}


/* -------------------------------------------------------------------------- */
/* RBsubstring: extract a substring */
/* -------------------------------------------------------------------------- */

PRIVATE void RB(substring)
(
    /* input */
    char *s,            /* input string */
    Int len,            /* length of string s [0..len] */
    Int start,          /* extract t [0:len-1] from s [start:start+len-1] */
    Int tlen,           /* length of substring t, excluding null terminator */

    /* output */
    char *t             /* size tlen+1 */
)
{
    Int i, end, k ;
    end = MIN (start+tlen, len) ;       /* do not go past the end of s */
    k = 0 ;
    for (i = start ; i < end ; i++)
    {
        t [k++] = s [i] ;
    }
    t [k] = '\0' ;                      /* ensure t is null-terminated */
}


/* -------------------------------------------------------------------------- */
/* RBxtoken: get the next token from a string, returning it as a double */
/* -------------------------------------------------------------------------- */

/* On input, the token to return from s [0..len] starts at s[k].  On output, k
   is updated so that the s[k] is the start of the token after this one. */

PRIVATE Int RB(xtoken)    /* TRUE if token found, FALSE othewise */
(
    /* input/output */
    char *s,            /* parse the next token in s [k..len] and update k */
    Int len,            /* length of s (input only) */
    Int *k,             /* start parsing at s [k] */
    /* output */
    double *x           /* value of the token, or 0 if not found */
)
{
    Int start ;

    *x = 0 ;

    /* consume leading spaces, if any */
    while ((*k) < len && s [*k] == ' ')
    {
        (*k)++ ;
    }

    /* the token starts here, if present */
    start = (*k) ;

    if (s [start] == '\0')
    {
        /* the end of s has been reached; there is no token */
        return (FALSE) ;
    }

    /* find where the token ends */
    while ((*k) < len && s [*k] != ' ')
    {
        (*k)++ ;
    }

    /* terminate the current token.  Note this might be s [len] */
    if (s [*k] != '\0')
    {
        s [(*k)++] = '\0' ;
    }

    /* parse the current token and return its value */
    return (sscanf (s+start, "%lg", x) == 1) ;
}


/* -------------------------------------------------------------------------- */
/* RBitoken: get the next token from a string, returning it as an Int */
/* -------------------------------------------------------------------------- */

/* Same as RBxtoken, except for the data type of the 4th parameter.  The
   return value i is checked for integer overflow of i+1. */

PRIVATE Int RB(itoken)
(
    /* input/output */
    char *s,            /* parse the next token in s [k..len] and update k */
    Int len,            /* length of s (input only) */
    Int *k,             /* start parsing at s [k] */
    /* output */
    Int *i              /* value of the token, or 0 if not found */
)
{
    double x ;
    Int ok = RB(xtoken) (s, len, k, &x) ;
    *i = (Int) x ;      /* convert to integer */
    return (ok && ((double) ((*i)+1) == (x+1))) ;
}

/* -------------------------------------------------------------------------- */
/* RBprune_space: remove trailing space from a string */
/* -------------------------------------------------------------------------- */

PRIVATE void RB(prune_space)
(
    /* input/output */
    char *s
)
{
    Int k ;
    for (k = strlen (s) - 1 ; k >= 0 ; k--)
    {
        if (isspace (s [k]))
        {
            s [k] = '\0' ;
        }
        else
        {
            return ;
        }
    }
}


/* -------------------------------------------------------------------------- */
/* RBheader:  read Rutherford/Boeing header lines */
/* -------------------------------------------------------------------------- */

/*
The Rutherford/Boeing matrix type is a 3-character string:

    (1) R: real, C: complex, P: pattern only, I: integer
        mkind: 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged
        Note that this function does not return mkind == 4, since the
        split/merged Ax and Az are considered by this function.
        It must be selected by the caller.

    (2) S: symmetric, U: unsymmetric, H: Hermitian, Z: skew symmetric,
        R: rectangular
        skind: R: -1, U: 0, S: 1, H: 2, Z: 3

    (3) A: assembled, E: element form
        nelnz = 0 for A, number of elements for E

        pattern matrices are given numerical values of 1 (except PZA).
        PZA matrices have +1 in the lower triangular part and -1 in
        the upper triangular part.
*/

PRIVATE Int RB(header)    /* 0: success, < 0: error, > 0: warning */
(
    /* input */
    FILE *file,         /* must be already open for reading */

    /* output */
    char title [73],    /* title, from first line of header */
    char key [9],       /* 8-character key, from first header line */
    char mtype [4],     /* RUA, RSA, PUA, PSA, RRA, etc */
    Int *nrow,          /* A is nrow-by-ncol */
    Int *ncol,
    Int *nnz,           /* number of entries in A (tril(A) if symmetric) */
    Int *nelnz,         /* number of finite-elements */
    char ptrfmt [21],   /* Fortran format for column pointers */
    char indfmt [21],   /* Fortran format for row indices */
    char valfmt [21],   /* Fortran format for numerical values */
    Int *mkind,         /* R__: 0, P__: 1, C__: 2, I__: 3 */
    Int *skind,         /* _R_: -1, _U_: 0, _S_: 1, _H_: 2, _Z_: 3 */
    Int *fem,           /* __A: false, __E: true */
    char *s,            /* first line of data after the header */
    Int slen            /* s is of length slen+1 */
)
{
    Int len, totcrd, ptrcrd, indcrd, valcrd, k, ok ;

    /* ---------------------------------------------------------------------- */
    /* get the 1st line: title, key */
    /* ---------------------------------------------------------------------- */

    len = RB(readline) (s, slen, file) ;
    if (len < 0) return (RBIO_HEADER_IOERROR) ;

    RB(substring) (s, len, 0, 72, title) ;
    RB(substring) (s, len, 72, 8, key) ;

    if (title [71] == '|')
    {
        /* remove the marker placed by RBwrite, if present */
        title [71] = '\0' ;
    }

    /* remove trailing spaces from the key and the title */
    RB(prune_space) (title) ;
    RB(prune_space) (key) ;

    /* ---------------------------------------------------------------------- */
    /* get the 2nd line: totcrd, ptrcrd, indcrd, valcrd */
    /* ---------------------------------------------------------------------- */

    len = RB(readline) (s, slen, file) ;
    if (len < 0) return (RBIO_HEADER_IOERROR) ;

    k = 0 ;
    ok = TRUE ;
    ok = ok && RB(itoken) (s, len, &k, &totcrd) ;
    ok = ok && RB(itoken) (s, len, &k, &ptrcrd) ;
    ok = ok && RB(itoken) (s, len, &k, &indcrd) ;
    ok = ok && RB(itoken) (s, len, &k, &valcrd) ;

    /* ---------------------------------------------------------------------- */
    /* get the 3rd line: mtype (RUA, etc), nrow, ncol, nnz, nelnz */
    /* ---------------------------------------------------------------------- */

    len = RB(readline)(s, slen, file) ;
    if (len < 0) return (RBIO_HEADER_IOERROR) ;

    RB(substring) (s, len, 0, 3, mtype) ;

    k = 3 ;
    ok = ok && RB(itoken) (s, len, &k, nrow) ;
    ok = ok && RB(itoken) (s, len, &k, ncol) ;
    ok = ok && RB(itoken) (s, len, &k, nnz) ;
    RB(itoken) (s, len, &k, nelnz) ;

    if (!ok || *nrow <= 0 || *ncol <= 0 || *nnz <= 0)
    {
        return (RBIO_DIM_INVALID) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the 4th line: ptrfmt, indfmt, valfmt */
    /* ---------------------------------------------------------------------- */

    len = RB(readline) (s, slen, file) ;
    if (len < 0) return (RBIO_HEADER_IOERROR) ;

    RB(substring) (s, len, 0,  16, ptrfmt) ;
    RB(substring) (s, len, 16, 16, indfmt) ;
    RB(substring) (s, len, 32, 20, valfmt) ;

    /* ---------------------------------------------------------------------- */
    /* skip the Harwell/Boeing header line 5, if present */
    /* ---------------------------------------------------------------------- */

    len = RB(readline) (s, slen, file) ;
    if (len <= 0) return (RBIO_HEADER_IOERROR) ;

    if ((s [0] == 'F') || (s [0] == 'f') || (s [0] == 'M') || (s [0] == 'm'))
    {
        len = RB(readline) (s, slen, file) ;
        if (len <= 0) return (RBIO_HEADER_IOERROR) ;
    }

    /* ---------------------------------------------------------------------- */
    /* determine if real, pattern, integer, or complex */
    /* ---------------------------------------------------------------------- */

    if (mtype [0] == 'R' || mtype [0] == 'r')
    {
        *mkind = 0 ;        /* real */
    }
    else if (mtype [0] == 'P' || mtype [0] == 'p')
    {
        *mkind = 1 ;        /* pattern */
    }
    else if (mtype [0] == 'C' || mtype [0] == 'c')
    {
        *mkind = 2 ;        /* complex (assume split-complex) */
    }
    else if (mtype [0] == 'I' || mtype [0] == 'i')
    {
        *mkind = 3 ;        /* integer */
    }
    else
    {
        return (RBIO_TYPE_INVALID) ;       /* error: invalid matrix mtype */
    }

    /* ---------------------------------------------------------------------- */
    /* determine the symmetry property */
    /* ---------------------------------------------------------------------- */

    if (mtype [1] == 'R' || mtype [1] == 'r')
    {
        *skind = -1 ;        /* rectangular: RRA, PRA, IRA, and CRA matrices */
    }
    else if (mtype [1] == 'U' || mtype [1] == 'u')
    {
        *skind = 0 ;         /* unsymmetric: RUA, PUA, IUA, and CUA matrices */
    }
    else if (mtype [1] == 'S' || mtype [1] == 's')
    {
        *skind = 1 ;         /* symmetric: RSA, PSA, ISA, and CSA matrices */
    }
    else if (mtype [1] == 'H' || mtype [1] == 'h')
    {
        *skind = 2 ;        /* Hermitian: CHA (PHA, IHA, and RHA are OK too) */
    }
    else if (mtype [1] == 'Z' || mtype [1] == 'z')
    {
        *skind = 3 ;        /* skew symmetric: RZA, PZA, IZA, and CZA */
    }
    else
    {
        return (RBIO_TYPE_INVALID) ;       /* error: invalid matrix mtype */
    }

    /* ---------------------------------------------------------------------- */
    /* assembled vs elemental matrices (**A vs **E) */
    /* ---------------------------------------------------------------------- */

    if (mtype [2] == 'A' || mtype [2] == 'a')
    {
        /* assembled - ignore nelnz */
        *fem = FALSE ;
        *nelnz = 0 ;
    }
    else if (mtype [2] == 'E' || mtype [2] == 'e')
    {
        /* finite-element */
        *fem = TRUE ;
    }
    else
    {
        return (RBIO_TYPE_INVALID) ;       /* error: invalid matrix mtype */
    }

    /* ---------------------------------------------------------------------- */
    /* assembled matrices must be square if skind is not R */
    /* ---------------------------------------------------------------------- */

    if (!(*fem) && (*skind) != -1 && (*nrow) != (*ncol))
    {
        return (RBIO_DIM_INVALID) ;       /* error: invalid matrix dimensions */
    }

    /* ---------------------------------------------------------------------- */
    /* matrix header is valid */
    /* ---------------------------------------------------------------------- */

    return (0) ;
}


/* -------------------------------------------------------------------------- */
/* RBread1i:  read a single integer value from the file */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(read1i)    /* TRUE if OK, false otherwise */
(
    FILE *file,         /* file to read from (must be already open) */
    char *s,            /* buffer to use */
    Int *len,           /* strlen(s) */
    Int slen,           /* size of s */
    Int *k,             /* read position in s */
    Int *x              /* value read from the file */
)
{
    /* read the token from the current line */
    Int ok = RB(itoken) (s, *len, k, x) ;
    if (!ok)
    {
        /* input line is exhausted; get the next one */
        *len = RB(readline) (s, slen, file) ;
        if (*len < 0) return (FALSE) ;
        /* read first token from the new line */
        *k = 0 ;
        ok = RB(itoken) (s, *len, k, x) ;
    }
    return (ok) ;
}


/* -------------------------------------------------------------------------- */
/* RBread1x:  read a single real value from the file */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(read1x)    /* TRUE if OK, false otherwise */
(
    FILE *file,         /* file to read from (must be already open) */
    char *s,            /* buffer to use */
    Int *len,           /* strlen(s) */
    Int slen,           /* size of s */
    Int *k,             /* read position in s */
    double *x           /* value read from the file */
)
{
    /* read the token from the current line */
    Int ok = RB(xtoken) (s, *len, k, x) ;
    if (!ok)
    {
        /* input line is exhausted; get the next one */
        *len = RB(readline) (s, slen, file) ;
        if (*len < 0) return (FALSE) ;
        /* read first token from the new line */
        *k = 0 ;
        ok = RB(xtoken) (s, *len, k, x) ;
    }
    return (ok) ;
}


/* -------------------------------------------------------------------------- */
/* RBiread:  read n integers from the file */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(iread)     /* TRUE if OK, FALSE otherwise */
(
    /* input */
    FILE *file,         /* file to read from (must be already open) */
    Int n,              /* number of integers to read */
    Int offset,         /* offset to subtract from each integer */

    /* output */
    Int *A,             /* read integers from file into A [0..n-1] */

    /* input/workspace */
    char *s,            /* first line of input may be present in s */
    Int slen            /* s is of size slen+1 */
)
{
    Int k, len, i, x, ok = TRUE ;

    /* the first token will be read from the current line, s, on input */
    len = strlen (s) ;
    k = 0 ;

    for (i = 0 ; ok && i < n ; i++)
    {
        /* read the next integer */
        ok = RB(read1i) (file, s, &len, slen, &k, &x) ;
        A [i] = x - offset ;
    }
    s [0] = '\0' ;      /* ignore remainder of current input string */
    return (ok) ;
}


/* -------------------------------------------------------------------------- */
/* RBxread:  read n real or complex values from the file */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(xread)     /* TRUE if OK, FALSE otherwise */
(
    /* input */
    FILE *file,         /* file to read from (must be already open) */
    Int n,              /* number of values to read (2*n if complex) */
    Int mkind,          /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */

    /* output */
    double *Ax,         /* read reals from file into Ax [0..n-1] */
    double *Az,         /* read reals from file into Az [0..n-1], may be NULL */

    /* input/workspace */
    char *s,            /* first line of input may be present in s */
    Int slen            /* s is of size slen+1 */
)
{
    double xr, xz ;
    Int k, len, i, ok = TRUE ;

    /* the first token will be read from the current line, s, on input */
    len = strlen (s) ;
    k = 0 ;
    xr = 1 ;
    xz = 0 ;

    for (i = 0 ; ok && i < n ; i++)
    {
        /* read the real and imaginary part of the kth entry */
        if (ok && mkind != 1)
        {
            /* read the real part, if present */
            ok = RB(read1x) (file, s, &len, slen, &k, &xr) ;
        }
        if (ok && (mkind == 2 || mkind == 4))
        {
            /* read the imaginary part, if present */
            ok = RB(read1x) (file, s, &len, slen, &k, &xz) ;
        }
        /* store the ith entry in Ax and/or Az */
        RB(put_entry) (mkind, Ax, Az, i, xr, xz) ;
    }
    s [0] = '\0' ;      /* ignore remainder of current input string */
    return (ok) ;
}


/* -------------------------------------------------------------------------- */
/* RBread: read a Rutherford/Boeing matrix from a file */
/* -------------------------------------------------------------------------- */

/*
    Space is allocated as needed for the output matrix A, the zero pattern Z,
    and required workspace.  To free the result of this function, do on the
    following:

        if (Ap) free (Ap) ;

    or

        SuiteSparse_free (Ap) ;

    etc, for Ap, Ai, Ax, Az, Zp, and Zi.

    Format of the numerical values:
    If Ax is NULL on input, only the pattern of the matrix is returned.  If
    Ax is not NULL but p_Az is NULL, then the real and imaginary parts (if
    complex) are both returned in Ax.  If both Ax and p_Az are non-NULL,
    imaginary parts (if present) are returned in Az.

    For real and integer matrices (R__, I__), Az is always returned as NULL.

    For pattern matrices (P__), Ax and Az are always returned as NULL.

    For complex matrices (C__), either Ax is used if p_Az is NULL
        (merged-complex) or both Ax and Az are used (split-complex).

    Only assembled matrices (__A) are handled.  Finite-element matrices
    (__E) are not handled.
 */

PUBLIC Int RB(read)              /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    char *filename,     /* filename to read from */
    Int build_upper,    /* if true, construct upper part for sym. matrices */
    Int zero_handling,  /* 0: do nothing, 1: prune zeros, 2: extract zeros */

    /* output */
    char title [73],
    char key [9],
    char mtype [4],     /* RUA, RSA, PUA, PSA, RRA, etc */
    Int *nrow,          /* A is nrow-by-ncol */
    Int *ncol,
    Int *mkind,         /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    Int *skind,         /* R: -1, U: 0, S: 1, H: 2, Z: 3 */
    Int *asize,         /* Ai array has size asize*sizeof(double) */
    Int *znz,           /* number of explicit zeros removed from A */

    /* output: these are malloc'ed below and must be freed by the caller */
    Int **p_Ap,         /* column pointers of A */
    Int **p_Ai,         /* row indices of A */
    double **p_Ax,      /* real values (ignored if NULL) of A */
    double **p_Az,      /* imaginary values (ignored if NULL) of A */
    Int **p_Zp,         /* column pointers of Z */
    Int **p_Zi          /* row indices of Z */
)
{
    Int nnz, nelnz, status, fem ;
    Int *w, *cp, *Ap, *Ai, *Zp, *Zi ;
    double *Ax, *Az ;
    FILE *file = NULL ;     /* read from stdin if NULL */
    int ok ;
    char s [SLEN+1], ptrfmt [21], indfmt [21], valfmt [21] ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (p_Ap) *p_Ap = NULL ;
    if (p_Ai) *p_Ai = NULL ;
    if (p_Ax) *p_Ax = NULL ;
    if (p_Az) *p_Az = NULL ;
    if (p_Zp) *p_Zp = NULL ;
    if (p_Zi) *p_Zi = NULL ;

    if (!title || !key || !mtype || !p_Ap || !p_Ai || !nrow || !ncol || !mkind
        || !skind || (zero_handling == 2 && (!p_Zp || !p_Zi)) || !znz || !asize)
    {
        /* one or more required arguments are missing (NULL) */
        return (RBIO_ARG_ERROR) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the header */
    /* ---------------------------------------------------------------------- */

    if (filename)
    {
        file = fopen (filename, "r") ;
        if (file == NULL)
        {
            return (RBIO_FILE_IOERROR) ;      /* cannot open file */
        }
    }

    status = RB(header) (file, title, key, mtype, nrow, ncol, &nnz, &nelnz,
        ptrfmt, indfmt, valfmt, mkind, skind, &fem, s, SLEN) ;

    /* close the file, so if there is an error the file is not left open */
    if (filename) fclose (file) ;

    if (status != 0)
    {
        return (status) ;
    }

    if (fem)
    {
        return (RBIO_UNSUPPORTED) ;
    }

    /* ---------------------------------------------------------------------- */
    /* adjust mkind based on the presence of Ax and Az input arguments */
    /* ---------------------------------------------------------------------- */

    if (!p_Ax)
    {
        /* the numerical values cannot be returned, so this is read as a
           pattern-only matrix */
        *mkind = 1 ;
    }

    if (*mkind == 2 && !p_Az)
    {
        /* The matrix is complex.  If p_Az is NULL, then the imaginary part
           will be returned in Ax, and the matrix is thus 4:merged-complex */
        *mkind = 4 ;
    }

    /* ---------------------------------------------------------------------- */
    /* allocate space for A */
    /* ---------------------------------------------------------------------- */

    *asize = ((build_upper) ? 2 : 1) * MAX (nnz, 1) ;
    Ap = (Int *) SuiteSparse_malloc ((*ncol) + 1, sizeof (Int)) ;
    Ai = (Int *) SuiteSparse_malloc (*asize, sizeof (Int)) ;
    ok = (Ap != NULL && Ai != NULL) ;
    Ax = NULL ;
    Az = NULL ;
    Zp = NULL ;
    Zi = NULL ;

    if (*mkind == 0 || *mkind == 3 || *mkind == 1)
    {
        /* return A as real, integer, or pattern */
        if (p_Ax)
        {
            Ax = (double *) SuiteSparse_malloc (*asize, sizeof (double)) ;
            ok = ok && (Ax != NULL) ;
        }
    }
    else if (*mkind == 2)
    {
        /* return A as split-complex */
        Ax = (double *) SuiteSparse_malloc (*asize, sizeof (double)) ;
        Az = (double *) SuiteSparse_malloc (*asize, sizeof (double)) ;
        ok = ok && (Ax != NULL && Az != NULL) ;
    }
    else /* if (*mkind == 4) */
    {
        /* return A as merged-complex */
        Ax = (double *) SuiteSparse_malloc (*asize, 2*sizeof (double)) ;
        ok = ok && (Ax != NULL) ;
    }

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    cp = (Int *) SuiteSparse_malloc ((*ncol) + 1, sizeof (Int)) ;
    w  = (Int *) SuiteSparse_malloc (MAX (*nrow, *ncol) + 1, sizeof (Int)) ;
    ok = ok && (cp != NULL && w != NULL) ;

    if (!ok)
    {
        FREE_ALL ;
        return (RBIO_OUT_OF_MEMORY) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the matrix */
    /* ---------------------------------------------------------------------- */

    if (filename)
    {
        file = fopen (filename, "r") ;
        if (file == NULL)
        {
            FREE_ALL ;
            return (RBIO_FILE_IOERROR) ;      /* cannot reopen file */
        }
    }

    status = RB(read2) (file, *nrow, *ncol, nnz, *mkind, *skind, build_upper,
        Ap, Ai, Ax, Az, w, cp, s, SLEN) ;

    if (filename) fclose (file) ;

    if (status != 0)
    {
        FREE_ALL ;
        return (status) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    FREE_WORK ;

    /* ---------------------------------------------------------------------- */
    /* prune or extract exact zeros */
    /* ---------------------------------------------------------------------- */

    if (zero_handling == 2)
    {
        /* allocate the Z matrix */
        *znz = RB(zcount) (Ap [*ncol], *mkind, Ax, Az) ;
        Zp = (Int *) SuiteSparse_malloc ((*ncol) + 1, sizeof (Int)) ;
        Zi = (Int *) SuiteSparse_malloc (*znz, sizeof (Int)) ;
        if (Zp == NULL || Zi == NULL)
        {
            FREE_ALL ;
            return (RBIO_OUT_OF_MEMORY) ;
        }
        /* remove zeros from A and store their pattern in Z */
        RB(extract) (*ncol, *mkind, Ap, Ai, Ax, Az, Zp, Zi) ;
    }
    else if (zero_handling == 1)
    {
        /* remove zeros from A and discard them */
        *znz = RB(extract) (*ncol, *mkind, Ap, Ai, Ax, Az, NULL, NULL) ;
    }
    else
    {
        *znz = 0 ;
    }

    /* ---------------------------------------------------------------------- */
    /* return results */
    /* ---------------------------------------------------------------------- */

    if (p_Ap) *p_Ap = Ap ;
    if (p_Ai) *p_Ai = Ai ;
    if (p_Ax) *p_Ax = Ax ;
    if (p_Az) *p_Az = Az ;
    if (p_Zp) *p_Zp = Zp ;
    if (p_Zi) *p_Zi = Zi ;
    return (RBIO_OK) ;
}


/* -------------------------------------------------------------------------- */
/* RBreadraw: read the raw contents of a Rutherford/Boeing file */
/* -------------------------------------------------------------------------- */

PUBLIC Int RB(readraw)           /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    char *filename,     /* filename to read from */

    /* output */
    char title [73],
    char key [9],
    char mtype [4],     /* RUA, RSA, PUA, PSA, RRA, etc */
    Int *nrow,          /* A is nrow-by-ncol */
    Int *ncol,
    Int *nnz,           /* size of Ai */
    Int *nelnz,
    Int *mkind,         /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    Int *skind,         /* R: -1, U: 0, S: 1, H: 2, Z: 3 */
    Int *fem,           /* 0:__A, 1:__E */
    Int *xsize,         /* size of Ax */

    /* output: these are malloc'ed below and must be freed by the caller */
    Int **p_Ap,         /* size ncol+1, column pointers of A */
    Int **p_Ai,         /* size nnz, row indices of A */
    double **p_Ax       /* size xsize, numerical values of A */
)
{
    FILE *file = NULL ; /* read from stdin if NULL */
    Int *Ap, *Ai ;
    double *Ax ;
    Int status ;
    int ok ;
    char s [SLEN+1], ptrfmt [21], indfmt [21], valfmt [21] ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (p_Ap) *p_Ap = NULL ;
    if (p_Ai) *p_Ai = NULL ;
    if (p_Ax) *p_Ax = NULL ;

    if (!title || !key || !mtype || !nrow || !ncol || !nnz || !nelnz || !mkind
        || !skind || !fem || !xsize || !p_Ap || !p_Ai || !p_Ax)
    {
        /* one or more required arguments are missing (NULL) */
        return (RBIO_ARG_ERROR) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the header */
    /* ---------------------------------------------------------------------- */

    if (filename)
    {
        file = fopen (filename, "r") ;
        if (file == NULL)
        {
            return (RBIO_FILE_IOERROR) ;      /* cannot open file */
        }
    }

    status = RB(header) (file, title, key, mtype, nrow, ncol, nnz, nelnz,
        ptrfmt, indfmt, valfmt, mkind, skind, fem, s, SLEN) ;

    /* close the file, so if there is an error the file is not left open */
    if (filename) fclose (file) ;

    if (status != 0)
    {
        return (status) ;
    }

    /* ---------------------------------------------------------------------- */
    /* allocate space for Ap, Ai, and Ax */
    /* ---------------------------------------------------------------------- */

    Ap = (Int *) SuiteSparse_malloc ((*ncol) + 1, sizeof (Int)) ;
    Ai = (Int *) SuiteSparse_malloc (*nnz, sizeof (Int)) ;
    ok = (Ap != NULL && Ai != NULL) ;

    if (*mkind == 1)
    {
        /* A is pattern-only */
        *xsize = 0 ;
        Ax = NULL ;
    }
    else
    {
        /* A has numerical values */
        *xsize = ((*fem) ? (*nelnz) : (*nnz)) * (((*mkind) == 2) ? 2 : 1) ;
        Ax = (double *) SuiteSparse_malloc (*xsize, sizeof (double)) ;
        ok = ok && (Ax != NULL) ;
    }

    if (!ok)
    {
        FREE_RAW ;
        return (RBIO_OUT_OF_MEMORY) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the matrix */
    /* ---------------------------------------------------------------------- */

    if (filename)
    {
        file = fopen (filename, "r") ;
        if (file == NULL)
        {
            FREE_RAW ;
            return (RBIO_FILE_IOERROR) ;        /* cannot reopen file */
        }
        RB(skipheader) (s, SLEN, file) ;        /* skip past the header */
    }

    if (!RB(iread) (file, (*ncol)+1, 1, Ap, s, SLEN))
    {
        FREE_RAW ;
        if (filename) fclose (file) ;
        return (RBIO_CP_IOERROR) ;      /* I/O error reading column pointers */
    }

    if (!RB(iread) (file, *nnz, 1, Ai, s, SLEN))
    {
        FREE_RAW ;
        if (filename) fclose (file) ;
        return (RBIO_ROW_IOERROR) ;     /* I/O error reading row indices */
    }

    if (*mkind != 1)
    {
        if (!RB(xread) (file, *xsize, 0, Ax, NULL, s, SLEN))
        {
            FREE_RAW ;
            if (filename) fclose (file) ;
            return (RBIO_VALUE_IOERROR) ;     /* I/O error reading values */
        }
    }

    /* ---------------------------------------------------------------------- */
    /* return results */
    /* ---------------------------------------------------------------------- */

    if (p_Ap) *p_Ap = Ap ;
    if (p_Ai) *p_Ai = Ai ;
    if (p_Ax) *p_Ax = Ax ;
    if (filename) fclose (file) ;
    return (RBIO_OK) ;
}


/* -------------------------------------------------------------------------- */
/* RBfix_mkind_in: adjust mkind_in, based on presence of Ax and Az */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(fix_mkind_in)      /* return revised mkind */
(
    Int mkind_in,       /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    double *Ax,
    double *Az
)
{
    if (!Ax)
    {
        /* matrix must be 1:pattern */
        mkind_in = 1 ;
    }
    if (mkind_in == 2 && !Az)
    {
        /* matrix must be 4:merged-complex */
        mkind_in = 4 ;
    }
    return (mkind_in) ;
}


/* -------------------------------------------------------------------------- */
/* RBwrite */
/* -------------------------------------------------------------------------- */

PUBLIC Int RB(write)         /* 0:OK, < 0: error, > 0: warning */
(
    /* input */
    char *filename, /* filename to write to (stdout if NULL) */
    char *title,    /* title (72 char max), may be NULL */
    char *key,      /* key (8 char max), may be NULL */
    Int nrow,       /* A is nrow-by-ncol */
    Int ncol,
    Int *Ap,        /* size ncol+1, column pointers */
    Int *Ai,        /* size anz=Ap[ncol], row indices (sorted) */
    double *Ax,     /* size anz or 2*anz, numerical values (binary if NULL) */
    double *Az,     /* size anz, imaginary part (real if NULL) */
    Int *Zp,        /* size ncol+1, column pointers for Z (or NULL) */
    Int *Zi,        /* size znz=Zp[ncol], row indices for Z (or NULL) */
    Int mkind_in,   /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */

    /* output */
    char mtype [4]  /* matrix type (RUA, RSA, etc), may be NULL */
)
{
    double xmin, xmax, zmin, zmax ;
    Int *w, *cp ;
    FILE *file = NULL ;     /* write to stdout if NULL */
    Int mkind, skind, zmkind, zskind, nnz2, vals, valn, indn, ptrn, valcrd,
        indcrd, ptrcrd, totcrd, anz, is_int, fmt, nbuf, j, njumbled, nzeros,
        znz, asize, status, nelnz = 0 ;
    int ok ;
    char zmtype [4], indfmt [21], indcfm [21], valfmt [21], valcfm [21],
        ptrfmt [21], ptrcfm [21], mtype2 [4] ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (!title || !key || !Ap || !Ax)
    {
        /* one or more required arguments are missing (NULL) */
        return (RBIO_ARG_ERROR) ;
    }

    mkind_in = RB(fix_mkind_in) (mkind_in, Ax, Az) ;

    anz = Ap ? (Ap [MAX (ncol,0)]) : 1 ;
    status = RB(ok) (nrow, ncol, anz, Ap, Ai, Ax, Az, NULL, mkind_in,
        &njumbled, &nzeros) ;
    if (status != RBIO_OK)
    {
        /* A matrix is corrupted */
        return (status) ;
    }

    if (Zp != NULL)
    {
        znz = Zp [MAX (ncol,0)] ;
        status = RB(ok) (nrow, ncol, znz, Zp, Zi, NULL, NULL, NULL, 3,
            &njumbled, &nzeros) ;
        if (status != RBIO_OK)
        {
            /* Z matrix is corrupted */
            return (status) ;
        }
    }

    if (mtype == NULL)
    {
        /* use internal array for mtype; do not return to caller */
        mtype = mtype2 ;
    }

    /* ---------------------------------------------------------------------- */
    /* clear the format strings */
    /* ---------------------------------------------------------------------- */

    RB(fill) (valfmt, 20, ' ') ;
    RB(fill) (valcfm, 20, ' ') ;
    RB(fill) (indfmt, 20, ' ') ;
    RB(fill) (indcfm, 20, ' ') ;
    RB(fill) (ptrfmt, 20, ' ') ;
    RB(fill) (ptrcfm, 20, ' ') ;
    indn = 0 ;
    valn = 0 ;
    ptrn = 0 ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    w  = SuiteSparse_malloc (MAX (nrow, ncol) + 1, sizeof (Int)) ;
    cp = SuiteSparse_malloc (ncol + 1, sizeof (Int)) ;
    if (cp == NULL || w == NULL)
    {
        FREE_WORK ;
        return (RBIO_OUT_OF_MEMORY) ;
    }

    /* ---------------------------------------------------------------------- */
    /* determine the matrix type (RSA, RUA, etc) of A+Z */
    /* ---------------------------------------------------------------------- */

    RB(kind) (nrow, ncol, Ap, Ai, Ax, Az, mkind_in, &mkind, &skind, mtype,
        &xmin, &xmax, cp) ;

    /* now use mkind instead of mkind_in */

    if (Zp != NULL && Zp [ncol] == 0)
    {
        /* Z has no entries; ignore it */
        Zp = NULL ;
        Zi = NULL ;
    }

    if (Zp != NULL)
    {
        /* determine if Z is symmetric or not */
        RB(kind) (nrow, ncol, Zp, Zi, NULL, NULL, 3, &zmkind, &zskind, zmtype,
            &zmin, &zmax, cp) ;
        if (zskind == 0)
        {
            /* Z is square and unsymmetric; force A unsymmetric too */
            mtype [1] = 'u' ;
            skind = 0 ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* determine the required precision for the numerical values */
    /* ---------------------------------------------------------------------- */

    asize = ((mkind == 4) ? 2 : 1) * anz ;
    is_int = (mkind == 3) ;
    if (mkind != 1)
    {
        /* A is real, split-complex, integer or merged-complex: check Ax */
        fmt = RB(format) (asize, Ax, is_int, xmin, xmax, 0,
            valfmt, valcfm, &valn) ;
    }
    if (mkind == 2)
    {
        /* A is split-complex: check Az */
        fmt = RB(format) (anz, Az, FALSE, 0, 0, fmt, valfmt, valcfm, &valn) ;
    }

    /* ---------------------------------------------------------------------- */
    /* determine the number of entries in the matrix A+Z */
    /* ---------------------------------------------------------------------- */

    /* task 1 does not write to the file */
    ok = RB(writeTask) (NULL, 1, nrow, ncol, mkind, skind, Ap, Ai, Ax, Az, Zp,
        Zi, indcfm, indn, valcfm, valn, &nnz2, w, cp) ;
    if (nnz2 <= 0)
    {
        FREE_WORK ;
        return (RBIO_DIM_INVALID) ;   /* matrix has no entries to print */
    }

    /* determine pointer format.  ncol+1 integers, in range 1 to nnz2+1 */
    RB(iformat) (1, nnz2+1, ptrfmt, ptrcfm, &ptrn) ;
    ptrcrd = RB(cards) (ncol+1, ptrn) ;

    /* determine row index format.  nnz2 integers, in range 1 to nrow */
    RB(iformat) (1, nrow, indfmt, indcfm, &indn) ;
    indcrd = RB(cards) (nnz2, indn) ;

    /* determine how many lines for the numerical values */
    if (mkind == 0 || mkind == 3)
    {
        /* real or integer */
        vals = 1 ;
    }
    else if (mkind == 1)
    {
        /* pattern */
        vals = 0 ;
    }
    else
    {
        /* complex (merged or split) */
        vals = 2 ;
    }
    valcrd = RB(cards) (vals*nnz2, valn) ;

    /* ---------------------------------------------------------------------- */
    /* determine total number of cards */
    /* ---------------------------------------------------------------------- */

    totcrd = ptrcrd + indcrd + valcrd ;

    /* ---------------------------------------------------------------------- */
    /* open the file */
    /* ---------------------------------------------------------------------- */

    if (filename)
    {
        file = fopen (filename, "w") ;
        if (file == NULL)
        {
            FREE_WORK ;
            return (RBIO_FILE_IOERROR) ;      /* cannot open file */
        }
    }

    /* ---------------------------------------------------------------------- */
    /* write the header */
    /* ---------------------------------------------------------------------- */

    ok = fprintf (file, "%-71.71s|%-8.8s\n",
        title ? title : "", key ? key : "") > 0;
    ok = ok && fprintf (file, "%14" IDD "%14" IDD "%14" IDD "%14" IDD "\n",
        totcrd, ptrcrd, indcrd, valcrd) > 0 ;
    ok = ok && fprintf (file,
        "%3s           %14" IDD "%14" IDD "%14" IDD "%14" IDD "\n",
        mtype, nrow, ncol, nnz2, nelnz) > 0 ;
    ok = ok && fprintf (file, "%.16s%.16s%.20s\n", ptrfmt, indfmt, valfmt) > 0 ;
    if (!ok)
    {
        /* file I/O error */
        FREE_WORK ;
        if (filename) fclose (file) ;
        return (RBIO_HEADER_IOERROR) ;
    }

    /* ---------------------------------------------------------------------- */
    /* write the column pointers (convert to 1-based) */
    /* ---------------------------------------------------------------------- */

    nbuf = 0 ;
    for (j = 0 ; ok && j <= ncol ; j++)
    {
        ok = RB(iprint) (file, ptrcfm, 1+ cp [j], ptrn, &nbuf) ;
    }
    ok = ok && fprintf (file, "\n") > 0 ;
    if (!ok)
    {
        /* file I/O error */
        FREE_WORK ;
        if (filename) fclose (file) ;
        return (RBIO_CP_IOERROR) ;
    }

    /* ---------------------------------------------------------------------- */
    /* write the row indices (convert to 1-based) */
    /* ---------------------------------------------------------------------- */

    ok = RB(writeTask) (file, 2, nrow, ncol, mkind, skind, Ap, Ai, Ax, Az,
        Zp, Zi, indcfm, indn, valcfm, valn, &nnz2, w, cp) ;
    if (!ok)
    {
        /* file I/O error */
        FREE_WORK ;
        if (filename) fclose (file) ;
        return (RBIO_ROW_IOERROR) ;
    }

    /* ---------------------------------------------------------------------- */
    /* write the numerical values */
    /* ---------------------------------------------------------------------- */

    if (mkind != 1)
    {
        ok = RB(writeTask) (file, 3, nrow, ncol, mkind, skind, Ap, Ai,
            Ax, Az, Zp, Zi, indcfm, indn, valcfm, valn, &nnz2, w, cp) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace and close the file */
    /* ---------------------------------------------------------------------- */

    FREE_WORK ;
    if (filename) fclose (file) ;
    return (ok ? RBIO_OK : RBIO_VALUE_IOERROR) ;
}


/* -------------------------------------------------------------------------- */
/* RBkind: determine the type of a sparse matrix */
/* -------------------------------------------------------------------------- */

PUBLIC Int RB(kind)          /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    Int nrow,       /* A is nrow-by-ncol */
    Int ncol,
    Int *Ap,        /* Ap [0...ncol]: column pointers */
    Int *Ai,        /* Ai [0...nnz-1]: row indices */
    double *Ax,     /* Ax [0...nnz-1]: real values.  Az holds imaginary part */
    double *Az,     /* if real, Az is NULL. if complex, Az is non-NULL */
    Int mkind_in,   /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */

    /* output */
    Int *mkind,     /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    Int *skind,     /* r: -1 (rectangular), u: 0 (unsymmetric), s: 1 symmetric,
                       h: 2 (Hermitian), z: 3 (skew symmetric) */
    char mtype [4], /* rua, psa, rra, cha, etc */
    double *xmin,   /* smallest value */
    double *xmax,   /* largest value */

    /* workspace: allocated internally if NULL */
    Int *cp         /* workspace of size ncol+1, undefined on input and output*/
)
{
    Int nnz, is_h, is_z, is_s, k, p, i, j, pt, get_workspace ;
    Int *w = NULL ;
    double aij_real, aij_imag, aji_real, aji_imag ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (!Ap || !Ai || !mkind || !skind || !mtype || !xmin || !xmax
        || ncol < 0 || nrow < 0)
    {
        /* one or more required arguments are missing (NULL) or invalid */
        return (RBIO_ARG_ERROR) ;
    }

    /* ---------------------------------------------------------------------- */
    /* allocate workspace, if needed */
    /* ---------------------------------------------------------------------- */

    get_workspace = (cp == NULL) ;
    if (get_workspace)
    {
        cp = (Int *) SuiteSparse_malloc (ncol + 1, sizeof (Int)) ;
    }
    if (cp == NULL)
    {
        return (RBIO_OUT_OF_MEMORY) ;
    }

    /* ---------------------------------------------------------------------- */
    /* determine numeric type (I*A, R*A, P*A, C*A) */
    /* ---------------------------------------------------------------------- */

    mkind_in = RB(fix_mkind_in) (mkind_in, Ax, Az) ;

    mtype [3] = '\0' ;
    nnz = Ap [ncol] ;
    *xmin = 0 ;
    *xmax = 0 ;

    if (mkind_in == 2 || mkind_in == 4)
    {
        /* A is a complex matrix (C*A), split or merged */
        mtype [0] = 'c' ;
        *mkind = mkind_in ;
    }
    else if (mkind_in == 1)
    {
        /* A is a pattern-only matrix (P*A) */
        mtype [0] = 'p' ;
        *mkind = 1 ;
    }
    else
    {
        /* The input matrix A is said to be 0:real or 3:integer, */
        /* Ax is not NULL */

        /* select P** format if all entries are equal to 1 */
        /* select I** format if all entries are integer and */
        /* between -99,999,999 and +999,999,999 */

        Int is_p = TRUE ;
        Int is_int = TRUE ;
        *xmin = Ax [0] ;
        *xmax = Ax [0] ;

        for (p = 0 ; (is_p || is_int) && p < nnz ; p++)
        {
            if (is_p)
            {
                if (Ax [p] != 1)
                {
                    is_p = FALSE ;
                }
            }
            if (is_int)
            {
                double x = Ax [p] ;
                k = (Int) x ;
                *xmin = MIN (x, *xmin) ;
                *xmax = MAX (x, *xmax) ;
                if (((double) k) != x)
                {
                    /* the entry is not an integer */
                    is_int = FALSE ;
                }
		if (x < -99999999 || x >= 999999999) 
                {
                    /* use real format for really big integers */
                    is_int = FALSE ;
                }
            }
        }

        if (is_p)
        {
            /* pattern-only matrix (P*A) */
            mtype [0] = 'p' ;
            *mkind = 1 ;
        }
        else if (is_int)
        {
            /* integer matrix (I*A) */
            mtype [0] = 'i' ;
            *mkind = 3 ;
        }
        else
        {
            /* real matrix (R*A) */
            mtype [0] = 'r' ;
            *mkind = 0 ;
        }
    }

    /* only assembled matrices are handled */
    mtype [2] = 'a' ;

    /* ---------------------------------------------------------------------- */
    /* determine symmetry (*RA, *UA, *SA, *HA, *ZA) */
    /* ---------------------------------------------------------------------- */

    /* Note that A must have sorted columns for this method to work.  */

    if (nrow != ncol)
    {
        /* rectangular matrix (*RA), no need to check values or pattern */
        mtype [1] = 'r' ;
        *skind = -1 ;
        if (get_workspace) FREE_WORK ;
        return (RBIO_OK) ;
    }

    /* if complex, the matrix is Hermitian until proven otherwise */
    is_h = (*mkind == 2 || *mkind == 4) ;

    /* the matrix is symmetric until proven otherwise */
    is_s = TRUE ;

    /* a non-pattern matrix is skew symmetric until proven otherwise */
    is_z = (*mkind != 1) ;

    /* if this method returns early, the matrix is unsymmetric */
    mtype [1] = 'u' ;
    *skind = 0 ;

    /* initialize the munch pointers (cp) */
    for (j = 0 ; j <= ncol ; j++)
    {
        cp [j] = Ap [j] ;
    }

    for (j = 0 ; j < ncol ; j++)
    {

        /* consider all entries not yet munched in column j */
        for (p = cp [j] ; p < Ap [j+1] ; p++)
        {
            /* get the row index of A(i,j) */
            i = Ai [p] ;

            if (i < j)
            {
                /* entry A(i,j) is unmatched, matrix is unsymmetric */
                if (get_workspace) FREE_WORK ;
                return (RBIO_OK) ;
            }

            /* get the A(j,i) entry, if it exists, and munch it */
            pt = cp [i]++ ;

            if (pt >= Ap [i+1] || Ai [pt] != j)
            {
                /* entry A(j,i) doesn't exist, matrix is unsymmetric */
                if (get_workspace) FREE_WORK ;
                return (RBIO_OK) ;
            }

            /* A(j,i) exists; check its value with A(i,j) */
            RB(get_entry) (*mkind, Ax, Az, p,  &aij_real, &aij_imag) ;
            RB(get_entry) (*mkind, Ax, Az, pt, &aji_real, &aji_imag) ;

            if (aij_real != aji_real || aij_imag != aji_imag)
            {
                is_s = FALSE ;      /* the matrix cannot be *SA */
            }
            if (aij_real != -aji_real || aij_imag != -aji_imag)
            {
                is_z = FALSE ;      /* the matrix cannot be *ZA */
            }
            if (aij_real != aji_real || aij_imag != -aji_imag)
            {
                is_h = FALSE ;      /* the matrix cannot be *HA */
            }

            if (! (is_s || is_z || is_h))
            {
                /* matrix is unsymmetric; terminate the test */
                if (get_workspace) FREE_WORK ;
                return (RBIO_OK) ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* return the symmetry */
    /* ---------------------------------------------------------------------- */

    if (is_h)
    {
        /* Hermitian matrix (*HA) */
        mtype [1] = 'h' ;
        *skind = 2 ;
    }
    else if (is_s)
    {
        /* symmetric matrix (*SA) */
        mtype [1] = 's' ;
        *skind = 1 ;
    }
    else if (is_z)
    {
        /* skew symmetric matrix (*ZA) */
        mtype [1] = 'z' ;
        *skind = 3 ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace, if allocated */
    /* ---------------------------------------------------------------------- */

    if (get_workspace) FREE_WORK ;
    return (RBIO_OK) ;
}


/* -------------------------------------------------------------------------- */
/* RBwrite data formats */
/* -------------------------------------------------------------------------- */

#define NFORMAT 19

/* Fortran formats */
static char *F_format [NFORMAT] = {
    "(8E9.1)             ",
    "(8E10.2)            ",
    "(7E11.3)            ",
    "(6E12.4)            ",
    "(6E13.5)            ",
    "(5E14.6)            ",
    "(5E15.7)            ",
    "(5E16.8)            ",
    "(4E17.9)            ",
    "(4E18.10)           ",
    "(4E19.11)           ",
    "(4E20.12)           ",
    "(3E21.13)           ",
    "(3E22.14)           ",
    "(3E23.15)           ",
    "(3E24.16)           ",
    "(3E25.17)           ",
    "(3E26.18)           ",
    "(2E30.18E3)         " } ;

/* corresponding C formats to use */
static char *C_format [NFORMAT] = {
    "%9.1E",
    "%10.2E",
    "%11.3E",
    "%12.4E",
    "%13.5E",
    "%14.6E",
    "%15.7E",
    "%16.8E",
    "%17.9E",
    "%18.10E",
    "%19.11E",
    "%20.12E",
    "%21.13E",
    "%22.14E",
    "%23.15E",
    "%24.16E",
    "%25.17E",
    "%26.18E",
    "%30.18E" } ;

/* Number of entries per line for each of the formats */
static Int entries_per_line [NFORMAT] = {
    8,
    8,
    7,
    6,
    6,
    5,
    5,
    5,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    3,
    2} ;


/* -------------------------------------------------------------------------- */
/* RBformat: determine the format required for an array of values */
/* -------------------------------------------------------------------------- */

/*
    This function ensures that a sufficiently wide format is used that
    accurately represent the data.  It also ensures that when printed,
    the numerical values all have at least one blank space between them.
*/

PRIVATE Int RB(format)  /* return format to use (index in F_, C_format) */
(
    /* input */
    Int nnz,            /* number of nonzeros */
    double *x,          /* of size nnz */
    Int is_int,         /* true if integer format is to be used */
    double xmin,        /* minimum value of x */
    double xmax,        /* maximum value of x */
    Int fmt,            /* initial format to use (index into F_format, ...) */

    /* output */
    char valfmt [21],   /* Fortran format to use */
    char valcfm [21],   /* C format to use */
    Int *valn           /* number of entries per line */
)
{
    Int i ;
    double a, b ;
    char s [1024] ;

    if (is_int)
    {

        /* ------------------------------------------------------------------ */
        /* use an integer format */
        /* ------------------------------------------------------------------ */

        RB(iformat) (xmin, xmax, valfmt, valcfm, valn) ;
        return (-1) ;

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* find the required precision for a real or complex matrix */
        /* ------------------------------------------------------------------ */

        fmt = 0 ;
        for (i = 0 ; i < nnz ; i++)
        {

            /* determine if the matrix has huge values, tiny values, or NaN's */
            a = ABS (x [i]) ;
            if (a != 0)
            {
                if (ISNAN (a) || a < 1e-90 || a > 1e90)
                {
                    fmt = NFORMAT-1 ;
                    break ;
                }
            }

            a = x ? x [i] : 1 ;
            for ( ; fmt < NFORMAT-1 ; fmt++)
            {
                /* write the value to a string, read back in, and check,
                 * using the kth format */
                sprintf (s, C_format [fmt], a) ;
                b = 0 ;
                sscanf (s, "%lg", &b) ;
                if (s [0] == ' ' && a == b)
                {
                    /* success, use this format (or wider) for all numbers */
                    break ;
                }
            }
        }

        strncpy (valfmt, F_format [fmt], 21) ;
        strncpy (valcfm, C_format [fmt], 21) ;
        *valn = entries_per_line [fmt] ;
        return (fmt) ;
    }
}


/* -------------------------------------------------------------------------- */
/* RBwriteTask: write portions of the matrix to the file */
/* -------------------------------------------------------------------------- */

/*
   task 0: just count the total number of entries in the matrix.  No file I/O.
   task 1: do task 0, and also construct w and cp.  No file I/O.
   task 2: write the row indices
   task 3: write the numerical values
*/

PRIVATE Int RB(writeTask)     /* returns TRUE if successful, FALSE on failure */
(
    /* input */
    FILE *file,     /* file to print to (already open) */
    Int task,       /* 0 to 3 (see above) */
    Int nrow,       /* A is nrow-by-ncol */
    Int ncol,
    Int mkind,      /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    Int skind,      /* -1:rect, 0:unsym, 1:sym, 2:hermitian, 3:skew */
    Int *Ap,        /* size ncol+1, column pointers */
    Int *Ai,        /* size anz=Ap[ncol], row indices */
    double *Ax,     /* size anz, real values */
    double *Az,     /* size anz, imaginary part (may be NULL) */
    Int *Zp,        /* size ncol+1, column pointers for Z (may be NULL) */
    Int *Zi,        /* size Zp[ncol], row indices for Z */
    char *indcfm,   /* C format for indices */
    Int indn,       /* # of indices per line */
    char *valcfm,   /* C format for values */
    Int valn,       /* # of values per line */

    /* output */
    Int *nnz,           /* number of entries that will be printed to the file */

    /* workspace */
    Int *w,         /* size MAX(nrow,ncol)+1 */
    Int *cp         /* size ncol+1 */
)
{
    double xr, xz ;
    Int j, pa, pz, paend, pzend, ia, iz, i, nbuf, ok ;

    /* ---------------------------------------------------------------------- */
    /* clear the nonzero counts */
    /* ---------------------------------------------------------------------- */

    *nnz = 0 ;
    for (j = 0 ; j < ncol ; j++)
    {
        w [j] = 0 ;
    }

    /* ---------------------------------------------------------------------- */
    /* print, or count, each column */
    /* ---------------------------------------------------------------------- */

    nbuf = 0 ;      /* number of characters in current line */
    ok = TRUE ;
    for (j = 0 ; ok && j < ncol ; j++)
    {

        /* find the set union of A (:,j) and Z (:,j) */
        pa = Ap [j] ;
        pz = Zp ? Zp [j] : 0 ;
        paend = Ap [j+1] ;
        pzend = Zp ? Zp [j+1] : 0 ;

        /* repeat while entries still exist in A(:,j) or Z(:,j) */
        while (ok)
        {
            /* get the next entry from A(:,j) */
            ia = (pa < paend) ? Ai [pa] : nrow ;

            /* get the next entry from Z(:,j) */
            iz = (pz < pzend) ? Zi [pz] : nrow ;

            /* exit loop if neither entry is present */
            if (ia >= nrow && iz >= nrow) break ;

            if (ia < iz)
            {
                /* get A (i,j) */
                i = ia ;
                RB(get_entry) (mkind, Ax, Az, pa, &xr, &xz) ;
                pa++ ;
            }
            else if (iz < ia)
            {
                /* get Z (i,j) */
                i = iz ;
                xr = 0 ;
                xz = 0 ;
                pz++ ;
            }
            else
            {
                /* get A (i,j), and delete its matched Z(i,j) */
                i = ia ;
                RB(get_entry) (mkind, Ax, Az, pa, &xr, &xz) ;
                pa++ ;
                pz++ ;
            }

            if (skind <= 0 || i >= j)
            {
                /* consider the (i,j) entry with value (xr,xz) */
                (*nnz)++ ;
                if (task == 1)
                {
                    /* only determining nonzero counts */
                    w [j]++ ;
                }
                else if (task == 2)
                {
                    /* printing the row indices (convert to 1-based) */
                    ok = RB(iprint) (file, indcfm, 1+ i, indn, &nbuf) ;
                }
                else if (task == 3)
                {
                    /* printing the numerical values */
                    ok = RB(xprint) (file, valcfm, xr, valn, mkind, &nbuf) ;
                    if (ok && (mkind == 2 || mkind == 4))
                    {
                        ok = RB(xprint) (file, valcfm, xz, valn, mkind, &nbuf) ;
                    }
                }
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* determine the new column pointers, or finish printing */
    /* ---------------------------------------------------------------------- */

    if (ok)
    {
        if (task == 1)
        {
            cp [0] = 0 ;
            for (j = 0 ; j < ncol ; j++)
            {
                cp [j+1] = cp [j] + w [j] ;
            }
        }
        else
        {
            ok = (fprintf (file ? file : stdout, "\n") > 0) ;
        }
    }
    return (ok) ;
}


/* -------------------------------------------------------------------------- */
/* RBiprint: print one integer value to the file */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(iprint)        /* returns TRUE if OK, FALSE otherwise */
(
    /* input */
    FILE *file,             /* which file to write to */
    char *indcfm,           /* C format to use */
    Int i,                  /* value to write */
    Int indn,               /* number of entries to write per line */

    /* input/output */
    Int *nbuf               /* number of entries written to current line */
)
{
    Int ok = TRUE ;
    if (!file) file = stdout ;                  /* file defaults to stdout */
    if (*nbuf >= indn)
    {
        *nbuf = 0 ;
        ok = (fprintf (file, "\n") > 0) ;
    }
    ok = ok && (fprintf (file, indcfm, i) > 0) ;
    (*nbuf)++ ;
    return (ok) ;
}


/* -------------------------------------------------------------------------- */
/* RBxprint: print one real value to the file */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(xprint)    /* returns TRUE if OK, FALSE otherwise */
(
    /* input */
    FILE *file,         /* which file to write to */
    char *valcfm,       /* C format to use */
    double x,           /* value to write */
    Int valn,           /* number of entries to write per line */
    Int mkind,          /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */

    /* input/output */
    Int *nbuf           /* number of entries written to current line */
)
{
    Int ok = TRUE ;
    if (!file) file = stdout ;                  /* file defaults to stdout */
    if (mkind == 3)
    {
        /* write out the value as an integer */
        ok = RB(iprint) (file, valcfm, (Int) x, valn, nbuf) ;
    }
    else
    {
        /* write out the value as a real */
        if (*nbuf >= valn)
        {
            *nbuf = 0 ;
            ok = (fprintf (file, "\n") > 0) ;
        }
        ok = ok && (fprintf (file, valcfm, x) > 0) ;
        (*nbuf)++ ;
    }
    return (ok) ;
}


/* -------------------------------------------------------------------------- */
/* RBiformat: determine format for printing an integer */
/* -------------------------------------------------------------------------- */

PRIVATE void RB(iformat)
(
    /* input */
    double xmin,            /* smallest integer to print */
    double xmax,            /* largest integer to print */

    /* output */
    char indfmt [21],       /* Fortran format to use */
    char indcfm [21],       /* C format to use */
    Int *indn               /* number of entries per line */
)
{
    if (xmin >= 0 && xmax <= 9.)
    {
        strncpy (indfmt, "(40I2)              ", 21) ;
        strncpy (indcfm, "%2" IDD              , 21) ;
        *indn = 40 ;
    }
    else if (xmin >= -9. && xmax <= 99.)
    {
        strncpy (indfmt, "(26I3)              ", 21) ;
        strncpy (indcfm, "%3" IDD              , 21) ;
        *indn = 26 ;
    }
    else if (xmin >= -99. && xmax <= 999.)
    {
        strncpy (indfmt, "(20I4)              ", 21) ;
        strncpy (indcfm, "%4" IDD              , 21) ;
        *indn = 20 ;
    }
    else if (xmin >= -999. && xmax <= 9999.)
    {
        strncpy (indfmt, "(16I5)              ", 21) ;
        strncpy (indcfm, "%5" IDD              , 21) ;
        *indn = 16 ;
    }
    else if (xmin >= -9999. && xmax <= 99999.)
    {
        strncpy (indfmt, "(13I6)              ", 21) ;
        strncpy (indcfm, "%6" IDD              , 21) ;
        *indn = 13 ;
    }
    else if (xmin >= -99999. && xmax <= 999999.)
    {
        strncpy (indfmt, "(11I7)              ", 21) ;
        strncpy (indcfm, "%7" IDD              , 21) ;
        *indn = 11 ;
    }
    else if (xmin >= -999999. && xmax <= 9999999.)
    {
        strncpy (indfmt, "(10I8)              ", 21) ;
        strncpy (indcfm, "%8" IDD              , 21) ;
        *indn = 10 ;
    }
    else if (xmin >= -9999999. && xmax <= 99999999.)
    {
        strncpy (indfmt, "(8I9)                ", 21) ;
        strncpy (indcfm, "%9" IDD              , 21) ;
        *indn = 8 ;
    }
    else if (xmin >= -99999999. && xmax <= 999999999.)
    {
        strncpy (indfmt, "(8I10)              ", 21) ;
        strncpy (indcfm, "%10" IDD             , 21) ;
        *indn = 8 ;
    }
    else if (xmin >= -999999999. && xmax <= 9999999999.)
    {
        strncpy (indfmt, "(7I11)              ", 21) ;
        strncpy (indcfm, "%11" IDD             , 21) ;
        *indn = 7 ;
    }
    else if (xmin >= -9999999999. && xmax <= 99999999999.)
    {
        strncpy (indfmt, "(6I12)              ", 21) ;
        strncpy (indcfm, "%12" IDD             , 21) ;
        *indn = 6 ;
    }
    else if (xmin >= -99999999999. && xmax <= 999999999999.)
    {
        strncpy (indfmt, "(6I13)              ", 21) ;
        strncpy (indcfm, "%13" IDD             , 21) ;
        *indn = 6 ;
    }
    else if (xmin >= -999999999999. && xmax <= 9999999999999.)
    {
        strncpy (indfmt, "(5I14)              ", 21) ;
        strncpy (indcfm, "%14" IDD             , 21) ;
        *indn = 5 ;
    }
    else
    {
        strncpy (indfmt, "(5I15)              ", 21) ;
        strncpy (indcfm, "%15" IDD             , 21) ;
        *indn = 5 ;
    }
}


/* -------------------------------------------------------------------------- */
/* RBcards: determine number of cards required */
/* -------------------------------------------------------------------------- */

PRIVATE Int RB(cards)
(
    Int nitems,         /* number of items to print */
    Int nperline        /* number of items per line */
)
{
    return ((nitems == 0) ? 0 : (((nitems-1) / nperline) + 1)) ;
}


/* -------------------------------------------------------------------------- */
/* RBfill: fill a string */
/* -------------------------------------------------------------------------- */

PRIVATE void RB(fill)
(
    char *s,            /* string to fill */
    Int len,            /* length of s (including trailing '\0') */
    char c              /* character to fill s with */
)
{
    Int i ;
    for (i = 0 ; i < len ; i++)
    {
        s [i] = c ;
    }
    s [len-1] = '\0' ;
}


/* -------------------------------------------------------------------------- */
/* RBok: verify a sparse matrix */
/* -------------------------------------------------------------------------- */

PUBLIC Int RB(ok)            /* 0:OK, < 0: error, > 0: warning */
(
    /* inputs, not modified */
    Int nrow,       /* number of rows */
    Int ncol,       /* number of columns */
    Int nzmax,      /* max # of entries */
    Int *Ap,        /* size ncol+1, column pointers */
    Int *Ai,        /* size nz = Ap [ncol], row indices */
    double *Ax,     /* real part, or both if merged-complex */
    double *Az,     /* imaginary part for split-complex */
    char *As,       /* logical matrices (useful for MATLAB caller only) */
    Int mkind,      /* 0:real, 1:logical/pattern, 2:split-complex, 3:integer,
                       4:merged-complex */

    /* outputs, not defined on input */
    Int *p_njumbled,   /* # of jumbled row indices (-1 if not computed) */
    Int *p_nzeros      /* number of explicit zeros (-1 if not computed) */
)
{
    double xr, xz ;
    Int i, j, p, pend, njumbled, nzeros, ilast ;

    /* ---------------------------------------------------------------------- */
    /* in case of early return */
    /* ---------------------------------------------------------------------- */

    if (p_njumbled) *p_njumbled = -1 ;
    if (p_nzeros  ) *p_nzeros = -1 ;

    if (mkind < 0 || mkind > 4)
    {
        return (RBIO_MKIND_INVALID) ;
    }

    /* ---------------------------------------------------------------------- */
    /* check the dimensions */
    /* ---------------------------------------------------------------------- */

    if (nrow < 0 || ncol < 0 || nzmax < 0)
    {
        return (RBIO_DIM_INVALID) ;
    }

    /* ---------------------------------------------------------------------- */
    /* check the column pointers */
    /* ---------------------------------------------------------------------- */

    if (Ap == NULL || Ap [0] != 0)
    {
        /* column pointers invalid */
        return (RBIO_CP_INVALID) ;
    }
    for (j = 0 ; j < ncol ; j++)
    {
        p = Ap [j] ;
        pend = Ap [j+1] ;
        if (pend < p || pend > nzmax)
        {
            /* column pointers not monotonically non-decreasing */
            return (RBIO_CP_INVALID) ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* check the row indices and numerical values */
    /* ---------------------------------------------------------------------- */

    if (Ai == NULL)
    {
        /* row indices not present */
        return (RBIO_ROW_INVALID) ;
    }

    njumbled = 0 ;
    nzeros = 0 ;

    for (j = 0 ; j < ncol ; j++)
    {
        ilast = -1 ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            if (i < 0 || i >= nrow)
            {
                /* row indices out of range */
                return (RBIO_ROW_INVALID) ;
            }
            if (i <= ilast)
            {
                /* row indices unsorted, or duplicates present */
                njumbled++ ;
            }
            if (mkind == 1 && As)
            {
                xr = (double) (As [p]) ;
                xz = 0 ;
            }
            else
            {
                RB(get_entry) (mkind, Ax, Az, p, &xr, &xz) ;
            }
            if (xr == 0 && xz == 0)
            {
                /* an explicit zero is present */
                nzeros++ ;
            }
            ilast = i ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* return results */
    /* ---------------------------------------------------------------------- */

    if (p_njumbled) *p_njumbled = njumbled ;
    if (p_nzeros  ) *p_nzeros = nzeros ;
    return ((njumbled > 0) ? RBIO_JUMBLED : RBIO_OK) ;
}

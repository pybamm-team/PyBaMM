/* ========================================================================== */
/* === metis_graph_mex ====================================================== */
/* ========================================================================== */

/*
METIS_GRAPH_MEX: reads a graph in METIS format.  This function is not normally
meant to be called directly by the user.  Use metis_graph.m instead.

    [i j x w fmt] = metis_graph_mex (graph_filename)

Reads a graph file in METIS format, returning the result as list of entries in
triplet form.  The first line of a METIS *.graph file specifies the format.  It
can be one of:

    n nz            # of nodes and # of edges
    n nz fmt        fmt defaults to 0 if not present
    n nz fmt ncon   ncon defaults to 0 if not present and fmt = 0, 1, or 100,
                    or 1 if fmt = 10 or 11.

    fmt:
        0:  no edge or node weights
        1:  no node weights, has edge weights
        10: has node weights, no edge weights
        11: has both node and edge weights
        100: graph may include self-edges and multiple edges (a multigraph)
            This is an extension to the METIS format.  No edge weights allowed.
            The fmt=100 option is treated just like fmt=0 here, since this
            mexFunction does not check for self-edges or duplicate/multiple
            edges.  That is done in metis_graph.m.

    ncon:  the number of weights associated with each node

The next n lines contain the adjacency list for each node.  Each line has the
following format.  Suppose the ith line contains:

    w1 w2 ... wncon  v1 e1 v2 e2 ... vk ek

then w1 to wncon are the node weights for node i, v1 is a node number and e1 is
the edge weight for the edge (i,v1).  Node i has k neighbors.
On output, w is an n-by-ncon dense matrix of node weights.

The nz = # of edges given in line 1 counts each edge (i,j) and (j,i) just once.
However, this mexFunction treats nz just as a hint.  The output vectors i, j,
and x are resized as needed.

[i j x] is a list of triplets, in arbitary order.  The kth triplet is an edge
between node i(k) and node j(k) with edge weight x(k).  For a graph with no
edge weights (fmt = 0, 10, or 100), x = 1 (a scalar) is returned.

Edge weights must be strictly > 0.  Node weights must be integers >= 0.
These conditions are not tested here, but in metis_graph.m.

If ncon = size(w,2) > 0 on output, then the graph also has node weights.
*/

#include "mex.h"
#include "stdio.h"
#include "ctype.h"
#define LEN 2000
#define Int mwSignedIndex
#define MAX(a,b) (((a) > (b)) ? (a) : (b))


/* ========================================================================== */
/* === get_token ============================================================ */
/* ========================================================================== */

/* get the next token from the file */

Int get_token           /* returns -1 on EOF, 0 on end of line, 1 otherwise */
(
    FILE *f,            /* file open for reading */
    char *s,            /* array of size maxlen */
    Int maxlen
)
{
    int c = ' ' ;
    Int len = 0 ;
    s [0] = '\0' ;      /* in case of early return */

    /* skip leading white space */
    while (isspace (c))
    {
        c = fgetc (f) ;
        if (c == EOF) return (-1) ;
        if (c == '\n') return (0) ;
    }

    /* read the token */
    while (c != EOF && !isspace (c) && len < maxlen)
    {
        s [len++] = c ;
        c = fgetc (f) ;
    }

    /* skip any trailing white space, stopping at the newline if found */
    while (c != EOF && c != '\n' && isspace (c))
    {
        c = fgetc (f) ;
    }
    if (c != EOF && c != '\n')
    {
        /* push back a valid character for the next token */
        ungetc (c, f) ;
    }

    /* terminate the string */
    s [len] = '\0' ;

    return ((c == '\n') ? 0 : 1) ;
}

/* ========================================================================== */
/* === eat_comments ========================================================= */
/* ========================================================================== */

/* remove any comment lines (first character is a '%') */
void eat_comments (FILE *f)
{
    int c ;
    while ((c = fgetc (f)) == '%')
    {
        while (1)
        {
            c = fgetc (f) ;
            if (c == EOF || c == '\n') break ;
        }
    }
    if (c != EOF)
    {
        ungetc (c, f) ;
    }
}

/* ========================================================================== */
/* === mexFunction ========================================================== */
/* ========================================================================== */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    FILE *f ;
    char s [LEN+1], msg [LEN+1] ;
    Int n, nzmax, fmt, ncon, nz, has_ew, i, j, k, status, t ;
    double e = 1, x, *Ti, *Tj, *Tx, *W, x1 = 0, x2 = 0, x3 = 0, x4 = 0 ;

    /* ---------------------------------------------------------------------- */
    /* get the filename */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 5 || !mxIsChar (pargin [0]))
    {
        mexErrMsgIdAndTxt ("metis_graph:usage",
            "usage: [i j x w fmt] = metis_graph_read_mex (filename)") ;
    }
    mxGetString (pargin [0], s, LEN) ;

    /* ---------------------------------------------------------------------- */
    /* open the file and remove leading comments */
    /* ---------------------------------------------------------------------- */

    f = fopen (s, "r") ;
    if (f == NULL)
    {
        sprintf (msg, "unable to open file: %s", s) ;
        mexErrMsgIdAndTxt ("metis_graph:no_such_file", msg) ;
    }
    eat_comments (f) ;

    /* ---------------------------------------------------------------------- */
    /* parse the format line */
    /* ---------------------------------------------------------------------- */

    s [0] = '\0' ;
    if (fgets (s, LEN, f) != NULL)
    {
        sscanf (s, "%lg %lg %lg %lg", &x1, &x2, &x3, &x4) ;
    }
    n = (Int) x1 ;
    nzmax = (Int) x2 ;
    fmt = (Int) x3 ;
    ncon = (Int) x4 ;
    nzmax = MAX (nzmax, 1) ;
    if (n < 0 || ncon < 0 ||
        !(fmt == 0 || fmt == 1 || fmt == 10 || fmt == 11 || fmt == 100))
    {
        sprintf (msg, "invalid header: %lg %lg %lg %lg", x1, x2, x3, x4) ;
        mexErrMsgIdAndTxt ("metis_graph:invalid_header", msg) ;
    }

    if (fmt == 10 || fmt == 11)
    {
        ncon = MAX (ncon, 1) ;
    }

    has_ew = (fmt == 1 || fmt == 11) ;

    /* ---------------------------------------------------------------------- */
    /* allocate the triplets */
    /* ---------------------------------------------------------------------- */

    nzmax = 2 * MAX (nzmax,1) ;
    pargout [0] = mxCreateDoubleMatrix (nzmax, 1, mxREAL) ;
    pargout [1] = mxCreateDoubleMatrix (nzmax, 1, mxREAL) ;
    pargout [2] = mxCreateDoubleMatrix (has_ew ? nzmax : 1, 1, mxREAL) ;
    Ti = mxGetPr (pargout [0]) ;
    Tj = mxGetPr (pargout [1]) ;
    Tx = mxGetPr (pargout [2]) ;
    Tx [0] = 1 ;

    pargout [3] = mxCreateDoubleMatrix (n, ncon, mxREAL) ;
    W = mxGetPr (pargout [3]) ;
    nz = 0 ;

    /* ---------------------------------------------------------------------- */
    /* read each line, one per adjacency list */
    /* ---------------------------------------------------------------------- */

    /* printf ("reading graph ...\n") ; */

    for (i = 1 ; i <= n ; i++)
    {

        /* ------------------------------------------------------------------ */
        /* remove leading comment lines */
        /* ------------------------------------------------------------------ */

        /* if (i % 10000 == 0) printf (".") ; */

        eat_comments (f) ;

        /* ------------------------------------------------------------------ */
        /* read each node weight */
        /* ------------------------------------------------------------------ */

        status = 1 ;
        for (k = 0 ; k < ncon ; k++)
        {
            x = 0 ;
            status = get_token (f, s, LEN) ;
            if (sscanf (s, "%lg", &x) != 1)
            {
                sprintf (msg, "node %lg: missing node weights", (double) i) ;
                mexWarnMsgIdAndTxt ("metis_graph:missing_node_weights", msg) ;
            }
            W [i-1 + k*n] = x ;
        }

        /* ------------------------------------------------------------------ */
        /* read each edge */
        /* ------------------------------------------------------------------ */

        while (status >= 1)
        {

            /* -------------------------------------------------------------- */
            /* get the node number */
            /* -------------------------------------------------------------- */

            status = get_token (f, s, LEN) ;
            if (status == EOF) break ;
            if (sscanf (s, "%lg", &x) != 1) break ;
            j = (Int) x ;
            if ((double) j != x || j <= 0 || j > n)
            {
                sprintf (msg, "node %lg: edge %lg invalid", (double) i, x) ;
                mexErrMsgIdAndTxt ("metis_graph:invalid_edge", msg) ;
            }

            /* -------------------------------------------------------------- */
            /* allocate more space if needed */
            /* -------------------------------------------------------------- */

            if (nz == nzmax)
            {
                /* double the space */
                /* printf ("nzmax %g ", nzmax) ; */
                nzmax = 2 * nzmax + n ;
                /* printf ("to %g\n", nzmax) ; */
                t = MAX (nzmax, 1) * sizeof (double) ;
                Ti = mxRealloc (Ti, t) ;
                mxSetPr (pargout [0], Ti) ;
                mxSetM  (pargout [0], nzmax) ;
                Tj = mxRealloc (Tj, t) ;
                mxSetPr (pargout [1], Tj) ;
                mxSetM  (pargout [1], nzmax) ;
                if (has_ew)
                {
                    Tx = mxRealloc (Tx, t) ;
                    mxSetPr (pargout [2], Tx) ;
                    mxSetM  (pargout [2], nzmax) ;
                }
            }

            /* -------------------------------------------------------------- */
            /* get the edge weight, if present */
            /* -------------------------------------------------------------- */

            if (has_ew)
            {
                s [0] = '\0' ;
                if (status == 1)
                {
                    status = get_token (f, s, LEN) ;
                }
                if (status == EOF || sscanf (s, "%lg", &e) != 1)
                {
                    sprintf (msg, "node %lg: missing edge weight", (double) i) ;
                    mexErrMsgIdAndTxt ("metis_graph:invalid_edge_weight", msg) ;
                }
                Tx [nz] = e ;
            }

            /* -------------------------------------------------------------- */
            /* add edge (i,j) to the triplet form */
            /* -------------------------------------------------------------- */

            Ti [nz] = i ;
            Tj [nz] = j ;
            nz++ ;
        }
    }

    fclose (f) ;

    /* ---------------------------------------------------------------------- */
    /* return the results */
    /* ---------------------------------------------------------------------- */

    /* printf ("returning results %d %d\n", nz, nzmax) ; */
    if (nz < nzmax)
    {
        t = MAX (nz, 1) * sizeof (double) ;
        mxSetPr (pargout [0], mxRealloc (Ti, t)) ;
        mxSetM  (pargout [0], nz) ;
        mxSetPr (pargout [1], mxRealloc (Tj, t)) ;
        mxSetM  (pargout [1], nz) ;
        if (has_ew)
        {
            mxSetPr (pargout [2], mxRealloc (Tx, t)) ;
            mxSetM  (pargout [2], nz) ;
        }
    }
    pargout [4] = mxCreateDoubleScalar ((double) fmt) ;
    /* printf ("done\n") ; */
}

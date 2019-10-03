/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

GLOBAL Int UMF_transpose
(
    Int n_row,
    Int n_col,
    const Int Ap [ ],
    const Int Ai [ ],
    const double Ax [ ],
    const Int P [ ],
    const Int Q [ ],
    Int nq,
    Int Rp [ ],
    Int Ri [ ],
    double Rx [ ],
    Int W [ ],
    Int check
#ifdef COMPLEX
    , const double Az [ ]
    , double Rz [ ]
    , Int do_conjugate
#endif
) ;

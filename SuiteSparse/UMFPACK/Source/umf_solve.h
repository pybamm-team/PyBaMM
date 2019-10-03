/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

GLOBAL Int UMF_solve
(
    Int sys,
    const Int Ap [ ],
    const Int Ai [ ],
    const double Ax [ ],
    double Xx [ ],
    const double Bx [ ],
#ifdef COMPLEX
    const double Az [ ],
    double Xz [ ],
    const double Bz [ ],
#endif
    NumericType *Numeric,
    Int irstep,
    double Info [UMFPACK_INFO],
    Int Pattern [ ],
    double SolveWork [ ]
) ;

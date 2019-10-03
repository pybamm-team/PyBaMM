/* ========================================================================== */
/* === ssmult_template.c ==================================================== */
/* ========================================================================== */

/* C = A*B, where A and B are sparse.  The column pointers for C (the Cp array)
 * have already been computed.  entries are dropped.  This code fragment is
 * #include'd into ssmult.c four times, with all four combinations of ACOMPLEX
 * (defined or not) and BCOMPLEX (defined or not).
 *
 * By default, C is returned with sorted column indices, and with explicit
 * zero entries dropped.  If C is complex with an all-zero imaginary part, then
 * the imaginary part is freed and C becomes real.  Thus, C is a pure MATLAB
 * sparse matrix.
 *
 * If UNSORTED is defined (-DUNSORTED), then the nonzero pattern of C is
 * returned with unsorted column indices.  This is much faster than returning a
 * pure MATLAB sparse matrix, but the result must eventually be sorted prior to
 * returning to MATLAB.
 *
 * If the compiler bug discussed below does not affect you, then uncomment the
 * following line, or compile with code with -DNO_GCC_BUG.

#define NO_GCC_BUG

 * The gcc bug occurs when cij underflows to zero:
 *
 *      cij = aik * bkj ;
 *      if (cij == 0)
 *      {
 *          drop this entry 
 *      }
 *
 * If cij underflows, cij is zero but the above test is incorrectly FALSE with
 * gcc -O, using gcc version 4.1.0 on an Intel Pentium.  The bug does not appear
 * on an AMD Opteron with the same compiler.  The solution is to store cij to
 * memory first, and then to read it back in and test it, which is slower.
 */

/* -------------------------------------------------------------------------- */
/* MULT: multiply (or multiply and accumulate, depending on op) */
/* -------------------------------------------------------------------------- */

/* op can be "=" or "+=" */

#ifdef ACOMPLEX
#ifdef BCOMPLEX
#define MULT(x,z,op) \
    azik = (ac ? (-Az [pa]) : (Az [pa])) ; \
    x op Ax [pa] * bkj - azik * bzkj ; \
    z op azik * bkj + Ax [pa] * bzkj ;
#else
#define MULT(x,z,op) \
    azik = (ac ? (-Az [pa]) : (Az [pa])) ; \
    x op Ax [pa] * bkj ; \
    z op azik * bkj ;
#endif
#else
#ifdef BCOMPLEX
#define MULT(x,z,op) \
    x op Ax [pa] * bkj ; \
    z op Ax [pa] * bzkj ;
#else
#define MULT(x,z,op) \
    x op Ax [pa] * bkj ;
#endif
#endif

/* -------------------------------------------------------------------------- */
/* ASSIGN_BKJ: copy B(k,j) into a local scalar */
/* -------------------------------------------------------------------------- */

#ifdef BCOMPLEX
#define ASSIGN_BKJ \
    bkj = Bx [pb] ; \
    bzkj = (bc ? (-Bz [pb]) : (Bz [pb])) ;
#else
#define ASSIGN_BKJ \
    bkj = Bx [pb] ;
#endif

/* -------------------------------------------------------------------------- */
/* DROP_CHECK: check if an entry must be dropped */
/* -------------------------------------------------------------------------- */

#if defined (ACOMPLEX) || defined (BCOMPLEX)
#define DROP_CHECK(x,z) \
    if (x == 0 && z == 0) drop = 1 ; \
    if (z != 0) zallzero = 0 ;
#else
#define DROP_CHECK(x,z) if (x == 0) drop = 1 ;
#endif

/* -------------------------------------------------------------------------- */
/* sparse matrix multiply template */
/* -------------------------------------------------------------------------- */

{

#ifdef ACOMPLEX
    double azik ;
#endif

    /* ---------------------------------------------------------------------- */
    /* initialize drop tests */
    /* ---------------------------------------------------------------------- */

    drop = 0 ;                  /* true if any entry in C is zero */
    zallzero = 1 ;              /* true if Cz is all zero */

    /* ---------------------------------------------------------------------- */
    /* quick check if A is diagonal, or a permutation matrix */
    /* ---------------------------------------------------------------------- */

    if (Anrow == Ancol && Ap [Ancol] == Ancol)
    {
        /* A is square, with n == nnz (A); check the pattern */
        A_is_permutation = 1 ;
        A_is_diagonal = 1 ;
        for (j = 0 ; j < Ancol ; j++)
        {
            if (Ap [j] != j)
            {
                /* A has a column with no entries, or more than 1 entry */
                A_is_permutation = 0 ;
                A_is_diagonal = 0 ;
                break ;
            }
        }
        mark-- ;                        /* Flag [0..n-1] != mark is now true */ 
        for (j = 0 ; j < Ancol && (A_is_permutation || A_is_diagonal) ; j++)
        {
            /* A has one entry in each column, so j == Ap [j] */
            i = Ai [j] ;
            if (i != j)
            {
                /* A is not diagonal, but might still be a permutation */
                A_is_diagonal = 0 ;
            }
            if (Flag [i] == mark)
            {
                /* row i appears twice; A is neither permutation nor diagonal */
                A_is_permutation = 0 ;
                A_is_diagonal = 0 ;
            }
            /* mark row i, so we know if we see it again */
            Flag [i] = mark ;
        }
    }
    else
    {
        /* A is not square, or nnz (A) is not equal to n */
        A_is_permutation = 0 ;
        A_is_diagonal = 0 ;
    }

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

#ifndef UNSORTED
    W = NULL ;
    if (!A_is_diagonal)
    {
#if defined (ACOMPLEX) || defined (BCOMPLEX)
        W = mxMalloc (Anrow * 2 * sizeof (double)) ;
        Wz = W + Anrow ;
#else
        W = mxMalloc (Anrow * sizeof (double)) ;
#endif
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* compute C one column at a time */
    /* ---------------------------------------------------------------------- */

    if (A_is_diagonal)
    {

        /* ------------------------------------------------------------------ */
        /* C = A*B where A is diagonal */
        /* ------------------------------------------------------------------ */

        pb = 0 ;
        for (j = 0 ; j < Bncol ; j++)
        {
            pcstart = pb ;
            pbend = Bp [j+1] ;  /* column B is in Bi,Bx,Bz [pb ... pbend+1] */
            for ( ; pb < pbend ; pb++)
            {
                k = Bi [pb] ;                   /* nonzero entry B(k,j) */
                ASSIGN_BKJ ;
                Ci [pb] = k ;
                pa = k ;
                MULT (Cx [pb], Cz [pb], =) ;    /* C(k,j) = A(k,k)*B(k,j) */
#ifdef NO_GCC_BUG
                DROP_CHECK (Cx [pb], Cz [pb]) ; /* check if C(k,j) == 0 */
#endif
            }

#ifndef NO_GCC_BUG
            for (pc = pcstart ; pc < pbend ; pc++)
            {
                DROP_CHECK (Cx [pc], Cz [pc]) ;   /* check if C(k,j) == 0 */
            }
#endif
        }

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* C = A*B, general case, or A permutation */
        /* ------------------------------------------------------------------ */

        pb = 0 ;
        cnz = 0 ;
        for (j = 0 ; j < Bncol ; j++)
        {

            /* -------------------------------------------------------------- */
            /* compute jth column of C: C(:,j) = A * B(:,j) */
            /* -------------------------------------------------------------- */

            pbend = Bp [j+1] ;  /* column B is in Bi,Bx,Bz [pb ... pbend+1] */
            pcstart = cnz ;     /* start of column j in C */
            blen = pbend - pb ; /* number of entries in B */
            needs_sorting = 0 ; /* true if column j needs sorting */

            if (blen == 0)
            {

                /* ---------------------------------------------------------- */
                /* nothing to do, B(:,j) and C(:,j) are empty */
                /* ---------------------------------------------------------- */

                continue ;

            }
            else if (blen == 1)
            {

                /* ---------------------------------------------------------- */
                /* B(:,j) contains only one nonzero */
                /* ---------------------------------------------------------- */

                /* since there is only one entry in B, just scale column A(:,k):
                 * C(:,j) = A(:,k) * B(k,j)
                 * C is sorted only if A is sorted on input */

                k = Bi [pb] ;                   /* nonzero entry B(k,j) */
                ASSIGN_BKJ ;
                paend = Ap [k+1] ;
                for (pa = Ap [k] ; pa < paend ; pa++, cnz++)
                {
                    Ci [cnz] = Ai [pa] ;            /* nonzero entry A(i,k) */
                    MULT (Cx [cnz], Cz [cnz], =) ;  /* C(i,j) = A(i,k)*B(k,j) */
#ifdef NO_GCC_BUG
                    DROP_CHECK (Cx [cnz], Cz [cnz]) ;   /* check C(i,j) == 0 */
#endif
                }
                pb++ ;

#ifndef NO_GCC_BUG
                for (pc = pcstart ; pc < cnz ; pc++)
                {
                    DROP_CHECK (Cx [pc], Cz [pc]) ;   /* check if C(i,j) == 0 */
                }
#endif

            }
            else
            {

                /* ---------------------------------------------------------- */
                /* B(:,j) has two or more entries */
                /* ---------------------------------------------------------- */

                if (A_is_permutation)
                {

                    /* ------------------------------------------------------ */
                    /* A is a permutation matrix */
                    /* ------------------------------------------------------ */

                    needs_sorting = 1 ;
                    for ( ; pb < pbend ; pb++)
                    {
                        k = Bi [pb] ;           /* nonzero entry B(k,j) */
                        ASSIGN_BKJ ;
                        i = Ai [k] ;            /* nonzero entry A(i,k) */
                        Ci [pb] = i ;
                        pa = k ;
                        /* C(i,j) = A(i,k)*B(k,j) */
#ifndef UNSORTED
                        MULT (W [i], Wz [i], =) ;
#else
                        MULT (Cx [pb], Cz [pb], =) ;
#endif
                    }
                    cnz = pbend ;

                }
                else
                {

                    /* ------------------------------------------------------ */
                    /* general case */
                    /* ------------------------------------------------------ */

                    /* first entry in jth column of B is simpler */
                    /* C(:,j) = A (:,k) * B (k,j) */
                    k = Bi [pb] ;                   /* nonzero entry B(k,j) */
                    ASSIGN_BKJ ;
                    paend = Ap [k+1] ;
                    for (pa = Ap [k] ; pa < paend ; pa++)
                    {
                        i = Ai [pa] ;               /* nonzero entry A(i,k) */
                        Flag [i] = cnz ;
                        Ci [cnz] = i ;              /* new entry C(i,j) */
                        /* C(i,j) = A(i,k)*B(k,j) */
#ifndef UNSORTED
                        MULT (W [i], Wz [i], =) ;
#else
                        MULT (Cx [cnz], Cz [cnz], =) ;
#endif
                        cnz++ ;
                    }
                    pb++ ;
                    for ( ; pb < pbend ; pb++)
                    {
                        k = Bi [pb] ;               /* nonzero entry B(k,j) */
                        ASSIGN_BKJ ;
                        /* C(:,j) += A (:,k) * B (k,j) */
                        paend = Ap [k+1] ;
                        for (pa = Ap [k] ; pa < paend ; pa++)
                        {
                            i = Ai [pa] ;           /* nonzero entry A(i,k) */
                            pc = Flag [i] ;
                            if (pc < pcstart)
                            {
                                pc = cnz++ ;
                                Flag [i] = pc ;
                                Ci [pc] = i ;           /* new entry C(i,j) */
                                /* C(i,j) = A(i,k)*B(k,j) */
#ifndef UNSORTED
                                MULT (W [i], Wz [i], =) ;
                                needs_sorting = 1 ;
#else
                                MULT (Cx [pc], Cz [pc], =) ;
#endif
                            }
                            else
                            {
                                /* C(i,j) += A(i,k)*B(k,j) */
#ifndef UNSORTED
                                MULT (W [i], Wz [i], +=) ;
#else
                                MULT (Cx [pc], Cz [pc], +=) ;
#endif
                            }
                        }
                    }
                }

                /* ---------------------------------------------------------- */
                /* sort the pattern of C(:,j) and gather the values of C(:,j) */
                /* ---------------------------------------------------------- */

#ifndef UNSORTED
                /* Sort the row indices in C(:,j).  Use Cx as Int workspace.
                 * This assumes sizeof (Int) < sizeof (double). If blen <= 1,
                 * or if subsequent entries in B(:,j) appended entries onto C,
                 * there is no need to sort C(:,j), assuming A is sorted. */
                if (needs_sorting)
                {
                    ssmergesort (Ci + pcstart, (Int *) (Cx + pcstart),
                        cnz - pcstart) ;
                }
                for (pc = pcstart ; pc < cnz ; pc++)
                {
#if defined (ACOMPLEX) || defined (BCOMPLEX)
                    i = Ci [pc] ;
                    cij = W [i] ;                   /* get C(i,j) from W */
                    czij = Wz [i] ;
                    Cx [pc] = cij ;                 /* copy C(i,j) into C */
                    Cz [pc] = czij ;
#else
                    cij = W [Ci [pc]] ;             /* get C(i,j) from W */
                    Cx [pc] = cij ;                 /* copy C(i,j) into C */
#endif
                    DROP_CHECK (cij, czij) ;        /* check if C(i,j) == 0 */
                }
#else
                /* no need to sort, but we do need to check for drop */
                for (pc = pcstart ; pc < cnz ; pc++)
                {
                    DROP_CHECK (Cx [pc], Cz [pc]) ; /* check if C(i,j) == 0 */
                }
#endif
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

#ifndef UNSORTED
    mxFree (W) ;
#endif


}

#undef ACOMPLEX
#undef BCOMPLEX
#undef MULT
#undef ASSIGN_BKJ
#undef DROP_CHECK

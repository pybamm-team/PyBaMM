// =============================================================================
// === spqr_analyze ============================================================
// =============================================================================

// Given the nonzero pattern of a sparse m-by-n matrix A, analyze it for
// subsequent numerical factorization.  This function operates on the pattern
// of A only; it does not need to be templatized.

#include "spqr.hpp"

#ifdef NSUPERNODAL
#error "SuiteSparseQR requires the CHOLMOD/Supernodal module"
#endif

// =============================================================================

#define FREE_WORK \
    cholmod_l_free_factor (&Sc, cc) ; \
    cholmod_l_free (2*(nf+1), sizeof (double), Flops,         cc) ; \
    cholmod_l_free (ns+2,     sizeof (Long),    Stack_stack,   cc) ; \
    cholmod_l_free (nf,       sizeof (Long),    Rh,            cc) ; \
    cholmod_l_free (ntasks,   sizeof (Long),    TaskParent,    cc) ;

// =============================================================================
// === spqr_analyze ============================================================
// =============================================================================

spqr_symbolic *spqr_analyze
(
    // inputs, not modified
    cholmod_sparse *A,
    int ordering,           // all options available
    Long *Quser,            // user provided ordering, if given (may be NULL)

    int do_rank_detection,  // if TRUE, then rank deficient matrices may be
                            // considered during numerical factorization,
    // with tol >= 0 (tol < 0 is also allowed).  If FALSE, then the tol
    // parameter is ignored by the numerical factorization, and no rank
    // detection is performed.  Ignored when using GPU acceleration
    // (no rank detection is performed in that case).

    int keepH,              // if TRUE, keep the Householder vectors

    // workspace and parameters
    cholmod_common *cc
)
{
    spqr_symbolic *QRsym ;
    Long *Parent, *Child, *Childp, *W, *Rj, *Rp, *Super, *Stair, *Fmap, *Sleft,
        *Post, *Ap, *Weight, *On_stack, *Task, *TaskParent,
        *TaskChildp, *TaskChild, *Fm, *Cm, *TaskFront, *TaskFrontp, *Rh,
        *Stack_stack, *Stack_maxstack, *Hip,
        *TaskStack, *InvPost ;
    Long nf, f, j, col1, col2, p, p1, p2, t, parent, anz, fp, csize_max,
        fmc, fnc, fpc, cm, cn, ci, fm, fn, cm_min, cm_max, csize_min, kf,
        rm, rn, col, c, pc, rsize, maxfn, csize, m, n, klast,
        stack, maxstack, rxsize, hisize,
        rhxsize, ctot, fsize, ns, ntasks, task ;
    cholmod_sparse *AT ;
    cholmod_factor *Sc ;
    int ok = TRUE, do_parallel_analysis ;
    double total_flops = 0 ;
    double *Flops, *Flops_subtree ;
    Long *Sp, *Sj;

#ifdef GPU_BLAS
    spqr_gpu *QRgpu ;
    Long *RjmapOffsets, *RimapOffsets ;
    Long RjmapSize, RimapSize;
    Long numStages;             // staging
    Long *Stagingp, *StageMap, *FOffsets, *ROffsets, *SOffsets;
    size_t *FSize, *RSize, *SSize;
#endif

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    PR (("\n ========================== QR analyze\n")) ;

    ASSERT (cc != NULL) ;
    ASSERT (A != NULL) ;
    ASSERT (cc->status == CHOLMOD_OK) ;

    m = A->nrow ;
    n = A->ncol ;
    Ap = (Long *) A->p ;
    // Ai = (Long *) A->i ;
    anz = Ap [n] ;

    do_parallel_analysis = (cc->SPQR_grain > 1) ;

    // The analysis for TBB parallelism attempts to construct a task graph with
    // leaf nodes with flop counts >= max ((total flops) / cc->SPQR_grain,
    // cc->SPQR_small).  If cc->SPQR_grain <= 1, or if the total flop
    // count is less than cc->SPQR_small, then no parallelism will be
    // exploited.  A decent value of cc->SPQR_grain is about 2 times the
    // number of cores.

    PR (("grainsize knobs %g %g\n", cc->SPQR_grain, cc->SPQR_small)) ;

    // -------------------------------------------------------------------------
    // GPU selection
    // -------------------------------------------------------------------------

#ifdef GPU_BLAS
    // See if the user wants to use GPU acceleration.
    bool useGPU ;

    if (cc->useGPU == -1)
    {
        // only CHOLMOD uses this option.  Not in use yet for SPQR
        useGPU = FALSE ;
    }
    else
    {
        useGPU = cc->useGPU ;
    }

    // Disable the GPU if the Householder vectors are requested, if we're
    // using TBB, if rank detection is requested, or if A is not real
    if (keepH || do_parallel_analysis || do_rank_detection ||
        A->xtype != CHOLMOD_REAL)
    {
        useGPU = FALSE ;
    }

    PR (("GPU: %d : %d %d %d %d", cc->useGPU,
        keepH, do_parallel_analysis,
        do_rank_detection, A->xtype != CHOLMOD_REAL)) ;

    if (useGPU)
    {
        PR ((" ============================= USE GPU")) ;
    }
    PR (("\n")) ;

#else
    // the GPU is not in use (not installed at compile-time)
    bool useGPU = FALSE ;
#endif

    // cc->for_GPU = (int) useGPU ;    // keep track of if we are using the GPU

    // -------------------------------------------------------------------------
    // allocate workspace for this function and cholmod_l_postorder
    // -------------------------------------------------------------------------

    // the number of frontal matrices (nf) is not yet known, but nf <= n will
    // hold.  cholmod_l_analyze_p2, below, will allocate m+6*n in cc->Iwork.

    nf = n ;    // just a placeholder; will be changed later to some nf <= n

    cholmod_l_allocate_work (n+1, MAX (m, 2*(n+1) + 2*(nf+2)) + 1, 0, cc) ;

    // workspace allocated later:
    Rh = NULL ;
    Flops = NULL ;
    Flops_subtree = NULL ;
    Sc = NULL ;
    Stack_stack = NULL ;
    TaskParent = NULL ;
    ns = 0 ;                    // number of stacks
    ntasks = 0 ;                // number of tasks in Task tree

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // supernodal Cholesky ordering and analysis of A'A
    // -------------------------------------------------------------------------

    AT = cholmod_l_transpose (A, 0, cc) ;   // AT = spones (A') [

    // save the current CHOLMOD settings
    Long save [6] ;
    save [0] = cc->supernodal ;
    save [1] = cc->nmethods ;
    save [2] = cc->postorder ;
    save [3] = cc->method [0].ordering ;
    save [4] = cc->method [1].ordering ;
    save [5] = cc->method [2].ordering ;

    // SuiteSparseQR requires a supernodal analysis to find its frontal matrices
    cc->supernodal = CHOLMOD_SUPERNODAL ;

    /* ---------------------------------------------------------------------- */
    /* determine ordering */
    /* ---------------------------------------------------------------------- */

    // 1:natural and 3:given become the same as 1:fixed
    if (ordering == SPQR_ORDERING_NATURAL ||
       (ordering == SPQR_ORDERING_GIVEN && Quser == NULL))
    {
        ordering = SPQR_ORDERING_FIXED ;
    }

    // 8:best: best of COLAMD(A), AMD(A'A), and METIS (if available)
    if (ordering == SPQR_ORDERING_BEST)
    {
        ordering = SPQR_ORDERING_CHOLMOD ;
        cc->postorder = TRUE ;
        cc->nmethods = 2 ;
        cc->method [0].ordering = CHOLMOD_COLAMD ;
        cc->method [1].ordering = CHOLMOD_AMD ;
#ifndef NPARTITION
        cc->nmethods = 3 ;
        cc->method [2].ordering = CHOLMOD_METIS ;
#endif
    }

    // 9:best: best of COLAMD(A) and AMD(A'A),
    if (ordering == SPQR_ORDERING_BESTAMD)
    {
        ordering = SPQR_ORDERING_CHOLMOD ;
        cc->postorder = TRUE ;
        cc->nmethods = 2 ;
        cc->method [0].ordering = CHOLMOD_COLAMD ;
        cc->method [1].ordering = CHOLMOD_AMD ;
    }

    /* ---------------------------------------------------------------------- */
    /* set up the ordering */
    /* ---------------------------------------------------------------------- */

    if (ordering == SPQR_ORDERING_FIXED)
    {
        // fixed column ordering
        cc->nmethods = 1 ;
        cc->method [0].ordering = CHOLMOD_NATURAL ;
        cc->postorder = FALSE ;
        Quser = NULL ;
    }
    else if (ordering == SPQR_ORDERING_GIVEN)
    {
        // user-provided column ordering
        cc->nmethods = 1 ;
        cc->method [0].ordering = CHOLMOD_GIVEN ;
        cc->postorder = FALSE ;
    }
    else if (ordering == SPQR_ORDERING_CHOLMOD)
    {
        // CHOLMOD default, defined in cc.  If not modified, this is
        // to try AMD(A'A), and then try METIS if available, and take the best
        // available.
        Quser = NULL ;
    }
    else if (ordering == SPQR_ORDERING_AMD)
    {
        // AMD (A'*A)
        cc->nmethods = 1 ;
        cc->method [0].ordering = CHOLMOD_AMD ;
        cc->postorder = TRUE ;
        Quser = NULL ;
    }
#ifndef NPARTITION
    else if (ordering == SPQR_ORDERING_METIS)
    {
        // METIS (A'*A), if installed
        cc->nmethods = 1 ;
        cc->method [0].ordering = CHOLMOD_METIS ;
        cc->postorder = TRUE ;
        Quser = NULL ;
    }
#endif
    else // if (ordering == SPQR_ORDERING_COLAMD)
         // or ordering == SPQR_ORDERING_DEFAULT
         // or ordering == SPQR_ORDERING_METIS and METIS not installed
    {
        // Version 1.2.0: default is COLAMD
        // COLAMD (A)
        ordering = SPQR_ORDERING_COLAMD ;
        cc->nmethods = 1 ;
        cc->method [0].ordering = CHOLMOD_COLAMD ;
        cc->postorder = TRUE ;
        Quser = NULL ;
    }

    // multifrontal QR ordering and analysis.
    // The GPU-accelerated SPQR requires additional supernodal analysis.
    Sc = cholmod_l_analyze_p2 (
        useGPU ? CHOLMOD_ANALYZE_FOR_SPQRGPU : CHOLMOD_ANALYZE_FOR_SPQR,
        AT, (SuiteSparse_long *) Quser, NULL, 0, cc) ;

    // record the actual ordering used
    if (Sc != NULL)
    {
        switch (Sc->ordering)
        {
            case CHOLMOD_NATURAL: ordering = SPQR_ORDERING_NATURAL ; break ;
            case CHOLMOD_GIVEN:   ordering = SPQR_ORDERING_GIVEN   ; break ;
            case CHOLMOD_AMD:     ordering = SPQR_ORDERING_AMD     ; break ;
            case CHOLMOD_COLAMD:  ordering = SPQR_ORDERING_COLAMD  ; break ;
            case CHOLMOD_METIS:   ordering = SPQR_ORDERING_METIS   ; break ;
        }
    }

    cc->SPQR_istat [7] = ordering ;

    // restore the CHOLMOD settings
    cc->supernodal              = save [0] ;
    cc->nmethods                = save [1] ;
    cc->postorder               = save [2] ;
    cc->method [0].ordering     = save [3] ;
    cc->method [1].ordering     = save [4] ;
    cc->method [2].ordering     = save [5] ;

    cholmod_l_free_sparse (&AT, cc) ;       // ]

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_WORK ;
        return (NULL) ;
    }

    if (Sc == NULL || !(Sc->is_super) || !(Sc->is_ll))
    {
        cholmod_l_free_factor (&Sc, cc) ;
        FREE_WORK ;
        ERROR (CHOLMOD_INVALID,
            "SuiteSparseQR requires the CHOLMOD/Supernodal module") ;
        return (NULL) ;
    }

    ASSERT (Sc != NULL && Sc->is_ll && Sc->is_super) ;

    // -------------------------------------------------------------------------
    // extract the contents of CHOLMOD's supernodal factorization
    // -------------------------------------------------------------------------

    QRsym = (spqr_symbolic *) cholmod_l_malloc (1, sizeof (spqr_symbolic), cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_WORK ;
        return (NULL) ;
    }

    QRsym->m = m ;
    QRsym->n = n ;
    QRsym->do_rank_detection = do_rank_detection ;
    QRsym->anz = anz ;
    QRsym->nf = nf = Sc->nsuper ;       // number of supernodes / fronts
    QRsym->rjsize = Sc->ssize ;         // size of int part of supernodal R
    QRsym->keepH = keepH ;
    QRsym->maxcsize = Sc->maxcsize;
    QRsym->maxesize = Sc->maxesize;

    QRsym->Qfill = (Long *) Sc->Perm ;           // size n column perm
    Sc->Perm = NULL ;

    QRsym->Super = Super = (Long *) Sc->super ;  // Super is size nf+1
    Sc->super = NULL ;

    QRsym->Rp = Rp = (Long *) Sc->pi ;           // Rp is size nf+1
    Sc->pi = NULL ;

    QRsym->Rj = Rj = (Long *) Sc->s ;            // Rj is size rjsize
    Sc->s = NULL ;

    // ColCount is required for the GPU factorization
    QRsym->ColCount = (Long *) Sc->ColCount ;
    Sc->ColCount = NULL ;

    cholmod_l_free_factor (&Sc, cc) ;

    // -------------------------------------------------------------------------
    // allocate the rest of QRsym
    // -------------------------------------------------------------------------

    ASSERT (nf <= n) ;
    QRsym->Parent = Parent = (Long *) cholmod_l_malloc (nf+1, sizeof(Long), cc);
    QRsym->Childp = Childp = (Long *) cholmod_l_calloc (nf+2, sizeof(Long), cc);
    QRsym->Child  = Child  = (Long *) cholmod_l_calloc (nf+1, sizeof(Long), cc);
    QRsym->Post   = Post   = (Long *) cholmod_l_malloc (nf+1, sizeof(Long), cc);
    QRsym->PLinv           = (Long *) cholmod_l_malloc (m,    sizeof(Long), cc);
    QRsym->Sleft  = Sleft  = (Long *) cholmod_l_malloc (n+2,  sizeof(Long), cc);
    QRsym->Sp     = Sp     = (Long *) cholmod_l_malloc (m+1,  sizeof(Long), cc);
    QRsym->Sj     = Sj     = (Long *) cholmod_l_malloc (anz,  sizeof(Long), cc);
    QRsym->Fm              = (Long *) cholmod_l_malloc (nf+1, sizeof(Long), cc);
    QRsym->Cm              = (Long *) cholmod_l_malloc (nf+1, sizeof(Long), cc);

    if (keepH)
    {
        QRsym->Hip = Hip = (Long *) cholmod_l_malloc (nf+1, sizeof (Long), cc) ;
    }
    else
    {
        QRsym->Hip = Hip = NULL ;
    }

    // allocated later (or skipped if no parallelism)
    QRsym->TaskChild  = NULL ;
    QRsym->TaskChildp = NULL ;
    QRsym->TaskFront = NULL ;
    QRsym->TaskFrontp = NULL ;
    QRsym->TaskStack = NULL ;
    QRsym->On_stack = NULL ;
    QRsym->Stack_maxstack = NULL ;
    QRsym->ntasks = EMPTY ;         // computed later
    QRsym->ns = EMPTY ;

    // allocated later (or skipped if not using GPU)
    QRsym->QRgpu = NULL ;

#ifdef GPU_BLAS

    QRgpu = NULL ;
    RjmapOffsets = NULL ;
    RimapOffsets = NULL ;
    Stagingp = NULL ;
    StageMap = NULL ;
    FSize = NULL ;
    RSize = NULL ;
    SSize = NULL ;
    FOffsets = NULL ;
    ROffsets = NULL ;
    SOffsets = NULL ;

    // FUTURE: Add a parameter that disables the GPU if the arithmetic
    // intensity drops below a user-definable threshold.

    if (useGPU)
    {
        // use calloc so that the pointers inside are all NULL
        QRgpu = (spqr_gpu *) cholmod_l_calloc (1, sizeof(spqr_gpu), cc) ;
        QRsym->QRgpu = QRgpu ;
        if(QRgpu)
        {
            RimapOffsets = (Long*) cholmod_l_malloc(nf, sizeof(Long), cc) ;
            QRgpu->RimapOffsets = RimapOffsets  ;

            RjmapOffsets = (Long*) cholmod_l_malloc(nf, sizeof(Long), cc) ;
            QRgpu->RjmapOffsets = RjmapOffsets ;

            // allocated later
            QRgpu->numStages = 0 ;
            QRgpu->Stagingp = (Long*)   NULL ;
            QRgpu->StageMap = (Long*)   NULL ;
            QRgpu->FSize    = (size_t*) NULL ;
            QRgpu->RSize    = (size_t*) NULL ;
            QRgpu->SSize    = (size_t*) NULL ;
            QRgpu->FOffsets = (Long*)   NULL ;
            QRgpu->ROffsets = (Long*)   NULL ;
            QRgpu->SOffsets = (Long*)   NULL ;
        }

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            spqr_freesym (&QRsym, cc) ;
            FREE_WORK ;
            return (NULL) ;
        }
    }

    // initialize assembly map sizes (used later)
    RimapSize = RjmapSize = 0;
#endif

    // -------------------------------------------------------------------------
    // allocate workspace needed for parallel analysis
    // -------------------------------------------------------------------------

    if (do_parallel_analysis)
    {
        // allocate Flops and Flops_subtree, each of size nf+1
        Flops = (double *) cholmod_l_malloc (2*(nf+1), sizeof (double), cc) ;
        Flops_subtree = Flops + (nf+1) ;
        if (keepH)
        {
            // Rh, size nf; Rh [f] is the size of R and H for front f
            Rh = (Long *) cholmod_l_malloc (nf, sizeof (Long), cc) ;
        }
    }

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freesym (&QRsym, cc) ;
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // construct the supernodal tree and its postordering
    // -------------------------------------------------------------------------

    // Parent is identical to Sparent of the relaxed supernodes in
    // cholmod_super_symbolic, but it was not kept.  It needs to be
    // reconstructed here.   FUTURE WORK: tell CHOLMOD to keep Sparent; some of
    // this code could be deleted.  Note that cholmod_analyze postorders the
    // children in order of increasing row count of R, so that bigger children
    // come later, and the biggest child of front f is front f-1.

    W = (Long *) cc->Iwork ;

    // use W [0:n-1] for SuperMap [

    for (f = 0 ; f < nf ; f++)
    {
        // front f has global pivot columns col1: col2-1
        col1 = Super [f] ;
        col2 = Super [f+1] ;
        for (j = col1 ; j < col2 ; j++)
        {
            // W [j] = f means that column j is a pivotal column in front f
            W [j] = f ;
        }
    }

    // find the parent of each front
    for (f = 0 ; f < nf ; f++)
    {
        col1 = Super [f] ;              // front f has columns col1:col2-1
        col2 = Super [f+1] ;
        p1 = Rp [f] ;                   // Rj [p1..p2-1] = columns in F
        p2 = Rp [f+1] ;
        fp = col2 - col1 ;              // first fp columns are pivotal
        p = p1 + fp ;                   // pointer to first non-pivotal col
        if (p < p2)
        {
            j = Rj [p] ;                // first non-pivot column in F
            ASSERT (j >= col2 && j < n) ;
            parent = W [j] ;            // get j's pivotal front
        }
        else
        {
            parent = nf ;               // placeholder front; root of all
        }
        Parent [f] = parent ;
        Childp [parent]++ ;             // increment count of children
    }
    Parent [nf] = EMPTY ;               // placeholder front; root of all

    // W no longer needed for SuperMap ]

    // -------------------------------------------------------------------------
    // postorder the frontal tree
    // -------------------------------------------------------------------------

    // use W (2*(nf+1):3*(nf+1)-1) for node weights [
    Weight = W + 2*(nf+1) ;

    for (f = 0 ; f < nf ; f++)
    {
        // Weight [f] = degree of node f (# of columns in front)
        Weight [f] = (Rp [f+1] - Rp [f]) ;
        // or ... reverse that order, so big nodes come first.  This can have
        // an affect on memory usage (sometimes less, sometimes more)
        // Weight [f] = n - (Rp [f+1] - Rp [f]) ;
    }
    Weight [nf] = 1 ;                   // any value will do

    // ensure that CHOLMOD does not reallocate the workspace [
    cc->no_workspace_reallocate = TRUE ;

    // uses CHOLMOD workspace: Head (nf+1), Iwork (2*(nf+1)).  Guaranteed
    // to succeed since enough workspace has already been allocated above.
    cholmod_l_postorder ((SuiteSparse_long *) Parent, nf+1,
        (SuiteSparse_long *) Weight, (SuiteSparse_long *) Post, cc) ;

    ASSERT (cc->status == CHOLMOD_OK) ;
    ASSERT (Post [nf] == nf) ;          // placeholder is last


    // CHOLMOD can now reallocate the workspace as needed ]
    cc->no_workspace_reallocate = FALSE ;

    // done using W (2*(nf+1):3*(nf+1)-1) for node weights ]

    // -------------------------------------------------------------------------
    // construct the lists of children of each front
    // -------------------------------------------------------------------------

    // replace Childp with cumsum ([0 Childp])
    spqr_cumsum (nf+1, Childp) ;

    // create the child lists
    for (kf = 0 ; kf < nf ; kf++)
    {
        // place the child c in the list of its parent
        c = Post [kf] ;
        ASSERT (c >= 0 && c < nf) ;
        parent = Parent [c] ;
        ASSERT (c < parent && parent != EMPTY && parent <= nf) ;
        Child [Childp [parent]++] = c ;
    }

    spqr_shift (nf+1, Childp) ;
    ASSERT (Childp [0] == 0) ;
    ASSERT (Childp [nf+1] == nf) ;

#ifndef NDEBUG
    for (kf = 0 ; kf <= nf ; kf++)
    {
        f = Post [kf] ;
        ASSERT (f >= 0 && f <= nf) ;
        PR (("Front %ld Parent %ld\n", f, Parent [f])) ;
        for (p = Childp [f] ; p < Childp [f+1] ; p++)
        {
            c = Child [p] ;                 // get the child c of front F
            ASSERT (c < f) ;
            PR (("   child: %ld\n", c)) ;
        }
    }
#endif

    // -------------------------------------------------------------------------
    // S = A (P,Q) in row form, where P = leftmost column sort of A(:,Q)
    // -------------------------------------------------------------------------

    // use W [0:m-1] workspace in spqr_stranspose1:
    spqr_stranspose1 (A, QRsym->Qfill, QRsym->Sp, QRsym->Sj, QRsym->PLinv,
        Sleft, W) ;

    // -------------------------------------------------------------------------
    // determine flop counts, front sizes, and sequential memory usage
    // -------------------------------------------------------------------------

    ASSERT (nf <= n) ;

    // was Iwork workspace usage, now saved in QRsym:
    Fm = QRsym->Fm ;
    Cm = QRsym->Cm ;

    // The space for Stair and Fmap (total size 2*n+2) is later used for
    // Task and InvPost (total size 2*nf+2)
    Stair = W ;             // size n+1 [
    Fmap = Stair + (n+1) ;  // size n+1 [

    stack = 0 ;             // current stack usage
    maxstack = 0 ;          // peak stack usage

    rxsize = 0 ;            // total size of the real part of R
    rhxsize = 0 ;           // total size of the real part of R+H
    maxfn = 0 ;             // maximum # of columns in any front

    PR (("\n\n =========== sequential\n")) ;

    for (kf = 0 ; ok && kf < nf ; kf++)
    {
        f = Post [kf] ;
        PR (("\n-------------------------- front: %ld\n", f)) ;
        ASSERT (f >= 0 && f < nf) ;

        // ---------------------------------------------------------------------
        // analyze the factorization of front F
        // ---------------------------------------------------------------------

        // pivotal columns Super [f] ... Super [f+1]-1
        col1 = Super [f] ;          // front f has pivot columns col1:col2-1
        col2 = Super [f+1] ;
        p1 = Rp [f] ;               // Rj [p1..p2-1] = columns in F
        p2 = Rp [f+1] ;
        fp = col2 - col1 ;          // first fp columns are pivotal
        fn = p2 - p1 ;              // exact number of columns of F
        maxfn = MAX (maxfn, fn) ;
        ASSERT (fp > 0) ;           // all fronts have at least one pivot col
        ASSERT (fp <= fn) ;
        ASSERT (fn <= n) ;
        PR (("  fn %ld fp %ld\n", fn, fp)) ;

        // ---------------------------------------------------------------------
        // create the Fmap for front F
        // ---------------------------------------------------------------------

        #ifndef NDEBUG
        for (col = 0 ; col < n ; col++) Fmap [col] = EMPTY ;
        #endif

        for (p = p1, j = 0 ; p < p2 ; p++, j++)
        {
            col = Rj [p] ;              // global column col is jth col of F
            Fmap [col] = j ;
            PR (("  column in front %ld: %ld\n", f, col)) ;
            #ifndef NPRINT
            if (j == fp-1) PR (("  -----\n")) ;
            #endif
        }

        // If do_rank_detection is false, then there are no possible pivot
        // column failures; in this case, the analysis is exact.  If
        // do_rank_detection is true, then worst-case pivot column failures are
        // considered; in this case, the analysis gives an upper bound.

        // ---------------------------------------------------------------------
        // initialize the staircase for front F
        // ---------------------------------------------------------------------

        // initialize the staircase with original rows of S
        for (j = 0 ; j < fp ; j++)
        {
            // global column j+col1 is the jth pivot column of front F
            col = j + col1 ;
            Stair [j] = Sleft [col+1] - Sleft [col] ;
#ifndef NDEBUG
            for (Long row = Sleft [col] ; row < Sleft [col+1] ; row++)
            {
                PR (("Assemble row %ld into stair [%ld] col %ld\n",
                    row,j, col)) ;
            }
#endif
        }

        // contribution blocks from children will be added here
        for ( ; j < fn ; j++)
        {
            Stair [j] = 0 ;
        }

        // ---------------------------------------------------------------------
        // assemble each child
        // ---------------------------------------------------------------------

        ctot = 0 ;
        for (p = Childp [f] ; p < Childp [f+1] ; p++)
        {
            c = Child [p] ;                 // get the child c of front F
            ASSERT (c >= 0 && c < f) ;
            pc = Rp [c] ;
            fnc = Rp [c+1] - pc ;           // total # cols in child F
            fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
            cn = fnc - fpc ;                // # of cols in child C
            fmc = Fm [c] ;                  // max or exact # rows in child

            if (do_rank_detection)
            {
                // with possible pivot failures
                // fmc is an upper bound on the # of rows in child F
                cm = MIN (fmc, cn) ;        // max # rows in child C
            }
            else
            {
                // with no pivot failures
                // fmc is the exact # of rows in child F
                Long rc = MIN (fmc, fpc) ;   // exact # of rows in child R
                cm = MAX (fmc - rc, 0) ;
                cm = MIN (cm, cn) ;         // exact # rows in C
            }
            ASSERT (cm >= 0 && cm <= cn) ;
            ASSERT (pc + cm <= Rp [c+1]) ;

            ASSERT (cm == Cm [c]) ;

            // add the child rows to the staircase
            for (ci = 0 ; ci < cm ; ci++)
            {
                col = Rj [pc + fpc + ci] ;  // leftmost col of row ci
                j = Fmap [col] ;            // global col is jth col of F
                ASSERT (j >= 0 && j < fn) ;
                Stair [j]++ ;               // add row ci to jth Stair
                #ifndef NDEBUG
                {
                    PR (("Assemble c %ld into stair [%ld] col %ld\n",
                        c,j,col)) ;
                }
                #endif
                ASSERT (col >= Super [c+1] && col < n) ;
            }

            // Keep track of total sizes of C blocks of all children.  The
            // C block of this child has at most cm rows, and always has
            // (fnc-fpc) columns.  Long overflow cannot occur because csize
            // < fsize of the child and fsize has already been checked.
            csize = cm*(cm+1)/2 + cm*(cn-cm) ;
            ctot += csize ;

            PR (("child %ld size %ld  = %ld * %ld\n", c, cm*cn, cm, cn)) ;
        }

        // ---------------------------------------------------------------------
        // replace Stair with cumsum (Stair), and find # rows of F
        // ---------------------------------------------------------------------

        fm = 0 ;
        for (j = 0 ; j < fn ; j++)
        {
            fm += Stair [j] ;
            Stair [j] = fm ;
            PR (("j %ld Stair %ld\n", j, Stair [j])) ;
        }
        PR (("fm %ld\n", fm)) ;

        // ---------------------------------------------------------------------
        // determine upper bounds on the size F
        // ---------------------------------------------------------------------

        // The front F is at most fm-by-fn, with fp pivotal columns.

        fsize = spqr_mult (fm, fn, &ok) ;
        PR (("fm %ld fn %ld rank %d\n", fm, fn, do_rank_detection)) ;
        if (!ok)
        {
            // problem too large
            break ;
        }

        ASSERT (fm >= 0) ;
        ASSERT (fn >= 0) ;
        ASSERT (fp >= 0 && fp <= fn) ;

        Fm [f] = fm ;        // keep track of # rows of F

        // ---------------------------------------------------------------------
        // determine upper bounds on the size R, if H not kept
        // ---------------------------------------------------------------------

        // The R block of F is at most rm-by-rn packed upper trapezoidal.
        rm = MIN (fm, fp) ;     // exact/max # of rows of R block
        rn = fn ;               // exact/max # of columns of R block
        ASSERT (rm <= rn) ;
        ASSERT (rm >= 0) ;
        ASSERT (rm <= fm) ;
        rsize = rm*(rm+1)/2 + rm*(rn-rm) ;  // Long overflow cannot occur
        ASSERT (rsize >= 0 && rsize <= fsize) ;
        rxsize += rsize ;
        PR ((" rm %ld rn %ld rsize %ld\n", rm, rn, rsize)) ;

        // ---------------------------------------------------------------------
        // determine upper bounds on the size C
        // ---------------------------------------------------------------------

        // The C block of F is at most cm-by-cn packed upper trapezoidal
        cn = fn - fp ;          // exact # of columns in C block

        // with rank detection and possible pivot failures
        cm_max = fm ;
        cm_max = MIN (cm_max, cn) ;         // max # rows in C block of F

        // with no pivot failures
        cm_min = MAX (fm - rm, 0) ;
        cm_min = MIN (cm_min, cn) ;         // exact # rows in C block

        // Long overflow cannot occur:
        csize_max = cm_max*(cm_max+1)/2 + cm_max*(cn-cm_max) ;
        csize_min = cm_min*(cm_min+1)/2 + cm_min*(cn-cm_min) ;
        csize = do_rank_detection ? csize_max : csize_min ;

        Cm [f] = do_rank_detection ? cm_max : cm_min ;

        PR (("this front cm: [ %ld %ld ], cn %ld csize %ld [%ld to %ld]\n",
            cm_min, cm_max, cn, csize, csize_min, csize_max)) ;
        ASSERT (csize >= 0 && csize <= fsize) ;
        ASSERT (cm_min >= 0) ;
        ASSERT (cm_max >= 0) ;
        ASSERT (cn >= 0) ;
        ASSERT (cm_min <= cn) ;
        ASSERT (cm_max <= cn) ;
        ASSERT (cm_min <= cm_max) ;
        ASSERT (csize_min <= csize_max) ;

#ifdef GPU_BLAS
        if(useGPU)
        {
            // Compute Rjmap Offsets.
            RjmapOffsets[f] = RjmapSize;
            RjmapSize += fn - fp;               // # cols - # piv cols

            // Compute Rimap Offsets.
            RimapOffsets[f] = RimapSize;
            RimapSize += Cm[f];                 // # rows of C

            // Munge Sj to cut down on assembly time.
            for(Long k=0 ; k<fp ; k++)
            {
                /* assemble all rows whose leftmost global column index is
                 * k+col1 */
                Long leftcol = k + col1 ;
                for (Long row = Sleft[leftcol] ; row < Sleft[leftcol+1] ; row++)
                {
                    /* scatter the row into F */
                    for (p=Sp[row] ; p<Sp[row+1] ; p++) Sj[p] = Fmap[Sj[p]];
                }
            }
        }
#endif

        // ---------------------------------------------------------------------
        // determine flop counts and upper bounds on the size R+H if H kept
        // ---------------------------------------------------------------------

        double fflops = 0 ;                 // flop count for this front
        Long rhsize = 0 ;                   // count entire staircase
        for (j = 0 ; j < fn ; j++)
        {
            t = MAX (j+1, Stair [j]) ;      // assume diagonal is present
            t = MIN (t, fm) ;               // except t cannot exceed fm
            PR (("   j %ld Stair %ld t %ld\n", j, Stair [j], t)) ;
            rhsize += t ;                   // Long overflow cannot occur
            if (t > j)
            {
                double h = (t-j) ;          // length of Householder vector
                fflops += 3*h               // compute Householder vector
                    + 4*h*(fn-j-1) ;        // apply to cols j+1:fn-1
            }
        }

        rhsize -= csize_min ;               // then exclude C block

        if (keepH && do_parallel_analysis)
        {
            Rh [f] = rhsize ;
        }

        rhxsize += rhsize ;

        PR (("\n --- front f %ld fsize %ld csize %ld\n",
            f, fsize, csize)) ;
        PR (("    rsize %ld  rhsize %ld\n", rsize, rhsize)) ;
        PR (("    parent %ld\n", Parent [f])) ;
        PR (("Front %ld RHSIZE predicted: %ld\n", f, rhsize)) ;
        ASSERT (rhsize >= 0 && rhsize <= fsize) ;

        ASSERT (rhsize >= rsize) ;

        // ---------------------------------------------------------------------
        // find the flop count for node f and all its descendants
        // ---------------------------------------------------------------------

        total_flops += fflops ;
        if (do_parallel_analysis)
        {
            Flops [f] = fflops ;            // flops for just f itself
            for (p = Childp [f] ; p < Childp [f+1] ; p++)
            {
                fflops += Flops_subtree [Child [p]] ;
            }
            // flops for entire subtree rooted at f, including f itself
            Flops_subtree [f] = fflops ;
        }

        // ---------------------------------------------------------------------
        // estimate stack usage for sequential case
        // ---------------------------------------------------------------------

        // allocate the front F
        stack = spqr_add (stack, fsize, &ok) ;
        maxstack = MAX (maxstack, stack) ;

        // assemble/delete children and factor F
        stack -= ctot ;
        ASSERT (stack >= 0) ;

        // create the C block for front F
        stack = spqr_add (stack, csize, &ok) ;
        maxstack = MAX (maxstack, stack) ;

        // delete F
        stack -= fsize ;
        ASSERT (stack >= 0) ;

        if (keepH)
        {
            // allocate RH in place of F (no additional memory required)
            stack += rhsize ;

        }
        else
        {
            // allocate R in place of F (no additional memory required)
            stack += rsize ;
        }
    }

#ifdef GPU_BLAS
    if(useGPU)
    {
        /* Save GPU size members. */
        QRgpu->RjmapSize = RjmapSize;
        QRgpu->RimapSize = RimapSize;
    }
#endif

    // -------------------------------------------------------------------------
    // compute Hip
    // -------------------------------------------------------------------------

    hisize = 0 ;            // total size of the row indices of H
    if (keepH)
    {
        for (f = 0 ; f < nf ; f++)
        {
            // allocate Hi
            fm = Fm [f] ;           // # rows of F
            Hip [f] = hisize ;
            hisize = spqr_add (hisize, fm, &ok) ;
        }
        Hip [nf] = hisize ;
    }

    // -------------------------------------------------------------------------
    // check for Long overflow
    // -------------------------------------------------------------------------

    PR (("stack     %ld\n", stack)) ;
    PR (("maxstack  %ld\n", maxstack)) ;
    PR (("rxsize    %ld\n", rxsize)) ;
    PR (("ok %d m %ld n %ld nf: %ld stacksize: %ld\n", ok,
        m, n, nf, maxstack)) ;

    if (!ok)
    {
        // Long overflow has occured
        spqr_freesym (&QRsym, cc) ;
        FREE_WORK ;
        ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // finalize flop counts and turn off parallelism if problem too small
    // -------------------------------------------------------------------------

    if (do_parallel_analysis)
    {
        Flops [nf] = 0 ;
        Flops_subtree [nf] = total_flops ;

#ifndef NDEBUG
        double fflops2 = 0 ;
        for (p = Childp [nf] ; p < Childp [nf+1] ; p++)
        {
            c = Child [p] ;                 // get a root (child of nf)
            fflops2 += Flops_subtree [c] ;  // add the flops of the root
        }
        PR (("fflops2 %g %g\n", fflops2, total_flops)) ;
        ASSERT (fflops2 == total_flops) ;
#endif

        if (total_flops < cc->SPQR_small)
        {
            // problem is too small for parallelism
            do_parallel_analysis = FALSE ;
        }
    }

    // -------------------------------------------------------------------------
    // sequential wrap-up
    // -------------------------------------------------------------------------

    ASSERT (IMPLIES (!keepH, stack == rxsize)) ;
    ASSERT (IMPLIES (keepH, stack == rhxsize)) ;

    QRsym->maxstack = maxstack ;    // max size of stack
    QRsym->hisize = hisize ;        // max size of int part of H
    QRsym->maxfn = maxfn ;          // max # of columns in any frontal matrix

    cc->SPQR_flopcount_bound = total_flops ;   // total bound on flop count

    cc->SPQR_istat [0] = rxsize ;              // nnz (R)
    cc->SPQR_istat [1] = rhxsize - rxsize ;    // nnz (H)
    cc->SPQR_istat [2] = nf ;                  // number of frontal matrices
    cc->SPQR_istat [3] = 1 ;                   // ntasks, for now

    DEBUG (spqrDebug_dump_Parent (nf+1, Parent, "pfile")) ;
    DEBUG (spqrDebug_dump_Parent (1, NULL, "tfile")) ;

    PR (("flops %g\n", total_flops)) ;

#ifdef GPU_BLAS

    // -------------------------------------------------------------------------
    // if we're using GPU acceleration, construct static gpu stages
    // -------------------------------------------------------------------------

    if(useGPU)
    {
        /* Compute a schedule based on memory requirements. */
        QRgpu->Stagingp = (Long*)   cholmod_l_malloc(nf+2, sizeof(Long), cc);
        QRgpu->StageMap = (Long*)   cholmod_l_malloc(nf, sizeof(Long), cc);
        QRgpu->FSize    = (size_t*) cholmod_l_malloc(nf+1, sizeof(size_t), cc);
        QRgpu->RSize    = (size_t*) cholmod_l_malloc(nf+1, sizeof(size_t), cc);
        QRgpu->SSize    = (size_t*) cholmod_l_malloc(nf+1, sizeof(size_t), cc);
        QRgpu->FOffsets = (Long*)   cholmod_l_malloc(nf, sizeof(Long), cc);
        QRgpu->ROffsets = (Long*)   cholmod_l_malloc(nf, sizeof(Long), cc);
        QRgpu->SOffsets = (Long*)   cholmod_l_malloc(nf, sizeof(Long), cc);

        Stagingp = QRgpu->Stagingp ;
        StageMap = QRgpu->StageMap ;
        FSize    = QRgpu->FSize    ;
        RSize    = QRgpu->RSize    ;
        SSize    = QRgpu->SSize    ;
        FOffsets = QRgpu->FOffsets ;
        ROffsets = QRgpu->ROffsets ;
        SOffsets = QRgpu->SOffsets ;

        /* Check for out of memory when allocating staging fields. */
        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            spqr_freesym (&QRsym, cc) ;
            FREE_WORK ;
            return (NULL) ;
        }

        /* Compute the front staging. */
        bool feasible;
        spqrgpu_computeFrontStaging (nf, Parent, Childp, Child, Fm, Cm, Rp,
            Sp, Sleft, Super, Post, RimapSize, RjmapSize,
            &feasible, &numStages, Stagingp, StageMap,
            FSize, RSize, SSize, FOffsets, ROffsets, SOffsets, cc);

        #ifndef NPRINT
        PR (("Is schedule feasible? %s with %ld stages.\n",
            feasible ? "yes" : "no", numStages)) ;
        #endif

        /* If we're not feasible then return that the problem is
           too large to run on the GPU. */
        if(!feasible)
        {
            spqr_freesym (&QRsym, cc) ;
            FREE_WORK ;
            return (NULL) ;
        }

        // save the total # of stages
        QRgpu->numStages = numStages;
    }

#endif

    // -------------------------------------------------------------------------
    // quick return if no parallel analysis
    // -------------------------------------------------------------------------

    if (!do_parallel_analysis)
    {
        // turn off parallelism and return the QRsym result
        QRsym->ntasks = 1 ;
        QRsym->ns = 1 ;
        FREE_WORK ;
        return (QRsym) ;
    }

    // =========================================================================
    // === parallel analysis ===================================================
    // =========================================================================

    // a front f is "big" if the work in the subtree rooted at f is > big_flops

    double big_flops = total_flops / cc->SPQR_grain ;
    big_flops = MAX (big_flops, cc->SPQR_small) ;
    double small_flops = big_flops / 16 ;
    PR (("total flops %g\n", total_flops)) ;
    PR (("big flops   %g\n", big_flops)) ;
    PR (("small flops %g\n", small_flops)) ;

    // -------------------------------------------------------------------------
    // initialize task assignment for each node
    // -------------------------------------------------------------------------

    // Stair, Fmap no longer needed ] ]
    // Still need Fm, Cm, Rh, Flops, and Flops_subtree however.
    // Since Stair and Fmap are no longer needed, we can use that workspace for
    // Task and InvPost.  Fm and Cm in W [0:...] are still needed.

    Task = W ;                          // size nf+1 [
    InvPost = Task + (nf+1) ;           // size nf+1 [

    for (Long k = 0 ; k <= nf ; k++)
    {
        f = Post [k] ;
        InvPost [f] = k ;
    }

    // Task [f] will be in one of four states: (a) small and unassigned, (b)
    // small and assigned, (c) pending with no small child task, (c) pending
    // with small child task.  For big nodes, the task assignment is pending,
    // and a pending task can have registered with it a small child task t.
    // This is done by setting Task [f] to -t-3; note that this value is always
    // < -2.  A pending node f with no small child task t is denoted with Task
    // [f] = -2.  For small nodes, the task assignment is empty (-1) or is
    // assigned to a task t = Task [f] where t >= 0.

#define PENDING (-2)
#define TASK_IS_PENDING(f) (Task [f] <= PENDING)
#define SMALL_TASK(f) (TASK_IS_PENDING (f) ? ((- Task [f]) - 3) : EMPTY)
#define ASSIGN_SMALL_TASK(f,task) { Task [f] = -(task) - 3 ; }

    for (f = 0 ; f < nf ; f++)
    {
        // a big node is marked pending, and will not be assigned in first pass.
        // Task [f] = PENDING means that the subtree rooted at f has more than
        // big_flops work in it.
        Task [f] = (Flops_subtree [f] > big_flops) ? PENDING : EMPTY ;
        ASSERT (SMALL_TASK (f) == EMPTY) ;
    }
    Task [nf] = PENDING ;
    ASSERT (SMALL_TASK (nf) == EMPTY) ;

    for (f = 0 ; f <= nf ; f++)
    {
        PR (("front %ld parent %ld flops %g subtree flops %g task %ld\n",
            f, Parent [f], Flops [f], Flops_subtree [f], Task [f])) ;
    }

    // -------------------------------------------------------------------------
    // assign small subtrees to tasks
    // -------------------------------------------------------------------------

    // The contents of Flops_subtree is required only for fronts marked with
    // Task [f] = EMPTY.  For a big front (one marked Task [f] = PENDING), this
    // information is no longer needed.  However, a large front f keeps track
    // of its mostly recently seen small child, and for this Flops_subtree [f]
    // is used.

    ntasks = 0 ;

    PR (("\n\n =========== parallel \n")) ;

    for (kf = 0 ; kf < nf ; kf++)
    {
        Long fstart = Post [kf] ;
        if (Task [fstart] != EMPTY)
        {
            // fstart is already assigned to a task, or it's a pending big
            // task; skip it
            continue ;
        }

        PR (("\nfstart: %ld\n", fstart)) ;
        ASSERT (!TASK_IS_PENDING (fstart)) ;

        // ---------------------------------------------------------------------
        // traverse f towards the root, stopping at a pending (big) node
        // ---------------------------------------------------------------------

        for (f = fstart ; Task [Parent [f]] == EMPTY ; f = Parent [f])
        {
            PR (("    front %ld flops %g %g parent %ld (%ld)\n", f,
                Flops [f], Flops_subtree [f],
                Parent [f], Task [Parent [f]])) ;
            ASSERT (f >= 0 && f < nf) ;
            ASSERT (Parent [f] >= 0 && Parent [f] <= nf) ;
            ASSERT (Task [f] == EMPTY) ;
            ASSERT (!TASK_IS_PENDING (f)) ;
        }

        Long flast = f ;
        parent = Parent [flast] ;

        PR (("    >>> flast is %ld  flops: %g parent: %ld\n", flast,
            Flops_subtree [flast], parent)) ;

        // we are guaranteed not to see an assigned node
        ASSERT (flast >= 0 && flast < nf) ;
        ASSERT (parent >= 0 && parent <= nf) ;

        // the task assignment of the parent is pending
        // ASSERT (Task [parent] == PENDING) ;
        ASSERT (TASK_IS_PENDING (parent)) ;

        // the path is fstart --> flast, all marked empty

        // ---------------------------------------------------------------------
        // start a new task, or merge with a prior sibling
        // ---------------------------------------------------------------------

        double flops = Flops_subtree [flast] ;
        if (SMALL_TASK (parent) == EMPTY)
        {
            // pending parent of flast has no small child task;  get new task
            task = ntasks++ ;
            if (flops <= small_flops)
            {
                // this task is small; register it and its flop count w/ parent
                PR (("    a new small task child of %ld\n", parent)) ;
                ASSIGN_SMALL_TASK (parent, task) ;
                Flops_subtree [parent] = flops ;
            }
        }
        else
        {
            // The parent of flast has a prior small child task, which is rooted
            // at a node that is a sibling of flast.  Reuse that task for
            // the subtree rooted at flast.  Keep track of the flops of this
            // new subtree and the old small_task, and remove the small task
            // if it gets large enough to be no longer "small".
            task = SMALL_TASK (parent) ;
            Flops_subtree [parent] += flops ;
            PR (("    merge with prior small task %ld of parent %ld\n",
                task, parent)) ;
            if (flops > small_flops)
            {
                // the merged task is now large enough to be no longer small;
                // remove it from the small child task of the parent so the
                // next child of this parent doesn't merge with it.
                ASSIGN_SMALL_TASK (parent, task) ;
                Flops_subtree [parent] = 0 ;
                PR (("    no longer small, flops is %g\n", flops)) ;
            }
        }

        // ---------------------------------------------------------------------
        // place all nodes in the subtree rooted at flast in the current task
        // ---------------------------------------------------------------------

        klast = InvPost [flast] ;
        for (Long k = kf ; k <= klast ; k++)
        {
            f = Post [k] ;
            PR (("    assign %ld to %ld\n", f, task)) ;
            ASSERT (Task [f] == EMPTY) ;
            Task [f] = task ;
        }
    }

    // InvPost [0:nf] no longer needed ]

    for (f = 0 ; f <= nf ; f++)
    {
        PR (("Front %ld task %ld : parent %ld task %ld\n",
            f, Task [f], Parent [f],
            (Parent [f] == EMPTY) ? -999 : Task [Parent [f]])) ;
        ASSERT (Task [f] != EMPTY) ;
    }

    // -------------------------------------------------------------------------
    // complete the assignment for large nodes
    // -------------------------------------------------------------------------

    for (f = 0 ; f < nf ; f++)
    {
        // if (Task [f] == PENDING)
        if (TASK_IS_PENDING (f))
        {
            // if f has no children, or children in more than one task
            // the create a new task for f.  Otherwise, assign f to the task
            // of its child(ren).
            PR (("\nassign pending front %ld:\n", f)) ;
            if ((Childp [f+1] - Childp [f]) == 0)
            {
                // task f has no children; create a new task for it
                task = ntasks++ ;
                PR (("Pending %ld has no children (new task %ld)\n", f,task)) ;
            }
            else
            {
                task = EMPTY ;
                for (p = Childp [f] ; p < Childp [f+1] ; p++)
                {
                    c = Child [p] ;
                    PR (("child %ld in task %ld\n", c, Task [c])) ;
                    ASSERT (!TASK_IS_PENDING (c)) ;
                    if (task == EMPTY)
                    {
                        // get the task of the first child of f
                        task = Task [c] ;
                        ASSERT (task >= 0 && task < ntasks) ;
                    }
                    else if (Task [c] != task)
                    {
                        // f has children from more than one task; create a new
                        // task for front f
                        task = ntasks++ ;
                        PR (("Pending %ld (new task %ld)\n", f, task)) ;
                        break ;
                    }
                }
            }
            PR (("Assign %ld to task %ld\n", f, task)) ;
            ASSERT (task != EMPTY) ;
            Task [f] = task ;
        }
    }

    // -------------------------------------------------------------------------
    // quick return if only one real task found
    // -------------------------------------------------------------------------

    if (ntasks <= 1)
    {
        // No parallelism found
        QRsym->ntasks = 1 ;
        QRsym->ns = 1 ;
        FREE_WORK ;
        return (QRsym) ;
    }

    // -------------------------------------------------------------------------
    // wrap-up the task assignment
    // -------------------------------------------------------------------------

    Task [nf] = ntasks++ ;      // give the placeholder its own task

#ifndef NDEBUG
    PR (("\nFinal assignment: ntasks %ld nf %ld\n", ntasks, nf)) ;
    ASSERT (ntasks <= nf+1) ;
    // reuse Flops_subtree to compute flops for each task
    for (task = 0 ; task < ntasks ; task++)
    {
        Flops_subtree [task] = 0 ;
    }
    for (f = 0 ; f < nf ; f++)
    {
        ASSERT (Task [f] >= 0 && Task [f] < ntasks) ;
        Flops_subtree [Task [f]] += Flops [f] ;
    }
    for (f = 0 ; f <= nf ; f++)
    {
        PR (("Front %ld task %ld : parent %ld task %ld\n",
            f, Task [f], Parent [f],
            (Parent [f] == EMPTY) ? -999 : Task [Parent [f]])) ;
        ASSERT (!TASK_IS_PENDING (f)) ;
        ASSERT (Task [f] >= 0 && Task [f] < ntasks) ;
        if (f < nf && Task [f] != Task [Parent [f]])
        {
            PR (("%ld task %ld parent %ld flops %g\n",
                f, Task [f],
                Task [Parent [f]],
                Flops_subtree [Task [f]])) ;
        }
    }
    PR (("\n")) ;
#endif

    // Flops no longer needed
    cholmod_l_free (2*(nf+1), sizeof (double), Flops, cc) ;
    Flops = NULL ;
    Flops_subtree = NULL ;

    // In the frontal tree, there is one node per front (0 to nf-1), plus one
    // placeholder node (nf) that is the root of the tree, for a total of nf+1
    // nodes.  Note that a tree can have nf=0 (a matrix with m=0 rows).  There
    // is at most one task per node, including the placeholder.
    ASSERT (ntasks <= nf + 1) ;
    ASSERT (ntasks >= 3) ;
    QRsym->ntasks = ntasks ;
    cc->SPQR_istat [3] = ntasks ;

    // -------------------------------------------------------------------------
    // allocate space for the parallel analysis
    // -------------------------------------------------------------------------

    // FUTURE: TaskStack could be deleted as temp workspace.
    // During factorization, just get On_stack [f] for first f in the task.

    // TaskParent is temporary workspace:
    TaskParent = (Long *) cholmod_l_malloc (ntasks,   sizeof (Long), cc) ;

    TaskChildp = (Long *) cholmod_l_calloc (ntasks+2, sizeof (Long), cc) ;
    TaskChild  = (Long *) cholmod_l_calloc (ntasks+1, sizeof (Long), cc) ;
    TaskFront  = (Long *) cholmod_l_malloc (nf+1,     sizeof (Long), cc) ;
    TaskFrontp = (Long *) cholmod_l_calloc (ntasks+2, sizeof (Long), cc) ;
    TaskStack  = (Long *) cholmod_l_malloc (ntasks+1, sizeof (Long), cc) ;
    On_stack   = (Long *) cholmod_l_malloc (nf+1,     sizeof (Long), cc) ;

    QRsym->TaskFront  = TaskFront ;
    QRsym->TaskFrontp = TaskFrontp ;
    QRsym->TaskChild  = TaskChild ;
    QRsym->TaskChildp = TaskChildp ;
    QRsym->TaskStack  = TaskStack ;
    QRsym->On_stack   = On_stack ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freesym (&QRsym, cc) ;
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // build the task tree
    // -------------------------------------------------------------------------

    for (task = 0 ; task < ntasks ; task++)
    {
        TaskParent [task] = EMPTY ;
    }

    for (f = 0 ; f < nf ; f++)
    {
        Long my_task = Task [f] ;
        Long parent_task = Task [Parent [f]] ;
        PR (("f %ld Task %ld parent %ld, Task of parent %ld\n",
            f, my_task, Parent [f], Task [Parent [f]])) ;
        if (my_task != parent_task)
        {
            TaskParent [my_task] = parent_task ;
        }
    }

    DEBUG (spqrDebug_dump_Parent (ntasks, TaskParent, "tfile")) ;

    // count the child tasks of each task
    for (task = 0 ; task < ntasks ; task++)
    {
        parent = TaskParent [task] ;
        if (parent != EMPTY)
        {
            TaskChildp [parent]++ ;
        }
    }

    // construct TaskChildp pointers
    spqr_cumsum (ntasks, TaskChildp) ;

    // create the child lists
    for (Long child_task = 0 ; child_task < ntasks ; child_task++)
    {
        // place the child c in the list of its parent
        parent = TaskParent [child_task] ;
        if (parent != EMPTY)
        {
            TaskChild [TaskChildp [parent]++] =  child_task ;
        }
    }

    spqr_shift (ntasks, TaskChildp) ;

#ifndef NPRINT
    for (task = 0 ; task < ntasks ; task++)
    {
        PR (("Task %ld children:\n", task)) ;
        for (p = TaskChildp [task] ; p < TaskChildp [task+1] ; p++)
        {
            PR (("  %ld\n", TaskChild [p])) ;
        }
    }
#endif

    // -------------------------------------------------------------------------
    // create lists of fronts for each task
    // -------------------------------------------------------------------------

    // count the number of fronts in each task
    for (f = 0 ; f < nf ; f++)
    {
        TaskFrontp [Task [f]]++ ;
    }

    // construct TaskFrontp pointers
    spqr_cumsum (ntasks+1, TaskFrontp) ;

    // create the front lists for each task, in postorder
    for (kf = 0 ; kf < nf ; kf++)
    {
        f = Post [kf] ;
        task = Task [f] ;
        TaskFront [TaskFrontp [task]++] = f ;
    }

    spqr_shift (ntasks+1, TaskFrontp) ;

    // Task [0:nf] no longer needed ]

    // -------------------------------------------------------------------------
    // assign a stack to each task
    // -------------------------------------------------------------------------

    PR (("\n\nAssign stacks to tasks\n")) ;

    for (task = 0 ; task < ntasks ; task++)
    {
        TaskStack [task] = EMPTY ;
    }

    for (Long task_start = 0 ; task_start < ntasks ; task_start++)
    {
        if (TaskStack [task_start] == EMPTY)
        {
            // start a new stack
            Long s = ns++ ;
            for (task = task_start ;
                task != EMPTY && TaskStack [task] == EMPTY ;
                task = TaskParent [task])
            {
                TaskStack [task] = s ;
            }
        }
        PR (("Task %ld on stack %ld   parent %ld\n",
            task_start, TaskStack [task_start], TaskParent [task_start])) ;
    }

    // first task (task = 0) always uses the first stack
    ASSERT (TaskStack [0] == 0) ;

    // There is exactly one stack per leaf in the task tree, and thus at most
    // one stack per task.
    ASSERT (ns > 1 && ns <= ntasks) ;
    QRsym->ns = ns ;

    for (task = 0 ; task < ntasks ; task++)
    {
        Long s = TaskStack [task] ;
        PR (("\nTask %ld children:\n", task)) ;
        for (p = TaskChildp [task] ; p < TaskChildp [task+1] ; p++)
        {
            PR (("  child Task %ld\n", TaskChild [p])) ;
        }
        PR (("\nTask %ld fronts: nf = %ld\n", task, nf)) ;
        for (p = TaskFrontp [task] ; p < TaskFrontp [task+1] ; p++)
        {
            f = TaskFront [p] ;
            PR (("  Front %ld task %ld stack %ld\n",f, task, TaskStack [task]));
            ASSERT (f >= 0 && f < nf) ;
            On_stack [f] = s ;
        }
    }

    // -------------------------------------------------------------------------
    // allocate the rest of QRsym
    // -------------------------------------------------------------------------

    // temporary workspace:
    // Stack_stack (s): current stack usage
    Stack_stack = (Long *) cholmod_l_calloc (ns+2, sizeof (Long), cc) ;

    // permanent part of QRsym:
    // Stack_maxstack (s): peak stack usage if H not kept
    Stack_maxstack = (Long *) cholmod_l_calloc (ns+2, sizeof (Long), cc) ;

    // FUTURE: keep track of maxfn for each stack

    QRsym->Stack_maxstack = Stack_maxstack ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freesym (&QRsym, cc) ;
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // find the stack sizes
    // -------------------------------------------------------------------------

    for (kf = 0 ; kf < nf ; kf++)
    {

        // ---------------------------------------------------------------------
        // analyze the factorization of front F on stack s
        // ---------------------------------------------------------------------

        f = Post [kf] ;
        Long s = On_stack [f] ;
        PR (("\n----------------------- front: %ld on stack %ld\n", f, s)) ;
        ASSERT (f >= 0 && f < nf) ;
        ASSERT (s >= 0 && s < ns) ;

        // fp = # of pivotal columns (Super [f] ... Super [f+1]-1)
        fp = Super [f+1] - Super[f] ;
        fn = Rp [f+1] - Rp [f] ;    // exact number of columns of F
        ASSERT (fp > 0) ;           // all fronts have at least one pivot col
        ASSERT (fp <= fn) ;
        ASSERT (fn <= n) ;
        PR (("  fn %ld fp %ld\n", fn, fp)) ;

        // ---------------------------------------------------------------------
        // assemble each child (ignore children from other stacks)
        // ---------------------------------------------------------------------

        ctot = 0 ;
        for (p = Childp [f] ; p < Childp [f+1] ; p++)
        {
            c = Child [p] ;                 // get the child c of front F
            ASSERT (c >= 0 && c < f) ;
            if (On_stack [c] != s)
            {
                // skip this child if not on my own stack
                PR (("child %ld skip\n", c)) ;
                continue ;
            }
            fnc = Rp [c+1] - Rp [c] ;       // total # cols in child F
            fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
            cn = fnc - fpc ;                // # of cols in child C
            cm = Cm [c] ;                   // # of rows in child C
            ASSERT (cm >= 0 && cm <= cn) ;
            csize = cm*(cm+1)/2 + cm*(cn-cm) ;
            ctot += csize ;
            PR (("child %ld size %ld  = %ld * %ld\n", c, cm*cn, cm, cn)) ;
        }

        // ---------------------------------------------------------------------
        // determine upper bounds on the size F
        // ---------------------------------------------------------------------

        // The front F is at most fm-by-fn, with fp pivotal columns.
        fm = Fm [f] ;                       // # rows of F
        fsize = fm * fn ;
        ASSERT (fm >= 0) ;
        ASSERT (fn >= 0) ;
        ASSERT (fp >= 0 && fp <= fn) ;
        PR (("fm %ld fn %ld rank %d\n", fm, fn, do_rank_detection)) ;
        ASSERT (IMPLIES (keepH, fm == (Hip [f+1] - Hip [f]))) ;

        // ---------------------------------------------------------------------
        // determine upper bounds on the size R
        // ---------------------------------------------------------------------

        // The R block of F is at most rm-by-rn packed upper trapezoidal.
        rm = MIN (fm, fp) ;     // exact/max # of rows of R block
        rn = fn ;               // exact/max # of columns of R block
        ASSERT (rm <= rn) ;
        ASSERT (rm >= 0) ;
        ASSERT (rm <= fm) ;
        rsize = rm*(rm+1)/2 + rm*(rn-rm) ;
        ASSERT (rsize >= 0 && rsize <= fsize) ;
        PR ((" rm %ld rn %ld rsize %ld\n", rm, rn, rsize)) ;

        // ---------------------------------------------------------------------
        // determine upper bounds on the size C
        // ---------------------------------------------------------------------

        // The C block of F is at most cm-by-cn packed upper trapezoidal
        cn = fn - fp ;          // exact # of columns in C block

        // with rank detection and possible pivot failures
        cm_max = fm ;
        cm_max = MIN (cm_max, cn) ;         // max # rows in C block of F

        // with no pivot failures
        cm_min = MAX (fm - rm, 0) ;
        cm_min = MIN (cm_min, cn) ;         // exact # rows in C block

        // Long overflow cannot occur:
        csize_max = cm_max*(cm_max+1)/2 + cm_max*(cn-cm_max) ;
        csize_min = cm_min*(cm_min+1)/2 + cm_min*(cn-cm_min) ;
        csize = do_rank_detection ? csize_max : csize_min ;

        ASSERT (Cm [f] == (do_rank_detection ? cm_max : cm_min)) ;

        PR (("this front cm: [ %ld %ld ], cn %ld csize %ld [%ld to %ld]\n",
            cm_min, cm_max, cn, csize, csize_min, csize_max)) ;
        ASSERT (csize >= 0 && csize <= fsize) ;
        ASSERT (cm_min >= 0) ;
        ASSERT (cm_max >= 0) ;
        ASSERT (cn >= 0) ;
        ASSERT (cm_min <= cn) ;
        ASSERT (cm_max <= cn) ;
        ASSERT (cm_min <= cm_max) ;
        ASSERT (csize_min <= csize_max) ;

        // ---------------------------------------------------------------------
        // determine upper bounds on the size R+H
        // ---------------------------------------------------------------------

        PR (("\n --- front f %ld fsize %ld ctot %ld csize %ld\n",
            f, fsize, ctot, csize)) ;
        PR (("    parent %ld\n", Parent [f])) ;

        // ---------------------------------------------------------------------
        // estimate stack usage for parallel case
        // ---------------------------------------------------------------------

        Long ss = Stack_stack [s] ;  // current size of stack
        Long sm = Stack_maxstack [s] ;  // max size of stack
        PR (("current ss: %ld fsize %ld ctot %ld csize %ld rsize %ld\n",
            ss, fsize, ctot, csize, rsize)) ;

        // allocate front F on stack s
        ss += fsize ;
        sm = MAX (sm, ss) ;

        // assemble and delete children on s, then factor F
        ss -= ctot ;
        ASSERT (ss >= 0) ;

        // create C for front F
        ss += csize ;
        sm = MAX (sm, ss) ;

        // delete F
        ss -= fsize ;
        ASSERT (ss >= 0) ;

        if (keepH)
        {
            // allocate R and H in place of F
            ss += Rh [f] ;
        }
        else
        {
            // allocate R in place of F (no memory allocated)
            ss += rsize ;
        }

        PR (("new ss %ld\n", ss)) ;
        Stack_stack [s] = ss ;      // new size of stack s
        Stack_maxstack [s] = sm ;   // new max of stack s
    }

    // Cm no longer needed in Iwork ]

    // -------------------------------------------------------------------------
    // free workspace and return result
    // -------------------------------------------------------------------------

    FREE_WORK ;
    return (QRsym) ;
}

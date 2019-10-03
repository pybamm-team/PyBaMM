// =============================================================================
// === spqrgpu_kernel ==========================================================
// =============================================================================

// Compute the QR factorization on the GPU.  Does not handle rank-deficient
// matrices (R is OK, but will not be 'squeezed').  Does not handle complex
// matrices.  Does not return the Householder vectors.

#ifdef GPU_BLAS
#include "spqr.hpp"

#ifdef SUITESPARSE_TIMER_ENABLED
#define INIT_TIME(x)    double x = 0.0; double time_ ## x;
#define TIC(x)          time_ ## x = SuiteSparse_time();
#define TOC(x)          x += SuiteSparse_time() - time_ ## x;
#else
#define INIT_TIME(x)
#define TIC(x)
#define TOC(x)
#endif

// -----------------------------------------------------------------------------
// numfronts_in_stage
// -----------------------------------------------------------------------------

void numfronts_in_stage
(
    // input, not modified
    Long stage,         // count the # of fronts in this stage
    Long *Stagingp,     // fronts are in the list
                        //      Post [Stagingp [stage]...Stagingp[stage+1]-1]
    Long *StageMap,     // front f is in stage StageMap [f]
    Long *Post,         // array of size nf (# of fronts)
    Long *Child,        // list of children for each front is
    Long *Childp,       //      in Child [Childp [f] ... Childp [f+1]-1]

    // output, not defined on input
    Long *p_numFronts,          // number of fronts in stage (a scalar)
    Long *p_leftoverChildren    // number of leftover children (a scalar)
)
{
    // the # of fronts at this stage is given by the Stagingp workspace
    // plus any children within the stage that must still be assembled
    Long sStart = Stagingp[stage];
    Long sEnd = Stagingp[stage+1];
    Long numFronts = (sEnd - sStart);
    Long leftoverChildren = 0;
    for(Long p=sStart; p<sEnd; p++) // for each front in the stage
    {
        Long f = Post[p];
        for(Long cp=Childp[f]; cp<Childp[f+1]; cp++)
        {
            Long c = Child[cp];
            if(StageMap[c] < stage) leftoverChildren++;
        }
    }
    numFronts += leftoverChildren ;
    *p_numFronts = numFronts ;
    *p_leftoverChildren = leftoverChildren ;
}


// -----------------------------------------------------------------------------
// spqrgpu_kernel
// -----------------------------------------------------------------------------

void spqrgpu_kernel
(
    spqr_blob <double> *Blob
)
{

    INIT_TIME(complete);
    TIC(complete);

    // -------------------------------------------------------------------------
    // get the Blob
    // -------------------------------------------------------------------------

    spqr_symbolic *QRsym = Blob->QRsym ;
    spqr_numeric <double> *QRnum = Blob->QRnum ;
    spqr_work <double> *Work = Blob->Work ;
    double *Sx = Blob->Sx ;
//  Long ntol = Blob->ntol ;        // no rank detection on the GPU
    cholmod_common *cc = Blob->cc ;

    // -------------------------------------------------------------------------
    // get the contents of the QR symbolic object
    // -------------------------------------------------------------------------

    Long *   Super = QRsym->Super ;      // size nf+1, gives pivot columns in F
    Long *   Rp = QRsym->Rp ;            // size nf+1, pointers for pattern of R
    Long *   Rj = QRsym->Rj ;            // size QRsym->rjsize, col indices of R
    Long *   Sleft = QRsym->Sleft ;      // size n+2, leftmost column sets
    Long *   Sp = QRsym->Sp ;            // size m+1, row pointers for S
    Long *   Sj = QRsym->Sj ;            // size anz, column indices for S
    Long *   Parent = QRsym->Parent ;    // size nf, for parent index
    Long *   Child = QRsym->Child ;      // size nf, for lists of children
    Long *   Childp = QRsym->Childp ;    // size nf+1, for lists of children
    Long *   Fm = QRsym->Fm ;            // number of rows in F
    Long *   Cm = QRsym->Cm ;            // # of rows in contribution blocks
    Long     nf = QRsym->nf ;            // number of fronts
    Long     n = QRsym->n ;              // number of columns
    Long     m = QRsym->m ;              // number of rows
    Long *   Post = QRsym->Post ;        // size nf

    // -------------------------------------------------------------------------
    // get the contents of the QR numeric object
    // -------------------------------------------------------------------------

    double ** Rblock = QRnum->Rblock ;
    char *    Rdead = QRnum->Rdead ;

    // -------------------------------------------------------------------------
    // get the stack for this task and the head/top pointers
    // -------------------------------------------------------------------------

    Long stack = 0 ;                    // no mixing of GPU and TBB parallelism
    ASSERT (QRnum->ntasks == 1) ;

    double * Stack_top = Work [stack].Stack_top ;
    double * Stack_head = Work [stack].Stack_head ;

    Long sumfrank = Work [stack].sumfrank ;
    Long maxfrank = Work [stack].maxfrank ;

    // -------------------------------------------------------------------------
    // get the SPQR GPU members from symbolic analysis
    // -------------------------------------------------------------------------

    spqr_gpu *QRgpu = QRsym->QRgpu;

    // assembly metadata
    Long *RjmapOffsets = QRgpu->RjmapOffsets;
    Long *RimapOffsets = QRgpu->RimapOffsets;
    Long RjmapSize = MAX(1, QRgpu->RjmapSize);
    Long RimapSize = MAX(1, QRgpu->RimapSize);

    // staging metadata
    Long numStages = QRgpu->numStages;
    Long *Stagingp = QRgpu->Stagingp;
    Long *StageMap = QRgpu->StageMap;
    size_t *FSize = QRgpu->FSize;
    size_t *RSize = QRgpu->RSize;
    size_t *SSize = QRgpu->SSize;
    Long *FOffsets = QRgpu->FOffsets;
    Long *ROffsets = QRgpu->ROffsets;
    Long *SOffsets = QRgpu->SOffsets;

    // gpu parameters
    size_t gpuMemorySize = cc->gpuMemorySize;

    // -------------------------------------------------------------------------
    // use a CUDA stream for asynchronous memory transfers
    // -------------------------------------------------------------------------

    cudaStream_t memoryStreamH2D;
    cudaError_t result = cudaStreamCreate(&memoryStreamH2D);
    if(result != cudaSuccess)
    {
        ERROR (CHOLMOD_GPU_PROBLEM, "cannot create CUDA stream to GPU") ;
        return ;
    }

    // -------------------------------------------------------------------------
    // use one mongo Stair for the entire problem
    // -------------------------------------------------------------------------

    Long stairsize = Rp [nf] ;
    Long *Stair = (Long*) cholmod_l_malloc (stairsize, sizeof(Long), cc);

    // -------------------------------------------------------------------------
    // use a workspace directory to store contribution blocks, if needed
    // semantics: create as needed, destroy after used
    // -------------------------------------------------------------------------

    Workspace **LimboDirectory =
        (Workspace**) cholmod_l_calloc (nf, sizeof(Workspace*), cc);

    // -------------------------------------------------------------------------
    // allocate, construct, and ship S, Rimap, and Rjmap for the entire problem
    // -------------------------------------------------------------------------

    // malloc wsMondoS on the CPU (pagelocked), not on the GPU
    Workspace *wsMondoS = Workspace::allocate (Sp[m],       // CPU pagelocked
        sizeof(SEntry), false, true, false, true) ;

    // malloc wsRimap and wsRjmap on both the CPU and GPU (not pagelocked)
    Workspace *wsRimap  = Workspace::allocate (RimapSize,   // CPU and GPU
        sizeof(int), false, true, true, false) ;
    Workspace *wsRjmap  = Workspace::allocate (RjmapSize,   // CPU and GPU
        sizeof(int), false, true, true, false) ;

    // use shared Long workspace (Iwork) for Fmap and InvPost [ [
    // Note that Iwork (0:nf-1) is already in use for Blob.Cm (size nf)
    // was: Long *Fmap = (Long*) cholmod_l_malloc (n, sizeof(Long), cc);
    cholmod_l_allocate_work (0, 2*nf + n + 1, 0, cc) ;
    Long *Wi = (Long *) cc->Iwork ;     // Cm is size nf, already in use
    Long *InvPost = Wi + nf ;           // InvPost is size nf+1
    Long *Fmap    = Wi + (nf+1) ;       // Fmap is size n

    Long numFronts = 0 ;
    Front *fronts = NULL ;
    Workspace *wsMondoF = NULL ;
    Workspace *wsMondoR = NULL ;
    Workspace *wsS = NULL ;

    Long wsMondoF_size = 0 ;
    Long wsMondoR_size = 0 ;
    Long wsS_size = 0 ;
    Long maxfronts_in_stage = 0 ;

    // -------------------------------------------------------------------------
    // check if out of memory
    // -------------------------------------------------------------------------

#define FREE_ALL_WORKSPACE \
        cudaStreamDestroy(memoryStreamH2D); \
        memoryStreamH2D = NULL ; \
        Stair = (Long*) cholmod_l_free (stairsize, sizeof(Long), Stair, cc) ; \
        if (LimboDirectory != NULL) \
        { \
            for (Long f2 = 0 ; f2 < nf ; f2++) \
            { \
                Workspace *wsLimbo2 = LimboDirectory[f2]; \
                if (wsLimbo2 != NULL) \
                { \
                    wsLimbo2->assign(wsLimbo2->cpu(), NULL); \
                    LimboDirectory[f2] = Workspace::destroy(wsLimbo2); \
                } \
            } \
        } \
        LimboDirectory = (Workspace**) \
            cholmod_l_free (nf, sizeof(Workspace*), LimboDirectory, cc); \
        wsMondoS = Workspace::destroy(wsMondoS); \
        wsRjmap  = Workspace::destroy(wsRjmap); \
        wsRimap  = Workspace::destroy(wsRimap); \
        fronts = (Front*) \
            cholmod_l_free (maxfronts_in_stage, sizeof(Front), fronts, cc) ; \
        wsMondoF = Workspace::destroy(wsMondoF); \
        wsMondoR = Workspace::destroy(wsMondoR); \
        if (wsS != NULL) wsS->assign(NULL, wsS->gpu()); \
        wsS = Workspace::destroy(wsS);

        // was: Fmap = (Long*) cholmod_l_free (n, sizeof(Long), Fmap, cc);

    if (cc->status < CHOLMOD_OK     // ensures Fmap and InvPost are allocated
        || !Stair || !LimboDirectory || !wsMondoS || !wsRimap || !wsRjmap)
    {
        // cholmod_l_allocate_work will set cc->status if it fails, but it
        // will "never" fail since the symbolic analysis has already allocated
        // a larger Iwork than is required by the factorization
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
        FREE_ALL_WORKSPACE ;
        return;
    }

    // -------------------------------------------------------------------------
    // build the assembly maps
    // -------------------------------------------------------------------------

    // construct/pack S, Rimap, and Rjmap
    spqrgpu_buildAssemblyMaps (
        nf, n,
        Fmap,       // workspace for spqrgpu_buildAssemblyMaps
        Post, Super, Rp, Rj,
        Sleft, Sp, Sj, Sx,
        Fm, Cm,
        Childp, Child,
        Stair,      // StairOffsets not needed because Rp suffices as offsets
        CPU_REFERENCE(wsRjmap, int*), RjmapOffsets,
        CPU_REFERENCE(wsRimap, int*), RimapOffsets,
        CPU_REFERENCE(wsMondoS, SEntry*)
                    // , Sp[m] (parameterno longer needed)
    ) ;

    // done using Iwork for Fmap ]
    // Fmap = (Long*) cholmod_l_free (n, sizeof(Long), Fmap, cc);

    // -------------------------------------------------------------------------
    // ship the assembly maps asynchronously along the H2D memory stream.
    // -------------------------------------------------------------------------

    wsRjmap->transfer(cudaMemcpyHostToDevice, false, memoryStreamH2D);
    wsRimap->transfer(cudaMemcpyHostToDevice, false, memoryStreamH2D);

    // Keep track of where we are in MondoS
    Long SOffset = 0;

    // -------------------------------------------------------------------------
    // allocate workspace for all stages
    // -------------------------------------------------------------------------

    for(Long stage=0; stage<numStages; stage++)
    {
        // find the memory requirements of this stage
        Long leftoverChildren ;
        numfronts_in_stage (stage, Stagingp, StageMap, Post, Child, Childp,
            &numFronts, &leftoverChildren) ;
        wsMondoF_size      = MAX (wsMondoF_size,      FSize [stage]) ;
        wsMondoR_size      = MAX (wsMondoR_size,      RSize [stage]) ;
        wsS_size           = MAX (wsS_size,           SSize [stage]) ;
        maxfronts_in_stage = MAX (maxfronts_in_stage, numFronts) ;
    }

    // The front listing has fronts in the stage at
    // the beginning with children needing assembly appearing at the end
    fronts = (Front*) cholmod_l_malloc (maxfronts_in_stage, sizeof(Front),cc) ;

    // This allocate is done once for each stage, but we still need
    // to set the first FSize[stage] entries in wsMondoF to zero, for each
    // stage.  So malloc the space on the GPU once, but see cudaMemset below.
    // This workspace is not on the CPU.
    wsMondoF = Workspace::allocate (wsMondoF_size,      // GPU only
        sizeof(double), false, false, true, false) ;

    // malloc R for each front on the CPU (not on the GPU)
    wsMondoR = Workspace::allocate (wsMondoR_size,      // CPU only
        sizeof(double), false, true, false, false) ;

    // malloc S on the GPU (not on the CPU)
    wsS = Workspace::allocate (wsS_size,                // GPU only
        sizeof(SEntry), false, false, true, false) ;

    if(!fronts || !wsMondoF || !wsMondoR || !wsS)
    {
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
        FREE_ALL_WORKSPACE ;
        return ;
    }

    // -------------------------------------------------------------------------
    // construct InvPost, the inverse postordering
    // -------------------------------------------------------------------------

    for (Long k = 0 ; k < nf ; k++)
    {
        Long f = Post [k] ;     // front f is the kth front in the postoder
        ASSERT (f >= 0 && f < nf) ;
        InvPost [f] = k ;       // record the same info in InvPost
    }
    InvPost [nf] = nf ;         // final placeholder

    // -------------------------------------------------------------------------
    // iterate over the staging schedule and factorize each stage
    // -------------------------------------------------------------------------

    for(Long stage=0; stage<numStages; stage++)
    {
        Long leftoverChildren ;

        PR (("Building stage %ld of %ld\n", stage+1, numStages));

        numfronts_in_stage (stage, Stagingp, StageMap, Post, Child, Childp,
            &numFronts, &leftoverChildren) ;

        Long sStart = Stagingp[stage];
        Long sEnd = Stagingp[stage+1];
        Long pNextChild = 0;

        PR (("# fronts in stage has %ld (%ld regular + %ld leftovers).\n",
            numFronts, numFronts-leftoverChildren, leftoverChildren));

        // set fronts on the GPU to all zero
        cudaMemset (wsMondoF->gpu(), (size_t) 0,
            FSize [stage] * sizeof (double)) ;

        // surgically transfer S input values to the GPU
        Workspace surgical_S (SSize [stage], sizeof (SEntry)) ;
        surgical_S.assign (((SEntry*) wsMondoS->cpu()) + SOffset, wsS->gpu());
        surgical_S.transfer(cudaMemcpyHostToDevice, false, memoryStreamH2D);
        surgical_S.assign (NULL, NULL) ;

        SOffset += SSize[stage];

        // get the fronts ready for GPUQREngine
        for(Long p=sStart; p<sEnd; p++)
        {
            // get the front, parent, and relative p
            Long f = Post[p];

            // compute basic front fields
            Long fm = Fm[f];                    // F is fm-by-fn
            Long fn = Rp [f+1] - Rp [f];
            Long col1 = Super [f] ;             // first global pivot col in F
            Long fp = Super[f+1] - col1 ;       // with fp pivot columns
            Long frank = MIN(fm, fp);

            // create the front
            Long frelp = leftoverChildren + (p - sStart);
            Long pid = Parent[f];
            Long gpid = Parent[pid];
            Long prelp = EMPTY;      // assume that we have no parent in stage

            // if we have a non-dummy parent in same stage.
            if(gpid != EMPTY && StageMap[pid] == stage)
            {
                Long pp = InvPost [pid] ;
                prelp = leftoverChildren + (pp - sStart);
            }

            PR (("building front %ld (%ld) with parent %ld (%ld)\n",
                f, frelp, pid, prelp));

            // using placement new
            Front *front = new (&fronts[frelp]) Front(frelp, prelp, fm, fn);
            front->fidg = f;
            front->pidg = pid;

            // override the default (dense) rank
            front->rank = frank;

            // attach the Stair
            front->Stair = Stair + Rp[f];

            // set pointer offsets into mondo workspaces
            front->gpuF = GPU_REFERENCE(wsMondoF, double*) + FOffsets[f];
            front->cpuR = CPU_REFERENCE(wsMondoR, double*) + ROffsets[f];

            front->sparseMeta = SparseMeta();
            SparseMeta *meta = &(front->sparseMeta);
            meta->isSparse = true;
            meta->fp = fp;
            meta->nc = Childp[f+1] - Childp[f];

            // S assembly
            Long pSStart, pSEnd;
            pSStart = Sp[Sleft[Super[f]]];
            pSEnd = Sp[Sleft[Super[f+1]]];
            meta->Scount = MAX(0, pSEnd - pSStart);
            meta->cpuS = CPU_REFERENCE(wsS, SEntry*) + SOffsets[f];
            meta->gpuS = GPU_REFERENCE(wsS, SEntry*) + SOffsets[f];

            // pack assembly
            meta->cn = fn - fp;
            meta->cm = Cm[f];
            meta->csize = meta->cm * meta->cn;
            if(prelp != EMPTY) // parent in stage
            {
                meta->pn = Rp[pid+1] - Rp[pid];
                meta->pc = Rp[f] + meta->fp;
                meta->gpuRjmap = GPU_REFERENCE(wsRjmap, int*) + RjmapOffsets[f];
                meta->gpuRimap = GPU_REFERENCE(wsRimap, int*) + RimapOffsets[f];
                meta->gpuC = front->gpuF + front->rank * front->fn + meta->fp;
            }
            else if(Parent[pid] != EMPTY) // parent not in stage
            {
                meta->isStaged = true;
            }

            // build any lingering children
            // the memory for the child fronts is at the end of its parent
            // the Front metadata resides at the beginning of the frontList
            Long ChildOffset = FOffsets[f] + fm*fn;
            for(Long cp=Childp[f]; cp<Childp[f+1]; cp++)
            {
                Long c = Child[cp];
                if(StageMap[c] == stage) continue; // only consider leftovers

                // Long cfm = Fm[c];
                Long cfn = Rp [c+1] - Rp [c];
                Long cfp = Super[c+1] - Super[c];
                // Long crank = MIN(cfm, cfp);
                Long ccn = cfn - cfp;
                Long ccm = Cm[c];

                // only copy the exact CBlock back to the GPU
                PR (("building child %ld (%ld) with parent %ld (%ld)\n",
                    c, pNextChild, f, frelp));
                Front *child = new (&fronts[pNextChild])
                                   Front(pNextChild, frelp, ccm, ccn);
                child->fidg = c;
                child->pidg = f;

                child->sparseMeta = SparseMeta();
                SparseMeta *cmeta = &(child->sparseMeta);
                cmeta->isSparse = true;
                cmeta->pushOnly = true;
                cmeta->nc = 0;
                cmeta->pn = fn;
                cmeta->cm = (int) ccm;
                cmeta->cn = (int) ccn;
                cmeta->csize = (int) ccm * (int) ccn;
                cmeta->gpuRimap = GPU_REFERENCE(wsRimap, int*)+RimapOffsets[c];
                cmeta->gpuRjmap = GPU_REFERENCE(wsRjmap, int*)+RjmapOffsets[c];

                // the memory layout for children can be right next to
                // its parent in memory since we're already charging our
                // staging algorithm for the space of a front and its children
                child->gpuF = cmeta->gpuC =
                    GPU_REFERENCE(wsMondoF, double*) + ChildOffset;

                // surgically copy the data to the GPU asynchronously
                Workspace *wsLimbo = LimboDirectory[c];
                wsLimbo->assign(wsLimbo->cpu(), child->gpuF);
                wsLimbo->transfer(cudaMemcpyHostToDevice, false,
                    memoryStreamH2D);

                // advance the child offset in case we have more children
                ChildOffset += ccm * ccn;

                // advance the next child index
                pNextChild++;
            }
        }

        // wait for data to go down to the GPU.
        cudaStreamSynchronize(memoryStreamH2D);

        // now we can free limbo children
        for(Long p=sStart; p<sEnd; p++)
        {
            Long f = Post[p];
            for(Long cp=Childp[f]; cp<Childp[f+1]; cp++)
            {
                Long c = Child[cp];
                if(StageMap[c] == stage) continue;
                Workspace *wsLimbo = LimboDirectory[c];
                wsLimbo->assign(wsLimbo->cpu(), NULL);
                LimboDirectory[c] = Workspace::destroy(wsLimbo);
            }
        }

        // ---------------------------------------------------------------------
        // Run the QR Engine
        // ---------------------------------------------------------------------

        INIT_TIME(engine);
        TIC(engine);

        QREngineStats stats;
        GPUQREngine(gpuMemorySize, fronts, numFronts, Parent, Childp, Child,
            &stats);
        cc->gpuKernelTime += stats.kernelTime;
        cc->gpuFlops += stats.flopsActual;
        cc->gpuNumKernelLaunches += stats.numLaunches;
        TOC(engine);
        PR (("%f engine time\n", engine));

#ifndef NDEBUG
        for(Long p=sStart; p<sEnd; p++)
        {
            Long f = Post[p];
            Long relp = leftoverChildren + (p - sStart);

            Long fm = Fm[f];                     // F has fm rows
            Long fn = Rp [f+1] - Rp [f] ;        // F has fn cols
            Long fp = Super [f+1] - Super [f] ;  // F has fp pivot columns
            Long frank = MIN(fm, fp);
            double *cpuF = (&fronts[relp])->cpuR;

            PR (("\n --- Front factorized, front %ld fm %ld fn %ld fp %ld frank %ld",
                f, fm, fn, fp, frank)) ;
            PR ((" : rows %ld to %ld of R (1-based)\n",
                1+ Super [f], 1+ Super [f+1]-1)) ;
            PR (("     Printing just R part, stored by row:\n")) ;
            for (Long j = 0 ; j < fn ; j++)
            {
                PR (("   --- column %ld of %ld\n", j, fn)) ;
                for (Long i = 0 ; i < frank ; i++)
                {
                    if (i == j) PR (("      [ diag:     ")) ;
                    else        PR (("      row %4ld    ", i)) ;
                    PR ((" %10.4g", cpuF [fn*i+j])) ;
                    if (i == j) PR ((" ]\n")) ;
                    else        PR (("\n")) ;
                }
                PR (("\n")) ;
            }
        }
#endif

        // ---------------------------------------------------------------------
        // Pack R from each front onto the Davis Stack.
        // ---------------------------------------------------------------------

        for(Long p=sStart; p<sEnd; p++)
        {
            Long f = Post[p];
            Long relp = leftoverChildren + (p - sStart);

            Long fm = Fm[f];                     // F has fm rows
            Long fn = Rp [f+1] - Rp [f] ;        // F has fn cols
            Long fp = Super [f+1] - Super [f] ;  // F has fp pivot columns
            Long frank = MIN(fm, fp);
            double *cpuF = (&fronts[relp])->cpuR;

            // cpuF for frontal matrix f has been factorized on the GPU and
            // copied to the CPU.  It is (frank+cm)-by-fn, stored by row.  The
            // entries in R for this front f are in the first frank rows, in
            // upper triangular form and stored in unpacked form.  On the Davis
            // stack, R is stored by columns in packed form.

            // Record where R is in the Davis stack
            Rblock [f] = Stack_head ;

            // copy the leading upper triangular part from cpuF to R
            for (Long j = 0 ; j < frank ; j++)
            {
                // copy column j of the front from cpuF to R
                for (Long i = 0 ; i <= j ; i++)
                {
                    (*Stack_head++) = cpuF [fn*i+j] ;
                }
            }
            // copy the rectangular part from cpuF to R
            for (Long j = frank ; j < fn ; j++)
            {
                // copy column j of the front from cpuF to R
                for (Long i = 0 ; i < frank ; i++)
                {
                    (*Stack_head++) = cpuF [fn*i+j] ;
                }
            }

            // When packed, R has frank*fn - (frank*(frank-1))/2 entries.
            ASSERT ((Stack_head - Rblock [f]) ==
                    (frank*fn - (frank*(frank-1))/2)) ;

            // If the front needs to be staged, copy the CBlock into limbo.
            SparseMeta *meta = &((&fronts[relp])->sparseMeta);
            if(meta->isStaged)
            {
                // offset into R to get the start of C
                double *C = cpuF + frank * fn + fp;

                PR (("   ===> move to Limbo\n")) ;

                // allocate a Limbo
                int cn = fn - fp;
                int cm = MIN (fm-frank, cn);

                PR (("frontList[%ld] is front %ld, staged;", relp, f)) ;
                PR (("CBlock is %d by %d\n", cm, cn)) ;

                // calloc pagelocked memory on the CPU
                Workspace *wsLimbo =
                    LimboDirectory[f] = Workspace::allocate (cm * cn,   // CPU
                    sizeof(double), true, true, false, true);

                if (wsLimbo == NULL)
                {
                    ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
                    FREE_ALL_WORKSPACE ;
                    return ;
                }

                double *L = CPU_REFERENCE(wsLimbo, double*);

                // copy from C into Limbo
                for(Long i=0; i<cm; i++)
                {
                    for(Long j=i; j<cn; j++)
                    {
                        double value = C[i*fn+j];
                        L[i*cn+j] = value;
                        PR (("%f\n", value));
                    }
                    PR (("\n"));
                }
            }
        }

        // cleanup after this stage

        #if 0
        // This is not needed since Front->F is NULL (nothing on the CPU)
        for(int f=0; f<numFronts; f++)
        {
            // invoke destructor without freeing memory
            Front *front = (&fronts[f]);
            front->~Front();
        }
        #endif

    }

    // -------------------------------------------------------------------------
    // cleanup the cuda memory transfer stream and free workspaces
    // -------------------------------------------------------------------------

    FREE_ALL_WORKSPACE ;
    // done using Iwork for InvPost ]

    // -------------------------------------------------------------------------
    // cleanup rank detection results (no rank detection done by the GPU)
    // -------------------------------------------------------------------------

    Work [stack].Stack_head = Stack_head  ;
    Work [stack].Stack_top = Stack_top ;

    // Compute sumfrank and maxfrank in a last-minute manner, and find Rdead
    for(Long f=0; f<nf; f++)
    {
        // compute basic front fields
        Long fm = Fm[f];                    // F is fm-by-fn
        // Long fn = Rp [f+1] - Rp [f];
        Long col1 = Super [f] ;             // first global pivot col in F
        Long fp = Super[f+1] - col1 ;       // with fp pivot columns
        Long frank = MIN(fm, fp);
        PR (("cleanup front %ld, %ld by %ld with %ld pivots, frank %ld\n",
            f, fm, Rp [f+1] - Rp [f], fp, frank)) ;

        if (fp > fm)
        {
            // hit the wall.  The front is fm-by-fn but has fp pivot columns
            PR (("hit the wall, npiv %ld k %ld rank %ld\n", fp, fm, fm)) ;
            for (Long k = fm ; k < fp ; k++)
            {
                Rdead [col1 + k] = 1 ;
            }
        }

        sumfrank += frank ;
        maxfrank = MAX (maxfrank, frank) ;
    }

    Work [stack].sumfrank = sumfrank ;  // sum rank of fronts for this stack
    Work [stack].maxfrank = maxfrank ;  // max rank of fronts for this stack

    TOC(complete) ;
    PR (("%f complete time\n", complete));
}

// -------------------------------------------------------------------------

void spqrgpu_kernel
(
    spqr_blob <Complex> *Blob
)
{
    // complex case not yet supported on the GPU
    cholmod_common *cc = Blob->cc ;
    ERROR (CHOLMOD_INVALID, "complex case not yet supported on the GPU") ;
    return ;
}
#endif

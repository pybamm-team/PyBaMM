// =============================================================================
// === GPUQREngine/Include/GPUQREngine_SparseMeta.hpp ==========================
// =============================================================================
//
// The SparseMeta class further wraps metadata required to perform a sparse
// factorization. SparseMeta sits within the Front class.
//
// =============================================================================

#ifndef GPUQRENGINE_SPARSEMETA_HPP
#define GPUQRENGINE_SPARSEMETA_HPP

#include "GPUQREngine_SEntry.hpp"

// Using int instead of Int or Long since we need to know the exact
// sizes on the CPU and GPU.

/* Has more stuff to support sparse multifrontal factorization */
class SparseMeta
{
public:
    int fp;               // # of pivotal columns in the front
    int nc;               // # of remaining children for the front
    bool isStaged;        // T/F indicating whether this front's
                          // parent is in a future stage
    bool pushOnly;        // Related to isStaged, pushOnly signals
                          // whether the front should only push its
                          // data to its parent
    bool isSparse;        // T/F indicating sparsiy

    /* Metadata for S Assembly */
    int lastSIndex;
    SEntry *cpuS;         // Packed S - pointer into scheduler's cpuS
    SEntry *gpuS;         // Packed S - pointer into scheduler's gpuS
    int Scount;           // # S entries to ship to GPU.

    /* Metadata for Pack Assembly */
    int cm;               // # rows of the contribution block
    int cn;               // # cols of the contribution block
    int csize;            // total size of the contribution block
                          //   (rows*cols)
    int pn;               // # of columns in the parent
    int pc;               // the p start for the contribution block
    int lastCiStart;      // Last contribution block row where we added a task
    int lastCjStart;      // Last contribution block col where we added a task
    int *gpuRjmap;        // The gpu location of the Rjmap
    int *gpuRimap;        // The gpu location of the Rimap
    double *gpuC;         // location of the front's contribution block
    double *gpuP;         // The location of the front's parent

    SparseMeta()
    {
        fp = 0;
        nc = 0;
        isStaged = false;
        pushOnly = false;
        isSparse = false;

        lastSIndex = 0;
        cpuS = NULL;
        gpuS = NULL;
        Scount = 0;

        cm = 0;
        cn = 0;
        csize = 0;
        pn = 0;
        pc = 0;
        lastCiStart = 0;
        lastCjStart = 0;
        gpuRjmap = NULL;
        gpuRimap = NULL;
        gpuC = NULL;
        gpuP = NULL;
    }

    ~SparseMeta()
    {
    }
};

#endif

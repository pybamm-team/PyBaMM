// =============================================================================
// === GPUQREngine/Source/GPUQREngine_ExpertSparse.cpp =========================
// =============================================================================
//
// This file contains the sparse GPUQREngine wrapper that simply calls down into
// the Internal GPUQREngine factorization routine.
//
// =============================================================================

#include "GPUQREngine_Internal.hpp"


QREngineResultCode GPUQREngine
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front *userFronts,      // The list of fronts to factorize
    Int numFronts,          // The number of fronts to factorize
    Int *Parent,            // The front-to-parent mapping
    Int *Childp,            // Front-to-child column pointers
    Int *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
)
{
    return (GPUQREngine_Internal (gpuMemorySize, userFronts, numFronts, Parent,
        Childp, Child, stats)) ;
}

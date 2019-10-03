// =============================================================================
// === GPUQREngine/Include/GPUQREngine_FrontState.hpp ==========================
// =============================================================================
//
// The front state refers to the finite state machine model for factorizing
// fronts using GPUQREngine.
//
// A front will progress through these fronts, as dictated by
//  FillWorkQueue and PostProcessing codes.
//
// =============================================================================

#ifndef GPUQRENGINE_FRONTSTATE_HPP
#define GPUQRENGINE_FRONTSTATE_HPP

enum FrontState
{
    ALLOCATE_WAIT = 0,      // Front not allocated yet
    ASSEMBLE_S = 1,         // Assembling rows of S
    CHILD_WAIT = 2,         // Waiting for children to be finished
    FACTORIZE = 3,          // Factorization under way
    FACTORIZE_COMPLETE = 4, // Records an event to mark the end of the
                            // factorize
    PARENT_WAIT = 5,        // Waits for the parent to be allocated
    PUSH_ASSEMBLE = 6,      // Pushes contribution blocks to the parent
    CLEANUP = 7,            // Frees the front
    DONE = 8                // Front totally finished
};

#endif

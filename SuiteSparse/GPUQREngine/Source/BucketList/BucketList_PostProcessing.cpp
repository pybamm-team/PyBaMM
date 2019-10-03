// =============================================================================
// === GPUQREngine/Source/BucketList_PostProcessing.cpp ========================
// =============================================================================
//
// PostProcess handles any cleanup operations following a kernel invocation
// including merging delta tiles with the main bundle and other fixups.
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"


void BucketList::PostProcess
(
    void
)
{
    for(int b=0; b<numBundles; b++)
    {
        LLBundle& bundle = Bundles[b];

        /* Get details about the task type. */
        TaskType type = bundle.CurrentTask;
        bool wasApply = (type == TASKTYPE_GenericApply);
        #ifdef GPUQRENGINE_PIPELINING
        wasApply = wasApply || (type == TASKTYPE_GenericApplyFactorize);
        #endif

        /* If the task was an apply or an apply-factorize: */
        if(wasApply)
        {
            /* Do the rearrange and find the index of the second min entry. */
            bundle.PipelinedRearrange();
            bundle.UpdateSecondMinIndex();
            bundle.UpdateMax();

            /* We're finished with VT[0], but we need to shuffle VT[0] <- VT[1]
             * so we can further pipeline apply-factorize tasks. */
            freeVT(bundle.VT[0]);
            bundle.VT[0] = bundle.VT[1];
            bundle.VT[1] = NULL;
        }
    }
}

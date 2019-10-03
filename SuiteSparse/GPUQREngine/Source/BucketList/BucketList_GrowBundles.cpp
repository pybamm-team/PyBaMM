// =============================================================================
// === GPUQREngine/Source/BucketList_GrowBundles.cpp ===========================
// =============================================================================
//
// GrowBundles looks for row tiles (or bundles) involved in a factorization
// and attempts to add those bundles or row tiles to a task currently set
// for a series of Apply tasks. This is also known as Pipelining.
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"
#ifdef GPUQRENGINE_PIPELINING

/* Grow existing bundles and advance the task type to APPLY_FACTORIZE. */
void BucketList::GrowBundles
(
    void
)
{
    //Console.WriteLine("=== GrowBundles");
    for (int i = 0; i < numBundles; i++ )
    {
        LLBundle& bundle = Bundles[i];

        /* The bundle is a phantom bundle if it only has a shadow entry.
         * In this case, we choose not to grow the bundle. */
        if(bundle.First == EMPTY) continue;

        /* Absorb any idle tiles from the bucket. */
        int nativeBucket = bundle.NativeBucket;
        bool hasIdleTiles = (head[nativeBucket] != EMPTY);

        //Console.WriteLine("Bundle " + bundle + "
        //  has NativeBucket " + nativeBucket);

        /* If there are idle tiles in the bucket, try to grow the bundle. */
        if (hasIdleTiles)
        {
            //Console.WriteLine("  Detected Idle Tiles:" + head[nativeBucket]);

            /* Set up the task to be an apply factorize. */
            bundle.CurrentTask = TASKTYPE_GenericApplyFactorize;

            /* Grow the bundle. */
            while (head[nativeBucket] != EMPTY && !bundle.IsFull())
            {
                int newTile = RemoveHead(nativeBucket);
                triu[newTile] = false;
                bundle.AddTileToDelta(newTile);
            }
        }
        /* Else upgrade the task if pipelining gives us an edge. */
        else if (bundle.Count > 1)
        {
            bundle.CurrentTask = TASKTYPE_GenericApplyFactorize;
        }
        else if (bundleCount[nativeBucket] == 1 && Wavefront == nativeBucket-1)
        {
            bundle.CurrentTask = TASKTYPE_GenericApplyFactorize;
        }

        /* If we upgraded to apply-factorize, we need another VT block. */
        if(bundle.CurrentTask == TASKTYPE_GenericApplyFactorize)
        {
            bundle.VT[1] = allocateVT();
        }
    }
}

#endif

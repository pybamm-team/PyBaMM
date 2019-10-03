// =============================================================================
// === GPUQREngine/Source/BucketList_AdvanceBundles.cpp ========================
// =============================================================================
//
// AdvanceBundles advances existing bundles, leaving the First tile behind
// and keeping a Shadow copy to support subsequent Apply tasks.
//
// =============================================================================


#include "GPUQREngine_BucketList.hpp"

void BucketList::AdvanceBundles()
{
    for (int i = 0; i < numBundles; i++)
    {
        LLBundle& bundle = Bundles[i];

        /* An advancing bundle is removed from its native bucket. */
        bundleCount[bundle.NativeBucket]--;

        /* Advance the bundle and check if it's still around (not evaporated) */
        bool stillAround = bundle.Advance();

        /* If the bundle didn't evaporate: */
        if (stillAround)
        {
            /* The advancing bundle arrives at the next bucket. */
            bundleCount[bundle.NativeBucket]++;

            /* Keep track of the last bucket. */
            LastBucket = MAX(LastBucket, bundle.NativeBucket);
        }
        /* Else the bundle evaporated. */
        else
        {
            Bundles[i] = Bundles[--numBundles];
            i--;
        }
    }
}

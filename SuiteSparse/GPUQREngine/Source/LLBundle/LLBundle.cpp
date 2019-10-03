// =============================================================================
// === GPUQREngine/Source/LLBundle.cpp =========================================
// =============================================================================
//
// This file contains the constructor and destructor for the LLBundle class.
// The constructor will attempt to reserve a VT tile automatically, since newly
// created bundles are immediately slated for factorization.
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"


LLBundle::LLBundle
(
    BucketList *buckets,
    Int panelSize,
    Int nativeBucket
)
{
    Buckets = buckets;
    PanelSize = panelSize;
    NativeBucket = nativeBucket;
    SecondMin = Shadow = First = Delta = Max = EMPTY;
    Count = ApplyCount = 0;
    VT[0] = VT[1] = NULL;

    /* Create the factorize task and allocate a VT block. */
    CurrentTask = TASKTYPE_GenericFactorize;
    VT[0] = buckets->allocateVT();
}

LLBundle::~LLBundle()
{
}

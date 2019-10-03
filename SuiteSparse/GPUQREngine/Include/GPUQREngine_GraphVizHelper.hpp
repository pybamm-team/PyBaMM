// =============================================================================
// === GPUQREngine/Include/GPUQREngine_GraphVisHelper.hpp ======================
// =============================================================================
//
// GraphVisHelper wraps logic to render the contents of a bucket list.
//   This is used primarilly in debugging efforts.
//
// =============================================================================

#ifndef GPUQRENGINE_GRAPHVIZHELPER_HPP
#define GPUQRENGINE_GRAPHVIZHELPER_HPP

#ifdef GPUQRENGINE_RENDER
#include "GPUQREngine_BucketList.hpp"

void GPUQREngine_RenderBuckets(BucketList *buckets);

#endif
#endif

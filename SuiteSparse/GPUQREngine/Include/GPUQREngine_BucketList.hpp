// =============================================================================
// === GPUQREngine/Include/GPUQREngine_BucketList.hpp ==========================
// =============================================================================
//
// The BucketList is a principal class in the GPUQREngine.
//
// The BucketList manages a set of LLBundle structures in a doubly-linked list.
// During factorization, the BucketList logically manipulates the LLBundles,
// and depending on the configuration of each, generates GPU tasks to be added
// to the GPU work queue.
//
// =============================================================================

#ifndef GPUQRENGINE_BUCKETLIST_HPP
#define GPUQRENGINE_BUCKETLIST_HPP

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"
#include "GPUQREngine_LLBundle.hpp"
#include "GPUQREngine_Front.hpp"

struct TaskDescriptor;
class LLBundle;

class BucketList
{
public:
    bool useFlag;            // A flag indicating whether to use this
    bool memory_ok;          // A flag indicating whether the object
                             // was constructed properly

    double *gpuF;            // The gpu front pointer

    Int *head;               // The head idle tile index in the bucket
    Int *next;               // The next idle tile index in the bucket
    Int *prev;               // The prev idle tile index in the bucket
    bool *triu;              // Flag indicating whether the tile index
                             // is upper triangular

    Int *bundleCount;        // The # of bundles native to bucket index
    Int *idleTileCount;      // The # of idle tiles in bucket index

    Front *front;
    Int numRowTiles;         // # row tiles of F
    Int numColTiles;         // # col tiles of F
    Int numBuckets;          // min(numRowTiles, numColTiles)
    Int numIdleTiles;        // Total # of idle tiles stored in buckets
    Int PanelSize;           // Max # of rowtiles that can fit in one bundle
    Int TileSize;            // Dimensions of tiles
    Int Wavefront;           // Index of first non-completed colBucket
    Int LastBucket;          // Index of last colBucket with idleTiles
                             // or bundles

    Int ApplyGranularity;    // The desired granularity (in col tiles)
                             // for applies

    LLBundle *Bundles;       // The bundles maintained by this scheduler
    Int numBundles;          // Total # of bundles

    Workspace *wsMongoVT;    // The VT blocks this bucket list scheduler owns
    double **gpuVT;          // Array of available VT slots within the VT struct
    int VThead;              // Index of the first available entry in VTlist

    // Constructors
    void *operator new(long unsigned int, BucketList* p)
    {
        return p;
    }
    BucketList(Front *f, Int minApplyGranularity);
    ~BucketList();

    // Bundle management functions
    void Insert(Int tile, Int bucket, bool upperTriangular = false);
    void Remove(Int tile, Int bucket);
    #ifdef GPUQRENGINE_PIPELINING
    Int RemoveHead(Int bucket);
    #endif

    // VT management functions
    double *allocateVT();
    double *freeVT(double *gpuVT);

    bool IsDone()
    {
        // We're done if we have no bundles left with tasks.
        return (numBundles == 0);
    }

//  // IsRReadyEarly experimental feature : not available in production use.
//  bool IsRReadyEarly()
//  {
//      // If we're doing a dense factorization, we're never done early.
//      if(front->isDense()) return false;
//
//      // We can't pull the R factor early if we also need the CBlock.
//      if(front->isStaged()) return false;
//
//      // If we're doing a sparse factorization, we're done early if we're
//      // past the pivot row.
//      return (TILESIZE * (Wavefront-1) > front->sparseMeta.fp);
//  }

    // Initialize takes the BucketList and adds rowtiles in positions
    // appropriate for the staircase of the problem.
    void Initialize
    (
        void
    );

    // AdvanceBundles advances existing bundles, leaving the First tile behind
    // and keeping a Shadow copy to support subsequent Apply tasks.
    void AdvanceBundles
    (
        void
    );

    #ifdef GPUQRENGINE_PIPELINING
    // GrowBundles looks for row tiles (or bundles) involved in a factorization
    // and attempts to add those bundles or row tiles to a task currently set
    // for a series of Apply tasks. This is also known as Pipelining.
    void GrowBundles
    (
        void
    );
    #endif

    // CreateBundles selects rowtiles up to PANELSIZE and creates a new bundle
    // ready for factorization.
    void CreateBundles
    (
        void
    );

    // PostProcess handles any cleanup operations following a kernel invocation
    // including merging delta tiles with the main bundle and other fixups.
    void PostProcess
    (
        void
    );

    // SkipBundleCreation determines whether we should skip creating a new
    // bundle for the specified tile in the specified column bucket.
    bool SkipBundleCreation
    (
        Int tile,
        Int colBucket
    );

    // IsInternal determines whether a tile is completely within the bounds
    // of the front because if it isn't then we will need to use the special
    // edge case kernels.
    bool IsInternal
    (
        LLBundle& bundle,
        int jLast
    );

    // FillWorkQueue is responsible for filling the work queue with items and
    // resolving generic TaskType entries on the bundles into concrete tasks
    // to be performed by the GPU.
    Int FillWorkQueue
    (
        TaskDescriptor *queue,  // The list of work items for the GPU
        Int *queueIndex         // The current index into the queue
    );
};

#endif

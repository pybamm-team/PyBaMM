// =============================================================================
// === SuiteSparse_GPURuntime/Source/SuiteSparseGPU_Workspace.cpp ==============
// =============================================================================

// The SuiteSparse_GPURuntime Workspace provides a convenient way to allocate
// and free memory on the CPU and/or GPU, and to transfer data between the
// CPU and GPU.

#include "SuiteSparseGPU_Runtime.hpp"

// -----------------------------------------------------------------------------
// Workspace constructor
// -----------------------------------------------------------------------------

Workspace::Workspace
(
    size_t _nitems,                 // number of items to allocate
    size_t _size_of_item            // size of each item
)
{
    nitems = _nitems;
    size_of_item = _size_of_item;

    totalSize = nitems * size_of_item;

    // check for integer overflow
    if (totalSize != ((double) nitems) * size_of_item)
    {
        totalSize = 0 ;            // size_t overflow
    }

    // lazyAllocate = false;       // always false, left for future use

    cpuReference = NULL;            // no CPU memory allocated yet
    gpuReference = NULL;            // no GPU memory allocated yet
}

// -----------------------------------------------------------------------------
// Workspace destructor
// -----------------------------------------------------------------------------

Workspace::~Workspace()
{
}

// -----------------------------------------------------------------------------
// allocate workspace
// -----------------------------------------------------------------------------

Workspace *Workspace::allocate
(
    size_t nitems,          // number of items
    size_t size_of_item,    // size of each item
    bool doCalloc,          // true if calloc instead of malloc
    bool cpuAlloc,          // true if allocating on the CPU
    bool gpuAlloc,          // true if allocating on the GPU
    bool pageLocked         // true if pagelocked on the CPU
)
{
    // Cannot use calloc directly since that's a member function,
    // and this is a static member function.
    Workspace *returner = (Workspace*)
        Workspace::cpu_calloc (1, sizeof(Workspace), false) ;

    if (returner)
    {
        new (returner) Workspace (nitems, size_of_item) ;

        /* Set whether the user wants the address page-locked. */
        returner->pageLocked = pageLocked ;

        /* Do the allocation & handle any errors. */
        bool okay = (doCalloc ? returner->ws_calloc (cpuAlloc, gpuAlloc)  :
                                returner->ws_malloc (cpuAlloc, gpuAlloc)) ;
        if (!okay)
        {
            returner = Workspace::destroy (returner) ;
        }
    }

    return (returner) ;
}

// -----------------------------------------------------------------------------
// destroy workspace, freeing memory
// -----------------------------------------------------------------------------

Workspace *Workspace::destroy
(
    Workspace *address
)
{
    if(address)
    {
        address->ws_free(address->cpu(), address->gpu());
        SuiteSparse_free(address);
    }
    return NULL;
}

// -----------------------------------------------------------------------------
// ws_malloc: malloc workspace on CPU and/or GPU
// -----------------------------------------------------------------------------

bool Workspace::ws_malloc(bool cpuAlloc, bool gpuAlloc)
{
    if(cpuAlloc)
    {
        cpuReference = Workspace::cpu_malloc(nitems, size_of_item, pageLocked);
    }
    if(gpuAlloc)
    {
        gpuReference = Workspace::gpu_malloc(nitems, size_of_item);
    }

    bool cpuSideOk = IMPLIES(cpuAlloc, cpuReference != NULL);
    bool gpuSideOk = IMPLIES(gpuAlloc, gpuReference != NULL)
        // || lazyAllocate
        ;
    return (cpuSideOk && gpuSideOk);
}

// -----------------------------------------------------------------------------
// ws_calloc: calloc workspace on CPU and/or GPU
// -----------------------------------------------------------------------------

bool Workspace::ws_calloc(bool cpuAlloc, bool gpuAlloc)
{
    if(cpuAlloc)
    {
        cpuReference = Workspace::cpu_calloc(nitems, size_of_item, pageLocked);
    }
    if(gpuAlloc)
    {
        gpuReference = Workspace::gpu_calloc(nitems, size_of_item);
    }

    bool cpuSideOk = IMPLIES(cpuAlloc, cpuReference != NULL);
    bool gpuSideOk = IMPLIES(gpuAlloc, gpuReference != NULL)
        // || lazyAllocate
        ;
    return (cpuSideOk && gpuSideOk);
}

// -----------------------------------------------------------------------------
// ws_free: free workspace on CPU and/or GPU
// -----------------------------------------------------------------------------

void Workspace::ws_free(bool cpuFree, bool gpuFree)
{
    if(cpuFree) Workspace::cpu_free(cpuReference, pageLocked);
    if(gpuFree) Workspace::gpu_free(gpuReference);
}

// =============================================================================
// === SuiteSparse_GPURuntime/Include/SuiteSparseGPU_Workspace.hpp =============
// =============================================================================

#ifndef SUITESPARSE_GPURUNTIME_WORKSPACE_HPP
#define SUITESPARSE_GPURUNTIME_WORKSPACE_HPP

#include "SuiteSparseGPU_Runtime.hpp"

class Workspace
{
private:

    size_t nitems;          // number of items to allocate
    size_t size_of_item;    // size of each item, in bytes
    size_t totalSize;       // nitems * size_of_item

//  bool lazyAllocate;      // no longer used (for possible future use)
    bool pageLocked;        // true if CPU memory is pagelocked

    void *cpuReference;     // pointer to CPU memory
    void *gpuReference;     // pointer to GPU memory

public:
    // Read-Only Properties
    size_t getCount(void){ return nitems; }
    size_t getStride(void){ return size_of_item; }
    void *cpu(void){ return cpuReference; }
    void *gpu(void){ return gpuReference; }

    // Constructor/Destructor
    void *operator new(size_t bytes, Workspace* ptr){ return ptr; }
    Workspace(size_t nitems, size_t size_of_item);
    ~Workspace();

    // Memory management wrappers
    static void *cpu_malloc(size_t nitems, size_t size_of_item,
        bool pageLocked=false);
    static void *cpu_calloc(size_t nitems, size_t size_of_item,
        bool pageLocked=false);
    static void *cpu_free(void *address, bool pageLocked = false);
    static void *gpu_malloc(size_t nitems, size_t size_of_item);
    static void *gpu_calloc(size_t nitems, size_t size_of_item);
    static void *gpu_free(void *);

    // Workspace management
    static Workspace *allocate
    (
        size_t nitems,              // number of items to allocate
        size_t size_of_item,        // size of each item, in bytes
        bool doCalloc = false,      // if true, then calloc; else malloc
        bool cpuAlloc = true,       // if true, then allocate CPU memory
        bool gpuAlloc = true,       // if true, then allocate GPU memory
        bool pageLocked = false     // true if CPU memory is pagelocked
    );

    // destroy workspace, freeing memory on both the CPU and GPU
    static Workspace *destroy
    (
        Workspace *address
    );

    // Reference manipulation functions
    template <typename T> void extract(T *cpu_arg, T *gpu_arg)
    {
        *cpu_arg = (T) cpuReference;
        *gpu_arg = (T) gpuReference;
    }
    void assign(void *cpu_arg, void *gpu_arg)
    {
        cpuReference = cpu_arg;
        gpuReference = gpu_arg;
    }

//  unused, left commented out for possible future use
//  void setLazy()
//  {
//      lazyAllocate = true;
//  }

    // Memory management for workspaces
    virtual bool ws_malloc(bool cpuAlloc = true, bool gpuAlloc = true);
    virtual bool ws_calloc(bool cpuAlloc = true, bool gpuAlloc = true);
    virtual void ws_free(bool cpuFree=true, bool gpuFree=true);

    // GPU-CPU transfer routines
    virtual bool transfer(cudaMemcpyKind direction, bool synchronous=true,
        cudaStream_t stream=0);

    // CPU & GPU memory functions
    // memset functions unused, left commented out for possible future use
    // bool gpu_memset(size_t newValue);
    // bool cpu_memset(size_t newValue);

    // Debug
#if DEBUG_ATLEAST_ERRORONLY
    static void print(Workspace *workspace)
    {
        printf (
            "(%ld,%ld) has %ld entries of size %ld each.\n",
            (size_t) workspace->cpu(),
            (size_t) workspace->gpu(),
            workspace->getCount(),
            workspace->getStride()
        );
    }
#endif
};

#endif

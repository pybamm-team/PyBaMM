// =============================================================================
// === GPUQREngine/Include/GPUQREngine_TaskDescriptor.hpp ======================
// =============================================================================
//
// The TaskType enum is used by the GPU UberKernel to determine which subkernel
// functionality to invoke.
//
// The TaskDescriptor struct wraps all metadata necessary to describe to the
// GPU how to perform one logical task.
//
// =============================================================================

#ifndef GPUQRENGINE_TASKDESCRIPTOR_HPP
#define GPUQRENGINE_TASKDESCRIPTOR_HPP

enum TaskType
{
    // Dummy Method (1 total)
    TASKTYPE_Dummy,                     // Used only for initializing a task

    // Factorize Methods (8 total)
    TASKTYPE_GenericFactorize,          // An uncategorized Factorize.
    TASKTYPE_FactorizeVT_3x1,           //   Factorize tasks are resolved
    TASKTYPE_FactorizeVT_2x1,           //   when the work queue is filled
    TASKTYPE_FactorizeVT_1x1,           //   based on the problem geometry,
    TASKTYPE_FactorizeVT_3x1e,          //   factorization state, and whether
    TASKTYPE_FactorizeVT_2x1e,          //   the factorization is at an
    TASKTYPE_FactorizeVT_1x1e,          //   "edge case."
    TASKTYPE_FactorizeVT_3x1w,

    // Apply Methods (4 total)
    TASKTYPE_GenericApply,              // An uncategorized Apply.
    TASKTYPE_Apply3,                    //   These tasks are likewise resolved
    TASKTYPE_Apply2,                    //   into their concrete types as the
    TASKTYPE_Apply1,                    //   work queue is filled.

    #ifdef GPUQRENGINE_PIPELINING
    // ApplyFactorize Methods (6 total)
    TASKTYPE_GenericApplyFactorize,     // An uncategorized Apply-Factorize.
    TASKTYPE_Apply3_Factorize3,         //   These tasks are likewise resolved
    TASKTYPE_Apply3_Factorize2,         //   into their concrete types as the
    TASKTYPE_Apply2_Factorize3,         //   work queue is filled.
    TASKTYPE_Apply2_Factorize2,
    TASKTYPE_Apply2_Factorize1,
    #endif

    // Assembly Methods (2 total)
    TASKTYPE_SAssembly,                 // Input matrix assembly
    TASKTYPE_PackAssembly               // Push assembly (child to parent)
};

class Scheduler;

struct TaskDescriptor
{
    /* Put pointers up front to guarantee word-alignment. */
    double *F;                          // Pointer to the frontal matrix
    double *AuxAddress[4];              // Usage Notes
                                        //   SAssembly:
                                        //     AuxAddress[0]    is SEntry*
                                        //   PackAssembly:
                                        //     AuxAddress[0]    is *C
                                        //     AuxAddress[1]    is *P
                                        //     AuxAddress[2]    is *Rjmap
                                        //     AuxAddress[3]    is *Rimap
                                        //   Apply, Factorize:
                                        //     AuxAddress[0]    is VT
                                        //   ApplyFactorize:
                                        //     AuxAddress[0:1] are VT

    TaskType Type;                      // The TaskType enum described above
    int fm;                             // # Rows in the front
    int fn;                             // # Cols in the front

    int extra[10];                      // Usage Notes
                                        //   SAssembly:
                                        //     extra[0]    is Scount    (unused)
                                        //     extra[1]    is pstart
                                        //     extra[2]    is pend
                                        //   PackAssembly:
                                        //     extra[0]    is pn
                                        //     extra[1]    is cm        (unused)
                                        //     extra[2]    is cn        (unused)
                                        //     extra[3]    is cTileSize
                                        //     extra[4]    is cistart
                                        //     extra[5]    is ciend
                                        //     extra[6]    is cjstart
                                        //     extra[7]    is cjend
                                        //   Apply:
                                        //     extra[0:2] are rowTiles
                                        //     extra[4:7] are colTiles
                                        //   Factorize:
                                        //     extra[0:2] are rowTiles
                                        //     extra[4]    is colTiles
                                        //   ApplyFactorize:
                                        //     extra[0:3] are rowTiles
                                        //     extra[4:7] are colTiles
                                        //     extra[8]    is delta
                                        //     extra[9]    is secondMin
                                        //
};


// These two methods are implemented in TaskDescriptor_flops.cpp.
// They are used to rearrange tasks in the WorkQueue to promote a
// uniform distribution of work items in the queue.
Int getFlops
(
    TaskDescriptor *task                // Task for which to compute the flops
);

Int getWeightedFlops
(
    TaskDescriptor *task                // Task for which to compute the flops
);

#endif

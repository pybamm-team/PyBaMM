// =============================================================================
// === GPUQREngine/Source/TaskDescriptor_flops.cpp =============================
// =============================================================================
//
// This file contains functions that are responsible for computing the actual
// flops performed by various GPU tasks in the GPUQREngine.
//
// =============================================================================

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"


// -----------------------------------------------------------------------------
// flopsFactorizeVT
// -----------------------------------------------------------------------------

Int flopsFactorizeVT(int numTiles)
{
    Int m = TILESIZE * numTiles;
    Int n = TILESIZE;
    Int v = TILESIZE;
    return 2 * (m-1)                       +
           v * (6 + 4*m*n + 5*n + 2*m)     +
           ((-4*n -2*m -5) * (v*(v+1)/2))  +
           3 * (v*(v+1)*(2*v+1)/6)         +
           (- 2*(m-v+1)*(n-v) - 2*(m-v-1)) ;
}

// -----------------------------------------------------------------------------
// flopsFactorize
// -----------------------------------------------------------------------------

Int flopsFactorize(int m, int n)
{
    Int v = MIN(m, n);
    return 2 * (m-1)                    +
           v * (6 + 4*m*n + 5*n + 2*m)  +
           ((-4*n -2*m -4) * v*(v+1)/2) +
           2 * v*(v+1)*(2*v+1)/6        +
           (- 2*(m-v+1)*(n-v) - 2*(m-v-1)) ;
}

// -----------------------------------------------------------------------------
// flopsApply
// -----------------------------------------------------------------------------

Int flopsApply(int numTiles, int n)
{
    Int m = TILESIZE * numTiles;
    Int k = TILESIZE;
    return k*n*(4*m - k + 3);
}

// -----------------------------------------------------------------------------
// flopsApplyFactorize
// -----------------------------------------------------------------------------

#ifdef GPUQRENGINE_PIPELINING
Int flopsApplyFactorize(int applyTiles, int factorizeTiles)
{
    return flopsApply(applyTiles, TILESIZE) + flopsFactorizeVT(factorizeTiles);
}
#endif

// -----------------------------------------------------------------------------
// getFlops
// -----------------------------------------------------------------------------

Int getFlops(TaskDescriptor *task)
{
    switch(task->Type)
    {
        case TASKTYPE_FactorizeVT_3x1:
        case TASKTYPE_FactorizeVT_3x1e:  return flopsFactorizeVT(3);
        case TASKTYPE_FactorizeVT_2x1:
        case TASKTYPE_FactorizeVT_2x1e:  return flopsFactorizeVT(2);
        case TASKTYPE_FactorizeVT_1x1:
        case TASKTYPE_FactorizeVT_1x1e:  return flopsFactorizeVT(1);

        case TASKTYPE_FactorizeVT_3x1w:
            return flopsFactorize(task->fm, task->fn);

        case TASKTYPE_Apply3:
            return flopsApply(3, task->extra[6] - task->extra[5]);

        case TASKTYPE_Apply2:
            return flopsApply(2, task->extra[6] - task->extra[5]);

        case TASKTYPE_Apply1:
            return flopsApply(1, task->extra[6] - task->extra[5]);

        #ifdef GPUQRENGINE_PIPELINING
        case TASKTYPE_Apply3_Factorize3: return flopsApplyFactorize(3, 3);
        case TASKTYPE_Apply3_Factorize2: return flopsApplyFactorize(3, 2);
        case TASKTYPE_Apply2_Factorize3: return flopsApplyFactorize(2, 3);
        case TASKTYPE_Apply2_Factorize2: return flopsApplyFactorize(2, 2);
        case TASKTYPE_Apply2_Factorize1: return flopsApplyFactorize(2, 1);
        #endif

        case TASKTYPE_SAssembly:         return 0;
        case TASKTYPE_PackAssembly:      return 0;
    }
}

// -----------------------------------------------------------------------------
// getWeightedFlops
// -----------------------------------------------------------------------------

Int getWeightedFlops(TaskDescriptor *task)
{
    Int flops = getFlops(task);
    switch(task->Type)
    {
        case TASKTYPE_FactorizeVT_3x1:
        case TASKTYPE_FactorizeVT_3x1e:  flops *= 1; break;
        case TASKTYPE_FactorizeVT_2x1:
        case TASKTYPE_FactorizeVT_2x1e:  flops *= 1; break;
        case TASKTYPE_FactorizeVT_1x1:
        case TASKTYPE_FactorizeVT_1x1e:  flops *= 1; break;

        case TASKTYPE_FactorizeVT_3x1w:  flops *= 1; break;

        case TASKTYPE_Apply3:            flops *= 0.5; break;
        case TASKTYPE_Apply2:            flops *= 0.5; break;
        case TASKTYPE_Apply1:            flops *= 0.5; break;

        #ifdef GPUQRENGINE_PIPELINING
        case TASKTYPE_Apply3_Factorize3: flops *= 10; break;
        case TASKTYPE_Apply3_Factorize2: flops *= 10; break;
        case TASKTYPE_Apply2_Factorize3: flops *= 10; break;
        case TASKTYPE_Apply2_Factorize2: flops *= 10; break;
        case TASKTYPE_Apply2_Factorize1: flops *= 10; break;
        #endif

        case TASKTYPE_SAssembly:         break;
        case TASKTYPE_PackAssembly:      break;
    }
    return flops;
}

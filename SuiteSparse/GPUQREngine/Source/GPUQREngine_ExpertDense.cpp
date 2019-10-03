// =============================================================================
// === GPUQREngine/Source/GPUQREngine_ExpertDense.cpp ==========================
// =============================================================================
//
// This file contains the dense GPUQREngine wrapper that finds the staircase,
// makes a copy of the user's front data, then calls down into the Internal
// GPUQREngine factorization routine.
//
// Other functions include:
//  - GPUQREngine_Cleanup:       Cleans up relevant workspaces in the dense
//                               factorization depending on how we're exiting.
//  - GPUQREngine_FindStaircase: Finds the staircase for a front and returns
//                               the staircase as an Int* list
// =============================================================================

#include "GPUQREngine_Internal.hpp"


QREngineResultCode GPUQREngine_Cleanup
(
    QREngineResultCode code,    // The result code that we're exiting with
    Front *userFronts,          // The user-provided list of fronts
    Front *fronts,              // The internal copy of the user's fronts
    Int numFronts,              // The number of fronts to be factorized
    Workspace *wsMongoF,        // Pointer to the total GPU Front workspace
    Workspace *wsMongoR         // Pointer to the total CPU R workspace
);

QREngineResultCode GPUQREngine
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front *userFronts,      // The list of fronts to factorize
    Int numFronts,          // The number of fronts to factorize
    QREngineStats *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
)
{
    /* Allocate workspaces */
    Front *fronts = (Front*) SuiteSparse_calloc(numFronts, sizeof(Front));
    if(!fronts)
    {
        return QRENGINE_OUTOFMEMORY;
    }

    size_t FSize, RSize;
    FSize = RSize = 0;
    for(int f=0; f<numFronts; f++)
    {
        /* Configure the front */
        Front *userFront = &(userFronts[f]);
        Int m = userFront->fm;
        Int n = userFront->fn;
        Front *front = new (&fronts[f]) Front(f, EMPTY, m, n);
        FSize += front->getNumFrontValues();
        RSize += front->getNumRValues();
    }

    // We have to allocate page-locked CPU-GPU space to leverage asynchronous
    // memory transfers.  This has to be done in a way that the CUDA driver is
    // aware of, which unfortunately means making a copy of the user input.

    // calloc pagelocked space on CPU, and calloc space on the GPU
    Workspace *wsMongoF = Workspace::allocate(FSize,    // CPU and GPU
        sizeof(double), true, true, true, true);

    // calloc pagelocked space on the CPU.  Nothing on the GPU
    Workspace *wsMongoR = Workspace::allocate(RSize,    // CPU
        sizeof(double), true, true, false, true);

    /* Cleanup and return if we ran out of memory. */
    if(!wsMongoF || !wsMongoR)
    {
        return GPUQREngine_Cleanup (QRENGINE_OUTOFMEMORY,
            userFronts, fronts, numFronts, wsMongoF, wsMongoR);
    }

    /* Prepare the fronts for GPU execution. */
    size_t FOffset, ROffset;
    FOffset = ROffset = 0;
    for(int f=0; f<numFronts; f++)
    {
        // Set the front pointers; make the copy from user data into front data.
        Front *front = &(fronts[f]);
        front->F    = CPU_REFERENCE(wsMongoF, double*) + FOffset;
        front->gpuF = GPU_REFERENCE(wsMongoF, double*) + FOffset;
        front->cpuR = CPU_REFERENCE(wsMongoR, double*) + ROffset;
        FOffset += front->getNumFrontValues();
        ROffset += front->getNumRValues();

        /* COPY USER DATA (user's F to our F) */
        Front *userFront = &(userFronts[f]);
        double *userF = userFront->F;
        double *F = front->F;
        Int m = userFront->fm;
        Int n = userFront->fn;
        bool isColMajor = userFront->isColMajor;
        Int ldn = userFront->ldn;
        for(Int i=0; i<m; i++)
        {
            for(Int j=0; j<n; j++)
            {
                F[i*n+j] = (isColMajor ? userF[j*ldn+i] : userF[i*ldn+j]);
            }
        }

        /* Attach either the user-specified Stair, or compute it. */
        front->Stair = userFront->Stair;
        if(!front->Stair) front->Stair = GPUQREngine_FindStaircase(front);

        /* Cleanup and return if we ran out of memory building the staircase */
        if(!front->Stair)
        {
            return GPUQREngine_Cleanup (QRENGINE_OUTOFMEMORY,
                userFronts, fronts, numFronts, wsMongoF, wsMongoR);
        }
    }

    /* Transfer the fronts to the GPU. */
    if(!wsMongoF->transfer(cudaMemcpyHostToDevice))
    {
        return GPUQREngine_Cleanup (QRENGINE_GPUERROR,
            userFronts, fronts, numFronts, wsMongoF, wsMongoR);
    }

    /* Do the factorization for this set of fronts. */
    QREngineResultCode result = GPUQREngine_Internal(gpuMemorySize, fronts,
        numFronts, NULL, NULL, NULL, stats);
    if(result != QRENGINE_SUCCESS)
    {
        return GPUQREngine_Cleanup (result,
            userFronts, fronts, numFronts, wsMongoF, wsMongoR);
    }

    /* COPY USER DATA (our R back to user's R) */
    for(int f=0; f<numFronts; f++)
    {
        Front *userFront = &(userFronts[f]);
        double *R = (&fronts[f])->cpuR;
        double *userR = userFront->cpuR;
        Int m = userFront->fm;
        Int n = userFront->fn;
        Int rank = userFront->rank;
        bool isColMajor = userFront->isColMajor;
        Int ldn = userFront->ldn;
        for(Int i=0; i<rank; i++)
        {
            for(Int j=0; j<n; j++)
            {
                userR[i*ldn+j] = (isColMajor ? R[j*n+i] : R[i*n+j]);
            }
        }
    }

    /* Return that the factorization was successful. */
    return GPUQREngine_Cleanup (QRENGINE_SUCCESS,
        userFronts, fronts, numFronts, wsMongoF, wsMongoR);
}

QREngineResultCode GPUQREngine_Cleanup
(
    QREngineResultCode code,    // The result code that we're exiting with
    Front *userFronts,          // The user-provided list of fronts
    Front *fronts,              // The internal copy of the user's fronts
    Int numFronts,              // The number of fronts to be factorized
    Workspace *wsMongoF,        // Pointer to the total GPU Front workspace
    Workspace *wsMongoR         // Pointer to the total CPU R workspace
)
{
    /* Cleanup fronts. */
    for(int f=0; f<numFronts; f++)
    {
        Front *userFront = (&userFronts[f]);
        Front *front = &(fronts[f]);
        if(front != NULL)
        {
            /* If we had to attach our own stair, clean it up. */
            if(userFront->Stair == NULL && front->Stair != NULL)
            {
                front->Stair = (Int *) SuiteSparse_free(front->Stair);
            }

            /* Detach front data since it's managed by the mongo. */
            front->F = NULL;
        }
    }
    fronts = (Front *) SuiteSparse_free(fronts);

    /* Free the mongo structures. Note that Workspace checks for NULL. */
    wsMongoF = Workspace::destroy(wsMongoF);
    wsMongoR = Workspace::destroy(wsMongoR);

    return code;
}

Int *GPUQREngine_FindStaircase
(
    Front *front                // The front whose staircase we are computing
)
{
    Int fm = front->fm;
    Int fn = front->fn;

    double *F = front->F;
    Int *Stair = (Int*) SuiteSparse_malloc(fn, sizeof(Int));
    if(!F || !Stair) return NULL;

    Int lastStair = 0;
    for(int j=0; j<fn; j++)
    {
        int i;
        for(i=fm-1; i>lastStair && F[i*fn+j] == 0.0; i--);
        Stair[j] = lastStair = i;
    }

    return Stair;
}

// =============================================================================
// === GPUQREngine/Include/Kernel/qrKernel.cu ==================================
// =============================================================================

__global__ void qrKernel
(
    TaskDescriptor* Queue,
    int QueueLength
)
{
    /* Copy the task details to shared memory. */
    if(threadIdx.x == 0)
    {
        IsApplyFactorize = 0;
        myTask = Queue[blockIdx.x];
    }
    __syncthreads();

    switch(myTask.Type)
    {
        case TASKTYPE_SAssembly:    sassemble();    return;
        case TASKTYPE_PackAssembly: packassemble(); return;

        case TASKTYPE_FactorizeVT_3x1:  factorize_3_by_1_tile_vt();      return;
        case TASKTYPE_FactorizeVT_2x1:  factorize_2_by_1_tile_vt();      return;
        case TASKTYPE_FactorizeVT_1x1:  factorize_1_by_1_tile_vt();      return;
        case TASKTYPE_FactorizeVT_3x1e: factorize_3_by_1_tile_vt_edge(); return;
        case TASKTYPE_FactorizeVT_2x1e: factorize_2_by_1_tile_vt_edge(); return;
        case TASKTYPE_FactorizeVT_1x1e: factorize_1_by_1_tile_vt_edge(); return;
        case TASKTYPE_FactorizeVT_3x1w: factorize_96_by_32();            return;

        case TASKTYPE_Apply3: block_apply_3(); return;
        case TASKTYPE_Apply2: block_apply_2(); return;
        case TASKTYPE_Apply1: block_apply_1(); return;

        #ifdef GPUQRENGINE_PIPELINING
        // Apply3_Factorize[3 or 2]: (note fallthrough to next case)
        case TASKTYPE_Apply3_Factorize3:
            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 3;

        case TASKTYPE_Apply3_Factorize2:
            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 2;
            block_apply_3_by_1();
            break;

        // Apply2_Factorize[3, 2, or 1]: (note fallthrough to next case)
        case TASKTYPE_Apply2_Factorize3:
            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 3;

        case TASKTYPE_Apply2_Factorize2:
            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 2;

        case TASKTYPE_Apply2_Factorize1:
            if(threadIdx.x == 0 && !IsApplyFactorize) IsApplyFactorize = 1;
            block_apply_2_by_1();
            break;
        #endif

        default: break;
    }

    #ifdef GPUQRENGINE_PIPELINING
    /* Tasks that get to this point are Apply-Factorize tasks
       because all other should have returned in the switch above. */
    switch(myTask.Type)
    {
        case TASKTYPE_Apply3_Factorize3: 
        case TASKTYPE_Apply2_Factorize3: factorize_3_by_1_tile_vt_edge(); break;
        case TASKTYPE_Apply3_Factorize2: 
        case TASKTYPE_Apply2_Factorize2: factorize_2_by_1_tile_vt_edge(); break;
        case TASKTYPE_Apply2_Factorize1: factorize_1_by_1_tile_vt_edge(); break;
    }
    #endif
}

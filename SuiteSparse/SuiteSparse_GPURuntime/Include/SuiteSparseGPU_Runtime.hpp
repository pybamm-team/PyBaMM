// =============================================================================
// === SuiteSparse_GPURuntime/Include/SuiteSparse_GPURuntime.hpp ===============
// =============================================================================

#ifndef SUITESPARSEGPU_RUNTIME_HPP
#define SUITESPARSEGPU_RUNTIME_HPP

#include "cuda_runtime.h"
#include "SuiteSparse_config.h"

#include <stdlib.h>

#include "SuiteSparseGPU_macros.hpp"

#if DEBUG_ATLEAST_ERRORONLY
#include <stdio.h>
#endif

class Workspace;

#include "SuiteSparseGPU_Workspace.hpp"

// Version information:
#define SUITESPARSE_GPURUNTIME_DATE "May 4, 2016"
#define SUITESPARSE_GPURUNTIME_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define SUITESPARSE_GPURUNTIME_MAIN_VERSION 1
#define SUITESPARSE_GPURUNTIME_SUB_VERSION 0
#define SUITESPARSE_GPURUNTIME_SUBSUB_VERSION 5
#define SUITESPARSE_GPURUNTIME_VERSION \
    SUITESPARSE_GPURUNTIME_VER_CODE(SUITESPARSE_GPURUNTIME_MAIN_VERSION,SUITESPARSE_GPURUNTIME_SUB_VERSION)

#endif

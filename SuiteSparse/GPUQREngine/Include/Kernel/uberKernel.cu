// =============================================================================
// === GPUQREngine/Include/Kernel/uberKernel.cu ================================
// =============================================================================

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_SEntry.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"

/*** Shared Memory Allocation ************************************************/

#include "sharedMemory.hpp"


/*** Assembly Device Functions ***********************************************/

#include "Kernel/Assemble/sAssemble.cu"
#include "Kernel/Assemble/packAssemble.cu"


/*** Apply Device Functions **************************************************/

#include "Kernel/Apply/params_apply.hpp"

#include "Kernel/Apply/block_apply_3.cu"
#include "Kernel/Apply/block_apply_2.cu"
#include "Kernel/Apply/block_apply_1.cu"

#ifdef GPUQRENGINE_PIPELINING
#include "Kernel/Apply/block_apply_3_by_1.cu"
#include "Kernel/Apply/block_apply_2_by_1.cu"
#endif

/*** Factorize Device Functions **********************************************/

#include "Kernel/Factorize/factorize_vt_3_by_1.cu"
#include "Kernel/Factorize/factorize_vt_2_by_1.cu"
#include "Kernel/Factorize/factorize_vt_1_by_1.cu"
#include "Kernel/Factorize/factorize_vt_3_by_1_edge.cu"
#include "Kernel/Factorize/factorize_vt_2_by_1_edge.cu"
#include "Kernel/Factorize/factorize_vt_1_by_1_edge.cu"
#include "Kernel/Factorize/factorize_3_by_1.cu"

/*** Main Uberkernel Global Function *****************************************/

#include "Kernel/qrKernel.cu"

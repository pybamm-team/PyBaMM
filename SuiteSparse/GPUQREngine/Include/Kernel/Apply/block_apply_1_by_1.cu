// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/block_apply_1_by_1.cu ==================
// =============================================================================

//------------------------------------------------------------------------------
// block_apply_1_by_1: handles all edge cases, but just a single column tile
//------------------------------------------------------------------------------

#define BLOCK_APPLY block_apply_1_by_1
#define ROW_PANELSIZE 1
#define COL_PANELSIZE 1
#include "block_apply.cu"

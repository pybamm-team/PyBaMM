// =============================================================================
// === GPUQREngine/Include/Kernel/Factorize/factorize_vt_3_by_1.cu =============
// =============================================================================

//------------------------------------------------------------------------------
// 3-by-1 factorize, with VT and tiles, no edge case.  384 threads
//------------------------------------------------------------------------------

#define FACTORIZE       factorize_3_by_1_tile_vt
#define ROW_PANELSIZE   (3)
#define M               (ROW_PANELSIZE * TILESIZE)
#define N               (TILESIZE)
#define BITTYROWS       (8)
#include "Kernel/Factorize/factorize_vt.cu"

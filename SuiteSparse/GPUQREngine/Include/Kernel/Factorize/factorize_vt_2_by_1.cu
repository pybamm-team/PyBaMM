// =============================================================================
// === GPUQREngine/Include/Kernel/Factorize/factorize_vt_2_by_1.cu =============
// =============================================================================

//------------------------------------------------------------------------------
// 2-by-1 factorize, with VT and tiles, no edge case.  256 threads
//------------------------------------------------------------------------------

#define FACTORIZE       factorize_2_by_1_tile_vt
#define ROW_PANELSIZE   (2)
#define M               (ROW_PANELSIZE * TILESIZE)
#define N               (TILESIZE)
#define BITTYROWS       (8)
#include "Kernel/Factorize/factorize_vt.cu"

// =============================================================================
// === GPUQREngine/Include/Kernel/Factorize/factorize_vt_1_by_1.cu =============
// =============================================================================

//------------------------------------------------------------------------------
// 1-by-1 factorize, with VT and tiles, no edge case.  256 threads
//------------------------------------------------------------------------------

#define FACTORIZE       factorize_1_by_1_tile_vt
#define ROW_PANELSIZE   (1)
#define M               (ROW_PANELSIZE * TILESIZE)
#define N               (TILESIZE)
#define BITTYROWS       (4)
#include "Kernel/Factorize/factorize_vt.cu"

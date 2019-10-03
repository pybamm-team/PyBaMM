// =============================================================================
// === GPUQREngine/Include/Kernel/Factorize/factorize_vt_2_by_1_edge.cu ========
// =============================================================================

//------------------------------------------------------------------------------
// 2-by-1 factorize, with VT and tiles, edge case.  256 threads
//------------------------------------------------------------------------------

#define FACTORIZE       factorize_2_by_1_tile_vt_edge
#define EDGE_CASE
#define ROW_PANELSIZE   (2)
#define M               (ROW_PANELSIZE * TILESIZE)
#define N               (TILESIZE)
#define BITTYROWS       (8)
#include "Kernel/Factorize/factorize_vt.cu"

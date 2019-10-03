// =============================================================================
// === GPUQREngine/Include/Kernel/Assemble/packAssemble.cu =====================
// =============================================================================

__device__ void packassemble ( )
{
    // Use shared memory for Rjmap and Rimap.
    int *shRimap = shMemory.packassemble.Rimap;
    int *shRjmap = shMemory.packassemble.Rjmap;

    double *C     = myTask.AuxAddress[0];
    double *P     = myTask.AuxAddress[1];
    int *Rjmap    = (int*) myTask.AuxAddress[2];
    int *Rimap    = (int*) myTask.AuxAddress[3];
//  int fm        = myTask.fm;
    int fn        = myTask.fn;
    int pn        = myTask.extra[0];
//  int cm        = myTask.extra[1];
//  int cn        = myTask.extra[2];
    int cTileSize = myTask.extra[3];
    int cistart   = myTask.extra[4];
    int ciend     = myTask.extra[5];
    int cjstart   = myTask.extra[6];
    int cjend     = myTask.extra[7];

    // Fill Rjmap and Rimaps.
    int ctm = ciend - cistart;      // # cell tile rows
    int ctn = cjend - cjstart;      // # cell tile cols
    for(int p=threadIdx.x; p<ctm; p+=blockDim.x)
    {
        shRimap[p] = Rimap[cistart+p];
    }
    for(int p=threadIdx.x; p<ctn; p+=blockDim.x)
    {
        shRjmap[p] = Rjmap[cjstart+p]; 
    }
    __syncthreads();

    for(int p=threadIdx.x; p<cTileSize; p+=blockDim.x)
    {
        // Translate local tile coordinates to contribution block ci, cj.
        int cil = p / ctn;          // ci local to the tile
        int cjl = p % ctn;          // cj local to the tile
        int ci = cistart + cil;     // ci is really the start plus local ci
        int cj = cjstart + cjl;     // cj is really the start plus local cj

        // Avoid copying the zeroes by only copying the upper-triangular bits.
        if(cj >= ci)
        {
            int fi = shRimap[cil];
            int fj = shRjmap[cjl];
            int cindex = fn*ci+cj;
            int pindex = pn*fi+fj;            
            P[pindex] = C[cindex];
        }
    }
}

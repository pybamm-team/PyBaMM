/* ========================================================================== */
/* ========================= CHOLMOD CUDA/C kernels ========================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/GPU Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

#include <stdio.h>
#include "SuiteSparse_config.h"
/* 64-bit version only */
#define Int SuiteSparse_long

extern "C" {

  __global__ void kernelCreateMap ( Int *d_Map, Int *d_Ls, 
				    Int psi, Int nsrow )
  /*
    Ls[supernode row] = Matrix Row
  */
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < nsrow ) {
      d_Map[d_Ls[psi+tid]] = ((Int) (tid));
    }
  }
  
  __global__ void kernelCreateRelativeMap ( Int *d_Map, Int *d_Ls, 
					    Int *d_RelativeMap, 
					    Int pdi1, Int ndrow )
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < ndrow ) {
      d_RelativeMap[tid] = d_Map[d_Ls[pdi1+tid]];
    }
  }
  
  __global__ void kernelAddUpdate ( double *d_A, double *devPtrC, 
				    Int *d_RelativeMap, 
				    Int ndrow1, Int ndrow2, 
				    Int nsrow )
  {
    int idrow = blockIdx.x * blockDim.x + threadIdx.x;
    int idcol = blockIdx.y * blockDim.y + threadIdx.y;
    if ( idrow < ndrow2  && idcol < ndrow1 ) {
      Int idx = d_RelativeMap[idrow] + d_RelativeMap[idcol] * nsrow;
      d_A[idx] += devPtrC[idrow+ndrow2*idcol];
    }
  }
  
  __global__ void kernelAddComplexUpdate ( double *d_A, double *devPtrC, 
					   Int *d_RelativeMap, 
					   Int ndrow1, Int ndrow2, 
					   Int nsrow )
  {
    int idrow = blockIdx.x * blockDim.x + threadIdx.x;
    int idcol = blockIdx.y * blockDim.y + threadIdx.y;
    if ( idrow < ndrow2  && idcol < ndrow1 ) {
      Int idx = d_RelativeMap[idrow] + d_RelativeMap[idcol] * nsrow;
      d_A[idx*2] += devPtrC[(idrow+ndrow2*idcol)*2];
      d_A[idx*2+1] += devPtrC[(idrow+ndrow2*idcol)*2+1];
    }
  }
  
  __global__ void kernelSumA ( double *a1, double *a2, const double alpha, 
			       int nsrow, int nscol ) {
    int isrow = blockIdx.x * blockDim.x + threadIdx.x;
    int iscol = blockIdx.y * blockDim.y + threadIdx.y;
    if ( isrow < nsrow && iscol < nscol ) {
      Int idx = iscol*nsrow + isrow;
      a1[idx] += alpha * a2[idx];
    }
  }

  __global__ void kernelSumComplexA ( double *a1, double *a2, 
				      const double alpha, int nsrow, 
				      int nscol ) {
    int isrow = blockIdx.x * blockDim.x + threadIdx.x;
    int iscol = blockIdx.y * blockDim.y + threadIdx.y;
    if ( isrow < nsrow && iscol < nscol ) {
      Int idx = iscol*nsrow + isrow;
      a1[idx*2] += alpha * a2[idx*2];
      a1[idx*2+1] += alpha * a2[idx*2+1];
    }
  }

  /* ======================================================================== */
  /* using Ls and Lpi data already on the device, construct Map */
  /* ======================================================================== */
  int createMapOnDevice ( Int *d_Map, Int *d_Ls, 
			  Int  psi, Int nsrow ) 
  {
    unsigned int kgrid = (nsrow+31)/32;
    unsigned int kblock = 32;
    kernelCreateMap <<<kgrid, kblock>>> ( d_Map, d_Ls, psi, nsrow );
    return 0;
  }


  int createRelativeMapOnDevice ( Int *d_Map, Int *d_Ls, 
				  Int *d_RelativeMap,Int  pdi1, 
				  Int ndrow, cudaStream_t* astream )
  {
    unsigned int kgrid = (ndrow+255)/256;
    unsigned int kblock = 256;
    kernelCreateRelativeMap <<<kgrid, kblock, 0, *astream>>> 
      ( d_Map, d_Ls, d_RelativeMap, pdi1, ndrow);
    return 0;
  }


  /* ======================================================================== */
  int addUpdateOnDevice ( double *d_A, double *devPtrC, 
			  Int *d_RelativeMap, Int ndrow1, 
			  Int ndrow2, Int nsrow, 
			  cudaStream_t* astream )
  /* ======================================================================== */
  /* Assemble the Schur complment from a descendant supernode into the current
     supernode */ 
  /* ======================================================================== */
{
  dim3 grids;
  dim3 blocks;

  blocks.x = 16;
  blocks.y = 16;
  blocks.z = 1;

  grids.x = (ndrow2+15)/16; 
  grids.y = (ndrow1+15)/16; 

  kernelAddUpdate <<<grids, blocks, 0, *astream>>> 
    ( d_A, devPtrC, d_RelativeMap, ndrow1, ndrow2, nsrow );

  return 0;
}

  /* ======================================================================== */
  int addComplexUpdateOnDevice ( double *d_A, double *devPtrC, 
				 Int *d_RelativeMap, Int ndrow1, 
				 Int ndrow2, Int nsrow, 
				 cudaStream_t* astream )
  /* ======================================================================== */
  /* Assemble the Schur complment from a descendant supernode into the current
     supernode */ 
  /* ======================================================================== */
{
  dim3 grids;
  dim3 blocks;

  blocks.x = 16;
  blocks.y = 16;
  blocks.z = 1;

  grids.x = (ndrow2+15)/16; 
  grids.y = (ndrow1+15)/16; 

  kernelAddComplexUpdate <<<grids, blocks, 0, *astream>>> 
    ( d_A, devPtrC, d_RelativeMap, ndrow1, ndrow2, nsrow );

  return 0;
}

  int sumAOnDevice ( double *a1, double *a2, const double alpha, 
		     int nsrow, int nscol )
  {
    dim3 grids;
    dim3 blocks;
    blocks.x = 16;
    blocks.y = 16;
    blocks.z = 1;
    grids.x = (nsrow+15)/16;
    grids.y = (nscol+15)/16;
    kernelSumA <<<grids, blocks, 0, 0>>> ( a1, a2, alpha, nsrow, nscol );
    return 0;
  }

  int sumComplexAOnDevice ( double *a1, double *a2, const double alpha, 
			    int nsrow, int nscol )
  {
    dim3 grids;
    dim3 blocks;
    blocks.x = 16;
    blocks.y = 16;
    blocks.z = 1;
    grids.x = (nsrow+15)/16;
    grids.y = (nscol+15)/16;
    kernelSumComplexA <<<grids, blocks, 0, 0>>> ( a1, a2, alpha, nsrow, nscol );
    return 0;
  }

}

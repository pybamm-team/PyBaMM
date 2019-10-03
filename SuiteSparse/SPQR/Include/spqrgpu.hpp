#ifndef SPQRGPU_HPP_
#define SPQRGPU_HPP_

#include "GPUQREngine.hpp"

void spqrgpu_kernel
(
    spqr_blob <double> *Blob    // contains the entire problem input/output
) ;

void spqrgpu_kernel             // placeholder, since complex case not supported
(
    spqr_blob <Complex> *Blob
) ;

void spqrgpu_computeFrontStaging
(
    // inputs, not modified on output
    Long numFronts,     // total number of fronts (nf in caller)
    Long *Parent,       // size nf+1, assembly tree (f=nf is placeholder)
    Long *Childp,       // size nf+2, children of f are
                        //      Child [Childp [f] ... Childp [f+1]-1]
    Long *Child,        // size nf+1.

    Long *Fm,           // size nf+1, front f has Fm [f] rows
    Long *Cm,           // size nf+1, front f has Cm [f] rows in contrib
    Long *Rp,           // size nf+1, Rj[Rp[f]...Rp[f+1]-1] are the cols in f
    Long *Sp,           // size m+1, row pointers for sparse matrix S
    Long *Sleft,        // size n+2, see spqr_stranspose for description
    Long *Super,        // size nf+1, front f pivotal cols are
                        //      Super[f]..Super[f+1]-1
    Long *Post,         // size nf+1, front f is kth, if f = Post [k]

    Long RimapSize,     // scalar, size of Rimap on the GPU (# of int's)
    Long RjmapSize,     // scalar, size of Rjmap on the GPU (# of int's)

    // output, not defined on input:
    bool *feasible,     // scalar, true if feasible, false if GPU memory too low
    Long *numStages,    // scalar, number of stages
    Long *Stagingp,     // size nf+2, fronts are in the list
                        //      Post [Stagingp [stage]...Stagingp[stage+1]-1]
    Long *StageMap,     // size nf, front f is in stage StageMap [f]

    size_t *FSize,      // size nf+1, FSize[stage]: size in bytes of MongoF
    size_t *RSize,      // size nf+1, Rsize[stage]: size in bytes of MongoR
    size_t *SSize,      // size nf+1, Ssize[stage]: size in bytes of S
    Long *FOffsets,     // size nf, front f in MondoF [FOffsets[f]...] on GPU
    Long *ROffsets,     // size nf, R block in MondoR [Roffsets[f]...] on CPU
    Long *SOffsets,     // size nf, S entries for front f are in
                        //      wsS [SOffsets[f]...]

    // input/output:
    cholmod_common *cc
);

void spqrgpu_buildAssemblyMaps
(
    Long numFronts,
    Long n,
    Long *Fmap,
    Long *Post,
    Long *Super,
    Long *Rp,
    Long *Rj,
    Long *Sleft,
    Long *Sp,
    Long *Sj,
    double *Sx,
    Long *Fm,
    Long *Cm,
    Long *Childp,
    Long *Child,
    Long *CompleteStair,
    int *CompleteRjmap,
    Long *RjmapOffsets,
    int *CompleteRimap,
    Long *RimapOffsets,
    SEntry *cpuS
);

#endif

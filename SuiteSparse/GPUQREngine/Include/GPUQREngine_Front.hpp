// =============================================================================
// === GPUQREngine/Include/GPUQREngine_Front.hpp ===============================
// =============================================================================
//
// The Front is a principal class in the GPUQREngine.
//
// Fronts wrap all metadata required to complete its factorization.
// When involved in a sparse factorization, additional metadata is present in
// the SparseMeta member, "sparseMeta."
//
// =============================================================================

#ifndef GPUQRENGINE_FRONT_HPP
#define GPUQRENGINE_FRONT_HPP

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_SparseMeta.hpp"
#include "GPUQREngine_FrontState.hpp"

class Front
{
public:
    Int fids;           // Front id within a stage
    Int pids;           // Parent front id within a stage
    Int fidg;           // Front id global to problem
    Int pidg;           // Parent id global to problem

    Int fm;             // # rows
    Int fn;             // # cols
    Int rank;           // (derived) MIN(fm, fn)

    // adv layout options
    bool isColMajor;    // default:F
    Int ldn;            // user-specified desired leading dim

    double *F;          // Front data
    double *gpuF;       // The frontal matrix on the GPU.
    double *cpuR;       // The cpu location of the R factor.

    FrontState state;   // The front factorization state

    Int *Stair;         // The staircase allows us to exploit block zeroes.

    /* Extension to support sparse factorization. */
    SparseMeta sparseMeta;

    /* Debug Fields */
    bool printMe;

    void* operator new(long unsigned int reqMem, Front* ptr){ return ptr; }

    Front(
        Int fids_arg,                   // the front identifier
        Int pids_arg,                   // the parent identifier
        Int fm_arg,                     // the # of rows in the front
        Int fn_arg,                     // the # of cols in the front
        bool isColMajor_arg=false,      // whether the front is col-major
        Int ldn_arg=EMPTY)              // the leading dimension
    {
        fids = fids_arg ;
        pids = pids_arg ;
        fidg = EMPTY;
        pidg = EMPTY;

        fm = fm_arg ;
        fn = fn_arg ;
        rank = MIN(fm, fn);

        isColMajor = isColMajor_arg ;
        ldn = (ldn_arg == EMPTY ? fn : ldn_arg) ;

        F = NULL;
        gpuF = NULL;
        cpuR = NULL;

        state = ALLOCATE_WAIT;

        Stair = NULL;

        sparseMeta = SparseMeta();

        printMe = false;
    }

    ~Front()
    {
        // for the sparse case, F is NULL on the CPU
        F = (double *) SuiteSparse_free (F) ;
    }

    bool isAllocated
    (
        void
    )
    {
        return gpuF != NULL;
    }

    bool isDense
    (
        void
    )
    {
        // NOTE: this code is tested by the SPQR/Tcov test, but that test does
        // not flag this line as being tested in the coverage output.  This is
        // determined by commenting out the following line, and seeing it
        // trigger under 'make' in SPQR/Tcov:
        //      { fprintf (stderr, "statement tested!\n") ; exit (0) ; }
        // This same problem occurs elsewhere in GPUQREngine/Include/*
        // and thus only affects *.hpp files #include'd in other files.
        // The optimizer must be getting in the way, or some related effect.
        return (!sparseMeta.isSparse);
    }

    bool isSparse
    (
        void
    )
    {
        // NOTE: also tested by SPQR/Tcov, but not flagged as such in cov output
        return (sparseMeta.isSparse);
    }

    bool isStaged
    (
        void
    )
    {
        // NOTE: also tested by SPQR/Tcov, but not flagged as such in cov output
        return (isSparse() && sparseMeta.isStaged);
    }

    bool isPushOnly
    (
        void
    )
    {
        return (isSparse() && sparseMeta.pushOnly);
    }

    size_t getNumFrontValues
    (
        void
    )
    {
        return fm * fn;
    }

    size_t getNumRValues
    (
        void
    )
    {
        return rank * fn;
    }

    bool isTooBigForSmallQR
    (
        void
    )
    {
        return (fm > 96 || fn > 32);
    }

};

#endif

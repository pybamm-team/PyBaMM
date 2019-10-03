/* ========================================================================== */
/* === Include/Mongoose_ImproveFM.hpp ======================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_IMPROVEFM_HPP
#define MONGOOSE_IMPROVEFM_HPP

#include "Mongoose_CutCost.hpp"
#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

/* Swap candidates have the following features: */
struct SwapCandidate
{
    Int vertex;
    bool partition;
    double vertexWeight;
    double gain;
    double heuCost;
    Int bhPosition;
    double imbalance;

    SwapCandidate()
    {
        vertex       = 0;
        partition    = false;
        vertexWeight = 0.0;
        gain         = -INFINITY;
        heuCost      = INFINITY;
        bhPosition   = -1;
        imbalance    = 0.0;
    }
};

void improveCutUsingFM(EdgeCutProblem *, const EdgeCut_Options *);

void fmSwap(EdgeCutProblem *, const EdgeCut_Options *, Int vertex, double gain,
            bool oldPartition);
void fmRefine_worker(EdgeCutProblem *graph, const EdgeCut_Options *options);
void calculateGain(EdgeCutProblem *, const EdgeCut_Options *, Int vertex, double *out_gain,
                   Int *out_externalDegree);

} // end namespace Mongoose

#endif

/* ========================================================================== */
/* === Source/Mongoose_GuessCut.cpp ========================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_GuessCut.hpp"
#include "Mongoose_ImproveQP.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Random.hpp"
#include "Mongoose_Waterdance.hpp"

namespace Mongoose
{

//-----------------------------------------------------------------------------
// This function takes a graph with options and computes the initial guess cut
//-----------------------------------------------------------------------------
bool guessCut(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    switch (options->initial_cut_type)
    {
    case InitialEdgeCut_QP:
        for (Int k = 0; k < graph->n; k++)
        {
            graph->partition[k] = false;
        }
        graph->partition[0] = true;

        bhLoad(graph, options);
        if (!improveCutUsingQP(graph, options, true))
        {
            return false;
            // Error - QP Failure
        }
        break;
    case InitialEdgeCut_Random:
        for (Int k = 0; k < graph->n; k++)
        {
            graph->partition[k] = (random() % 2 == 0);
        }

        bhLoad(graph, options);
        break;
    case InitialEdgeCut_NaturalOrder:
        for (Int k = 0; k < graph->n; k++)
        {
            graph->partition[k] = (k < graph->n / 2);
        }
        bhLoad(graph, options);
        break;
    }

    /* Do the waterdance refinement. */
    waterdance(graph, options);

    return true;
}

} // end namespace Mongoose

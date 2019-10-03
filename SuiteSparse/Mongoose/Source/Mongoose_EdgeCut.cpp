/* ========================================================================== */
/* === Source/Mongoose_EdgeCut.cpp ========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_EdgeCut.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_Coarsening.hpp"
#include "Mongoose_GuessCut.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"
#include "Mongoose_Random.hpp"
#include "Mongoose_Refinement.hpp"
#include "Mongoose_Waterdance.hpp"

#include <algorithm>

namespace Mongoose
{

bool optionsAreValid(const EdgeCut_Options *options);
void cleanup(EdgeCutProblem *graph);

EdgeCut::~EdgeCut()
{
    SuiteSparse_free(partition);
    SuiteSparse_free(this);
}

EdgeCut *edge_cut(const Graph *graph)
{
    // use default options if not present
    EdgeCut_Options *options = EdgeCut_Options::create();

    if (!options)
        return NULL;

    EdgeCut *result = edge_cut(graph, options);

    options->~EdgeCut_Options();

    return (result);
}

EdgeCut *edge_cut(const Graph *graph, const EdgeCut_Options *options)
{
    // Check inputs
    if (!optionsAreValid(options))
        return NULL;

    setRandomSeed(options->random_seed);

    if (!graph)
        return NULL;

    // Create an EdgeCutProblem
    EdgeCutProblem *problem = EdgeCutProblem::create(graph);

    if (!problem)
        return NULL;

    EdgeCut *result = edge_cut(problem, options);

    problem->~EdgeCutProblem();

    return result;
}

EdgeCut *edge_cut(EdgeCutProblem *problem, const EdgeCut_Options *options)
{
    // Check inputs
    if (!optionsAreValid(options))
        return NULL;

    setRandomSeed(options->random_seed);

    if (!problem)
        return NULL;

    /* Finish initialization */
    problem->initialize(options);

    /* Keep track of what the current graph is at any stage */
    EdgeCutProblem *current = problem;

    /* If we need to coarsen the graph, do the coarsening. */
    while (current->n >= options->coarsen_limit)
    {
        match(current, options);
        EdgeCutProblem *next = coarsen(current, options);

        /* If we ran out of memory during coarsening, unwind the stack. */
        if (!next)
        {
            while (current != problem)
            {
                next = current->parent;
                current->~EdgeCutProblem();
                current = next;
            }
            return NULL;
        }

        current = next;
    }

    /*
     * Generate a guess cut and do FM refinement.
     * On failure, unwind the stack.
     */
    if (!guessCut(current, options))
    {
        while (current != problem)
        {
            EdgeCutProblem *next = current->parent;
            current->~EdgeCutProblem();
            current = next;
        }
        return NULL;
    }

    /*
     * Refine the guess cut back to the beginning.
     */
    while (current->parent != NULL)
    {
        current = refine(current, options);
        waterdance(current, options);
    }

    cleanup(current);

    EdgeCut *result = (EdgeCut*)SuiteSparse_malloc(1, sizeof(EdgeCut));

    if (!result)
    {
        return NULL;
    }

    result->partition = current->partition;
    current->partition = NULL; // Unlink pointer
    result->n         = current->n;
    result->cut_cost  = current->cutCost;
    result->cut_size  = current->cutSize;
    result->w0        = current->W0;
    result->w1        = current->W1;
    result->imbalance = current->imbalance;

    return result;
}

bool optionsAreValid(const EdgeCut_Options *options)
{
    if (!options)
    {
        LogError("Fatal Error: options struct cannot be NULL.");
        return (false);
    }

    if (options->coarsen_limit < 1)
    {
        LogError("Fatal Error: options->coarsen_limit cannot be less than one.");
        return (false);
    }

    if (options->high_degree_threshold < 0)
    {
        LogError("Fatal Error: options->high_degree_threshold cannot be less "
                 "than zero.");
        return (false);
    }

    if (options->num_dances < 0)
    {
        LogError("Fatal Error: options->num_dances cannot be less than zero.");
        return (false);
    }

    if (options->FM_search_depth < 0)
    {
        LogError(
            "Fatal Error: options->fmSearchDepth cannot be less than zero.");
        return (false);
    }

    if (options->FM_consider_count < 0)
    {
        LogError(
            "Fatal Error: options->FM_consider_count cannot be less than zero.");
        return (false);
    }

    if (options->FM_max_num_refinements < 0)
    {
        LogError("Fatal Error: options->FM_max_num_refinements cannot be less "
                 "than zero.");
        return (false);
    }

    if (options->gradproj_tolerance < 0)
    {
        LogError("Fatal Error: options->gradproj_tolerance cannot be less than "
                 "zero.");
        return (false);
    }

    if (options->gradproj_iteration_limit < 0)
    {
        LogError("Fatal Error: options->gradProjIterationLimit cannot be less "
                 "than zero.");
        return (false);
    }

    if (options->target_split < 0 || options->target_split > 1)
    {
        LogError(
            "Fatal Error: options->target_split must be in the range [0, 1].");
        return (false);
    }

    if (options->soft_split_tolerance < 0)
    {
        LogError("Fatal Error: options->soft_split_tolerance cannot be less than "
                 "zero.");
        return (false);
    }

    return (true);
}

void cleanup(EdgeCutProblem *G)
{
    Int cutSize = 0;
    for (Int p = 0; p < 2; p++)
    {
        Int *bhHeap = G->bhHeap[p];
        for (Int i = 0; i < G->bhSize[p]; i++)
        {
            cutSize += G->externalDegree[bhHeap[i]];
        }
    }

    G->imbalance = fabs(G->imbalance);
    G->cutSize   = cutSize / 2;
    G->cutCost   = G->cutCost / 2;
}

} // end namespace Mongoose

/* ========================================================================== */
/* === Include/Mongoose_EdgeCutOptions.hpp ================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_EDGECUTOPTIONS_HPP
#define MONGOOSE_EDGECUTOPTIONS_HPP

#include "Mongoose_Internal.hpp"

namespace Mongoose
{

struct EdgeCut_Options
{
    Int random_seed;

    /** Coarsening Options ***************************************************/
    Int coarsen_limit;
    MatchingStrategy matching_strategy;
    bool do_community_matching;
    double high_degree_threshold;

    /** Guess Partitioning Options *******************************************/
    InitialEdgeCutType initial_cut_type; /* The guess cut type to use */

    /** Waterdance Options ***************************************************/
    Int num_dances; /* The number of interplays between FM and QP
                      at any one coarsening level. */

    /**** Fidducia-Mattheyes Options *****************************************/
    bool use_FM;              /* Flag governing the use of FM               */
    Int FM_search_depth;       /* The # of non-positive gain move to make    */
    Int FM_consider_count;     /* The # of heap entries to consider          */
    Int FM_max_num_refinements; /* Max # of times to run Fiduccia-Mattheyses  */

    /**** Quadratic Programming Options **************************************/
    bool use_QP_gradproj;         /* Flag governing the use of gradproj       */
    double gradproj_tolerance;   /* Convergence tol for projected gradient   */
    Int gradproj_iteration_limit; /* Max # of iterations for gradproj         */

    /** Final Partition Target Metrics ***************************************/
    double target_split;        /* The desired split ratio (default 50/50)  */
    double soft_split_tolerance; /* The allowable soft split tolerance.      */
                               /* Cuts within this tolerance are treated   */
                               /* equally.                                 */

    /* Constructor & Destructor */
    static EdgeCut_Options *create();
    ~EdgeCut_Options();
};

} // end namespace Mongoose

#endif

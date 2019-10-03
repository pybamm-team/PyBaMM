/* ========================================================================== */
/* === Source/Mongoose_EdgeCutOptions.cpp =================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

/* Constructor & Destructor */
EdgeCut_Options *EdgeCut_Options::create()
{
    EdgeCut_Options *ret
        = static_cast<EdgeCut_Options *>(SuiteSparse_malloc(1, sizeof(EdgeCut_Options)));

    if (ret != NULL)
    {
        ret->random_seed = 0;

        ret->coarsen_limit        = 64;
        ret->matching_strategy    = HEMSR;
        ret->do_community_matching = false;
        ret->high_degree_threshold = 2.0;

        ret->initial_cut_type = InitialEdgeCut_Random;

        ret->num_dances = 1;

        ret->use_FM               = true;
        ret->FM_search_depth       = 50;
        ret->FM_consider_count     = 3;
        ret->FM_max_num_refinements = 20;

        ret->use_QP_gradproj          = true;
        ret->gradproj_tolerance      = 0.001;
        ret->gradproj_iteration_limit = 50;

        ret->target_split        = 0.5;
        ret->soft_split_tolerance = 0;
    }

    return ret;
}

EdgeCut_Options::~EdgeCut_Options()
{
    SuiteSparse_free(this);
}

} // end namespace Mongoose

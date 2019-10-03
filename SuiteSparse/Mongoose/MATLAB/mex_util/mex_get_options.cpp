#include "mongoose_mex.hpp"

namespace Mongoose
{

#define MEX_STRUCT_READINT(F)    returner->F = (Int) readField(matOptions, #F);
#define MEX_STRUCT_READDOUBLE(F) returner->F = readField(matOptions, #F);
#define MEX_STRUCT_READBOOL(F)   returner->F = static_cast<bool>((readField(matOptions, #F) != 0.0));
#define MEX_STRUCT_READENUM(F,T) returner->F = (T) (readField(matOptions, #F));
    
EdgeCut_Options *mex_get_options
(
    const mxArray *matOptions
)
{
    EdgeCut_Options *returner = EdgeCut_Options::create();

    if(!returner)
        return NULL;
    if(matOptions == NULL)
        return returner;

    MEX_STRUCT_READINT(random_seed);
    MEX_STRUCT_READINT(coarsen_limit);
    MEX_STRUCT_READENUM(matching_strategy, MatchingStrategy);
    MEX_STRUCT_READBOOL(do_community_matching);
    MEX_STRUCT_READDOUBLE(high_degree_threshold);
    
    /** Guess Partitioning Options *******************************************/
    MEX_STRUCT_READENUM(initial_cut_type, InitialEdgeCutType);

    /** Waterdance Options ***************************************************/
    MEX_STRUCT_READINT(num_dances);

    /**** Fidducia-Mattheyes Options *****************************************/
    MEX_STRUCT_READBOOL(use_FM);
    MEX_STRUCT_READINT(FM_search_depth);
    MEX_STRUCT_READINT(FM_consider_count);
    MEX_STRUCT_READINT(FM_max_num_refinements);

    /**** Quadratic Programming Options **************************************/
    MEX_STRUCT_READBOOL(use_QP_gradproj);
    MEX_STRUCT_READDOUBLE(gradproj_tolerance);
    MEX_STRUCT_READINT(gradproj_iteration_limit);

    /** Final Partition Target Metrics ***************************************/
    MEX_STRUCT_READDOUBLE(target_split);
    MEX_STRUCT_READDOUBLE(soft_split_tolerance);

    return returner;
}

}
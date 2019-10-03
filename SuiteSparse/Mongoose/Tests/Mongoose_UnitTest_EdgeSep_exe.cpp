
#define LOG_ERROR 1
#define LOG_WARN 1
#define LOG_INFO 0
#define LOG_TEST 1

#include "Mongoose_Test.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_IO.hpp"
#include "Mongoose_EdgeCut.hpp"

using namespace Mongoose;

int main(int argn, char** argv)
{
    (void)argn; // Unused variable
    (void)argv; // Unused variable

    SuiteSparse_start();

    // Set Logger to report all messages and turn off timing info
    Logger::setDebugLevel(All);
    Logger::setTimingFlag(false);

    // Test with NULL graph
    EdgeCut *result = edge_cut(NULL);
    assert(result == NULL);

    Graph *G = read_graph("../Matrix/bcspwr02.mtx");

    // Test with no options struct
    result = edge_cut(G);
    result->~EdgeCut();

    // Test with NULL options struct
    EdgeCut_Options *O = NULL;
    result = edge_cut(G, O);
    assert(result == NULL);

    O = EdgeCut_Options::create();

    // Test with invalid coarsen_limit
    O->coarsen_limit = 0;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->coarsen_limit = 50;

    // Test with invalid high_degree_threshold
    O->high_degree_threshold = -1;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->high_degree_threshold = 2.0;

    // Test with invalid num_dances
    O->num_dances = -1;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->num_dances = 1;

    // Test with invalid FM_search_depth
    O->FM_search_depth = -1;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->FM_search_depth = 50;

    // Test with invalid FM_consider_count
    O->FM_consider_count = -1;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->FM_consider_count = 3;

    // Test with invalid FM_max_num_refinements
    O->FM_max_num_refinements = -1;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->FM_max_num_refinements = 20;

    // Test with invalid gradproj_tolerance
    O->gradproj_tolerance = -1;
    edge_cut(G, O);
    O->gradproj_tolerance = 0.001;

    // Test with invalid gradproj_iteration_limit
    O->gradproj_iteration_limit = -1;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->gradproj_iteration_limit = 50;

    // Test with invalid target_split
    O->target_split = 1.2;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->target_split = 0.4;

    // Test with invalid tolerance
    O->soft_split_tolerance = -1;
    result = edge_cut(G, O);
    assert(result == NULL);
    O->soft_split_tolerance = 0.01;

    // Test with no QP
    O->use_QP_gradproj = false;
    result = edge_cut(G, O);
    assert(result->partition != NULL);
    result->~EdgeCut();
    O->use_QP_gradproj = true;

    // Test with no FM
    O->use_FM = false;
    result = edge_cut(G, O);
    assert(result->partition != NULL);
    result->~EdgeCut();
    O->use_FM = true;

    // Test with no coarsening
    O->coarsen_limit = 1E15;
    result = edge_cut(G, O);
    assert(result->partition != NULL);
    result->~EdgeCut();

    // Test with x = NULL (assume pattern matrix)
    G->x = NULL;
    result = edge_cut(G, O);
    assert(result->partition != NULL);
    result->~EdgeCut();
    O->coarsen_limit = 50;

    O->~EdgeCut_Options();
    G->~Graph();

    SuiteSparse_finish();

    return 0;
}

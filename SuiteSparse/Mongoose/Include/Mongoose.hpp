/* ========================================================================== */
/* === Include/Mongoose.hpp ================================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_HPP
#define MONGOOSE_HPP

#include "SuiteSparse_config.h"
#include <string>

namespace Mongoose
{

/* Type definitions */
typedef SuiteSparse_long Int;

typedef struct cs_sparse /* matrix in compressed-column or triplet form */
{
    Int nzmax; /* maximum number of entries */
    Int m;     /* number of rows */
    Int n;     /* number of columns */
    Int *p;    /* column pointers (size n+1) or col indices (size nzmax) */
    Int *i;    /* row indices, size nzmax */
    double *x; /* numerical values, size nzmax */
    Int nz;    /* # of entries in triplet matrix, -1 for compressed-col */
} cs;

/* Enumerations */
enum MatchingStrategy
{
    Random,
    HEM,
    HEMSR,
    HEMSRdeg
};

enum InitialEdgeCutType
{
    InitialEdgeCut_QP,
    InitialEdgeCut_Random,
    InitialEdgeCut_NaturalOrder
};

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
    bool use_FM;              /* Flag governing the use of FM             */
    Int FM_search_depth;       /* The # of non-positive gain move to make  */
    Int FM_consider_count;     /* The # of heap entries to consider        */
    Int FM_max_num_refinements; /* Max # of times to run FidduciaMattheyes  */

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

class Graph
{
public:
    /** Graph Data ***********************************************************/
    Int n;     /** # vertices                      */
    Int nz;    /** # edges                         */
    Int *p;    /** Column pointers                 */
    Int *i;    /** Row indices                     */
    double *x; /** Edge weight                     */
    double *w; /** Node weight                     */

    /* Constructors & Destructor */
    static Graph *create(const Int _n, const Int _nz, Int *_p = NULL,
                         Int *_i = NULL, double *_x = NULL, double *_w = NULL);
    static Graph *create(cs *matrix);
    ~Graph();

private:
    Graph();

    /** Memory Management Flags ***********************************************/
    bool shallow_p;
    bool shallow_i;
    bool shallow_x;
    bool shallow_w;
};

/**
 * Generate a Graph from a Matrix Market file.
 *
 * Generate a Graph class instance from a Matrix Market file. The matrix
 * contained in the file must be sparse, real, and square. If the matrix
 * is not symmetric, it will be made symmetric with (A+A')/2. If the matrix has
 * more than one connected component, the largest will be found and the rest
 * discarded. If a diagonal is present, it will be removed.
 *
 * @param filename the filename or path to the Matrix Market File.
 */
Graph *read_graph(const std::string &filename);

/**
 * Generate a Graph from a Matrix Market file.
 *
 * Generate a Graph class instance from a Matrix Market file. The matrix
 * contained in the file must be sparse, real, and square. If the matrix
 * is not symmetric, it will be made symmetric with (A+A')/2. If the matrix has
 * more than one connected component, the largest will be found and the rest
 * discarded. If a diagonal is present, it will be removed.
 *
 * @param filename the filename or path to the Matrix Market File.
 */
Graph *read_graph(const char *filename);

struct EdgeCut
{
    bool *partition;     /** T/F denoting partition side     */
    Int n;               /** # vertices                      */

    /** Cut Cost Metrics *****************************************************/
    double cut_cost;    /** Sum of edge weights in cut set    */
    Int cut_size;       /** Number of edges in cut set        */
    double w0;          /** Sum of partition 0 vertex weights */
    double w1;          /** Sum of partition 1 vertex weights */
    double imbalance;   /** Degree to which the partitioning
                            is imbalanced, and this is
                            computed as (0.5 - W0/W).         */

    // desctructor (no constructor)
    ~EdgeCut();
};

EdgeCut *edge_cut(const Graph *);
EdgeCut *edge_cut(const Graph *, const EdgeCut_Options *);

/* Version information */
int major_version();
int minor_version();
int patch_version();
std::string mongoose_version();

} // end namespace Mongoose

#endif

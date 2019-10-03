/**
 * demo.cpp
 * Runs a variety of computations on several input matrices and outputs
 * the results. Does not take any input. This application can be used to
 * test that compilation was successful and that everything is working
 * properly.
 */

#include "Mongoose.hpp"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace Mongoose;
using namespace std;

int main(int argn, const char **argv)
{
    #define NMAT 17
    const std::string demo_files[NMAT] = {
        "Erdos971.mtx",
        "G51.mtx",
        "GD97_b.mtx",
        "Pd.mtx",
        "bcspwr01.mtx",
        "bcspwr02.mtx",
        "bcspwr03.mtx",
        "bcspwr04.mtx",
        "bcspwr05.mtx",
        "bcspwr06.mtx",
        "bcspwr07.mtx",
        "bcspwr08.mtx",
        "bcspwr09.mtx",
        "bcspwr10.mtx",
        "dwt_992.mtx",
        "jagmesh7.mtx",
        "NotreDame_www.mtx"
    };

    cout << "********************************************************************************" << endl;
    cout << "Mongoose Graph Partitioning Library, Version " << mongoose_version() << endl;
    cout << "Copyright (C) 2017-2018" << endl;
    cout << "Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager" << endl;
    cout << "Mongoose is licensed under Version 3 of the GNU General Public License." << endl;
    cout << "Mongoose is also available under other licenses; contact authors for details." << endl;

    clock_t start = clock();
    double duration;

    for (int k = 0; k < NMAT; k++)
    {
        cout << "********************************************************************************" << endl;
        cout << "Computing an edge cut for " << demo_files[k] << "..." << endl;
        
        clock_t trial_start = clock();
        EdgeCut_Options *options = EdgeCut_Options::create();
        if (!options) return EXIT_FAILURE; // Return an error if we failed.

        options->matching_strategy = HEMSRdeg;
        options->initial_cut_type = InitialEdgeCut_QP;

        Graph *graph = read_graph("../Matrix/" + demo_files[k]);
        if (!graph)
        {
            return EXIT_FAILURE;
        }

        EdgeCut *result = edge_cut(graph, options);

        cout << "Cut Cost:       " << setprecision(2) << result->cut_cost << endl;
        if (result->imbalance < 1e-12)
        {
            // imbalance is zero; this is just a roundoff epsilon in the statistic
            cout << "Cut Imbalance:  zero (a perfect balance)" << endl;
        }
        else
        {
            cout << "Cut Imbalance:  " << setprecision(2) << 100*(result->imbalance) << "%" << endl;
        }

        double trial_duration = (std::clock() - trial_start) / (double) CLOCKS_PER_SEC;
        cout << "Trial Time:     " << trial_duration*1000 << "ms" << endl;

        options->~EdgeCut_Options();
        graph->~Graph();
        result->~EdgeCut();
    }

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    cout << "********************************************************************************" << endl;
    cout << "Total Demo Time:  " << setprecision(2) << duration << "s" << endl;

    cout << endl;
    cout << "Demo complete; all tests passed" << endl ;

    /* Return success */
    return EXIT_SUCCESS;
}

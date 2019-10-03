#include <string>
#include "Mongoose_IO.hpp"
#include "Mongoose_EdgeCut.hpp"
#include "Mongoose_Test.hpp"
#include <fstream>

using namespace Mongoose;

int runEdgeSeparatorTest(const std::string &inputFile, const double targetSplit)
{
    LogTest("Running Edge Separator Test on " << inputFile);
        
    // Given a symmetric matrix...
    EdgeCut_Options *options;
    Graph *graph;
    
    options = EdgeCut_Options::create();
    if (!options)
    {
        // Ran out of memory
        LogTest("Error creating Options struct in Edge Separator Test");
        return EXIT_FAILURE;
    }

    options->target_split = targetSplit;
    
    // Read graph from file
    graph = read_graph(inputFile);

    if (!graph)
    {
        // Ran out of memory
        LogTest("Error reading Graph from file in Edge Separator Test");
        options->~EdgeCut_Options();
        return EXIT_FAILURE;
    }

    // An edge separator should be computed with default options
    EdgeCut *result = edge_cut(graph, options);

    options->~EdgeCut_Options();

    if (!result)
    {
        // Error occurred
        LogTest("Error computing edge cut in Edge Separator Test");
        return EXIT_FAILURE;
    }
    else
    {
        // The graph should be partitioned
        assert (result->partition != NULL);
        int count = 0;
        for (int i = 0; i < result->n; i++)
        {
            bool equals_0 = (result->partition[i] == 0);
            bool equals_1 = (result->partition[i] == 1);
            assert(equals_0 != equals_1);

            count += result->partition[i];
        }

        double split = (double) count / (double) graph->n;
        double target = targetSplit;
        if (split > 0.5)
        {
            split = 1 - split;
        }
        if (targetSplit > 0.5)
        {
            target = 1 - target;
        }

        Logger::printTimingInfo();
        LogTest("Cut Properties:");
        LogTest("  Cut Cost:  " << result->cut_cost);
        LogTest("  Imbalance: " << result->imbalance);
    }

    graph->~Graph();
    result->~EdgeCut();

    LogTest("Edge Separator Test Completed Successfully");

    return EXIT_SUCCESS;
}


#include "Mongoose_Internal.hpp"
#include "Mongoose_EdgeCut.hpp"
#include "Mongoose_IO.hpp"
#include "Mongoose_Logger.hpp"
#include "Mongoose_Version.hpp"

#include <fstream>

using namespace Mongoose;

int main(int argn, const char **argv)
{
    SuiteSparse_start();

    clock_t t;
    
    // Set Logger to report only Error messages
    Logger::setDebugLevel(Error);

    if (argn < 2 || argn > 3)
    {
        // Wrong number of arguments - return error
        LogError("Usage: mongoose <MM-input-file.mtx> [output-file]");
        SuiteSparse_finish();
        return EXIT_FAILURE;
    }

    // Read in input file name
    std::string inputFile = std::string(argv[1]);

    std::string outputFile;
    if (argn == 3)
    {
        outputFile = std::string(argv[2]);
    }
    else
    {
        outputFile = "mongoose_out.txt";
    }

    // Turn timing information on
    Logger::setTimingFlag(true);

    EdgeCut_Options *options = EdgeCut_Options::create();
    if (!options)
    {
        // Ran out of memory
        LogError("Error creating Options struct");
        return EXIT_FAILURE;
    }

    Graph *graph = read_graph(inputFile);

    if (!graph)
    {
        // Ran out of memory or problem reading the graph from file
        LogError("Error reading Graph from file");

        options->~EdgeCut_Options();

        return EXIT_FAILURE;
    }

    // Print version and license information
    std::cout << "********************************************************************************" << std::endl;
    std::cout << "Mongoose Graph Partitioning Library, Version " << mongoose_version() << std::endl;
    std::cout << "Copyright (C) 2017-2018" << std::endl;
    std::cout << "Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager" << std::endl;
    std::cout << "Mongoose is licensed under Version 3 of the GNU General Public License." << std::endl;
    std::cout << "Mongoose is also available under other licenses; contact authors for details." << std::endl;
    std::cout << "********************************************************************************" << std::endl;

    // An edge separator should be computed with default options
    t = clock();
    EdgeCut *result = edge_cut(graph, options);
    t = clock() - t;

    if (!result)
    {
        // Error occurred
        LogError("Error computing edge separator");
        options->~EdgeCut_Options();
        graph->~Graph();
        result->~EdgeCut();
        return EXIT_FAILURE;
    }
    else
    {
        double test_time = ((double) t)/CLOCKS_PER_SEC;
        std::cout << "Total Edge Separator Time: " << test_time << "s\n";
        Logger::printTimingInfo();
        std::cout << "Cut Properties:\n";
        std::cout << " Cut Size:       " << result->cut_size << "\n";
        std::cout << " Cut Cost:       " << result->cut_cost << "\n";
        std::cout << " Imbalance:      " << result->imbalance << "\n";

        // Write results to file
        if (!outputFile.empty())
        {
            LogTest("Writing results to file: " << outputFile);
            std::ofstream ofs (outputFile.c_str(), std::ofstream::out);
            ofs << "{" << std::endl;
            ofs << "  \"InputFile\": \"" << inputFile << "\"," << std::endl;
            ofs << "  \"Timing\": {" << std::endl;
            ofs << "    \"Total\": " << test_time << "," << std::endl;
            ofs << "    \"Matching\": " << Logger::getTime(MatchingTiming) << "," << std::endl;
            ofs << "    \"Coarsening\": " << Logger::getTime(CoarseningTiming) << "," << std::endl;
            ofs << "    \"Refinement\": " << Logger::getTime(RefinementTiming) << "," << std::endl;
            ofs << "    \"FM\": " << Logger::getTime(FMTiming) << "," << std::endl;
            ofs << "    \"QP\": " << Logger::getTime(QPTiming) << "," << std::endl;
            ofs << "    \"IO\": " << Logger::getTime(IOTiming) << std::endl;
            ofs << "  }," << std::endl;
            ofs << "  \"CutSize\": " << result->cut_size << "," << std::endl;
            ofs << "  \"CutCost\": " << result->cut_cost << "," << std::endl;
            ofs << "  \"Imbalance\": " << result->imbalance << std::endl;
            ofs << "}" << std::endl;

            ofs << std::endl;
            for (Int i = 0; i < graph->n; i++)
            {
                ofs << i << " " << result->partition[i] << std::endl;
            }
            ofs << std::endl;

            ofs.close();
        }
    }

    options->~EdgeCut_Options();
    graph->~Graph();
    result->~EdgeCut();

    SuiteSparse_finish();

    return 0 ;
}

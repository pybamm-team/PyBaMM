
#include "Mongoose_EdgeCut.hpp"
#include "Mongoose_IO.hpp"
#include <iostream>
#include "Mongoose_Test.hpp"

using namespace Mongoose;

int RunAllTests(const std::string &inputFile, EdgeCut_Options*);

int RunTest(const std::string &inputFile, const EdgeCut_Options*, int allowedMallocs);

/* Custom memory management functions allow for memory testing. */
int AllowedMallocs;

void *myMalloc(size_t size)
{
    if(AllowedMallocs <= 0) return NULL;
    AllowedMallocs--;
    return malloc(size);
}

void *myCalloc(size_t count, size_t size)
{
    if(AllowedMallocs <= 0) return NULL;
    AllowedMallocs--;
    return calloc(count, size);
}

void *myRealloc(void *ptr, size_t newSize)
{
    if(AllowedMallocs <= 0) return NULL;
    AllowedMallocs--;
    return realloc(ptr, newSize);
}

void myFree(void *ptr)
{
    if(ptr != NULL) free(ptr);
}

int runMemoryTest(const std::string &inputFile)
{
    EdgeCut_Options *options = EdgeCut_Options::create();

    if(!options)
    {
        LogTest("Error creating Options struct in Memory Test");
        return EXIT_FAILURE; // Return early if we failed.
    }

    /* Override SuiteSparse memory management with custom testers. */
    SuiteSparse_config.malloc_func = myMalloc;
    SuiteSparse_config.calloc_func = myCalloc;
    SuiteSparse_config.realloc_func = myRealloc;
    SuiteSparse_config.free_func = myFree;

    int status = RunAllTests(inputFile, options);

    options->~EdgeCut_Options();

    return status;
}

int RunAllTests (const std::string &inputFile, EdgeCut_Options *options)
{
    LogTest("Running Memory Test on " << inputFile);

    int m = 0;
    int remainingMallocs;

    MatchingStrategy matchingStrategies[4] = {Random, HEM, HEMSR, HEMSRdeg};
    InitialEdgeCutType guessCutStrategies[3] = {InitialEdgeCut_QP, InitialEdgeCut_Random, InitialEdgeCut_NaturalOrder};
    Int coarsenLimit[3] = {64, 256, 1024};

    for(int c = 0; c < 2; c++)
    {
        options->do_community_matching = static_cast<bool>(c);

        for(int i = 0; i < 4; i++)
        {
            options->matching_strategy = matchingStrategies[i];

            for(int j = 0; j < 3; j++)
            {
                options->initial_cut_type = guessCutStrategies[j];
                for(int k = 0; k < 3; k++)
                {
                    options->coarsen_limit = coarsenLimit[k];
                    m = 0;
                    do {
                        remainingMallocs = RunTest(inputFile, options, m);
                        if (remainingMallocs == -1)
                        {
                            // Error!
                            LogTest("Terminating Memory Test Early");
                            return EXIT_FAILURE;
                        }
                        m += 1;
                    } while (remainingMallocs < 1);
                }
            }
        }
    }

    // Run once with no options struct
    m = 0;
    do {
        remainingMallocs = RunTest(inputFile, NULL, m);
        if (remainingMallocs == -1)
        {
            // Error!
            LogTest("Terminating Memory Test Early");
            return EXIT_FAILURE;
        }
        m += 1;
    } while (remainingMallocs < 1);

    LogTest("Memory Test Completed Successfully");
    return EXIT_SUCCESS;
}

int RunTest (const std::string &inputFile, const EdgeCut_Options *O, int allowedMallocs)
{
    /* Set the # of mallocs that we're allowed. */
    AllowedMallocs = allowedMallocs;

    /* Read and condition the matrix from the MM file. */
    Graph *U = read_graph(inputFile);
    if (!U) return AllowedMallocs;

    EdgeCut *result;

    if (O)
    {
        result = edge_cut(U, O);
    }
    else
    {
        result = edge_cut(U);
    }

    U->~Graph();

    if (result != NULL)
        result->~EdgeCut();

    return AllowedMallocs;
}

#define LOG_ERROR 1
#define LOG_WARN 1
#define LOG_INFO 0
#define LOG_TEST 1

#include "Mongoose_Test.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_IO.hpp"
#include "Mongoose_EdgeCutProblem.hpp"

using namespace Mongoose;

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

int main(int argn, char** argv)
{
    (void)argn; // Unused variable
    (void)argv; // Unused variable

    SuiteSparse_start();

    // Set Logger to report all messages and turn off timing info
    Logger::setDebugLevel(All);
    Logger::setTimingFlag(false);

    // Test Graph(n, nz) static constructor
    Graph *G2 = Graph::create(10, 20);
    EdgeCutProblem *prob = EdgeCutProblem::create(G2);

    prob->clearMarkArray(LONG_MAX);
    Int markValue = prob->getMarkValue();
    assert(markValue == 1);

    prob->clearMarkArray(LONG_MAX-1);
    prob->clearMarkArray();
    markValue = prob->getMarkValue();
    assert(markValue >= 1);
    prob->~EdgeCutProblem();

    MM_typecode matcode;
    cs *M4 = read_matrix("../Matrix/bcspwr01.mtx", matcode);
    M4->x = NULL;
    Graph *G7 = Graph::create(M4);
    assert(G7 != NULL);

    // Tests to increase coverage
    /* Override SuiteSparse memory management with custom testers. */
    SuiteSparse_config.malloc_func = myMalloc;
    SuiteSparse_config.calloc_func = myCalloc;
    SuiteSparse_config.realloc_func = myRealloc;
    SuiteSparse_config.free_func = myFree;

    // Simulate failure to allocate return arrays
    AllowedMallocs = 0;
    EdgeCutProblem *G3 = EdgeCutProblem::create(G7);
    assert(G3 == NULL);

    AllowedMallocs = 1;
    EdgeCutProblem *G4 = EdgeCutProblem::create(G7);
    assert(G4 == NULL);

    AllowedMallocs = 5;
    EdgeCutProblem *G5 = EdgeCutProblem::create(G7);
    assert(G5 == NULL);

    AllowedMallocs = 10;
    EdgeCutProblem *G6 = EdgeCutProblem::create(G7);
    assert(G6 == NULL);

    G7->~Graph();

    SuiteSparse_finish();

    return 0;
}

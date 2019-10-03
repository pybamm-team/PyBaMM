#include "Mongoose_Test.hpp"
#include "Mongoose_IO.hpp"
#include "Mongoose_Sanitize.hpp"

using namespace Mongoose;

#undef LOG_ERROR
#undef LOG_WARN
#undef LOG_INFO
#undef LOG_TEST
#define LOG_ERROR 1
#define LOG_WARN 1
#define LOG_INFO 0
#define LOG_TEST 1

int main(int argn, char** argv)
{
    (void)argn; // Unused variable
    (void)argv; // Unused variable

    SuiteSparse_start();

    // Set Logger to report all messages and turn off timing info
    Logger::setDebugLevel(All);
    Logger::setTimingFlag(false);

    Graph *G;

    // Nonexistent file
    G = read_graph("../Tests/Matrix/no_such_file.mtx");
    assert (G == NULL);

    // Bad header 
    G = read_graph("../Tests/Matrix/bad_header.mtx");
    assert (G == NULL);

    // Bad matrix type
    G = read_graph("../Tests/Matrix/bad_matrix_type.mtx");
    assert (G == NULL);

    // Bad dimensions
    G = read_graph("../Tests/Matrix/bad_dimensions.mtx");
    assert (G == NULL);
      
    // Rectangular matrix     
    G = read_graph("../Tests/Matrix/Trec4.mtx");
    assert (G == NULL);

    // C-style string filename
    MM_typecode matcode;
    std::string filename = "../Matrix/bcspwr01.mtx";
    cs *M = read_matrix(filename, matcode);
    assert(M != NULL);

    cs *binaryM = sanitizeMatrix(M, true, true);
    for (Int j = 0; j < M->nz; j++)
    {
        assert(binaryM->x[j] == 0 || binaryM->x[j] == 1);
    }
    SuiteSparse_free(M);

    SuiteSparse_finish();

    return 0;
}
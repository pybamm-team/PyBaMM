// =============================================================================
// === GPUQREngine/Demo/dense_demo.cpp =========================================
// =============================================================================

// GPUQREngine can be used to factorize a set of dense matrices of
// various sizes.  This is the demo for this 'dense' usage case.
// The 'sparse' case is exercised by SuiteSparseQR.

#ifndef GPU_BLAS
#define GPU_BLAS
#endif

#include "GPUQREngine.hpp"
#include "cholmod.h"
#include <time.h>
#include <stdio.h>
#include "GPUQREngine_Timing.hpp"

//------------------------------------------------------------------------------

void randfill(double *x, Int m, Int n)
{
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            x[i*n+j] = rand() / 1e8;
        }
    }
}

//------------------------------------------------------------------------------

void randfill_stair(double *x, Int m, Int n, Int aggression)
{
    int i = 0;
    for(int j=0; j<n; j++)
    {
        int iend = MIN(i + (rand() % aggression) + 1, m);
        for(i=0; i<m; i++)
        {
            x[i*n+j] = (i < iend ? rand() / 1e8 : 0.0);
        }
        i = iend;
    }
}

//------------------------------------------------------------------------------

void randfill_blocktriu(double *x, Int m, Int n)
{
    int ybmax = CEIL(m, TILESIZE);
    int xbmax = CEIL(n, TILESIZE);

    for(int yb=0; yb<ybmax; yb++)
    {
        for(int xb=0; xb<xbmax; xb++)
        {
            if(yb > xb) continue;

            int imax = MIN(m, TILESIZE*(yb+1));
            for(int i=TILESIZE*yb; i<imax; i++)
            {
                int jmax = MIN(n, TILESIZE*(xb+1));
                for(int j=TILESIZE*xb; j<jmax; j++)
                {
                    x[i*n+j] = rand() / 1e8;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------

void printMatrix(FILE *troll, const char *name,
    int which, int f, double *inM, int m, int n)
{

    double (*M)[n] = (double (*)[n]) inM;

    fprintf(troll, "%s%d {%d} = [\n", name, which, 1+f);
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            fprintf(troll, "%30.16e ", M[i][j]);
        }
        fprintf(troll, "\n");
    }
    fprintf(troll, "];\n");
}

//------------------------------------------------------------------------------

void printQRScript (FILE *troll)
{

    fprintf (troll, "format compact;\n");
    fprintf (troll, "res = norm(triu(R)'*triu(R)-A'*A) / norm(A'*A)\n");
    fprintf (troll, "if isnan(res), error ('failure'); end\n");
    fprintf (troll, "if res > 1e-12, error ('failure'); end\n");

//  fprintf (troll, "[%d %d] = size (A) ;\n") ;
//  int rank = MIN(m, n);
//  fprintf (troll, "[Davis_R V1 T] = qrsigma_concise(A);\n");
//  fprintf (troll, "norm(triu(Davis_R) - triu(R))\n");
//  fprintf (troll, "norm(Davis_R(%d:end,:)-R(%d:end,:))\n", rank+1, rank+1);

    fprintf (troll, "MR = qr(A);\n");
    fprintf (troll, "tR = triu(abs(R));");
    fprintf (troll, "tMR = triu(abs(MR));");
    fprintf (troll, "norm(tMR - tR)\n");
}

//------------------------------------------------------------------------------

void driver
(
    cholmod_common *cc,
    const char *filename,
    Front *fronts,
    int numFronts,
    QREngineStats *stats,
    int which
)
{
    FILE *troll = NULL ;

    if (filename != NULL)
    {
        troll = fopen(filename, "a");
    }
    if (troll != NULL)
    {
        fprintf (troll, "clear ;\n") ;
        fprintf (troll, "A_%d = cell (1,%d) ;\n", which, numFronts) ;
        fprintf (troll, "R_%d = cell (1,%d) ;\n", which, numFronts) ;
    }

    for(int f=0; f<numFronts; f++)
    {
        Front *front = (&fronts[f]);
        int m = front->fm;
        int n = front->fn;

        /* Attach the front data. */
        double *F = front->cpuR = front->F = (double*)
            SuiteSparse_calloc(m*n, sizeof(double));
        randfill(F, m, n);

        /* Print the A matrix. */
        if (troll != NULL)
        {
            printMatrix (troll, "A_", which, f, F, m, n);
        }
    }

    /* Run the QREngine code */
    QREngineResultCode result = GPUQREngine (cc->gpuMemorySize,
        fronts, numFronts, stats);
    if (result != QRENGINE_SUCCESS)
    {
        printf ("test failure!\n'") ;
        exit (0) ;
    }

    /* Do something with R factors. */
    for(int f=0; f<numFronts; f++)
    {
        Front *front = (&fronts[f]);
        int m = front->fm;
        int n = front->fn;
        double *R = front->cpuR;

        /* Print the R matrix. */
        if (troll != NULL)
        {
            printMatrix(troll, "R_", which, f, R, m, n);
            fprintf (troll, "R = R_%d {%d} ; A = A_%d {%d} ;\n", 
                which, 1+f, which, 1+f) ;
            printQRScript (troll) ;
        }

        /* Detach the front data. */
        SuiteSparse_free(front->F);
        front->F = NULL;
    }

    if (troll != NULL)
    {
        fprintf (troll, "disp ('all tests passed') ;\n") ;
        fclose (troll) ;
    }
}

//------------------------------------------------------------------------------

void printStats(QREngineStats stats, int numFronts, double m, double n)
{
    float kernelTime = stats.kernelTime;
    Int numLaunches = stats.numLaunches;
    Int gpuFlops = stats.flopsActual;

    /* Compute & Print FLOPS */
    double time = (double) kernelTime;
    double flops;
    if(m >= n)
    {
        flops = 2.0 * n*n * (m - (n/3));
    }
    else
    {
        flops = 2.0 * m*m * (n - (m/3));
    }

    flops *= (double) numFronts;
    flops /= (time / 1e3);
    double gflops = flops / 1e9;
    double gpugflops = (gpuFlops / (time / 1e3)) / 1e9;
    printf("m: %.0f, n: %.0f, nf: %d, nl: %ld, gpuFlops: %ld, t: %fms, gflops: %f, gpugflops: %f\n", m, n, numFronts, numLaunches, gpuFlops, time, gflops, gpugflops);
}

//------------------------------------------------------------------------------

void experiment1(cholmod_common *cc, int which, int numFronts, int m, int n)
{
    /* Configure problem set. */
    Front *fronts = (Front*) SuiteSparse_calloc(numFronts, sizeof(Front));
    for(int f=0; f<numFronts; f++)
    {
        new (&fronts[f]) Front(f, EMPTY, m, n);
    }

    /* Run the driver code. */
    QREngineStats stats;
    driver(cc, "troll.m", fronts, numFronts, &stats, which);
//  driver(cc, NULL,      fronts, numFronts, &stats, which);
    printStats(stats, numFronts, m, n);

    /* Cleanup resources. */
    for(int f=0; f<numFronts; f++) (&fronts[f])->~Front();
    SuiteSparse_free(fronts);
}

//------------------------------------------------------------------------------

void experiment2(cholmod_common *cc, int m, int n, int numFronts)
{
    /* See if this experiment would blow out the memory. */
    size_t threshold = 3.50 * 1024 * 1024 * 1024;
    size_t memoryReq = (size_t) (numFronts * (CEIL(m, 32) * 32 * 33 + m * n)) ;
    if(memoryReq * sizeof(double) > threshold) return;

    /* Configure problem set. */
    Front *fronts = (Front*) SuiteSparse_calloc(numFronts, sizeof(Front));
    for(int f=0; f<numFronts; f++)
    {
        new (&fronts[f]) Front(f, EMPTY, m, n);
    }

    /* Run the driver code if we won't run out of memory. */
    QREngineStats stats;
    driver(cc, NULL,      fronts, numFronts, &stats, 0);   // no troll.m output
    printStats(stats, numFronts, m, n);

    /* Cleanup resources. */
    for(int f=0; f<numFronts; f++) (&fronts[f])->~Front();
    SuiteSparse_free(fronts);
}


//------------------------------------------------------------------------------

int main(int argn, char **argv)
{
    double t ;
    size_t total_mem, available_mem ;

    /* Clear the troll file. */
    FILE *troll;
    troll = fopen("troll.m", "w");
    fclose(troll);

    srand(1);

    // start CHOLMOD
    cholmod_common *cc, Common ;
    cc = &Common ;
    cholmod_l_start (cc) ;

    // warmup the GPU.  This can take some time, but only needs
    // to be done once
    cc->useGPU = true ;
    t = SuiteSparse_time ( ) ;
    cholmod_l_gpu_memorysize (&total_mem, &available_mem, cc) ;
    cc->gpuMemorySize = available_mem ;
    t = SuiteSparse_time ( ) - t ;
    if (cc->gpuMemorySize <= 1)
    {
        printf ("no GPU available\n") ;
        return (0) ;
    }
    printf ("available GPU memory: %g MB, warmup time: %g\n",
        (double) (cc->gpuMemorySize) / (1024 * 1024), t) ;

    experiment1(cc, 1, 2, 8, 8);
    experiment1(cc, 2, 2, 12, 8);
    experiment1(cc, 3, 2, 64, 32);
    experiment1(cc, 4, 1, 100, 200);

    printf ("to check results, run 'troll.m' in MATLAB\n") ;

#if 0
    for(int numFronts=1; numFronts<=128; numFronts*=2)
    {
        for(int dim=128; dim<=6144; dim+=128)
        {
            experiment2(cc, dim, dim, numFronts);
        }
    }
#endif
#if 0
    for(int numFronts=1; numFronts<=128; numFronts*=2)
    {
        for(int smdim=128; smdim<=6144/4; smdim+=128)
        {
            experiment2(cc, smdim, 4*smdim, numFronts);
            experiment2(cc, 4*smdim, smdim, numFronts);
        }
        for(int smdim=128; smdim<=6144/16; smdim+=128)
        {
            experiment2(cc, smdim, 16*smdim, numFronts);
            experiment2(cc, 16*smdim, smdim, numFronts);
        }
    }
#endif
}


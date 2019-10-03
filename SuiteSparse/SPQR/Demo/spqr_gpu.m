function info = spqr_gpu (ordering, A)
% info = spqr_gpu (ordering)
%   ordering: 1 colamd
%   ordering: 2 metis

if (nargin > 1)
    % write the matrix to a file
    mwrite ('A.mtx', A) ;
end

if (exist ('gpu_results.txt', 'file'))
    delete ('gpu_results.txt') ;
end

setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64:/usr/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib64')
if (ordering == 1)
    system ('tcsh demo_colamd.sh') ;
else
    system ('tcsh demo_metis.sh') ;
end

load ('gpu_results.txt') ;
% delete ('gpu_results.txt') ;

[m n] = size (A) ;
B = ones (m,1) ;
X = mread ('X.mtx') ;

r = A*X-B ;
atr = A'*r ;
atanorm = norm (A'*A,1) ;
atrnorm = norm (atr) / atanorm ;
fprintf ('relative norm (A''*(A*x-b)): %g\n', atrnorm) ;

% qrdemo_gpu.cpp writes the following:
%
%   fprintf(info, "%ld\n", cc->SPQR_istat[7]);        // ordering method
%   fprintf(info, "%ld\n", cc->memory_usage);         // memory usage (bytes)
%   fprintf(info, "%30.16e\n", cc->SPQR_flopcount);   // flop count
%   fprintf(info, "%lf\n", cc->SPQR_analyze_time);    // analyze time
%   fprintf(info, "%lf\n", cc->SPQR_factorize_time);  // factorize time
%   fprintf(info, "-1\n") ;                           // cpu memory (bytes)
%   fprintf(info, "-1\n") ;                           // gpu memory (bytes)
%   fprintf(info, "%8.1e\n", rnorm);                  // residual
%   fprintf(info, "%ld\n", cholmod_l_nnz (A, cc));    // nnz(A)
%   fprintf(info, "%ld\n", cc->SPQR_istat [0]);       // nnz(R)
%   fprintf(info, "%ld\n", cc->SPQR_istat [2]);       // # of frontal matrices
%   fprintf(info, "%ld\n", cc->SPQR_istat [3]);       // ntasks, for now
%   fprintf(info, "%lf\n", cc->gpuKernelTime);        // kernel time (ms)
%   fprintf(info, "%ld\n", cc->gpuFlops);             // "actual" gpu flops
%   fprintf(info, "%d\n", cc->gpuNumKernelLaunches);  // # of kernel launches
%   fprintf(info, "%8.1e\n", atrnorm);                // norm (A'*(Ax-b))

info.ordering              = gpu_results (1) ;
info.memory_usage_in_bytes = gpu_results (2) ;
info.flops                 = gpu_results (3) ;
info.analyze_time          = gpu_results (4) ;
info.factorize_time        = gpu_results (5) ;
info.cpuWatermark          = gpu_results (6) ;
info.gpuWatermark          = gpu_results (7) ;
info.resid                 = gpu_results (8) ;
info.nnzA                  = gpu_results (9) ;
info.nnzR                  = gpu_results (10) ;
info.numFronts             = gpu_results (11) ;
info.numTasks              = gpu_results (12) ;
info.kerneltime            = gpu_results (13) ;
info.gpuFlops              = gpu_results (14) ;
info.kernellaunches        = gpu_results (15) ;
info.atrnorm               = gpu_results (16) / atanorm ;

% /* ordering options */
%   #define SPQR_ORDERING_FIXED 0
%   #define SPQR_ORDERING_NATURAL 1
%   #define SPQR_ORDERING_COLAMD 2
%   #define SPQR_ORDERING_GIVEN 3       /* only used for C/C++ interface */
%   #define SPQR_ORDERING_CHOLMOD 4     /* CHOLMOD best-effort (COLAMD, METIS,...)*/
%   #define SPQR_ORDERING_AMD 5         /* AMD(A'*A) */
%   #define SPQR_ORDERING_METIS 6       /* metis(A'*A) */
%   #define SPQR_ORDERING_DEFAULT 7     /* SuiteSparseQR default ordering */
%   #define SPQR_ORDERING_BEST 8        /* try COLAMD, AMD, and METIS; pick best */
%   #define SPQR_ORDERING_BESTAMD 9     /* try COLAMD and AMD; pick best */

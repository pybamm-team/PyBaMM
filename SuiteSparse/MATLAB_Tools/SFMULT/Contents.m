% SSMULT:  sparse matrix multiplication (sparse times sparse)
%
% SSMULT computes C=A*B where A and B are sparse.  It is typically faster
% than C=A*B in MATLAB 7.4 (or earlier), and always uses less memory.
%
%   ssmult          - multiplies two sparse matrices.
%   ssmult_install  - compiles, installs, and tests ssmult.
%   ssmult_unsorted - multiplies two sparse matrices, returning non-standard result.
%   ssmultsym       - computes nnz(C), memory, and flops to compute C=A*B; A and B sparse.
%   sstest          - exhaustive performance test for SSMULT.
%   sstest2         - exhaustive performance test for SSMULT.  Requires ssget.
%
% Example:
%   C = ssmult(A,B) ;    % computes C = A*B
%
% Copyright 2007, Timothy A. Davis, University of Florida

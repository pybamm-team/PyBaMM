% SPARSEINV  The sparseinv function computes the sparse inverse subset of a
% sparse matrix A.  These entries in the inverse subset correspond to nonzero
% entries in the factorization of A.  They can be computed without computing
% all of the entries in inv(A), so this method is much faster and takes much
% less memory than inv(A).  If A is symmetric and positive definite, then all
% entries of the diagona of inv(A) are computed (as well as many off-diagonal
% terms.  This version is restricted to real sparse matrices.  A complex
% version is left for future work.
%
% Copyright (c) 2011, Timothy A. Davis
%
% Files
%   sparseinv         - computes the sparse inverse subset of a real sparse square matrix A.
%   sparseinv_install - compiles and installs the sparseinv function.
%   sparseinv_test    - tests the sparseinv function.

% SuiteSparseQR : a multifrontal, multithreaded, rank-revealing sparse QR
% factorization method.  Works for both real and complex sparse matrices.
%
% Files
%   spqr         - multithreaded multifrontal rank-revealing sparse QR.
%   spqr_demo    - short demo of SuiteSparseQR 
%   spqr_install - compile and install SuiteSparseQR
%   spqr_make    - compiles the SuiteSparseQR mexFunctions
%   spqr_qmult   - computes Q'*X, Q*X, X*Q', or X*Q with Q in Householder form.
%   spqr_solve   - solves a linear system or least squares problem via QR factorization.
%   spqr_singletons - finds the singleton permutation of a sparse matrix A.
%
% Example:
%   x = spqr_solve (A,b) ;  % solves a least-squares problem (like x=A\b)

% Copyright 2008-2012, Timothy A. Davis, http://www.suitesparse.com

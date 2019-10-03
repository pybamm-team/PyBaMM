function [p q m1 n1 tol] = spqr_singletons (A,tol)                          %#ok
%SPQR_SINGLETONS finds the singleton permutation of a sparse matrix A.
% [p q m1 n1 tol] = spqr_singletons (A, tol) finds permutation vectors p and q
% so that C=A(p,q) is a block 2-by-2 matrix in the form [C11 C12 ; 0 C22] where
% C11 is upper triangular (or "squeezed" upper trapezoidal), of dimension
% m1-by-n1.  The columns of C11 are the column singletons of A.  If C11 is
% square then all of its diagonal entries are larger in magnitude than tol.
%
% The input tol is optional; it defaults to 20*(m+n)*eps*maxcol2norm where
% [m n] = size(A) and maxcol2norm is the maximum 2-norm of the columns of A.
% The output tol is the tolerance used.
%
% If C11 is rectangular, then some of the column singletons have no
% corresponding row singleton.  A column j in C11 has a corresponding row i if
% if i = max(find(C(:,j))) > max(find(C(:,j-1))).  If present, abs(C(i,j)) will
% be larger than tol.
%
% Example:
%
%   A = [1 8 0 0
%        2 9 3 7
%        0 4 0 0
%        0 6 0 0 ]
%   [p q m1 n1 tol] = spqr_singletons (sparse (A))
%   C = A(p,q)
%
% In this example, C11 is 2-by-3.  One of the 3 column singletons (the 2nd on
% in C) has no corresponding row singleton.  This is an auxiliary routine that
% illustrates the singleton removal step used by SuiteSparseQR.  It is not
% needed to solve a least-squares problem or find a sparse QR factorization;
% use SPQR_SOLVE or SPQR for those tasks, respectively.
%
% See also DMPERM, SPQR, SPQR_SOLVE.

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

error ('spqr_singletons mexFunction not found') ;

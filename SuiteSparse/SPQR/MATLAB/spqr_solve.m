function [X,info] = spqr_solve (A,B,opts)                                   %#ok
%SPQR_SOLVE solves a linear system or least squares problem via QR factorization.
% A is a sparse m-by-n matrix A and B is a sparse or full m-by-k matrix.
% If m == n, x = spqr_solve(A,B) solves A*X=B.  If m < n, a basic solution is
% found for the underdetermined system A*X=B.  If m > n, the least squares
% solution is found.  An optional third input argument specifies non-default
% options (see "help spqr" for details).  Only opts.tol, opts.ordering, and
% opts.solution are used; the others are implicitly set to opts.econ = 0,
% opts.Q = 'Householder', and opts.permutation = 'vector'.
%
% opts.solution:  'basic' (default) or 'min2norm'.  To obtain a minimum 2-norm
% solution to an undetermined system (m < n), use 'min2norm'.  For m >= n,
% these two options find the same solution.
%
% An optional second output provides statistics on the solution.
%
% Example: 
%   x = spqr_solve (A,B) 
%   [x,info] = spqr_solve (A,B) 
%   x = spqr_solve (A,B,opts) 
%   x = spqr_solve (A,B, struct ('solution','min2norm')) ;
%
% See also SPQR, SPQR_QMULT, QR, MLDIVIDE.

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

type spqr_solve
error ('spqr_solve mexFunction not found') ;

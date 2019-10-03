function [x,stats] = cholmod2 (A, b, ordering)				    %#ok
%CHOLMOD2 supernodal sparse Cholesky backslash, x = A\b
%
%   Example:
%   x = cholmod2 (A,b)
%
%   Computes the LL' factorization of A(p,p), where p is a fill-reducing
%   ordering, then solves a sparse linear system Ax=b. A must be sparse,
%   symmetric, and positive definite).  Uses only the upper triangular part
%   of A.  A second output, [x,stats]=cholmod2(A,b), returns statistics:
%
%       stats(1)    estimate of the reciprocal of the condition number
%       stats(2)    ordering used:
%                   0: natural, 1: given, 2:amd, 3:metis, 4:nesdis,
%                   5:colamd, 6: natural but postordered.
%       stats(3)    nnz(L)
%       stats(4)    flop count in Cholesky factorization.  Excludes solution
%                   of upper/lower triangular systems, which can be easily
%                   computed from stats(3) (roughly 4*nnz(L)*size(b,2)).
%       stats(5)    memory usage in MB.
%
%   The 3rd argument select the ordering method to use.  If not present or -1,
%   the default ordering strategy is used (AMD, and then try METIS if AMD finds
%   an ordering with high fill-in, and use the best method tried).
%
%   Other options for the ordering parameter:
%
%       0   natural (no etree postordering)
%       -1  use CHOLMOD's default ordering strategy (AMD, then try METIS)
%       -2  AMD, and then try NESDIS (not METIS) if AMD has high fill-in
%       -3  use AMD only
%       -4  use METIS only
%       -5  use NESDIS only
%       -6  natural, but with etree postordering
%       p   user permutation (vector of size n, with a permutation of 1:n)
%
%   See also CHOL, MLDIVIDE.

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('cholmod2 mexFunction not found\n') ;

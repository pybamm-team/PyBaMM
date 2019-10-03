function [count, h, parent, post, L] = symbfact2 (A, mode, Lmode)	    %#ok
%SYMBFACT2  symbolic factorization
%
%   Analyzes the Cholesky factorization of A, A'*A, or A*A'.
%
%   Example:
%   count = symbfact2 (A)               returns row counts of R=chol(A)
%   count = symbfact2 (A,'col')         returns row counts of R=chol(A'*A)
%   count = symbfact2 (A,'sym')         same as symbfact2(A)
%   count = symbfact2 (A,'lo')          same as symbfact2(A'), uses tril(A)
%   count = symbfact2 (A,'row')         returns row counts of R=chol(A*A')
%
%   The flop count for a subsequent LL' factorization is sum(count.^2)
%
%   [count, h, parent, post, R] = symbfact2 (...) returns:
%
%       h: height of the elimination tree
%       parent: the elimination tree itself
%       post: postordering of the elimination tree
%       R: a 0-1 matrix whose structure is that of chol(A) for the symmetric
%           case, chol(A'*A) for the 'col' case, or chol(A*A') for the
%           'row' case.
%
%   symbfact2(A) and symbfact2(A,'sym') uses the upper triangular part of A
%   (triu(A)) and assumes the lower triangular part is the transpose of
%   the upper triangular part.  symbfact2(A,'lo') uses tril(A) instead.
%
%   With one to four output arguments, symbfact2 takes time almost proportional
%   to nnz(A)+n where n is the dimension of R, and memory proportional to
%   nnz(A).  Computing the 5th argument takes more time and memory, both
%   O(nnz(L)).  Internally, the pattern of L is computed and R=L' is returned.
%
%   The following forms return L = R' instead of R.  They are faster and take
%   less memory than the forms above.  They return the same count, h, parent,
%   and post outputs.
%
%   [count, h, parent, post, L] = symbfact2 (A,'col','L')
%   [count, h, parent, post, L] = symbfact2 (A,'sym','L')
%   [count, h, parent, post, L] = symbfact2 (A,'lo', 'L')
%   [count, h, parent, post, L] = symbfact2 (A,'row','L')
%
%   See also CHOL, ETREE, TREELAYOUT, SYMBFACT

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('symbfact2 mexFunction not found!') ;


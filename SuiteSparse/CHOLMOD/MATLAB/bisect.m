function p = bisect (A, mode)						    %#ok
%BISECT computes a node separator based on METIS_ComputeVertexSeparator.
%
%   Example:
%   s = bisect(A)       bisects A. Uses tril(A) and assumes A is symmetric.
%   s = bisect(A,'sym') the same as p=bisect(A).
%   s = bisect(A,'col') bisects A'*A.
%   s = bisect(A,'row') bisects A*A'.
%
%   A must be square for p=bisect(A) and bisect(A,'sym').
%
%   s is a vector of length equal to the dimension of A, A'*A, or A*A',
%   depending on the matrix bisected.  s(i)=0 if node i is in the left subgraph,
%   s(i)=1 if it is in the right subgraph, and s(i)=2 if node i is in the node
%   separator.
%
%   Requires METIS, authored by George Karypis, Univ. of Minnesota.  This
%   MATLAB interface, via CHOLMOD, is by Tim Davis.
%
%   See also METIS, NESDIS

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('bisect mexFunction not found') ;

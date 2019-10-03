function S = sparse2 (i,j,s,m,n,nzmax)					    %#ok
%SPARSE2 replacement for SPARSE
%
%   Example:
%   S = sparse2 (i,j,s,m,n,nzmax)
%   
%   Identical to the MATLAB sparse function (just faster).
%   An additional feature is added that is not part of the MATLAB sparse
%   function, the Z matrix.  With an extra output,
%
%   [S Z] = sparse2 (i,j,s,m,n,nzmax)
%
%   the matrix Z is a binary real matrix whose nonzero pattern contains the
%   explicit zero entries that were dropped from S.  Z only contains entries
%   for the sparse2(i,j,s,...) usage.  [S Z]=sparse2(X) where X is full always
%   returns Z with nnz(Z) = 0, as does [S Z]=sparse2(m,n).  More precisely,
%   Z is the following matrix (where ... means the optional m, n, and nzmax
%   parameters).
%
%       S = sparse (i,j,s, ...)
%       Z = spones (sparse (i,j,1, ...)) - spones (S)
%
%   See also sparse.

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('sparse2 mexFunction not found') ;

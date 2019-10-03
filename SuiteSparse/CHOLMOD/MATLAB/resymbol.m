function L = resymbol (L, A)						    %#ok
%RESYMBOL recomputes the symbolic Cholesky factorization of the matrix A.
%
%   Example:
%   L = resymbol (L, A)
%
%   Recompute the symbolic Cholesky factorization of the matrix A.  A must be
%   symmetric.  Only tril(A) is used.  Entries in L that are not in the Cholesky
%   factorization of A are removed from L.  L can be from an LL' or LDL'
%   factorization (lchol or ldlchol).  resymbol is useful after a series of
%   downdates via ldlupdate or ldlrowmod, since downdates do not remove any
%   entries in L.  The numerical values of A are ignored; only its nonzero
%   pattern is used.
%
% See also LCHOL, LDLUPDATE, LDLROWMOD

%   Copyright 2006-2015, Timothy A. Davis, http://www.suitesparse.com

error ('resymbol not found') ;

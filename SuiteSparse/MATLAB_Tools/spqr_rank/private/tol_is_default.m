function s = tol_is_default (tol)
%TOL_IS_DEFAULT return true if tol is default, false otherwise
% usage: s = tol_is_default (tol)

% Copyright 2012, Leslie Foster and Timothy A Davis.

s = (isempty (tol) || ischar (tol) || (isreal (tol) && tol < 0)) ;


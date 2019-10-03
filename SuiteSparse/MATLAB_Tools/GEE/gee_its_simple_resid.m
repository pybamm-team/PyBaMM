function r = gee_its_simple_resid (A, x, b)
%GEE_ITS_SIMPLE_RESID compute the relative residual for x=A\b
% For non-singular matrices with reasonable condition number, the relative
% residual is typically O(eps).
%
% Example:
%
%   r = gee_its_simple_resid (A, x, b) ;
%
% See also: norm, gee_its_simple

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

r = norm (A*x-b,inf) / (norm (A,inf) * norm (x, inf) + norm (b, inf)) ;


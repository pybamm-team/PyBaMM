function [x, rcnd] = gee_its_simple (A, b)
%GEE_ITS_SIMPLE solves A*x=b using a Gaussian Elimination Example (it's simple!)
% For details on the algorithm used (Gaussian elimination with partial
% pivoting), see gee_its_simple_factorize, gee_its_simple_forwardsolve,
% and gee_its_simple_backsolve.
%
% Example:
%
%   x = gee_its_simple (A,b) ;
%   [x,rcnd] = gee_its_simple (A,b) ;
%
%   % which is the same as:
%   x = A\b ;
%
%   % or using LU:
%   [L U p] = lu (A) ;
%   x = U \ (L \ (b (p,:))) ;
%   rcnd = min (abs (diag (U))) / max (abs (diag (U))) ;
%
% See also: lu, mldivide, rcond, gee_its_short

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

% check inputs
if (nargin ~= 2 | nargout > 2)                                              %#ok
    error ('Usage: [x,rcnd] = gee_its_simple (A,b)') ;
end

% ensure A is square, and that A and b have the same number of rows
gee_its_simple_check (A, 'A', b) ;

% LU factorization, using Gaussian Elimination with partial pivoting, same as
% [L,U,p] = lu (A) ; rcnd = rcond (A) ; except return L and U in one matrix LU.
[LU p rcnd] = gee_its_simple_factorize (A) ;

% forward/backsolve, same as x = U \ (L \ b (p,:))
x = gee_its_simple_backsolve (LU, gee_its_simple_forwardsolve (LU, b (p,:))) ;


function b = gee_its_simple_backsolve (U, b)
%GEE_ITS_SIMPLE_BACKSOLVE x=U\b where U is upper triangular
% Perform back substitution to solve x=U\b.  U must be square.  The lower
% triangular part of U is ignored (this allows L and U to be packed into a
% single matrix).
%
% Example:
%
%   x = gee_its_simple_backsolve (U,b) ;
%
%   % which is the same as
%   x = triu (U) \ b ;
%
% See also: mldivide, gee_its_simple, gee_its_short, gee_its_simple_forwardsolve

% Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

if (nargin ~= 2 | nargout > 1)                                              %#ok
    error ('Usage: x = ge_its_simple_backsolve (U,b)') ;
end

% ensure U is square, and that U and b have the same number of rows
gee_its_simple_check (U, 'U', b) ;

%-------------------------------------------------------------------------------
% backsolve solve, overwriting b with the solution U\b
%-------------------------------------------------------------------------------

n = size (U,1) ;
for k = n:-1:1
    b (k,:) = b (k,:) / U (k,k) ;
    b (1:k-1,:) = b (1:k-1,:) - U (1:k-1,k) * b (k,:) ;
end


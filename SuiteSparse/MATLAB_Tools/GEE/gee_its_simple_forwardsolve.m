function b = gee_its_simple_forwardsolve (L, b)
%GEE_ITS_SIMPLE_FORWARDSOLVE computes x=L\b where L is unit lower triangular
% Perform forward substitution to solve x=L\b.  L must be square.  The diagonal
% and upper triangular part of L is ignored (this allows L and U to be packed
% into a single matrix).
%
% Example:
%
%   x = gee_its_simple_forwardsolve (L,b) ;
%
%   % which is the same as:
%   L2 = tril (L,-1) + eye (size (L,1)) ;
%   x = L2\b ;
%
% See also: mldivide, gee_its_simple, gee_its_short, gee_its_simple_backsolve

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

if (nargin ~= 2 | nargout > 1)                                              %#ok
    error ('Usage: x = ge_its_simple_forwardsolve (L,b)') ;
end

% ensure L is square, and that L and b have the same number of rows
gee_its_simple_check (L, 'L', b) ;

%-------------------------------------------------------------------------------
% forward solve, overwriting b with the solution L\b
%-------------------------------------------------------------------------------

n = size (L,1) ;
for k = 1:n
    b (k+1:n,:) = b (k+1:n,:) - L (k+1:n,k) * b (k,:) ;
end


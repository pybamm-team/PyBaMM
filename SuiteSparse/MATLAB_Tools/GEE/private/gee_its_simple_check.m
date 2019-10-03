function gee_its_simple_check (A, name, b)
%GEE_ITS_SIMPLE_CHECK private function to check input arguments
% Ensures the matrix A is square, and that the right-hand-side b has the same
% number of rows as A (if present).  All matrices must be 2D, as well.
%
% Example:
%   gee_its_simple_check (A, 'A', b)
%
% See also: gee_its_simple

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

[m n] = size (A) ;
if (m ~= n)
    error ('%s must be square', name) ;
end

if (ndims (A) ~= 2)
    error ('%s must be a 2D matrix', name) ;
end

if (nargin > 2)
    if (m ~= size (b,1))
        error ('%s and b must have the same number of rows', name) ;
    end
    if (ndims (b) ~= 2)
        error ('b must be a 2D matrix') ;
    end
end


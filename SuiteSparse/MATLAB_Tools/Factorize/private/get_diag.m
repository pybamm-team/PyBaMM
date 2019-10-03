function d = get_diag (A)
%GET_DIAG extracts the diagonal of a matrix or vector.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (isempty (A))
    d = [ ] ;
elseif (isvector (A))
    % diag (A) would return a matrix for this case, which we do not want.
    d = A (1) ;
else
    d = diag (A) ;
end
d = full (d) ;
% assert (isvector (d)) ;

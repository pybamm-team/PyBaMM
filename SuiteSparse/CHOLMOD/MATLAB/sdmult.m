function C = sdmult (S,F,transpose)					    %#ok
%SDMULT sparse matrix times dense matrix
%   Compute C = S*F or S'*F where S is sparse and F is full (C is also sparse).
%   S and F must both be real or both be complex.  This function is
%   substantially faster than the MATLAB expression C=S*F when F has many
%   columns.
%
%   Example:
%       C = sdmult (S,F) ;       C = S*F
%       C = sdmult (S,F,0) ;     C = S*F
%       C = sdmult (S,F,1) ;     C = S'*F
%
%   See also MTIMES

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('sdmult mexFunction not found') ;



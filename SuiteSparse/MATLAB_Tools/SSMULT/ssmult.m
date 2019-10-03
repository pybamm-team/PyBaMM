function C = ssmult (A,B, at,ac, bt,bc, ct,cc)                              %#ok
%SSMULT multiplies two sparse matrices.
% C = ssmult (A,B) computes C=A*B where A and B are sparse.  This function is
% typically faster than C=A*B in MATLAB 7.4, and always uses less memory.
% Either A or B, or both, can be complex.
%
% Example:
%   load west0479
%   A = west0479 ;
%   B = sprand (west0479) ;
%   C = A*B ;
%   D = ssmult (A,B) ;
%   C-D
%
% This function can also compute any of the 64 combinations of
% C = op (op(A) * op(B)) where op(A) is A, A', A.', or conj(A).
% The general form is
%
%   C = ssmult (A,B, at,ac, bt,bc, ct,cc)
%
% where at = 0 or 1 to transpose A or not, and ac = 0 or 1 to use the conjugate
% of A, or not.  If not present, these 6 terms default to 0.  For example,
% these pairs of expressions are identical:
%
%   ssmult (A,B, 1,1, 0,0, 0,0)             A'*B
%   ssmult (A,B, 0,1, 0,0, 0,0)             conj(A)*B
%   ssmult (A,B, 1,0, 0,0, 0,0)             A.'*B
%   ssmult (A,B, 0,0, 0,0, 0,0)             A*B
%   ssmult (A,B, 0,0, 1,1, 0,0)             A*B'
%   ssmult (A,B, 0,1, 0,0, 1,1)             (conj(A)*B)'
%
% See also ssmultsym, mtimes.  See sstest3 for a list of all 64 variants.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

error ('ssmult mexFunction not found') ;

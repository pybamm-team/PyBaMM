function [mtype, mkind, skind] = RBtype (A)                                 %#ok
%RBTYPE determine the Rutherford/Boeing type of a sparse matrix
% Usage:
%   [mtype mkind skind] = RBtype (A)
%
% A must be a sparse matrix.  RBtype determines the Rutherford/Boeing type of A.
% Very little memory is used (just size(A,2) integer workspace), so this can
% succeed where a test such as nnz(A-A')==0 will fail.
%
%       mkind:  R: (0), A is real, and not binary
%               P: (1), A is binary (all values or 0 or 1)
%               C: (2), A is complex
%               I: (3), A is integer
%
%       skind:  R: (-1), A is rectangular
%               U: (0), A is unsymmetric (not S, H, or Z)
%               S: (1), A is symmetric (nnz(A-A.') is 0)
%               H: (2), A is Hermitian (nnz(A-A') is 0)
%               Z: (3), A is skew symmetric (nnz(A+A.') is 0)
%
% mtype is a 3-character string, where mtype(1) is the mkind
% ('R', 'P', or 'C').  mtype(2) is the skind ('R', 'U', 'S', 'H', or 'Z'),
% and mtype(3) is 'A'.
%
% Example:
%   load west0479
%   A = west0479 ;
%   RBtype (A)
%   RBtype (spones (A))
%   RBtype (2*spones (A))
%   C = A+A' ;
%   RBtype (C)
%
% See also RBread, RBwrite.

% Copyright 2009-2011, Timothy A. Davis, http://www.suitesparse.com

error ('RBtype mexFunction not found') ;


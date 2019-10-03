function [p,q,r] = btf (A)                                                  %#ok
%BTF permute a square sparse matrix into upper block triangular form
% with a zero-free diagonal, or with a maximum number of nonzeros along the
% diagonal if a zero-free permutation does not exist.
%
% Example:
%       [p,q,r] = btf (A) ;
%       [p,q,r] = btf (A,maxwork) ;
%
% If the matrix has structural full rank, this is essentially identical to
%
%       [p,q,r] = dmperm (A)
%
% except that p, q, and r will differ in trivial ways.  Both return an upper
% block triangular form with a zero-free diagonal, if the matrix is
% structurally non-singular.  The number and sizes of the blocks will be
% identical, but the order of the blocks, and the ordering within the blocks,
% can be different.
% 
% If the matrix is structurally singular, the q from btf will contain negative
% entries.  The permuted matrix is C = A(p,abs(q)), and find(q<0) gives a list
% of indices of the diagonal of C that are equal to zero.  This differs from
% dmperm, which does not place the maximum matching along the main diagonal
% of C=A(p,q), but places it above the diagonal instead.
%
% The second input limits the maximum amount of work the function does to
% be maxwork*nnz(A), or no limit at all if maxwork <= 0.  If the function
% terminates early as a result, a maximum matching may not be found, and the
% diagonal of A(p,abs(q)) might not have the maximum number of nonzeros
% possible.  Also, the number of blocks (length(r)-1) may be larger than
% what btf(A) or dmperm(A) would compute.
%
% See also maxtrans, strongcomp, dmperm, sprank

% Copyright 2004-2007, University of Florida
% with support from Sandia National Laboratories.  All Rights Reserved.

error ('btf mexFunction not found') ;

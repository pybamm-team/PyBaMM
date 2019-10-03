function [p,q,r] = strongcomp (A, qin)                                      %#ok
%STRONGCOMP symmetric permutation to upper block triangular form
% The matrix must be sparse and square.
%
% Example:
%    [p,r] = strongcomp (A) ;
%    [p,q,r] = strongcomp (A,qin) ;
%
% In the first usage, the permuted matrix is C = A (p,p).  In the second usage,
% the matrix A (:,qin) is symmetrically permuted to upper block triangular
% form, where qin is an input column permutation, and the final permuted
% matrix is C = A (p,q).  This second usage is equivalent to
%
%    [p,r] = strongcomp (A (:,qin)) ;
%    q = qin (p) ;
%
% That is, if qin is not present it is assumed to be qin = 1:n.
%
% C is the permuted matrix, with a number of blocks equal to length(r)-1.
% The kth block is from row/col r(k) to row/col r(k+1)-1 of C.
% r(1) is one and the last entry in r is equal to n+1.
% The diagonal of A (or A (:,qin)) is ignored.
%
% strongcomp is normally proceeded by a maximum transversal.
% Assuming A is square and structurally nonsingular,
%
%    [p,q,r] = strongcomp (A, maxtrans (A))
%
% is essentially identical to
%
%    [p,q,r] = dmperm (A)
%
% except that p, q, and r will differ.  Both return an upper block triangular
% form with a zero-free diagonal.  The number and sizes of the blocks will be
% identical, but the order of the blocks, and the ordering within the blocks,
% can be different.  If the matrix is structurally singular, both strongcomp
% and maxtrans return a vector q containing negative entries.  abs(q) is a
% permutation of 1:n, and find(q<0) gives a list of the indices of the
% diagonal of A(p,q) that are zero.
%
% See also btf, maxtrans, dmperm

% Copyright 2004-2007, University of Florida

error ('strongcomp mexFunction not found') ;

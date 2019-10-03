function q = maxtrans (A)                                                   %#ok
%MAXTRANS permute the columns of a sparse matrix so it has a zero-free diagonal
% (if it exists).   If no zero-free diagonal exists, then a maximum matching is
% found.  Note that this differs from p=dmperm(A), which returns a row
% permutation.
%
% Example:
%   q = maxtrans (A)
%   q = maxtrans (A,maxwork)
%
% If row i and column j are matched, then q(i) = j.  Otherwise, if row is
% unmatched, then q(i) = 0.  This is similar to dmperm, except that
% p = dmperm(A) returns p(j)=i if row i and column j are matched, or p(j)=0 if
% column j is unmatched.
%
% If A is structurally nonsingular, then A(:,maxtrans(A)) has a zero-free
% diagonal, as does A (dmperm(A),:).
%
% The second input limits the maximum amount of work the function does
% (excluding the O(nnz(A)) cheap match phase) to be maxwork*nnz(A), or no limit
% at all if maxwork <= 0.  If the function terminates early as a result, a
% maximum matching may not be found.  An optional second output
% [q,work] = maxtrans (...) returns the amount of work performed, or -1 if the
% maximum work limit is reached.
%
% See also: btf, strongcomp, dmperm, sprank

% Copyright 2004-2007, University of Florida

error ('maxtrans mexfunction not found') ;

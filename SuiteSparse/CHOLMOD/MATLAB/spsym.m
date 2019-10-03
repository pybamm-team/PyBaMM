function result = spsym (A, quick)					    %#ok
%SPSYM determine if a sparse matrix is symmetric, Hermitian, or skew-symmetric.
%   If so, also determine if its diagonal has all positive real entries.
%   A must be sparse.
%
%   Example:
%   result = spsym (A) ;
%   result = spsym (A,quick) ;
%
%   If quick = 0, or is not present, then this routine returns:
%
%       1: if A is rectangular
%       2: if A is unsymmetric
%       3: if A is symmetric, but with one or more A(j,j) <= 0
%       4: if A is Hermitian, but with one or more A(j,j) <= 0 or with
%           nonzero imaginary part
%       5: if A is skew symmetric (and thus the diagonal is all zero as well)
%       6: if A is symmetric with real positive diagonal
%       7: if A is Hermitian with real positive diagonal
%
%   If quick is nonzero, then the function can return more quickly, as soon as
%   it finds a diagonal entry that is <= 0 or with a nonzero imaginary part.
%   In this case, it returns 2 for a square matrix, even if the matrix might
%   otherwise be symmetric or Hermitian.
%
%   Regardless of the value of "quick", this function returns 6 or 7 if A is
%   a candidate for sparse Cholesky.
%
%   For an MATLAB M-file function that computes the same thing as this
%   mexFunction (but much slower), see the get_symmetry function by typing
%   "type spsym".
%
%   This spsym function does not compute the transpose of A, nor does it need
%   to examine the entire matrix if it is unsymmetric.  It uses very little
%   memory as well (just size-n workspace, where n = size (A,1)).
%
%   Examples:
%       load west0479
%       A = west0479 ;
%       spsym (A)
%       spsym (A+A')
%       spsym (A-A')
%       spsym (A+A'+3*speye(size(A,1)))
%
%   See also mldivide.

%       function result = get_symmetry (A,quick)
%       %GET_SYMMETRY: does the same thing as the spsym mexFunction.
%       % It's just a lot slower and uses much more memory.  This function
%       % is meant for testing and documentation only.
%       [m n] = size (A) ;
%       if (m ~= n)
%           result = 1 ;            % rectangular
%           return
%       end
%       if (nargin < 2)
%           quick = 0 ;
%       end
%       d = diag (A) ;
%       posdiag = all (real (d) > 0) & all (imag (d) == 0) ;
%       if (quick & ~posdiag)
%           result = 2 ;            % Not a candidate for sparse Cholesky.
%       elseif (~isreal (A) & nnz (A-A') == 0)
%           if (posdiag)
%               result = 7 ;        % complex Hermitian, with positive diagonal
%           else
%               result = 4 ;        % complex Hermitian, nonpositive diagonal
%           end
%       elseif (nnz (A-A.') == 0)
%           if (posdiag)
%               result = 6 ;        % symmetric with positive diagonal
%           else
%               result = 3 ;        % symmetric, nonpositive diagonal
%           end
%       elseif (nnz (A+A.') == 0)
%           result = 5 ;            % skew symmetric
%       else
%           result = 2 ;            % unsymmetric
%       end

% With additional outputs, spsym computes the following for square matrices:
% (in this case "quick" is ignored, and set to zero):
%
% [result xmatched pmatched nzoffdiag nnzdiag] = spsym(A)
%
%   xmatched is the number of nonzero entries for which A(i,j) = conj(A(j,i)).
%   pmatched is the number of entries (i,j) for which A(i,j) and A(j,i) are
%   both in the pattern of A (the value doesn't matter).  nzoffdiag is the
%   total number of off-diagonal entries in the pattern.  nzdiag is the number
%   of diagonal entries in the pattern.  If the matrix is rectangular,
%   xmatched, pmatched, nzoffdiag, and nzdiag are not computed (all of them are
%   returned as zero).  Note that a matched pair, A(i,j) and A(j,i) for i != j,
%   is counted twice (once per entry).

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('spsym mexFunction not found') ;

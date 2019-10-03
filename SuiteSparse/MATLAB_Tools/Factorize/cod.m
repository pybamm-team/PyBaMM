function [U, R, V, r] = cod (A, tol)
%COD complete orthogonal decomposition of a full matrix A = U*R*V'
%
%   [U, R, V, r] = cod (A)
%   [U, R, V, r] = cod (A, tol)
%
% The full m-by-n matrix A is factorized into U*R*V' where R is r-by-r and
% upper triangular and where r is the estimated rank of A.  The diagonal
% entries of R have magnitude greater than tol.  The default tol of
% 20*(m+n)*eps(max(diag(R))) is used if tol is not present or if tol < 0.
% Use COD_SPARSE for sparse matrices.
%
% COD provides a 'rank-sized' economy factorization, where R is r-by-r, U is
% m-by-r and V is n-by-r.  U and V are matrices with orthonormal columns.  That
% is, U'*U = V'*V = eye (r).  If A is rank deficient, then U and V are full
% matrices.  If A has full rank, either U or V are sparse permutation matrices
% (V if m >= n, U if m < n).  QR with column 2-norm pivoting is used on A
% if m >= n, or on A' if m < n, and thus abs(diag(R)) is decreasing if m >= n
% and increasing if m < n.
%
% If condest(R) is high (> 1e12 or so), then the estimated rank of A
% might be incorrect.  Try increasing tol in that case, which will make R
% better conditioned and reduce the estimated rank.
%
% Example:
%
%   A = magic (4),   [U, R, V] = cod (A),  norm (A - U*R*V')
%   A = rand (4,3),  [U, R, V] = cod (A),  norm (A - U*R*V')
%   A = rand (3,4),  [U, R, V] = cod (A),  norm (A - U*R*V')
%
% See also qr, svd, rq, spqr, cod_sparse.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (issparse (A))
    error ('FACTORIZE:cod:sparse', ...
        'COD is not designed for sparse matrices.  Use COD_SPARSE instead.') ;
end
[m, n] = size (A) ;
if (nargin < 2)
    tol = -1 ;                  % default tol will be used
end

if (m >= n)

    % factorize A into U*R*V' with column 2-norm pivoting
    [U, R, V] = qr (A, 0) ;     % economy U*R = A(V,:) with column pivoting V
    V = sparse (V, 1:n, 1) ;    % R n-by-n and triu, U m-by-n, V n-by-n

    % find a rough estimate of the rank of A
    r = rank_est (R, m, n, tol) ;

    if (r < n)
        % A is rank deficient.  R upper trapezoidal with R(r+1:end,:) tiny
        [R, Q] = rq (R, r, n) ; % RQ factorization, R now upper triangular
        U = U (:, 1:r) ;        % discard all but the first r columns of U
        V = V * Q' ;            % merge Q and V
        % R is now r-by-r, U is m-by-r, and V is n-by-r.
    end

else

    % compute the cod of A' and permute the result
    [V, R, U, r] = cod (A', tol) ;
    U = U (:, end:-1:1) ;
    R = R (end:-1:1, end:-1:1)' ;
    V = V (:, end:-1:1) ;

end

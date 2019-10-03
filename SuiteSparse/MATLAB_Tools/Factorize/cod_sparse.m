function [U, R, V, r] = cod_sparse (A, arg)
%COD_SPARSE complete orthogonal decomposition of a sparse matrix A = U*R*V'
%
%   [U, R, V, r] = cod_sparse (A)
%   [U, R, V, r] = cod_sparse (A, opts)
%
% The sparse m-by-n matrix A is factorized into U*R*V' where R is m-by-n and
% all zero except for R(1:r,1:r), which is upper triangular.  The first r
% diagonal entries of R have magnitude greater than tol, where r is the
% estimated rank of A.  All other diagonal entries are zero.  The default tol
% of 20*(m+n)*eps(max(diag(R))) is used if tol is not present or if tol<0.
% Use COD for full matrices.
%
% By default, U and V are not returned as sparse matrices, but as structs that
% represent a sequence of Householder transformations (U of size m-by-m and V
% of size n-by-n).  They can be passed to COD_QMULT to multiply them with other
% matrices or to convert them into matrices.  Alternatively, you can have U and
% V returned as matrices with opts.Q='matrix'.
%
% If A has full rank and m >= n, then this function simply returns the QR
% factorization Q*R*P' = U*R*V' = A where V=P is the fill-reducing ordering.
% If m < n, then U is the fill-reducing ordering and V' the orthgonal factor in
% Householder form.  If A is rank deficient, then both U and V contain
% non-trivial Householder transformations.
%
% If condest (R (1:r,1:r)) is large (> 1e12 or so) then the estimated rank of A
% might be incorrect.  Try increasing tol in that case, which will make R
% better conditioned and reduce the estimated rank of A.
%
% If the opts input parameter is a scalar, then it is used as the value of tol.
% If it is a struct, it can contain non-default options:
%
%   opts.tol    the tolerance to be used.  tol < 0 means the default is used.
%   opts.Q      'Householder' to return U and V as structs (default), 'matrix'
%               to return them as sparse matrices.  In their matrix form, U and
%               V can take a huge amount of memory, however.
%
% Example:
%
%   A = sparse (magic (4))
%   [U, R, V] = cod_sparse (A)
%   norm (A - cod_qmult (U, cod_qmult (V, R, 2),1),1)   % 1-norm of A - U*R*V'
%   U = cod_qmult (U, speye (size (A,1)), 1) ;      % convert U into a matrix
%   V = cod_qmult (V, speye (size (A,2)), 1) ;      % convert V into a matrix
%   norm (A - U*R*V',1)
%   opts.Q = 'matrix'
%   [U, R, V] = cod_sparse (A,opts)
%   norm (A - U*R*V',1)
%
%   A = sparse (rand (4,3)),  [U, R, V] = cod_sparse (A)
%   norm (A - cod_qmult (U, cod_qmult (V, R, 2), 1), 1)
%   A = sparse (rand (4,3)),  [U, R, V] = cod_sparse (A)
%   norm (A - cod_qmult (U, cod_qmult (V, R, 2), 1), 1)
%
% Requires the SPQR and SPQR_QMULT functions from SuiteSparse,
% http://www.suitesparse.com
%
% See also qr, cod, cod_qmult, spqr, spqr_qmult.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% get the inputs
%-------------------------------------------------------------------------------

if (~issparse (A))
    error ('FACTORIZE:cod_sparse', ...
        'COD_SPARSE is not designed for full matrices.  Use COD instead.') ;
end

[m, n] = size (A) ;
opts = struct ;
if (nargin > 1)
    if (isreal (arg) && arg >= 0)
        opts.tol = arg ;
    else
        if (isfield (arg, 'Q'))
            opts.Q = arg.Q ;
        end
        if (isfield (arg, 'tol') && arg.tol >= 0)
            opts.tol = arg.tol ;
        end
    end
end
if (~isfield (opts, 'Q'))
    opts.Q = 'Householder' ;        % return Q as a struct
end
ismatrix = isequal (opts.Q, 'matrix') ;

%-------------------------------------------------------------------------------
% compute the COD
%-------------------------------------------------------------------------------

if (m >= n)

    %---------------------------------------------------------------------------
    % A is square, or tall and thin
    %---------------------------------------------------------------------------

    % U*R*P1' = A where R is m-by-n, P1 is n-by-n, and U is a struct
    % of Householder transformations representing an m-by-m matrix.
    [U, R, P1, info] = spqr (A, opts) ;
    r = info.rank_A_estimate ;
    if (r < n)
        % A is rank deficient.  R is m-by-n and upper trapezoidal.
        opts.tol = 0 ;
        [V, R, P2] = spqr (R', opts) ;      % R' is m-by-n and lower triangular
        rn = reversal (r, n) ;
        rm = reversal (r, m) ;
        R = R (rn, rm)' ;                   % reverse and transpose R
        if (ismatrix)
            U = U * P2 (:, rm) ;            % return U and V as sparse matrices
            V = P1 * V ;
            V = V (:, rn) ;
        else
            U.Pc = P2 (:, rm) ;             % U = U * P2 (:,rm)
            V.P = (P1 * V.P')' ;            % V = P1 * V ;
            V.Pc = sparse (1:n, rn, 1) ;    % V = V (:,rn)
        end
    else
        % the factorization is A = U*R*V' with R upper triangular
        if (ismatrix)
            V = P1 ;                        % return V as a matrix, P1
        else
            V = Qpermutation (P1) ;         % V = P1, as a struct.
        end
    end

else

    %---------------------------------------------------------------------------
    % A is short and fat
    %---------------------------------------------------------------------------

    % V*R*P1' = A' where R is n-by-m, P1 is m-by-m, and V is a struct
    % of Householder transformations representing an n-by-n matrix.
    [V, R, P1, info] = spqr (A', opts) ;
    r = info.rank_A_estimate ;
    if (r < m)
        % A is rank deficient.  R is n-by-m and upper trapezoidal.
        opts.tol = 0 ;
        [U, R, P2] = spqr (R', opts) ;      % R is m-by-n and upper triangular
        if (ismatrix)
            U = P1 * U ;
            V = V * P2 ;
        else
            U.P = (P1 * U.P')' ;            % U = P1 * U
            V.Pc = P2 ;                     % V = V * P2
        end
    else
        % A is full rank, with A = P1*R'*U'.  Transpose and reverse R.
        rm = reversal (m, m) ;
        rn = reversal (m, n) ;
        R = R (rn, rm)' ;                   % reverse and transpose R
        if (ismatrix)
            V = V (:, rn) ;
            U = P1 (:, rm) ;
        else
            V.Pc = sparse (rn, 1:n, 1) ;    % V = V (:,rn)
            U = Qpermutation (P1 (:, rm)) ; % U = P1 (:,rm)
        end
    end
end

%-------------------------------------------------------------------------------

function p = reversal (r,n)
%REVERSAL return a vector that reverses the first r entries of 1:n
p = [(r:-1:1) (r+1:n)] ;

function Q = Qpermutation (P)
%QPERMUTATION convert a permutation matrix P into a struct for cod_qmult
% The output Q contains no Householder transformations.
n = size (P,1) ;
Q.H = sparse (n,0) ;
Q.Tau = zeros (1,0) ;
Q.P = (P * (1:n)')' ;

function F = factorize (A,strategy,burble)
%FACTORIZE an object-oriented method for solving linear systems
% and least-squares problems, and for representing operations with the
% inverse of a square matrix or the pseudo-inverse of a rectangular matrix.
%
% F = factorize(A) returns an object F that holds the factorization of A.
% x=F\b then solves a linear system or a least-squares problem.  S=inverse(F)
% or S=inverse(A) returns a factorized representation of the inverse of A so
% that inverse(A)*b is mathematically equivalent to pinv(A)*b, but the former
% does not actually compute the inverse or pseudo-inverse of A.
%
% Example
%
%   F = factorize(A) ;      % LU, QR, or Cholesky factorization of A
%   x = F\b ;               % solve A*x=b; same as x=A\b
%   S = inverse (F) ;       % S represents the factorization of inv(A)
%   x = S*b ;               % same as x = A\b.
%   E = A-B*inverse(D)*C    % efficiently computes the Schur complement
%   E = A-B*inv(D)*C        % bad method for computing the Schur complement
%   S = inverse(A) ; S(:,1) % compute just the first column of inv(A),
%                           % without computing inv(A)
%
%   F = factorize (A, strategy, burble) ;   % optional 2nd and 3rd inputs
%
% A string can be specified as a second input parameter to select the strategy
% used to factorize the matrix.  The first two are meta-strategies:
%
%   'default'   if rectangular
%                   use QR for sparse A or A' (whichever is tall and thin);
%                   use COD for full A
%               else
%                   if symmetric
%                       if positive real diagonal: try CHOL
%                       else (or if CHOL fails): try LDL
%                   end
%                   if not yet factorized: try LU (fails if rank-deficient)
%               end
%               if all else fails, or if QR or LU report that the matrix
%               is singular (or nearly so): use COD
%               This strategy mimics backslash, except that backslash never
%               uses COD.  Backslash also exploits other solvers, such as
%               specialized tridiagonal and banded solvers.
%
%   'symmetric' as 'default' above, but assumes A is symmetric without
%               checking, which is faster if you already know A is symmetric.
%               Uses tril(A) and assumes triu(A) is its transpose.  Results
%               will be incorrect if A is not symmetric.  If A is rectangular,
%               the 'default' strategy is used instead.
%
%   'unsymmetric'  as 'default', but assumes A is unsymmetric.
%
% The next "strategies" just select a single method, listed in decreasing order
% of generality and increasing order of speed and memory efficiency.  All of
% them except the SVD can exploit sparsity.
%
%   'svd'       use SVD.  Never fails ... unless it runs out of time or memory.
%                   Coerces a sparse matrix A to full.
%
%   'cod'       use COD.  Almost as accurate as SVD, and much faster.  Based
%                   on dense or sparse QR with rank estimation.  Handles
%                   rank-deficient matrices, as long as it correctly estimates
%                   the rank.  If the rank is ill-defined, use the SVD instead.
%                   Sparse COD requires the SPQR package to be installed
%                   (see http://www.suitesparse.com).
%
%   'qr'        use QR.  Reports a warning if A is singular.
%
%   'lu'        use LU.  Fails if A is rectangular; warning if A singular.
%
%   'ldl'       use LDL.  Fails if A is rank-deficient or not symmetric, or if
%                   A is sparse and complex.  Uses tril(A) and assumes triu(A)
%                   is the transpose of tril(A).
%
%   'chol'      use CHOL.  Fails if A is rank-deficient or not symmetric
%                   positive definite.  If A is sparse, it uses tril(A) and
%                   assumes triu(A) is the transpose of tril(A).  If A is dense,
%                   triu(A) is used instead.
%
% A third input, burble, can be provided to tell this function to print what
% methods it tries (if burble is nonzero).
%
% For a demo type "fdemo" in the Demo directory or see the Doc/ directory.
%
% See also inverse, slash, linsolve, spqr.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

assert (ndims (A) == 2, 'Matrix must be 2D.') ;

if (nargin < 2 || isempty (strategy))
    strategy = 'default' ;
end
if (nargin < 3)
    burble = 0 ;
end

if (burble)
    fprintf ('\nfactorize: strategy %s, A has size %d-by-%d, ', ...
        strategy, size (A)) ;
    if (issparse (A))
        fprintf ('sparse with %d nonzeros.\n', nnz (A)) ;
    else
        fprintf ('full.\n') ;
    end
end

switch strategy

    case 'default'
        [F, me] = backslash_mimic (A, burble, 0) ;

    case 'symmetric'
        [F, me] = backslash_mimic (A, burble, 1) ;

    case 'unsymmetric'
        [F, me] = backslash_mimic (A, burble, 2) ;

    case 'svd'
        [F, me] = factorize_svd (A, burble) ;

    case 'cod'
        [F, me] = factorize_cod (A, burble) ;

    case 'qr'
        [F, me] = factorize_qr (A, burble, 0) ;

    case 'lu'
        % do not report a failure if the matrix is singular
        [F, me] = factorize_lu (A, burble, 0) ;

    case 'ldl'
        [F, me] = factorize_ldl (A, burble) ;

    case 'chol'
        [F, me] = factorize_chol (A, burble) ;

    otherwise
        error ('FACTORIZE:invalidStrategy', 'unrecognized strategy.') ;

end

if (~isobject (F))
    throw (me) ;
end

%-------------------------------------------------------------------------------

function [F, me] = backslash_mimic (A, burble, strategy)
%BACKSLASH_MIMIC automatically select a method to factorize A.
F = [ ] ;
me = [ ] ;
[m, n] = size (A) ;

% If the following condition is true, then the QR, QRT, or LU factorizations
% will report a failure if A is singular (or nearly so).  This allows COD
% or COD_SPARSE to then be used instead.  COD_SPARSE relies on the SPQR
% mexFunction in SuiteSparse, which might not be installed.  In this case,
% QR, QRT, and LU do not report failures for sparse matrices that are singular
% (or nearly so), since there is no COD_SPARSE to fall back on.
fail_if_singular = ~issparse (A) || (exist ('spqr') == 3) ;                 %#ok
try_cod = true ;

if (m ~= n)
    if (issparse (A))
        % Use QR for the sparse rectangular case (ignore 'strategy' argument).
        [F, me] = factorize_qr (A, burble, fail_if_singular) ;
    else
        % Use COD for the full rectangular case (ignore 'strategy' argument).
        % If this fails, there's no reason to retry the COD below.  If A has
        % full rank, then COD is the same as QR with column pivoting (with the
        % same cost in terms of run time and memory).  Backslash in MATLAB uses
        % QR with column pivoting alone, so this is just as fast as x=A\b in
        % the full-rank case, but gives a more reliable result in the rank-
        % deficient case.
        try_cod = false ;
        [F, me] = factorize_cod (A, burble) ;
    end
else
    % square case: Cholesky, LDL, or LU factorization of A
    switch strategy
        case 0
            is_symmetric = (nnz (A-A') == 0) ;
        case 1
            is_symmetric = true ;
        case 2
            is_symmetric = false ;
    end
    if (is_symmetric)
        % A is symmetric (or assumed to be so)
        d = diag (A) ;
        if (all (d > 0) && nnz (imag (d)) == 0)
            % try a Cholesky factorization
            [F, me] = factorize_chol (A, burble) ;
        end
        if (~isobject (F) && (~issparse (A) || isreal (A)))
            % try an LDL factorization.
            % complex sparse LDL does not yet exist in MATLAB
            [F, me] = factorize_ldl (A, burble) ;
        end
    end
    if (~isobject (F))
        % use LU if Cholesky and/or LDL failed, or were skipped.
        [F, me] = factorize_lu (A, burble, fail_if_singular) ;
    end
end
if (~isobject (F) && try_cod)
    % everything else failed, matrix is rank-deficient.  Use COD
    [F, me] = factorize_cod (A, burble) ;
end


%-------------------------------------------------------------------------------

function [F, me] = factorize_qr (A, burble, fail_if_singular)
% QR fails if the matrix is rank-deficient.
F = [ ] ;
me = [ ] ;
try
    [m, n] = size (A) ;
    if (m >= n)
        if (burble)
            fprintf ('factorize: try QR of A ... ') ;
        end
        if (issparse (A))
            F = factorization_qr_sparse (A, fail_if_singular) ;
        else
            F = factorization_qr_dense (A, fail_if_singular) ;
        end
    else
        if (burble)
            fprintf ('factorize: try QR of A'' ... ') ;
        end
        if (issparse (A))
            F = factorization_qrt_sparse (A, fail_if_singular) ;
        else
            F = factorization_qrt_dense (A, fail_if_singular) ;
        end
    end
    if (burble)
        fprintf ('OK.\n') ;
    end
catch me
    if (burble)
        fprintf ('failed.\nfactorize: %s\n', me.message) ;
    end
end


%-------------------------------------------------------------------------------

function [F, me] = factorize_chol (A, burble)
% LDL fails if the matrix is rectangular, rank-deficient, or not positive
% definite.  Only the lower triangular part of A is used.
F = [ ] ;
me = [ ] ;
try
    if (burble)
        fprintf ('factorize: try CHOL ... ') ;
    end
    if (issparse (A))
        F = factorization_chol_sparse (A) ;
    else
        F = factorization_chol_dense (A) ;
    end
    if (burble)
        fprintf ('OK.\n') ;
    end
catch me
    if (burble)
        fprintf ('failed.\nfactorize: %s\n', me.message) ;
    end
end


%-------------------------------------------------------------------------------

function [F, me] = factorize_ldl (A, burble)
% LDL fails if the matrix is rectangular or rank-deficient.
% As of MATLAB R2012a, ldl does not work for complex sparse matrices.
% Only the lower triangular part of A is used.
F = [ ] ;
me = [ ] ;
try
    if (burble)
        fprintf ('factorize: try LDL ... ') ;
    end
    if (issparse (A))
        F = factorization_ldl_sparse (A) ;
    else
        F = factorization_ldl_dense (A) ;
    end
    if (burble)
        fprintf ('OK.\n') ;
    end
catch me
    if (burble)
        fprintf ('failed.\nfactorize: %s\n', me.message) ;
    end
end


%-------------------------------------------------------------------------------

function [F, me] = factorize_lu (A, burble, fail_if_singular)
% LU fails if the matrix is rectangular or rank-deficient.
F = [ ] ;
me = [ ] ;
try
    if (burble)
        fprintf ('factorize: try LU ... ') ;
    end
    if (issparse (A))
        F = factorization_lu_sparse (A, fail_if_singular) ;
    else
        F = factorization_lu_dense (A, fail_if_singular) ;
    end
    if (burble)
        fprintf ('OK.\n') ;
    end
catch me
    if (burble)
        fprintf ('failed.\nfactorize: %s\n', me.message) ;
    end
end


%-------------------------------------------------------------------------------

function [F, me] = factorize_cod (A, burble)
% COD only fails when it runs out of memory.
F = [ ] ;
me = [ ] ;
try
    if (burble)
        fprintf ('factorize: try COD ... ') ;
    end
    if (issparse (A))
        F = factorization_cod_sparse (A) ;
    else
        F = factorization_cod_dense (A) ;
    end
    if (burble)
        fprintf ('OK.\n') ;
    end
catch me
    if (burble)
        fprintf ('failed.\nfactorize: %s\n', me.message) ;
    end
end

%-------------------------------------------------------------------------------

function [F, me] = factorize_svd (A, burble)
% SVD only fails when it runs out of memory.
F = [ ] ;
me = [ ] ;
try
    if (burble)
        fprintf ('factorize: try SVD ... ') ;
    end
    F = factorization_svd (A) ;
    if (burble)
        fprintf ('OK.\n') ;
    end
catch me
    if (burble)
        fprintf ('failed.\nfactorize: %s\n', me.message) ;
    end
end

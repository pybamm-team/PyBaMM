function [result, t] = linfactor (arg1, arg2)
%LINFACTOR factorize a matrix, or use the factors to solve Ax=b.
% Uses LU or CHOL to factorize A, or uses a previously computed factorization to
% solve a linear system.  This function automatically selects an LU or Cholesky
% factorization, depending on the matrix.  A better method would be for you to
% select it yourself.  Note that mldivide uses a faster method for detecting
% whether or not A is a candidate for sparse Cholesky factorization (see spsym
% in the CHOLMOD package, for example).
%
% Example:
%   F = linfactor (A) ;     % factorizes A into the object F
%   x = linfactor (F,b) ;   % uses F to solve Ax=b
%   norm (A*x-b)
%
% A second output is the time taken by the method, ignoring the overhead of
% determining which method to use.  This makes for a fairer comparison between
% methods, since normally the user will know if the matrix is supposed to be
% symmetric positive definite or not, and whether or not the matrix is sparse.
% Also, the overhead here is much higher than mldivide or spsym.
%
% This function has its limitations:
%
% (1) determining whether or not the matrix is symmetric via nnz(A-A') is slow.
%     mldivide (and spsym in CHOLMOD) do it much faster.
%
% (2) MATLAB really needs a sparse linsolve.  See cs_lsolve, cs_ltsolve, and
%     cs_usolve in CSparse, for example.
%
% (3) this function really needs to be written as a mexFunction.
%
% (4) the full power of mldivide is not brought to bear.  For example, UMFPACK
%     is not very fast for sparse tridiagonal matrices.  It's about a factor of
%     four slower than a specialized tridiagonal solver as used in mldivide.
%
% (5) permuting a sparse vector or matrix is slower in MATLAB than it should be;
%     a built-in linfactor would reduce this overhead.
%
% (6) mldivide when using UMFPACK uses relaxed partial pivoting and then
%     iterative refinement.  This leads to sparser LU factors, and typically
%     accurate results.  linfactor uses sparse LU without iterative refinement.
%
% The primary purpose of this function is to answer The Perennially Asked
% Question (or The PAQ for short (*)):  "Why not use x=inv(A)*b to solve Ax=b?
% How do I use LU or CHOL to solve Ax=b?"  The full answer is below.  The short
% answer to The PAQ (*) is "PAQ=LU ... ;-) ... never EVER use inv(A) to solve
% Ax=b."
%
% The secondary purpose of this function is to provide a prototype for some of
% the functionality of a true MATLAB built-in linfactor function.
% 
% Finally, the third purpose of this function is that you might find it actually
% useful for production use, since its syntax is simpler than factorizing the
% matrix yourself and then using the factors to solve the system.
%
% See also lu, chol, mldivide, linsolve, umfpack, cholmod.
%
% Oh, did I tell you never to use inv(A) to solve Ax=b?
%
% Requires MATLAB 7.3 (R2006b) or later.

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1 | nargin > 2 | nargout > 2)          %#ok
    error ('Usage: F=linfactor(A) or x=linfactor(F,b)') ;
end

if (nargin == 1)

    %---------------------------------------------------------------------------
    % F = linfactor (A) ;
    %---------------------------------------------------------------------------

    A = arg1 ;
    [m n] = size (A) ;
    if (m ~= n)
        error ('linfactor: A must be square') ;
    end

    if (issparse (A))

        % try sparse Cholesky (CHOLMOD): L*L' = P*A*P'
        if (nnz (A-A') == 0 & all (diag (A) > 0))   %#ok
            try
                tic ;
                [L, g, PT] = chol (A, 'lower') ;
                t = toc ;
                if (g == 0)
                    result.L = L ;
                    result.LT = L' ;    % takes more memory, but solve is faster
                    result.P = PT' ;    % ditto.  Need a sparse linsolve here...
                    result.PT = PT ;
                    result.kind = 'sparse Cholesky: L*L'' = P*A*P''' ;
                    result.code = 0 ;
                    return
                end
            catch
		% matrix is symmetric, but not positive definite
		% (or we ran out of memory)
            end
        end

        % try sparse LU (UMFPACK, with row scaling): L*U = P*(R\A)*Q
        tic ;
        [L, U, P, Q, R] = lu (A) ;
        t = toc ;
        result.L = L ;
        result.U = U ;
        result.P = P ;
        result.Q = Q ;
        result.R = R ;
        result.kind = 'sparse LU: L*U = P*(R\A)*Q where R is diagonal' ;
        result.code = 1 ;

    else

        % try dense Cholesky (LAPACK): L*L' = A
        if (nnz (A-A') == 0 & all (diag (A) > 0))                           %#ok
            try
                tic ;
                L = chol (A, 'lower') ;
                t = toc ;
                result.L = L ;
                result.kind = 'dense Cholesky: L*L'' = A' ;
                result.code = 2 ;
                return
            catch
		% matrix is symmetric, but not positive definite
		% (or we ran out of memory)
            end
        end

        % try dense LU (LAPACK): L*U = A(p,:)
        tic ;
        [L, U, p] = lu (A, 'vector') ;
        t = toc ;
        result.L = L ;
        result.U = U ;
        result.p = p ;
        result.kind = 'dense LU: L*U = A(p,:)' ;
        result.code = 3 ;

    end

else

    %---------------------------------------------------------------------------
    % x = linfactor (F,b)
    %---------------------------------------------------------------------------

    F = arg1 ;
    b = arg2 ;

    if (F.code == 0)

        % sparse Cholesky: MATLAB could use a sparse linsolve here ...
        tic ;
        result = F.PT * (F.LT \ (F.L \ (F.P * b))) ;
        t = toc ;

    elseif (F.code == 1)

        % sparse LU: MATLAB could use a sparse linsolve here too ...
        tic ;
        result = F.Q * (F.U \ (F.L \ (F.P * (F.R \ b)))) ;
        t = toc ;

    elseif (F.code == 2)

        % dense Cholesky: result = F.L' \ (F.L \ b) ;
        lower.LT = true ;
        upper.LT = true ;
        upper.TRANSA = true ;
        tic ;
        result = linsolve (F.L, linsolve (F.L, b, lower), upper) ;
        t = toc ;

    elseif (F.code == 3)

        % dense LU: result = F.U \ (F.L \ b (F.p,:)) ;
        lower.LT = true ;
        upper.UT = true ;
        tic ;
        result = linsolve (F.U, linsolve (F.L, b (F.p,:), lower), upper) ;
        t = toc ;
    end
end

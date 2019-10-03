function [Z, Zpattern, L, D, U, P, Q, stats] = sparseinv (A)
%SPARSEINV computes the sparse inverse subset of a real sparse square matrix A.
% This function is typically much faster than computing all of inv(A).
%
% [Z, Zpattern, L, D, U, P, Q, stats] = sparseinv (A)
%
% Z is a subset of the inverse a sparse matrix A of full rank.  On output, if
% Zpattern(i,j)=1, it means that Z(i,j) has been computed.  That is, the norm
% of (Zpattern .* (Z - inv (A))) will be small.
%
% Method: The permuted matrix C = P*A*Q is first factorized into C =
% (L+I)*D*(U+I) where D is diagonal, L+I is lower triangular with unit
% diagonal, and U+I is upper triangular with unit diagonal (I = speye (n)).  If
% A is symmetric and positive definite, then a Cholesky factorization is used
% (in which case P=Q' and L=U', and Z will include all diagonal entries of
% inv(A)).  Next, the entries in the inverse of C that correspond to nonzero
% values in Zpattern are found via Takahashi's method.  Zpattern is the
% symbolic Cholesky factorization of C+C', so it includes all entries in L+U
% and its transpose.
%
% stats is an optional struct containing statistics on the factorization.
%
% Example:
%   load west0479
%   A = west0479 ;
%   [Z, Zpattern] = sparseinv (A) ;
%   S = inv (A) ;
%   err = norm (Zpattern .* (Z - S), 1) / norm (S, 1)
%
% See also inv, lu, chol.

% Copyright 2011, Timothy A. Davis, http://www.suitesparse.com

get_stats = (nargout > 7) ;
if (get_stats)
    t1 = tic ;
end

% check inputs
if (~issparse (A))
    error ('A must be sparse') ;
end
[m n] = size (A) ;
if (m ~= n)
    error ('A must be square') ;
end
if (~isreal (A))
    error ('complex matrices not supported') ;
end

% construct the factorization: C = P*A*Q = (L+I)*D*(U+I)
p = 1 ;
if (all (diag (A)) > 0 && nnz (A-A') == 0)
    [L,p,Q] = chol (A, 'lower') ;
end
if (p == 0)
    % Cholesky worked.
    P = Q' ;
    d = diag (L) ;
    L = tril (L / diag (d), -1) ;
    U = L' ;
    d = d.^2 ;
    D = diag (d) ;
else
    % Cholesky failed, or wasn't attempted.  Use LU instead.
    [L,U,P,Q] = lu (A) ;
    d = diag (U) ;
    if (any (d == 0))
        error ('A must be full-rank') ;
    end
    D = diag (d) ;
    U = triu (D \ U, 1) ;
    L = tril (L, -1) ;
end
d = full (d) ;

% find the symbolic Cholesky of C+C'
S = spones (P*A*Q) ;
[c,h,pa,po,R] = symbfact (S+S') ;
clear h pa po
Zpattern = spones (R+R') ;
clear R S

if (get_stats)
    t1 = toc (t1) ;
    t2 = tic ;
end

% compute the sparse inverse subset
[Z takflops] = sparseinv_mex (L, d, U', Zpattern) ;
if (p == 0)
    % Force Z to be symmetric.  This is because sparseinv_mex does not
    % exploit the symmetry in the factorization, but computes both upper and
    % lower triangular parts of Z separately.  The work for the Takahashi
    % equations could be cut in half as a result.
    Z = (Z + Z') / 2 ;
end

% permute the result
Z = Q*Z*P ;
Zpattern = Q*Zpattern*P ;

% return stats, if requested
if (nargout > 7)
    t2 = toc (t2) ;
    if (p == 0)
        stats.kind = 'Cholesky' ;
        fl = 2*n + sum (c.^2) ;
        stats.nnz_factors = sum (c) ;
    else
        stats.kind = 'LU' ;
        Lnz = full (sum (spones (L))) ;	        % off diagonal nz in cols of L
        Unz = full (sum (spones (U')))' ;	% off diagonal nz in rows of U
        fl = n + 2*Lnz*Unz + sum (Lnz) ;
        stats.nnz_factors = nnz (L) + nnz (U) + n ;
    end
    stats.flops_factorization = fl ;
    stats.flops_Takahashi = takflops ;
    stats.time_factorization = t1 ;
    stats.time_Takahashi = t2 ;
end


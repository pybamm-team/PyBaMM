function [A, p, rcnd] = gee_its_simple_factorize (A)
%GEE_ITS_SIMPLE_FACTORIZE Gaussian Elimination Example, with partial pivoting
%
% gee_its_simple_factorize factorizes the n-by-n matrix A(p,:) in to the
% product of a unit lower triangular matrix L and an upper triangular matrix U.
% "Unit" means that the diagonal entires of L are all equal to one.  The
% entries on the diagonal of U are not equal to one.  The permutation vector p
% is a permutation of 1:n which arises from the row swaps performed by partial
% pivoting.  A(p,:) can also be written as the product of an n-by-n permutation
% matrix P times A; or P*A where P=I(p,:) where I=eye(n), or P=sparse(1:n,p,1)
% for a more concise representation.  Thus, L*U equals P*A or equivalently
% A(p,:).
%
% This factorization can be used to solve a linear system of equations, A*x=b.
% using the following derivation, where below, "=" is mathematical equality,
% not MATLAB assignment:
%
%   A*x = b             (1) original system
%   L*U = P*A           (2) factorization of P*A or A(p,:) into the product L*U
%   P*A*x = P*b         (3) multiply both sides of (1) by P
%   L*U*x = P*b         (4) substitute (2) into (3)
%   let y = U*x         (5) define y as U*x
%   let c = P*b         (6) define c as P*b
%   L*y = c             (7) subsitute (5) and (6) into (4)
%   U*x = y             (8) a rewrite of (5)
%
% These expressions can be used to compute x, where below "=" is MATLAB
% assignment, not mathematical equality:
%
%   [L U p] = lu (A) ;  % factorize
%   y = L \ (P*b) ;     % forward solve of (7), a lower triangular system
%   x = U \ y ;         % backsolve of (8), an upper triangular system
%
% The book "Direct Methods for Sparse Linear Systems" by T. Davis, SIAM, 2006,
% includes a complete derivation of Gaussian Elimination (producing an LU
% factorization) and partial pivoting, forward solve, and back solve, for both
% the full and sparse cases.  Refer to Section 3.1 for forward/backsolve, and
% Section 6.3 for right-looking LU factorization (aka Gaussian elimination).
%
% See also "Numerical Computing with MATLAB" (Chapter 2, "Linear Equations"),
% by Cleve Moler, SIAM, 2004.  You can also download the book for free from
% http://www.mathworks.com/moler .  The NCM Toolbox includes the lutx and
% bslashtx functions which are very similar to the algorithms used here
% (the backsolve in bslashtx works by rows of U; here it's by columns).
%
% In contrast to the MATLAB lu function, this function returns L and U packed
% into a single n-by-n matrix called LU, where L is contained in the strictly
% lower triangular part of LU (the unit diagonal of L is not stored) and U is
% stored in the strictly upper triangular part of LU.
%
% A very cheap estimate of the reciprocal condition number is also returned,
% which is merely the smallest absolute value on the diagonal of U divided by
% the largest absolute value.  This is the same estimate used in x=A\b to
% decide when to print the warning "matrix is close to singular or badly
% scaled."
%
% Example:
%
%   [LU p rcnd] = gee_its_simple_factorize (A) ;
%
%   % which is the same as
%   [L U p] = lu (A) ;
%   LU = tril (L,-1) + U ;
%   rcnd = rcond (A) ;          % this gives a better estimate
%
% See also: lu, rcond

% Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% check the inputs
%-------------------------------------------------------------------------------

if (nargin < 1 | nargout > 3)                                               %#ok
    error ('Usage: [LU p rcnd] = gee_its_simple_factorize (A)') ;
end

% ensure A is square
gee_its_simple_check (A, 'A') ;

%-------------------------------------------------------------------------------
% LU factorization, using Gaussian elimination with partial pivoting
%-------------------------------------------------------------------------------

% start with the identity permutation
n = size (A,1) ;
p = 1:n ;

% compute L, U, and p (overwriting A with L and U)
for k = 1:n
    % partial pivoting: look in A(k:n,k) for the largest entry A(i,k)
    [x i] = max (abs (A (k:n,k))) ;
    i = i+k-1 ;
    % swap row i and k of A (and L)
    A ([k i],:) = A ([i k],:) ;
    % record the pivot row swap just made
    p ([k i]) = p ([i k]) ;
    % divide the pivot column (the kth column of L) by the pivot entry
    A (k+1:n,k) = A (k+1:n,k) / A (k,k) ;
    % subtract rank-1 outer product from the (n-k)-by-(n-k) trailing submatrix
    A (k+1:n,k+1:n) = A (k+1:n,k+1:n) - A (k+1:n,k) * A (k,k+1:n) ;
end

%-------------------------------------------------------------------------------
% compute reciprocal condition number estimate
%-------------------------------------------------------------------------------

if (n == 0)
    rcnd = 1 ;
else
    d = max (abs (diag (A))) ;
    if (d == 0)
        rcnd = 0 ;
    else
        rcnd = min (abs (diag (A))) / d ;
    end
end

%-------------------------------------------------------------------------------
% check the result
%-------------------------------------------------------------------------------

if (rcnd == 0)
    warning ('MATLAB:singularMatrix', 'matrix is singular') ;
elseif (~isfinite (rcnd) | rcnd < eps)                                      %#ok
    warning ('MATLAB:nearlySingluarMatrix', ...
             'matrix is close to singular or badly scaled') ;
end


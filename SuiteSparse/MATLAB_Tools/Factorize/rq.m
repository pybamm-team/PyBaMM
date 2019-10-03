function [R, Q] = rq (A, m, n)
%RQ economy RQ or QL factorization of a full matrix A.
%   No special handling is done for rank-deficient matrices.
%
%   [R,Q] = rq (A)
%   [R,Q] = rq (A,m,n)
%
% If A is m-by-n with m <= n, then R*Q=A is factorized with R upper triangular
%   and m-by-m.  Q is m-by-n with orthonormal rows, where Q*Q' = eye (m), but
%   Q'*Q is not identity.  RQ works quickly when A is upper trapezoidal, but
%   also works in the general case.  With n=3 and m=5, an upper trapezoidal A:
%
%       x x x x x
%       . x x x x
%       . . x x x
%
%   The factorization is R*Q = A where R is upper triangular and m-by-m,
%   and Q is m-by-n:
%
%         R    *      Q      =      A
%       x x x     x x x x x     x x x x x
%       . x x     x x x x x     . x x x x
%       . . x     x x x x x     . . x x x
%
%   Q also happens to be upper trapezoidal if A is upper trapezoidal.
%   With two optional input arguments (m,n), only A (1:m,1:n) is factorized.
%
% If m > n, then Q*R=A is computed where "R" is lower triangular and Q
%   has orthonormal columns (Q'*Q is identity).
%
% Example
%
%   A = rand (3,4),   [R, Q] = rq (A),   norm (R*Q-A), norm (Q*Q'-eye(3))
%   C = rand (4,3),   [L, Q] = rq (C),   norm (Q*L-C), norm (Q'*Q-eye(3))
%
% See also qr.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (issparse (A))
    % RQ would actually work, but it would be very inefficient since no fill
    % reducing ordering is used.  That would require a row permutation of R.
    error ('FACTORIZE:rq:sparse', 'RQ is not designed for sparse matrices.') ;
end

if (nargin == 1)
    [m, n] = size (A) ;
end

if (m <= n)

    %---------------------------------------------------------------------------
    % RQ factorization of a short-and-fat matrix A
    %---------------------------------------------------------------------------

    [Q, R] = qr (A (m:-1:1, n:-1:1)', 0) ;
    R = R (end:-1:1, end:-1:1)' ;
    Q = Q (end:-1:1, end:-1:1)' ;

    % Below is a step-by-step working description of the algorithm.  Each of
    % the error norms will be small.  This code will operate correctly if
    % uncommented, it will just be slower than the 3 lines of code above.

    % (1) The A matrix is transposed and its rows and columns are reversed.
    %   The row/column reversal can be viewed as multiplication of A by row and
    %   column permutations, so this operation makes sense in terms of linear
    %   algebra: H = (Pm*A*Pn)' where Pm and Pn are permutation matrices of
    %   size m and n, respectively.

    %           H = A (m:-1:1, n:-1:1)' ;

    % H now has the following form.  This is good, because qr(H) can exploit
    % the 3 zeros in the lower triangular part, to reduce the computation time.
    %
    %   x x x
    %   x x x
    %   x x x
    %   . x x
    %   . . x
    %
    % We could instead factorize A', which has the following shape:
    %
    %   x . .
    %   x x .
    %   x x x
    %   x x x
    %   x x x
    %
    % but the QR method in MATLAB cannot exploit the zeros in upper triangular
    % part A'.

    % (2) The QR factorzation of H is computed.  QR in MATLAB takes advantage
    % of the zeros in H.  The resulting Q is n-by-m, R is m-by-m.

    %           [Q, R] = qr (H, 0) ;
    %           err = norm (Q*R-H)

    %     Q    *    R    =    H
    %
    %   x x x     1 x x     x x x       (a "1" denotes the R(1,1) entry,
    %   x x x     . x x     x x x        so it can be followed in the
    %   x x x     . . x     x x x        operations below)
    %   . x x               . x x
    %   . . x               . . x

    % (3) The columns of R and H are reversed.  This is the same as multiplying
    % both sides of the equation by the Pm m-by-m permutation matrix on the
    % right.

    %           R = R (:, end:-1:1) ;
    %           H = H (:, end:-1:1) ;
    %           err = norm (Q*R-H)

    %     Q    *    R    =    H
    %
    %   x x x     x x 1     x x x
    %   x x x     x x .     x x x
    %   x x x     x . .     x x x
    %   . x x               x x .
    %   . . x               x . .

    % (4) Both sides of the equation are transposed.

    %           H = H' ;
    %           R = R' ;
    %           Q = Q' ;
    %           err = norm (R*Q-H)

    %     R    *      Q     =      H
    %
    %   x x x     x x x . .    x x x x x
    %   x x .     x x x x .    x x x x .
    %   1 . .     x x x x x    x x x . .

    % (5) The columns of Q and H are reversed.  This is the same as multiplying
    %   both sides of the equation by the Pn n-by-n permutation matrix on the
    %   right.  H is now equal to A again.

    %           H = H (:, end:-1:1)  ;
    %           Q = Q (:, end:-1:1)  ;
    %           err = norm (A-H)
    %           err = norm (R*Q-H)

    %     R    *      Q     =      A
    %
    %   x x x     . . x x x    x x x x x
    %   x x .     . x x x x    . x x x x
    %   1 . .     x x x x x    . . x x x

    % (6) The columns of R and rows of Q are reversed.  This the same as
    %   inserting the product of Pm*Pm' = I between R and Q, where Pm is the
    %   m-by-m permutation matrix.

    %           R = R (:, end:-1:1) ;
    %           Q = Q (end:-1:1, :) ;
    %           err = norm (R*Q-H)
    %           err = norm (Q*Q' - eye (m))

    %     R    *      Q     =      A
    %
    %   x x x     x x x x x    x x x x x
    %   . x x     . x x x x    . x x x x
    %   . . 1     . . x x x    . . x x x

else

    %---------------------------------------------------------------------------
    % QL factorization of a tall-and-thin matrix A
    %---------------------------------------------------------------------------

    [R, Q] = rq (A', n, m) ;
    R = R' ;
    Q = Q' ;

end

function [L, D] = ldlrow (A)
%LDLROW an m-file description of the algorithm used by LDL
%
% Example:
%  [L, D] = ldlrow (A)
%
%  Compute the L*D*L' factorization of A, by rows.  Returns
%  full L and D matrices.  This routine serves as an outline
%  of the numerical factorization performed by ldl.c.
%
%  Here is a diagram of how L is computed.  "a" means an
%  entry that is accessed at the kth step, and "c" means an
%  entry that is computed.  A "-" means neither accessed nor
%  computed.  A "1" means the value of the entry is L (the
%  unit diagonal of L), and it is accessed at the kth step.
%  A "." means the value is zero.
%
%  The L matrix
%
%     1 . . . . . . .
%     a 1 . . . . . .
%     a a 1 . . . . .
%     a a a 1 . . . .
%     c c c c c . . .  <- kth row of L
%     - - - - - - . .
%     - - - - - - - .
%     - - - - - - - -
%
%  The A matrix:
%
%             the kth column of A
%             v
%     - - - - a - - -
%     - - - - a - - -
%     - - - - a - - -
%     - - - - a - - -
%     - - - - a - - -  <- kth row of A
%     - - - - - - - - 
%     - - - - - - - -
%     - - - - - - - -
%
%  The D matrix:
%
%             the kth column of D
%             v
%     a . . . . . . .
%     . a . . . . . .
%     . . a . . . . .
%     . . . a . . . .
%     . . . . c . . .  <- kth row of D
%     . . . . . . . . 
%     . . . . . . . .
%     . . . . . . . .
%
% See also ldlsparse.

% Copyright 2006-2007 by Timothy A. Davis, http://www.suitesparse.com

[m n] = size (A) ;
L = zeros (n, n) ;
D = zeros (n, 1) ;
A = full (A) ;

L (1, 1) = 1 ;
D (1) = A (1,1) ;

for k = 2:n

    % note the sparse triangular solve.  For the sparse
    % case, the pattern of y is the same as the pattern of
    % the kth row of L.
    y = L (1:k-1, 1:k-1) \ A (1:k-1, k) ;

    % scale row k of L
    L (k, 1:k-1) = (y ./ D (1:k-1))' ;
    L (k, k) = 1 ;

    % compute the diagonal
    D (k) = A (k,k) - L (k, 1:k-1) * y ;
end

D = diag (D) ;

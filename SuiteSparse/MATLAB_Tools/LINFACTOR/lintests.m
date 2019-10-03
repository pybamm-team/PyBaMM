function lintests
%LINTESTS test linfactor with many different kinds of systems.
% Compares x=A\b, linfactor and (ack!) inv(A)*b.  You should never, ever use
% inv(A) to solve a linear system.
%
% Example
%   lintests
%
% See also lintest, linfactor, mldivide.

% Copyright 2007, Timothy A. Davis

rand ('state', 0) ;
help linfactor

for n = [100 1000 2000]

    fprintf ('\nn: %d (with all nonzero matrix A)\n', n) ;

    % dense LU
    A = rand (n) ;
    b = rand (n,1) ;
    lintest (A,b) ;

    % sparse LU
    A = sparse (A) ;
    lintest (A,b) ;

    % dense Cholesky
    A = A*A' + 10*eye(n) ;
    lintest (A,b) ;

    % sparse Cholesky
    A = sparse (A) ;
    lintest (A,b) ;

end

for n = [1000 2000]

    % note that UMFPACK is not particularly fast for tridiagonal matrices
    % (see "doc mldivide", which uses a specialized tridiagonal solver)
    fprintf ('\nn: %d (sparse tridiagonal matrix)\n', n) ;

    % sparse LU
    e = rand (n, 1) ;
    b = rand (n, 1) ;
    A = spdiags ([e 4*e e], -1:1, n, n) ;
    lintest (A,b) ;

    % sparse Cholesky
    e = ones (n, 1) ;
    A = spdiags ([e 4*e e], -1:1, n, n) ;
    lintest (A,b) ;

end

% sparse LU again
fprintf ('\nwest0479:\n') ;
load west0479 ;
n = size (west0479, 1) ;
b = rand (n, 1) ;
lintest (west0479, b) ;

% completely break inv(A) with a simple 2-by-2 matrix ...
fprintf ('\nbreak inv(A) with a trivial 2-by-2 matrix:\n') ;
s = warning ('off', 'MATLAB:singularMatrix') ;
lintest (rand(2) * realmin/2, ones(2,1)) ;
warning (s) ;

function test13
%TEST13 test cholmod2 and MATLAB on large tridiagonal matrices
% Example:
%   test13
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test13: test cholmod2 and MATLAB on large tridiagonal matrices\n') ;

for n = [10000 1e4 1e5 1e6]

    e = ones (n,1) ;
    A = spdiags ([e 4*e e], -1:1, n, n) ;
    clear e
    b = rand (n,1) ;

    tic ;
    x = cholmod2 (A,b) ;
    t2 = toc ;
    e = norm (A*x-b,1) ;
    fprintf ('n %9d   cholmod2 %8.2f  err %6.1e\n', n, t2, e) ;

    tic ;
    x = A\b ;
    t1 = toc ;
    e = norm (A*x-b,1) ;
    fprintf ('n %9d   matlab  %8.2f  err %6.1e\n', n, t1, e) ;

    clear A b

end



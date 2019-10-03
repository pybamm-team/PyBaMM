function test25
%TEST25 test sdmult on a large matrix
% Example:
%   test25
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test25: test sdmult on a large matrix\n') ;

Prob = UFget (936) ;
A = Prob.A ;
n = size (A,1) ;

nz = nnz (A) ;
fprintf ('\nTest matrix: %d-by-%d, nnz %d\n', n, n, nz) ;

Z = rand (n, 500) ;

fprintf ('\nA*X where X is %d-by-k\n', n) ;

for k = [0:10 10:10:50 100:100:500]

    X = Z (:, 1:k) ;

    tic ;
    D = A*X ;
    t1 = toc ;

    tic ;
    C = sdmult (A,X) ;
    t2 = toc ;

    err = norm (C-D,1) ;
    fprintf (...
	'k: %3d time: MATLAB %8.2f CHOLMOD %8.2f speedup %8.2f err %6.0e',...
	k, t1, t2, t1/t2, err) ;
    fl = 2*nz*k ;
    fprintf ('  mflop: MATLAB %8.1f CHOLMOD %8.1f\n', 1e-6*fl/t1, 1e-6*fl/t2) ;
	
    clear C D X
end

fprintf ('\nFor comparison, here is CHOLMOD''s x=A\\b time:\n') ;
for k = [1 100:100:500]
    B = Z (:, 1:k) ;
    tic
    x = cholmod2 (A,B) ;
    t2 = toc ;
    err2 = norm (sdmult(A,x)-B,1) ;
    fprintf (...
       'CHOLMOD x=A\\b time: %8.2f (b is n-by-%d) resid %6.0e\n', t2, k, err2) ;
    clear x B
end

b = Z (:,1) ;
clear Z

tic
x = A\b ;
t1 = toc ;
err1 = norm (A*x-b,1) ;
fprintf ('\nMATLAB  x=A\\b time: %8.2f (b is n-by-1) resid %6.0e\n', t1, err1) ;

fprintf ('test25 passed\n') ;

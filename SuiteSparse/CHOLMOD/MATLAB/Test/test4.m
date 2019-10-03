function test4
%TEST4 test cholmod2 with multiple and sparse right-hand-sides
% Example:
%   test4
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test4: test cholmod2 with multiple and sparse right-hand-sides\n') ;

Prob = UFget ('HB/bcsstk01') ;
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;
x = cholmod2 (A,b) ;
m2 = norm (A*x-b,1) ;
b = sparse (b) ; 
x = cholmod2 (A,b) ;
m2 = max (m2, norm (A*x-b,1)) ;
m1 = 0 ;

for nrhs = 1:80
    b = sparse (rand (n,nrhs)) ;
    x = A\b ;
    e1 = norm (A*x-b,1) ;
    x = cholmod2 (A,b) ;
    e2 = norm (A*x-b,1) ;
    if (e2 > 1e-11)
	error ('!') ;
    end
    m1 = max (m1, e1) ;
    m2 = max (m2, e2) ;
end

for nrhs = 1:80
    b = sprandn (n, nrhs, 0.01) ;
    x = A\b ;
    % nnz (x) / (n*nrhs)
    e1 = norm (A*x-b,1) ;
    x = cholmod2 (A,b) ;
    e2 = norm (A*x-b,1) ;
    if (e2 > 1e-11)
	error ('!') ;
    end
    m1 = max (m1, e1) ;
    m2 = max (m2, e2) ;
end

fprintf ('maxerr %e %e\n', m1, m2) ;

if (m1 > 1e-11 | m2 > 1e-11)						    %#ok
    error ('!') ;
end

fprintf ('test4 passed\n') ;

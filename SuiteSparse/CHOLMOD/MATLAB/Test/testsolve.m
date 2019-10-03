function [x1,x2,e1,e2] = testsolve (A,b)
%TESTSOLVE test CHOLMOD and compare with x=A\b 
% [x1,x2,e1,e2] = testsolve (A,b) ;
% Compare CHOLMOD and MATLAB's x=A\b
% x1 = A\b, x2 = cholmod2(A,b), e1 = norm(A*x1-b), e2 = norm(A*x2-b)
% Example:
%   [x1,x2,e1,e2] = testsolve (A,b) ;
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('A: [n %6d real %d]    B: [sp:%d nrhs %d real %d]  ', ...
    size(A,1), isreal(A), issparse(b), size(b,2), isreal(b)) ;
tic
x1 = A\b ;
t1 = toc ;
tic
x2 = cholmod2(A,b) ;
t2 = toc ;
tic
e1 = norm (A*x1-b,1) ;
t3 = toc ;
e2 = norm (A*x2-b,1) ;
if (e2 == 0 | e1 == 0)							    %#ok
    e12 = 0 ;
else
    e12 = log2 (e1/e2) ;
end
if (t2 == 0)
    t12 = 1 ;
else
    t12 = t1 / t2 ;
end
if (t2 == 0)
    t32 = 1 ;								    %#ok
else
    t32 = t3 / t2 ;							    %#ok
end
fprintf (' [e1: %5.0e : %5.1f] [t1: %8.2f t2 %8.2f : %5.1f]\n', ...
    e1, e12, t1, t2, t12) ;
if (e2 > max (1e-8, 1e3*e1))
    error ('!') ;
end

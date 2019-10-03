function test23
%TEST23 test chol and cholmod2 on the sparse matrix used in "bench"
% Example:
%   test23
% See also cholmod_test

% Copyright 2007-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test23: test chol & cholmod2 on the sparse matrix used in "bench"\n');

n = 120 ;
A = delsq (numgrid ('L', n)) ;
b = sum (A)' ;

fprintf ('Using each method''s internal fill-reducing ordering:\n') ;

tic ;
x = A\b ;
t1 = toc ;
e1 = norm (A*x-b) ;

tic ;
x = cholmod2 (A,b) ; 
t2 = toc ;
e2 = norm (A*x-b) ;

fprintf ('MATLAB  x=A\\b      time: %8.4f  resid: %8.0e\n', t1, e1) ;
fprintf ('CHOLMOD x=A\\b      time: %8.4f  resid: %8.0e\n', t2, e2) ;
fprintf ('CHOLMOD speedup: %8.2f\n', t1/t2) ;

% get CHOLMOD's ordering (best of AMD and METIS)
p = analyze (A) ;
S = A (p,p) ;

tic ;
R = chol (S) ;
t1 = toc ;
x = R \ (R' \ b (p)) ;
x (p) = x ;
e1 = norm (A*x-b) ;

tic ;
L = lchol (S) ;
t2 = toc ;
x = L' \ (L \ b (p)) ;
x (p) = x ;
e2 = norm (A*x-b) ;

fprintf ('\nS = A(p,p) where p is CHOLMOD''s ordering:\n') ;
fprintf ('MATLAB  R=chol(S)  time: %8.4f  resid: %8.0e\n', t1, e1) ;
fprintf ('CHOLMOD L=lchol(S) time: %8.4f  resid: %8.0e\n', t2, e2) ;
fprintf ('CHOLMOD speedup: %8.2f\n', t1/t2) ;

% get MATLABS's ordering (symmmd in v7.0.4).  If that fails then use amd.
% A future version of MATLAB will remove symmmd, since it is declared
% "deprecated" in v7.0.4.
try % symmmd, use amd if it fails
    method = 'symmmd' ;
    p = symmmd (A) ;
catch
    % use AMD from SuiteSparse
    method = 'amd' ;
    fprintf ('\nsymmmd not available, using amd instead.\n') ;
    p = amd2 (A) ;
end
S = A (p,p) ;

tic ;
R = chol (S) ;
t1 = toc ;
x = R \ (R' \ b (p)) ;
x (p) = x ;
e1 = norm (A*x-b) ;

tic ;
L = lchol (S) ;
t2 = toc ;
x = L' \ (L \ b (p)) ;
x (p) = x ;
e2 = norm (A*x-b) ;

fprintf ('\nS = A(p,p) where p is MATLAB''s ordering in x=A\\b (%s):\n',method);
fprintf ('MATLAB  R=chol(S)  time: %8.4f  resid: %8.0e\n', t1, e1) ;
fprintf ('CHOLMOD L=lchol(S) time: %8.4f  resid: %8.0e\n', t2, e2) ;
fprintf ('CHOLMOD speedup: %8.2f\n', t1/t2) ;

fprintf ('\n\nWith no fill-reducing orderings:\n') ;
tic ;
R = chol (A) ;
t1 = toc ;
x = R \ (R' \ b) ;
e1 = norm (A*x-b) ;

tic ;
L = lchol (A) ;
t2 = toc ;
x = L' \ (L \ b) ;
e2 = norm (A*x-b) ;

fprintf ('MATLAB  R=chol(A)  time: %8.4f  resid: %8.0e\n', t1, e1) ;
fprintf ('CHOLMOD L=lchol(A) time: %8.4f  resid: %8.0e\n', t2, e2) ;
fprintf ('CHOLMOD speedup: %8.2f\n', t1/t2) ;

fprintf ('\n\nWith no fill-reducing orderings (as used in "bench"):\n') ;

spparms ('autommd',0) ;
tic ;
x = A\b ;
t1 = toc ;
e1 = norm (A*x-b) ;

tic ;
x = cholmod2 (A,b,0) ; 
t2 = toc ;
e2 = norm (A*x-b) ;

fprintf ('MATLAB  x=A\\b      time: %8.4f  resid: %8.0e\n', t1, e1) ;
fprintf ('CHOLMOD x=A\\b      time: %8.4f  resid: %8.0e\n', t2, e2) ;
fprintf ('CHOLMOD speedup: %8.2f\n', t1/t2) ;

spparms ('default') ;

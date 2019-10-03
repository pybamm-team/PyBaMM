function cholmod_updown_demo
%CHOLMOD_UPDOWN_DEMO a demo for CHOLMOD's update/downdate and row add/delete
%
% Provides a short demo of CHOLMOD's update/downdate and row add/delete
% functions.
%
% Example:
%   cholmod_updown_demo
%
% See also CHOLMOD_DEMO

%   Copyright 2015, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\n\n------------------------------------------------------------\n') ;
fprintf ('demo of CHOLMOD update/downdate and row add/delete functions\n') ;
fprintf ('------------------------------------------------------------\n\n') ;

% A is the matrix used in the MATLAB 'bench' demo
A = delsq (numgrid ('L', 300)) ;
n = size (A,1) ;
% make b small so we can use absolute error norms, norm (A*x-b)
b = rand (n,1) / n ;

% L*D*L' = A(p,p).  The 2nd argument is zero if A is positive definite.
[LD,ignore,p] = ldlchol (A) ;

% solve Ax=b using the L*D*L' factorization of A(p,p)
c = b (p) ;
x = ldlsolve (LD, c) ;
x (p) = x ;
err = norm (A*x-b) ;
fprintf ('norm (A*x-b) using ldlsolve:   %g\n', err) ;
if (err > 1e-12)
    error ('!')
end

% solve again, just using backslash
tic
x = A\b ;
t1 = toc ;
err = norm (A*x-b) ;
fprintf ('norm (A*x-b) using backslash:  %g\n', err) ;
fprintf ('time for x=A\\b: %g\n', t1) ; 
if (err > 1e-12)
    error ('!')
end

% check the norm of LDL'-S
[L,D] = ldlsplit (LD) ;
S = A (p,p) ;
err = norm (L*D*L' - S, 1) ;
fprintf ('nnz (L) for original matrix:   %d\n', nnz (L)) ;

% rank-1 update of the L*D*L' factorization
% A becomes A2=A+W*W', and S becomes S2=S+C*C'.  Some fillin occurs.
fprintf ('\n\n------------------------------------------------------------\n') ;
fprintf ('Update A to A+W*W'', and the permuted S=A(p,p) becomes S+C*C'':\n') ;
W = sparse ([1 2 3 152], [1 1 1 1], [5 -1 -1 -1], n, 1)
C = W (p,:)
tic
LD2 = ldlupdate (LD, C, '+') ;
t2 = toc ;
S2 = S + C*C' ;
A2 = A + W*W' ;
[L,D] = ldlsplit (LD2) ;
err = norm (L*D*L' - S2, 1) ;
fprintf ('norm (LDL-S2) after update:    %g\n', err) ;
fprintf ('nnz (L) after rank-1 update:   %d\n', nnz (L)) ;
if (err > 1e-12)
    error ('!')
end

% solve A2*x=b using the updated LDL factorization
tic
c = b (p) ;
x = ldlsolve (LD2, c) ;
x (p) = x ;
t3 = toc ;
err = norm (A2*x-b) ;
fprintf ('norm (A*x-b) using ldlsolve:   %g\n', err) ;
if (err > 1e-12)
    error ('!')
end
fprintf ('time to solve x=A\\b using update: %g\n', t2 + t3) ; 

% solve again, just using backslash
tic
x = A2\b ;
t1 = toc ;
err = norm (A2*x-b) ;
fprintf ('norm (A*x-b) using backslash:  %g\n', err) ;
if (err > 1e-12)
    error ('!')
end
fprintf ('time for x=A\\b: %g\n', t1) ; 
fprintf ('speedup of update vs backslash: %g\n', t1 / (t2 + t3)) ;

% invert the permutation
invp = zeros (1,n) ;
invp (p) = 1:n ;

% delete row 3 of A to get A3.  This corresponds to row invp(3) in S and LDL
fprintf ('\n\n------------------------------------------------------------\n') ;
fprintf ('Delete row 3\n') ;
k = 3 ;
pk = invp (k) ;
I = speye (n) ;
A3 = A ;
A3 (:,k) = I (:,k) ;
A3 (k,:) = I (k,:) ;
tic
LD3 = ldlrowmod (LD, pk) ;
t2 = toc ;

S3 = S ;
S3 (:,pk) = I (:,pk) ;
S3 (pk,:) = I (pk,:) ;
[L,D] = ldlsplit (LD3) ;
err = norm (L*D*L' - S3, 1) ;
fprintf ('norm (LDL-S3) after row del:   %g\n', err) ;
if (err > 1e-12)
    error ('!')
end

% solve A3*x=b using the modified LDL factorization
tic
c = b (p) ;
x = ldlsolve (LD3, c) ;     % x = S3\c
x (p) = x ;                 % now x = A3\b
t3 = toc ;
err = norm (A3*x-b) ;
fprintf ('norm (A3*x-b) after row del:   %g\n', err) ;
fprintf ('nnz (L) after row del:         %d\n', nnz (L)) ;
if (err > 1e-12)
    error ('!')
end

% solve again using backslash
tic
x = A3\b ;
t1 = toc ;
err = norm (A3*x-b) ;
fprintf ('norm (A3*x-b) with backslash:  %g\n', err) ;
fprintf ('time for x=A\\b with backslash %g\n', t1) ;
fprintf ('time for x=A\\b using rowdel:  %g\n', t2 + t3) ; 
fprintf ('speedup of rowdel vs backslash: %g\n', t1 / (t2 + t3)) ;

% add row 3 back to A3 to get A4.  This corresponds to row invp(3) in S and LDL
fprintf ('\n\n------------------------------------------------------------\n') ;
fprintf ('Add row 3 back\n') ;
W = sparse ([1 3 7 9], [1 1 1 1], [-1 8 -1 -1], n, 1) ;
A4 = A3 ;
A4 (:,k) = W ;
A4 (k,:) = W' ;
C = W (p) ;             % permuted version of row/column 3
S4 = S3 ;
S4 (:,pk) = C ;
S4 (pk,:) = C' ;

tic
LD4 = ldlrowmod (LD3, pk, C) ;
t2 = toc ;
fprintf ('nnz (L) after row add:         %d\n', nnz (L)) ;

[L,D] = ldlsplit (LD4) ;
err = norm (L*D*L' - S4, 1) ;
fprintf ('norm (LDL-S4) after row add:   %g\n', err) ;
if (err > 1e-12)
    error ('!')
end

% solve A4*x=b using the modified LDL factorization
tic
c = b (p) ;
x = ldlsolve (LD4, c) ;     % x = S4\c
x (p) = x ;                 % now x = A4\b
t3 = toc ;
err = norm (A4*x-b) ;
fprintf ('norm (A4*x-b) after row add:   %g\n', err) ;
if (err > 1e-12)
    error ('!')
end

% solve A4*x=b using backslash
tic
x = A4\b ;
t1 = toc ;
err = norm (A4*x-b) ;
fprintf ('norm (A4*x-b) with backslash:  %g\n', err) ;
if (err > 1e-12)
    error ('!')
end

fprintf ('time for x=A\\b with backslash %g\n', t1) ;
fprintf ('time for x=A\\b using rowadd:  %g\n', t2 + t3) ; 
fprintf ('speedup of rowadd vs backslash: %g\n', t1 / (t2 + t3)) ;


function [C,R,E,B,X, err] = spqr_gpu3 (ordering, A)
% info = spqr_gpu2 (ordering,A)
%   ordering: 1 colamd
%   ordering: 2 metis

% write the matrix to a file
mwrite ('A.mtx', A) ;

if (exist ('R.mtx','file'))
    delete ('R.mtx') ;
end
if (exist ('C.mtx','file'))
    delete ('C.mtx') ;
end
if (exist ('E.txt','file'))
    delete ('E.txt') ;
end

setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64:/usr/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib64')
if (ordering == 1)
    system ('tcsh demo_colamd3.sh') ;
else
    system ('tcsh demo_metis3.sh') ;
end

atanorm = norm (A'*A,1) ;

[m n] = size (A) ;
R = mread ('R.mtx') ;
C = mread ('C.mtx') ;
B = ones (m,1) ;
E = load ('E.txt') ;
S = A (:,E) ;
err (1) = norm (R'*R - S'*S, 1) / atanorm  ;
X = R\C ;
X (E) = X ;

[C2, R2] = qr (S, B, 0) ;
whos
err (2) = norm (R2'*R2 - S'*S, 1) / atanorm ;
X2 = R2\C2 ;
X2 (E) = X2 ;
err (3) = norm (X-X2) / norm (X2) ;

x = A\B ;
err (4) = norm (A'*(A*x-B))  / atanorm ;
err (5) = norm (A'*(A*X-B))  / atanorm ;
err (6) = norm (A'*(A*X2-B)) / atanorm ;


% normalize the diagonal of R
e = spdiags (sign (full (diag (R))), 0, n, n) ;
R = e*R ;
err (7) = norm (R'*R - S'*S, 1) / atanorm ;

e = spdiags (sign (full (diag (R2))), 0, n, n) ;
R2 = e*R2 ;
err (8) = norm (R2'*R2 - S'*S, 1) / atanorm ;


% err
% log10(err)

fprintf ('1: GPU    norm(R''R-S''S)               %12.4e\n', err (1)) ;
fprintf ('2: matlab norm(R''R-S''S)               %12.4e\n', err (2)) ;
fprintf ('3: GPU vs matlab norm (X-X2)/norm(X)  %12.4e\n', err (3)) ;
fprintf ('4: x=A\\b norm(A''(Ax-b))/norm(A''A)     %12.4e\n', err (4)) ;
fprintf ('5: GPU    norm(A''(Ax-b))/norm(A''A)    %12.4e\n', err (5)) ;
fprintf ('6: matlab norm(A''(Ax-b))/norm(A''A)    %12.4e\n', err (6)) ;
fprintf ('7: GPU    norm(R''R-S''S) nonneg        %12.4e\n', err (7)) ;
fprintf ('8: matlab norm(R''R-S''S) nonneg        %12.4e\n', err (8)) ;
fprintf ('\n-------------------------------------------------------\n') ;

if (err (2) > 1e-10 || err (5) > max (1e-12, 1e3 * err (4)))
    warning ('!')
end

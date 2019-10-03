function spqr_demo
%SPQR_DEMO short demo of SuiteSparseQR 
%
% Example:
%   spqr_demo
%
% See also SPQR, SPQR_SOLVE, SPQR_QMULT, SPQR_MAKE, SPQR_INSTALL.

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

help spqr
help spqr_solve
help spqr_qmult
help spqr_demo

% input ('Hit enter to start the SuiteSparseQR demo: ', 's') ;
fprintf ('\nTesting SuiteSparseQR functions ... please wait ...\n') ;

load west0479 ;
A = west0479 ;
rand ('state', 0) ;     %#ok
m = size (A,1) ;

clf

maxerr = 0 ;

for acomplex = 0:1

    if (acomplex)
        A = A + 1i * sprand (A) ;
    end
    anorm = norm (A,1) ;

    R1 = spqr (A) ;
    err = norm (R1'*R1 - A'*A, 1) / anorm^2 ;
    maxerr = max (maxerr, err) ;

    [Q,R,E] = spqr (A) ;
    err = norm (Q*R-A*E, 1) / anorm ;
    maxerr = max (maxerr, err) ;

    [H,R,E] = spqr (A, struct ('Q', 'Householder')) ;
    err = norm (spqr_qmult (H,R,1) - A*E, 1) / anorm ;
    maxerr = max (maxerr, err) ;

    if (acomplex)
        subplot (2,5,7)  ; spy (R1)  ; title ('R, no permutation (complex)') ;
        subplot (2,5,8)  ; spy (R)   ; title ('R with colamd (complex)') ;
        subplot (2,5,9)  ; spy (Q)   ; title ('Q with colamd (complex)') ;
        subplot (2,5,10) ; spy (H.H) ; title ('H with colamd (complex)') ;
    else
        subplot (2,5,1)  ; spy (A)   ; title ('A') ;
        subplot (2,5,2)  ; spy (R1)  ; title ('R, no permutation') ;
        subplot (2,5,3)  ; spy (R)   ; title ('R with colamd') ;
        subplot (2,5,4)  ; spy (Q)   ; title ('Q with colamd') ;
        subplot (2,5,5)  ; spy (H.H) ; title ('H with colamd') ;
    end
    drawnow

    % test spqr_solve: real/complex, sparse/full,
    % single/multiple right-hand-sides
    for bcomplex = 0:1
        for bsparse = 0:1
            for nrhs = 0:10
                if (bsparse)
                    b = sprand (m, nrhs, 0.1) ;
                else
                    b = rand (m, nrhs) ;
                end
                if (bcomplex)
                    b = b + 1i * sprand (b) ;
                end
                x = spqr_solve (A,b) ;
                err = norm (A*x-b,1) / max (anorm * norm(x,1) + norm (b,1), 1) ;
                maxerr = max (maxerr, err) ;
            end
        end
    end
end

fprintf ('\nQR maximum error: %g\n\n', maxerr) ;
if (maxerr > 1e-12)
    error ('One or more tests failed; error is high!') ;
end

% ------------------------------------------------------------------------------
% compare spqr_solve
% ------------------------------------------------------------------------------

fprintf ('Compare performance with MATLAB on a dense least-squares problem:\n');
A = rand (2000,1000) ;
b = rand (2000,1) ;
S = sparse (A) ;
fprintf ('\nA = rand (2000,1000) ;\nb = rand (2000,1) ;\nS = sparse (A) ;\n') ;

fprintf ('tic, x = spqr_solve(S,b) ; toc  ') ;
tic
x = spqr_solve (S, b) ;
t1 = toc ;
r1 = norm (A*x-b,1) ;
r1b = norm (A'*(A*x)-A'*b,1) / norm (A'*A,1) ;
fprintf ('%% time %8.3f residual %8.3e normal eqn resid %8.3e\n', t1,r1, r1b) ;

fprintf ('tic, x = A\\b ; toc              ') ;
tic
x = A\b ;
t2 = toc ;
r2 = norm (A*x-b,1) ;
r2b = norm (A'*(A*x)-A'*b,1) / norm (A'*A,1) ;
fprintf ('%% time %8.3f residual %8.3e normal eqn resid %8.3e\n', t2,r2,r2b) ;

fprintf ('tic, x = S\\b ; toc              ') ;
tic
x = S\b ;
t3 = toc ;
r3 = norm (A*x-b,1) ;
r3b = norm (A'*(A*x)-A'*b,1) / norm (A'*A,1) ;
fprintf ('%% time %8.3f residual %8.3e normal eqn resid %8.3e\n', t3,r3,r3b) ;

% ------------------------------------------------------------------------------
% compare spqr with 1000-by-2000 system
% ------------------------------------------------------------------------------

fprintf ('\nA = rand (1000,2000) ;\nS = sparse (A) ;\n') ;
A = rand (1000,2000) ;
S = sparse (A) ;

fprintf ('tic, R = spqr(S) ; toc          ') ;
tic
R = spqr (S) ;
t1 = toc ;
r1 = norm (R'*R-A'*A,1) / norm(A,1)^2 ;
fprintf ('%% time %8.3f error %8.3e\n', t1,r1) ;
clear R

fprintf ('tic, R = qr(A)   ; toc          ') ;
tic
R = qr (A) ;
t2 = toc ;
R = triu (R) ;
r2 = norm (R'*R-A'*A,1) / norm(A,1)^2 ;
fprintf ('%% time %8.3f error %8.3e\n', t2,r2) ;
clear R

fprintf ('tic, R = qr(S)   ; toc          ') ;
tic
R = qr (S) ;
t3 = toc ;
r3 = norm (R'*R-A'*A,1) / norm(A,1)^2 ;
fprintf ('%% time %8.3f error %8.3e\n', t3,r3) ;
clear R

% ------------------------------------------------------------------------------
% compare spqr with 100-by-20000 system
% ------------------------------------------------------------------------------

A = rand (100,20000) ;
S = sparse (A) ;
fprintf ('\nA = rand (100,20000) ;\nS = sparse (A) ;\n') ;

fprintf ('tic, R = spqr(S) ; toc          ') ;
tic
R = spqr (S) ;                                                              %#ok
t1 = toc ;
fprintf ('%% time %8.3f\n', t1) ;
clear R

fprintf ('tic, R = qr(A)   ; toc          ') ;
tic
R = qr (A) ;                                                                %#ok
t2 = toc ;
fprintf ('%% time %8.3f\n', t2) ;
clear R

% skip the old MATLAB QR ... it's way to slow ...
%   fprintf ('tic, R = qr(S)   ; toc          ') ;
%   try
%       tic
%       R = qr (S) ;                                                        %#ok
%       t3 = toc ;
%       fprintf ('%% time %8.3f\n', t3) ;
%   catch                                                                   %#ok
%       fprintf ('%% MATLAB sparse qr failed ...\n') ;
%   end
%   clear R

fprintf ('All spqr tests passed\n') ;

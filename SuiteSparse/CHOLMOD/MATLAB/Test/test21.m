function test21
%TEST21 test cholmod2 on diagonal or ill-conditioned matrices
% Example:
%   test21
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test21: test cholmod2 on diagonal or ill-conditioned matrices\n') ;

f = [
 72	% HB/bcsstm22
 315	% Bai/mhdb416
 64	% HB/bcsstm09
 71	% HB/bcsstm21
 1207	% Oberwolfach/t2dal_e
 354	% Boeing/crystm02
 1211	% Oberwolfach/t3dl_e
 ]' ;

for i = f

    Prob = UFget (i)							    %#ok
    A = Prob.A ;
    n = size (A,1) ;
    x = ones (n,2) ;
    b = A*x ;
    fprintf ('nnz: %d\n', nnz (A)) ;

    x1 = A\b ;
    x2 = cholmod2 (A,b) ;

    s = norm (A,1) * norm (x,1) + norm (b,1) ;
    resid1 = norm (A*x1-b,1) / s ; 
    resid2 = norm (A*x2-b,1) / s ; 

    err1 = norm (x-x1,1) ;
    err2 = norm (x-x2,1) ;

    fprintf ('MATLAB  resid %6.1e err %6.1e\n', resid1, err1) ;
    fprintf ('CHOLMOD resid %6.1e err %6.1e\n', resid2, err2) ;
    fprintf ('condest %6.1e\n', condest (A)) ;

end

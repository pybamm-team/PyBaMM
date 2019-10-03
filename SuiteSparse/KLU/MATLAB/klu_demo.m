function klu_demo
% KLU demo
%
% Example:
%   klu_demo
%
% See also klu, btf

% Copyright 2004-2009, Univ. of Florida

load west0479
A = west0479 ;

n = size (A,1) ;
b = rand (n,1) ;

clf
subplot (2,2,1) ;
spy (A)
title ('west0479') ;

subplot (2,2,2) ;
[p, q, r] = btf (A) ;
drawbtf (A, p, q, r) ;
title ('BTF form') ;

[x,info,c] = klu (A, '\', b) ;
matlab_condest = condest (A) ;
matlab_cond = cond (full (A)) ;
fprintf ('MATLAB condest: %g KLU condest: %g cond: %g\n', ...
    matlab_condest, c, matlab_cond) ;

fprintf ('\nKLU with scaling, AMD ordering and condition number estimate:\n') ;
[LU,info] = klu (A, struct ('ordering',0, 'scale', 1)) ;
x = klu (LU, '\', b) ;
resid = norm (A*x-b,1) / norm (A,1) ;
rgrowth = full (min (max (abs ((LU.R \ A (LU.p,LU.q)) - LU.F)) ./ ...
    max (abs (LU.U)))) ;
fprintf ('resid: %g KLU condest: %g rgrowth: %g\n', resid, c, rgrowth) ;
disp (info) ;

subplot (2,2,3) ;
spy (LU.L + LU.U + LU.F) ;
title ('KLU+AMD factors') ;

fprintf ('\nKLU with COLAMD ordering\n') ;
[LU,info] = klu (A, struct ('ordering',1)) ;
x = klu (LU, '\', b) ;
resid = norm (A*x-b,1) / norm (A,1) ;
fprintf ('resid: %g\n', resid) ;
disp (info) ;

subplot (2,2,4) ;
spy (LU.L + LU.U + LU.F) ;
title ('KLU+COLAMD factors') ;

fprintf ('\nKLU with natural ordering (lots of fillin)\n') ;
[x,info] = klu (A, '\', b, struct ('ordering',2)) ;
resid = norm (A*x-b,1) / norm (A,1) ;
fprintf ('resid: %g\n', resid) ;
disp (info) ;

try

    fprintf ('\nKLU with CHOLMOD(A''*A) ordering\n') ;
    [x,info] = klu (A, '\', b, struct ('ordering',3)) ;
    resid = norm (A*x-b,1) / norm (A,1) ;
    fprintf ('resid: %g\n', resid) ;
    disp (info) ;

    fprintf ('\nKLU with CHOLMOD(A+A'') ordering\n') ;
    [x,info] = klu (A, '\', b, struct ('ordering',4)) ;
    resid = norm (A*x-b,1) / norm (A,1) ;
    fprintf ('resid: %g\n', resid) ;
    disp (info) ;

catch
    fprintf ('KLU test with CHOLMOD skipped (CHOLMOD not installed)\n') ;
end

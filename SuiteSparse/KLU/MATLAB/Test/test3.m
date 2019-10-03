function test3
%test3: KLU test
% Example:
%   test3
% See also klu

% Copyright 2004-2012, University of Florida

h = waitbar (1/12, 'KLU test 3 of 5') ;

% rand ('state', 0) ;

load west0479
A = west0479 ;
% A = sparse (rand (4)) ;
% A (3:4, 1:2) = 0 ;

n = size (A,1) ;
b = rand (n,1) ;
spparms ('spumoni',2)
x = A\b ;
spparms ('spumoni',0)
fprintf ('MATLAB resid %g\n', norm (A*x-b,1)) ;

[LU,info,cond_estimate] = klu (A) ;
fprintf ('\nLU = \n') ; disp (LU) ;
fprintf ('\ninfo = \n') ; disp (info) ;
fprintf ('KLU condest    %g\n', cond_estimate) ;
matlab_condest = condest (A) ;
matlab_cond = cond (full (A)) ;
fprintf ('MATLAB condest %g cond %g\n', matlab_condest, matlab_cond) ;

for nrhs = 1:10
    waitbar (nrhs/12, h) ;
    b = rand (n,nrhs) ;
    x = klu (LU,'\',b) ;
    fprintf ('nrhs: %d resid: %g\n', ...
        nrhs, norm (A*x-b,1) / norm (A,1)) ;
end

[x,info,cond_estimate] = klu (A, '\', b) ;                                  %#ok
fprintf ('\ninfo = \n') ; disp (info) ;
fprintf ('KLU cond_estimate %g\n', cond_estimate) ;

waitbar (11/12, h) ;

[x,info] = klu (A, '\', b, struct ('ordering',1)) ;                         %#ok
fprintf ('\ninfo = \n') ; disp (info) ;
[x,info,cond_estimate] = klu (A, '\', b, struct ('ordering',2)) ;           %#ok
fprintf ('\ninfo = \n') ; disp (info) ;
try
    [x,info,cond_estimate] = klu (A, '\', b, struct ('ordering',3)) ;       %#ok
    fprintf ('\ninfo = \n') ; disp (info) ;
    [x,info,cond_estimate] = klu (A, '\', b, struct ('ordering',4)) ;       %#ok
    fprintf ('\ninfo = \n') ; disp (info) ;
catch me
    disp (me.message) ;
    fprintf ('test with CHOLMOD skipped (CHOLMOD or METIS not installed)\n') ;
end

close (h) ;

function cholmod_test (nmat, do_diary)
%CHOLMOD_TEST test the CHOLMOD mexFunctions
%
% Example:
% cholmod_test(nmat,do_diary)
%
% The UFget interface to the UF sparse matrix collection is required.
%
% nmat is optional.  If present, it is the # of matrices used in
%   tests 0, 8, 10, 11, 12, and 12.  tests 14 and 15 use 2*nmat matrices.
%   default nmat is 50.
%
% do_diary: 1 to save results in a diary, 0 otherwise.  Default 0.
%
% cholmod_demo: run tests on a few random matrices
% graph_demo: graph partitioning demo
% test0:  test most CHOLMOD functions
% test1:  test sparse2
% test2:  test sparse2
% test3:  test sparse on int8, int16, and logical
% test4:  test cholmod2 with multiple and sparse right-hand-sides
% test5:  test sparse2
% test6:  test sparse with large matrix, both real and complex, compare w/MATLAB
% test7:  test sparse2
% test8:  order many sparse matrices, test symbfact2, compare amd and metis
% test9:  test metis, etree, bisect, nesdis
% test10: test cholmod2's backslash on real and complex matrices
% test11: test analyze, compare CHOLMOD and MATLAB, save results in Results.mat
% test12: test etree2 and compare with etree
% test13: test cholmod2 and MATLAB on large tridiagonal matrices
% test14: test metis, symbfact2, and etree2
% test15: test symbfact2 vs MATLAB
% test16: test cholmod2 on a large matrix
% test17: test lchol on a few large matrices
% test18: test cholmod2 on a few large matrices
% test19: look for NaN's from lchol (caused by Intel MKL 7.x bug)
% test20: test symbfact2, cholmod2, and lu on a few large matrices
% test21: test cholmod2 on diagonal or ill-conditioned matrices
% test22: test chol and chol2 and singular and indefinite matrices
% test23: test chol and cholmod2 on the sparse matrix used in "bench"
% test24: test sdmult
% test25: test sdmult on a large matrix
% test26: test logical full and sparse matrices
% test27: test nesdis
% ltest:  test lxbpattern
% lxtest: test lsubsolve
%
% See also test0, test1, ... test28.

% This extensive test is not included:
% test28: test nesdis

% Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 2)
    do_diary = 0 ;
end

if (do_diary)
    diary off
    s = date ;
    t = clock ;
    s = sprintf ('diary cholmod_test_%s_%d-%d-%d.txt\n', s, t (4), t(5), fix(t(6)));
    eval (s) ;
end

fprintf ('Running CHOLMOD tests.\n') ;
help cholmod_test

test_path = pwd ;
% addpath (test_path) ;
cd ('..') ;
cholmod_path = pwd ;
addpath (cholmod_path)
cd ('../../AMD/MATLAB') ;
amd_path = pwd ;
addpath (amd_path)
cd ('../../COLAMD') ;
colamd_path = pwd ;
addpath (colamd_path)
cd ('../CCOLAMD') ;
ccolamd_path = pwd ;
addpath (ccolamd_path)
cd ('../CAMD/MATLAB') ;
camd_path = pwd ;
addpath (camd_path)

cd (test_path)
fprintf ('Added the following paths.  You may wish to add them\n') ;
fprintf ('permanently using the MATLAB pathtool command.\n') ;
fprintf ('%s\n', cholmod_path) ;
fprintf ('%s\n', amd_path) ;
fprintf ('%s\n', colamd_path) ;
fprintf ('%s\n', ccolamd_path) ;
fprintf ('%s\n', camd_path) ;

if (nargin < 1)
    nmat = 50 ;
end

try
    s = metis (sparse (1)) ;
    do_metis = 1 ;
catch
    fprintf ('METIS not installed\n') ;
    do_metis = 0 ;
end

tt = 35 ;

h = waitbar (0.5/tt, 'CHOLMOD demo:') ;

try

    cholmod_demo                    ; waitbar ( 2/tt, h, 'CHOLMOD graph demo');
    if (do_metis)
        graph_demo ;
    end
    waitbar ( 2/tt, h, 'CHOLMOD test0') ;
    test0 (nmat)                    ; waitbar ( 3/tt, h, 'CHOLMOD test1') ;
    test1                           ; waitbar ( 4/tt, h, 'CHOLMOD test2') ;
    test2                           ; waitbar ( 5/tt, h, 'CHOLMOD test3') ;
    test3                           ; waitbar ( 6/tt, h, 'CHOLMOD test4') ;
    test4                           ; waitbar ( 7/tt, h, 'CHOLMOD test5') ;
    test5                           ; waitbar ( 8/tt, h, 'CHOLMOD test6') ;
    test6                           ; waitbar ( 9/tt, h, 'CHOLMOD test7') ;
    test7                           ; waitbar (10/tt, h, 'CHOLMOD test8') ;

    if (do_metis)
        % these tests require METIS
        test8 (nmat)                ; waitbar (11/tt, h, 'CHOLMOD test9') ;
        test9 ;
    end

    waitbar (12/tt, h, 'CHOLMOD test10') ;
    test10 (nmat)                   ; waitbar (13/tt, h, 'CHOLMOD test11') ;
    test11 (nmat)                   ; waitbar (14/tt, h, 'CHOLMOD test12') ;
    test12 (nmat)                   ; waitbar (15/tt, h, 'CHOLMOD test13') ;
    test13                          ; waitbar (16/tt, h, 'CHOLMOD test14') ;

    if (do_metis)
        % this test requires METIS
        test14 (2*nmat) ;
    end

    waitbar (17/tt, h, 'CHOLMOD test15') ;
    test15 (2*nmat)                 ; waitbar (18/tt, h, 'CHOLMOD test16') ;
    test16                          ; waitbar (19/tt, h, 'CHOLMOD test17') ;
    test17                          ; waitbar (20/tt, h, 'CHOLMOD test18') ;
    test18                          ; waitbar (21/tt, h, 'CHOLMOD test19') ;
    test19                          ; waitbar (22/tt, h, 'CHOLMOD test20') ;
    test20                          ; waitbar (23/tt, h, 'CHOLMOD test21') ;
    test21                          ; waitbar (24/tt, h, 'CHOLMOD test22a') ;
    test22 (nmat)                   ; waitbar (25/tt, h, 'CHOLMOD test22b') ;
    test22 (0)                      ; waitbar (26/tt, h, 'CHOLMOD test23') ;
    test23                          ; waitbar (27/tt, h, 'CHOLMOD test24') ;
    test24                          ; waitbar (28/tt, h, 'CHOLMOD test25') ;
    test25                          ; waitbar (29/tt, h, 'CHOLMOD test26') ;
    test26 (do_metis)               ; waitbar (31/tt, h, 'CHOLMOD test27') ;

    if (do_metis)
        test27 ;
    end

    % this test requires METIS
    % test28 ;                      % (disabled)

    ltest                           ; waitbar (32/tt, h, 'CHOLMOD ltest') ;
    lxtest                          ; waitbar (33/tt, h, 'CHOLMOD lxtest') ;
    test29                          ; waitbar (34/tt, h, 'CHOLMOD test29') ;

    waitbar (tt/tt, h, 'CHOLMOD test done') ;
    fprintf ('=============================================================\n');
    fprintf ('all tests passed\n') ;

catch
    % out-of-memory is OK, other errors are not
    disp (lasterr) ;
    if (isempty (strfind (lasterr, 'Out of memory')))
        error (lasterr) ;                                                   %#ok
    else
        fprintf ('test terminated early, but otherwise OK\n') ;
    end
end

close (h) ;
if (do_diary)
    diary off
end

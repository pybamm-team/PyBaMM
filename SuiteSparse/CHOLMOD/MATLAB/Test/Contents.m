% CHOLMOD TEST functions
%
%   cholmod_test  - test the CHOLMOD mexFunctions
%   dg            - order and plot A*A', using CHOLMOD's nested dissection
%   n2            - script to test CHOLMOD septree function
%   nn            - Compare nesdis with metis, in both quality and run time
%   test0         - test most CHOLMOD functions
%   test1         - test sparse2
%   test2         - test sparse2
%   test3         - test sparse on int8, int16, and logical
%   test4         - test cholmod2 with multiple and sparse right-hand-sides
%   test5         - test sparse2
%   test6         - test sparse with large matrix, both real and complex
%   test7         - test sparse2
%   test8         - order a large range of sparse matrices, test symbfact2
%   test9         - test metis, etree, bisect, nesdis
%   test10        - test cholmod2's backslash on real and complex matrices
%   test11        - compare CHOLMOD and MATLAB, save results in Results.mat
%   test11results - analyze results from test11.m
%   test12        - test etree2 and compare with etree
%   test13        - test cholmod2 and MATLAB on large tridiagonal matrices
%   test14        - test metis, symbfact2, and etree2
%   test15        - test symbfact2 vs MATLAB
%   test16        - test cholmod2 on a large matrix
%   test17        - test lchol on a few large matrices
%   test18        - test cholmod2 on a few large matrices
%   test19        - look for NaN's from lchol (caused by Intel MKL 7.x bug)
%   test20        - test symbfact2, cholmod2, and lu on a few large matrices
%   test21        - test cholmod2 on diagonal or ill-conditioned matrices
%   test22        - test pos.def and indef. matrices
%   test23        - test chol and cholmod2 on the sparse matrix used in "bench"
%   test24        - test sdmult
%   test25        - test sdmult on a large matrix
%   test26        - test logical full and sparse matrices
%   test27        - test nesdis with one matrix (HB/west0479)
%   test28        - test nesdis
%   testmm        - compare mread and mmread for entire Matrix Market collection
%   testsolve     - test CHOLMOD and compare with x=A\b 
%   ltest         - test lxbpattern
%   lxtest        - test lsubsolve
%   ltest2        - test lsubsolve
%
% Example:
%   cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com


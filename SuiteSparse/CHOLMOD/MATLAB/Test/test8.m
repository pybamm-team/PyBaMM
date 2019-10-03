function test8 (nmat)
%TEST8 order a large range of sparse matrices, test symbfact2
% compare AMD and METIS
% Example:
%   test8(nmat)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test8: factorize a large range of sparse matrices\n') ;

% get list of test matrices

index = UFget ;

% GHS posdef test set (more or less)
f = find (...
    ((index.numerical_symmetry == 1 & index.isBinary) | (index.posdef)) ...
    & (index.nnzdiag == index.nrows) ...
    & (index.nrows > 10000 | index.nrows == 9000) ...
    & (index.nrows < 600000) & (index.nnz > index.nrows)) ;		    %#ok

% include small matrices
f = find (...
    ((index.numerical_symmetry == 1 & index.isBinary) | (index.posdef)) ...
    & (index.nnzdiag == index.nrows) ...
    & (index.nrows < 600000) & (index.nnz > index.nrows)) ;

for k = 1:length (f) 
    names {k} = index.Name {f(k)} ;			%#ok
end

[ignore i] = sort (names) ;

f = f (i) ;

% fprintf ('test matrices sorted by name:\n') ;
% for i = f
%     fprintf ('%4d: %-20s %-20s %12d %d\n', i,  ...
% 	index.Group {i}, index.Name {i}, index.nrows (i), index.posdef (i)) ;
% end

[ignore i] = sort (index.nrows (f)) ;
f = f (i) ;

if (nargin > 0)
    nmat = max (0,nmat) ;
    nmat = min (nmat, length (f)) ;
    f = f (1:nmat) ;
end

fprintf ('test matrices sorted by dimension:\n') ;
for i = f
    fprintf ('%4d: %-20s %-20s %12d %d\n', i,  ...
	index.Group {i}, index.Name {i}, index.nrows (i), index.posdef (i)) ;
end

junk = sparse (1) ;

% input ('hit enter to continue: ') ;

for k = 1:length (f)

    Problem = UFget (f(k)) ;
    A = Problem.A ; 
    fprintf ('\n================== Problem: %s  n: %d nnz: %d\n', ...
	Problem.name, size (A,1), nnz (A)) ;
    fprintf ('title: %s\n\n', Problem.title) ;
    clear Problem
    n = size (A,1) ;							    %#ok

    amd2 (junk) ;
    metis (junk) ;

    tic ;
    [p1,info] = amd2 (A) ;						    %#ok
    t1 = toc ;
    S1 = A (p1,p1) ;
    tic ;
    c1 = symbfact (S1) ;
    ts1 = toc ;
    tic ;
    d1 = symbfact (S1) ;
    ts2 = toc ;
    if (any (c1 ~= d1))
	error ('!')
    end
    fprintf ('symbfact time: MATLAB %9.4f  CHOLMOD %9.4f  speedup %8.2f\n', ...
	ts1, ts2, ts1/ts2) ;

    lnz1 = sum (c1) ;
    fl1 = sum (c1.^2) ;
    fprintf ('time: amd     %10.4f mnnz(L) %8.1f mfl %8.0f  fl/nnz(L) %8.1f\n', ...
	t1, lnz1/1e6, fl1 /1e6, fl1/lnz1) ;

    tic ;
    p2 = metis (A) ;
    t2 = toc ;
    S2 = A (p2,p2) ;
    c2 = symbfact (S2) ;
    lnz2 = sum (c2) ;
    fl2 = sum (c2.^2) ;

    fprintf ('time: metis   %10.4f mnnz(L) %8.1f mfl %8.0f  fl/nnz(L) %8.1f\n', ...
	t2, lnz2/1e6, fl2/1e6, fl2/lnz2) ;

    r = lnz2 / lnz1 ;							    %#ok
    fprintf ('\nmetis/amd time: %8.4f nnz(L): %8.4f\n', t2/t1, lnz2/lnz1) ;

    % save results
    lnz (k,1) = lnz1 ;			    %#ok
    lnz (k,2) = lnz2 ;			    %#ok
    fl1 (k,1) = fl1 ;			    %#ok
    fl2 (k,2) = fl2 ;			    %#ok
    t (k,1) = t1 ;			    %#ok
    t (k,2) = t2 ;			    %#ok

end

fprintf ('test8 passed\n') ;

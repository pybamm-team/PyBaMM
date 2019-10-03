function test15 (nmat)
%TEST15 test symbfact2 vs MATLAB
% Example:
%   test15(nmat)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');

index = UFget ;

% only test matrices with nrows = 109000 or less.  large ones nearly always
% cause a MATLAB segfault.
f = find (index.nrows < 109000 & index.ncols < 109000) ;

% sort by row /col dimension
s = max (index.nrows, index.ncols) ;
[ignore i] = sort (s (f)) ;
f = f (i) ;

if (nargin > 0)
    nmat = max (0,nmat) ;
    nmat = min (nmat, length (f)) ;
    f = f (1:nmat) ;
end

fprintf ('Matrices to test: %d\n', length (f)) ;

for i = f

    % try

	Problem = UFget (i) ;
	A = spones (Problem.A) ;
	[m n] = size (A) ;
	fprintf ('\n%4d: %-20s nrow: %6d ncol: %6d nnz: %10d\n', ...
	    i, Problem.name, m, n, nnz(A)) ;

	% warmup, for accurate timing
	etree (sparse (1)) ;
	etree2 (sparse (1)) ;
	amd2 (sparse (1)) ;
	symbfact (sparse (1)) ;
	symbfact2 (sparse (1)) ;

	% test symmetric case
	if (m == n)

	    % permute the matrix first
	    p = amd2 (A) ;
	    A = A (p,p) ;

	    % test with triu(A)
	    tic
	    co = symbfact (A) ;
	    t1 = toc ;
	    tic
	    co2 = symbfact2 (A) ;
	    t2 = toc ;

	    fprintf ('c=symbfact(A):         %10.4f %10.4f  speedup %8.2f lnz %d\n', ...
		t1, t2, t1/t2, sum (co)) ;

	    if (any (co ~= co2))
		error ('!') ;
	    end

	    tic
	    [co h parent post R] = symbfact (A) ;
	    t1 = toc ;
	    tic
	    [co2 h2 parent2 post2 R2] = symbfact2 (A) ;
	    t2 = toc ;

	    fprintf ('R=symbfact(A):         %10.4f %10.4f  speedup %8.2f\n',...
		t1, t2, t1/t2) ;

	    checkem(co,co2,parent,parent2,post,post2,R,R2,h,h2) ;

	    % test with tril(A)
	    tic
	    co = symbfact (A') ;
	    t1 = toc ;
	    tic
	    co2 = symbfact2 (A,'lo') ;
	    t2 = toc ;

	    fprintf (...
	    'c=symbfact(A''):        %10.4f %10.4f  speedup %8.2f lnz %d\n',...
		t1, t2, t1/t2, sum (co)) ;

	    if (any (co ~= co2))
		error ('!') ;
	    end

	    tic
	    [co h parent post R] = symbfact (A') ;
	    t1 = toc ;
	    tic
	    [co2 h2 parent2 post2 R2] = symbfact2 (A,'lo') ;
	    t2 = toc ;

	    fprintf (...
		'R=symbfact(A''):        %10.4f %10.4f  speedup %8.2f\n',...
		t1, t2, t1/t2) ;

	    checkem(co,co2,parent,parent2,post,post2,R,R2,h,h2) ;

	end

	% permute the matrix first
	p = colamd (A) ;
	[parent post] = etree2 (A (:,p), 'col') ;
	p = p (post) ;
	A = A (:,p) ;

	% test column case
	tic
	co = symbfact (A,'col') ;
	t1 = toc ;
	tic
	co2 = symbfact2 (A,'col') ;
	t2 = toc ;

	fprintf ('c=symbfact(A,''col''):   %10.4f %10.4f  speedup %8.2f lnz %d\n', ...
	    t1, t2, t1/t2, sum (co)) ;

	if (any (co ~= co2))
	    error ('!') ;
	end

	tic
	[co h parent post R] = symbfact (A,'col') ;
	t1 = toc ;
	tic
	[co2 h2 parent2 post2 R2] = symbfact2 (A,'col') ;
	t2 = toc ;

	fprintf ('R=symbfact(A,''col''):   %10.4f %10.4f  speedup %8.2f\n', ...
	    t1, t2, t1/t2) ;

	checkem(co,co2,parent,parent2,post,post2,R,R2,h,h2) ;

%    catch
%    	fprintf ('%d failed\n', i) ;
%    end
end

fprintf ('test15 passed\n') ;

%-------------------------------------------------------------------------------

function checkem(co,co2,parent,parent2,post,post2,R,R2,h,h2)
% checkem compare results from symbfact and symbfact2
if (any (co ~= co2))
    error ('count!') ;
end
if (any (parent ~= parent2))
    error ('parent!') ;
end
if (any (post ~= post2))
    error ('post!') ;
end
if (nnz (R2) ~= nnz (R))
    error ('lnz!') ;
end
if (h ~= h2)
    error ('h!') ;
end
% this may run out of memory
try % compute nnz(R-R2)
    err = nnz (R-R2) ;
catch
    err = -1 ;
    fprintf ('nnz(R-R2) not computed\n')  ;
end
if (err > 0)
    error ('R!') ;
end

function test0 (nmat)
%TEST0 test most CHOLMOD functions
% Example:
%   test0(nmat)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test0: test most CHOLMOD functions\n') ;

% This test requires UFget, the MATLAB interface to the UF sparse matrix
% collection.  You can obtain UFget from http://www.suitesparse.com.

try
    index = UFget ;
catch
    error ('Test aborted.  UF sparse matrix collection not available.\n') ;
end

f = find (index.posdef) ;
[ignore i] = sort (index.nrows (f)) ;
f = f (i) ;

rand ('state', 0) ;
randn ('state', 0) ;

doplots = 0 ;

if (doplots)
    clf
end

% skip = [937:939 1202:1211] ;
skip = 937:939 ;
if (nargin > 0)
    nmat = max (0,nmat) ;
    nmat = min (nmat, length (f)) ;
    f = f (1:nmat) ;
end

% f= 229

fprintf ('test matrices sorted by dimension:\n') ;
for i = f
    if (any (i == skip))
	continue
    end
    fprintf ('%4d: %-20s %-20s %12d %d\n', i,  ...
	index.Group {i}, index.Name {i}, index.nrows (i), index.posdef (i)) ;
end

% pause

for i = f

    if (any (i == skip))
	continue
    end

    % try

	Problem = UFget (i) ;
	A = Problem.A ; 
	fprintf ('\n================== Problem: %d: %s  n: %d nnz: %d\n', ...
	    i, Problem.name, size (A,1), nnz (A)) ;
	fprintf ('title: %s\n', Problem.title) ;
	clear Problem
	n = size (A,1) ;

        % use AMD from SuiteSparse
        tic
        p = amd2 (A) ;
        t0 = toc ;
        fprintf ('time: amd     %10.4f\n', t0) ;

	S = A (p,p) ;

	if (doplots)
	    subplot (3,2,1) ;   spy (A) ;	title ('A original') ;
	    subplot (3,2,2) ;   spy (S) ;	title ('A permuted') ;
	    drawnow ;
	end

	% ensure chol, chol2, and lchol are loaded, for more accurate timing
	R = chol2 (sparse (1)) ;	    %#ok
	R = chol (sparse (1)) ;		    %#ok
	R = lchol (sparse (1)) ;	    %#ok
	R = ldlchol (sparse (1)) ;	    %#ok
	R = ldlupdate (sparse (1), sparse (1)) ;	    %#ok
	c = symbfact (sparse (1)) ;	    %#ok

	tic ;
	L = lchol (S) ;
	t3 = toc ;
	if (doplots)
	    subplot (3,2,5) ;   spy (L) ;	title ('L=lchol') ;
	    drawnow ;
	end
	fprintf ('CHOLMOD time: L=lchol  %10.4f  nnz(L): %d\n', t3, nnz (L)) ;
	lnorm = norm (L, 1) ;

	err = ldl_normest (S, L) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end
	clear L

	tic ;
	R = chol2 (S) ;
	t2 = toc ;
	if (doplots)
	    subplot (3,2,3) ;   spy (R) ;	title ('R=chol2') ;
	    drawnow ;
	end
	fprintf ('CHOLMOD time: R=chol2  %10.4f  nnz(R): %d\n', t2, nnz (R)) ;

	err = ldl_normest (S, R') / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end
	clear R

	tic ;
	R = chol (S) ;
	t1 = toc ;
	fprintf ('MATLAB time:  R=chol   %10.4f  nnz(R): %d\n', t1, nnz (R)) ;
	if (doplots)
	    subplot (3,2,4) ;   spy (R) ;	title ('chol') ;
	    drawnow ;
	end

	err = ldl_normest (S, R') / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end
	clear R

	tic ;
	[count,h,parent,post,R] = symbfact (S) ;
	t7 = toc ;
	fprintf ('MATLAB [..,R]=symbfact %10.4f  nnz(R): %d\n', t7, nnz (R)) ;

	fprintf ('\nCHOLMOD speedup vs MATLAB chol:         R: %8.2f L: %8.2f\n\n', ...
	    t1/t2, t1/t3) ;

	fprintf ('\nCHOLMOD numeric lchol vs MATLAB symbfact:  %8.2f\n', t7/t3) ;

	clear R S

	% use AMD or METIS, doing the ordering in CHOLMOD
	tic
	[L,p,q] = lchol (A) ;
	t4 = toc ;
	fprintf ('CHOLMOD time: [L,,q]=lchol   %10.4f  nnz(L): %d\n', ...
	    t4, nnz (L)) ;
	if (doplots)
	    subplot (3,2,6) ;   spy (L) ;	title ('[L,p,q]=lchol') ;
	    drawnow ;
	end

	err = ldl_normest (A (q,q), L) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end
	clear L

	% try an LDL' factorization, LD has LDL' factorization of S = A(q,q)
	tic
	[LD,p,q] = ldlchol (A) ;
	t5 = toc ;
	fprintf ('CHOLMOD time: [L,,q]=ldlchol %10.4f  nnz(L): %d\n', ...
	    t5, nnz (LD)) ;
	[L,D] = ldlsplit (LD) ;
	S = A (q,q) ;

	err = ldl_normest (S, L, D) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end
	clear L D A

	% update the LDL' factorization (rank 1 to 8).  Pick a C that has
	% the same pattern as a random set of columns of L, so no fill-in
	% occurs.  Then add one arbitrary entry, to add some fill-in to L.
	k = 1 + floor (rand (1) * 8) ;
	cols = randperm (n) ;
	cols = cols (1:k) ;
	C = sprandn (LD (:,cols)) ;
	row = 1 + floor (rand (1) * n) ;
	C (row,1) = 1 ;

	if (~isreal (C) | ~isreal (LD))					    %#ok
	    fprintf ('skip update/downdate of complex matrix ...\n') ;
	    continue ;
	end

	tic
	LD2 = ldlupdate (LD, C) ;
	t = toc ;
	fprintf ('\nCHOLMOD time: rank-%d ldlupdate    %10.4f  nnz(L) %d', ...
	    k, t, nnz (LD2)) ;

	if (nnz (LD2) > nnz (LD))
	    fprintf ('  with fill-in\n') ;
	else
	    fprintf ('  no fill-in\n') ;
	end

	% check the factorization, LD2 has LDL' factorization of S+C*C'
	[L,D] = ldlsplit (LD2) ;
	err = ldl_normest (S + C*C', L, D) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end
	clear L D

	% downate the LDL' factorization, with just part of C
	% no change to the pattern occurs.
	k = max (1, floor (k/2)) ;
	C1 = C (:, 1:k) ;
	C2 = C (:, k+1:end) ;		%#ok
	tic
	LD3 = ldlupdate (LD2, C1, '-') ;
	t = toc ;
	clear LD2
	fprintf ('CHOLMOD time: rank-%d ldldowndate  %10.4f  nnz(L) %d', ...
	    k, t, nnz (LD3)) ;
	fprintf ('  no fill-in\n') ;

	% check the factorization, LD3 has LDL' factorization of A(q,q)+C2*C2'
	[L,D] = ldlsplit (LD3) ;
	S2 = S + C*C' - C1*C1' ;
	err = ldl_normest (S2, L, D) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end

	% now test resymbol
	LD4 = resymbol (LD3, S2) ;
	[L,D] = ldlsplit (LD4) ;
	err = ldl_normest (S2, L, D) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end
	fprintf ('after resymbol: %d\n', nnz (LD4)) ;

	% compare resymbol with ldlchol
	LD5 = ldlchol (S2) ;
	if (nnz (LD5) ~= nnz (LD4))
	    error ('!') ;
	end
	if (nnz (spones (LD5) - spones (LD4)) ~= 0)
	    error ('!') ;
	end

	b = rand (n,2) ;
	x = ldlsolve (LD4, b) ;
	err1 = norm (S2*x-b,1) / norm (S,1) ;

	fprintf ('CHOLMOD residual:  %6.1e\n', err1) ;

	x = S2\b ;
	err2 = norm (S2*x-b,1) / norm (S,1) ;
	fprintf ('MATLAB  residual:  %6.1e\n', err2) ;

	b = sprandn (n,3,0.4) ;
	x = ldlsolve (LD4, b) ;
	err1 = norm (S2*x-b,1) / norm (S,1) ;

	fprintf ('CHOLMOD residual:  %6.1e (sparse b)\n', err1) ;

	x = S2\b ;
	err2 = norm (S2*x-b,1) / norm (S,1) ;
	fprintf ('MATLAB  residual:  %6.1e (sparse b)\n', err2) ;

        % ----------------------------------------------------------------------
        % test the row delete
        k = max (1, fix (n/2)) ;
        tic
        LD6 = ldlrowmod (LD,k) ;
        t6 = toc ;
	fprintf ('\nCHOLMOD time: ldlrowmod, delete    %10.4f  nnz(L) %d', ...
	    t6, nnz (LD6)) ;

        I = speye (n) ;
        S2 = S ;
        S2 (k,:) = I (k,:) ;
        S2 (:,k) = I (:,k) ;

	% check the factorization, LD6 has LDL' factorization of S2
	[L,D] = ldlsplit (LD6) ;
	err = ldl_normest (S2, L, D) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end

        % test the row add, by adding the row back in again
        Ck = S (:,k) ;
        S2 (:,k) = Ck ;
        S2 (k,:) = Ck' ;

        tic
        LD7 = ldlrowmod (LD6,k,Ck) ;
        t7 = toc ;
	fprintf ('\nCHOLMOD time: ldlrowmod, add       %10.4f  nnz(L) %d', ...
	    t7, nnz (LD7)) ;

	% check the factorization, LD7 has LDL' factorization of S
	[L,D] = ldlsplit (LD7) ;
	err = ldl_normest (S, L, D) / lnorm ;
	if (err > 1e-6)
	    error ('!') ;
	end

        % ----------------------------------------------------------------------

    % catch
    %	fprintf ('failed\n') ;
    % end

    clear A S C L R LD LD2 LD3 D p q C1 C2 LD3 S2 LD4 b x LD5 I LDL6 LD7 Ck

end

fprintf ('test0 passed\n') ;

function test22(nmat)
%TEST22 test pos.def and indef. matrices
% Example:
% test22(nmat)
%
% if nmat <= 0, just test problematic matrices
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test22: test pos.def and indef. matrices\n') ;

index = UFget ;

[ignore f] = sort (index.nrows) ;

if (nargin > 0)
    problematic = (nmat <= 0) ;
    if (problematic)
	% Matrices for which MATLAB and CHOLMOD differ, for which the error
	% is high, or other issues arose during debugging
	fprintf ('testing matrices for which MATLAB and CHOLMOD differ\n') ;
	f = [
	    186 % HB/jgl011
	    109 % HB/curtis54
	    793 % Qaplib/lp_nug05
	    607 % LPnetlib/lp_brandy
	    707 % LPnetlib/lp_bore3d
	    231 % HB/plskz362
	    794 % Qaplib/lp_nug06
	    673 % LPnetlib/lp_scorpion
	   1156 % Sandia/oscil_dcop_45 (also 1157:1168)
	    795 % Qaplib/lp_nug07
	    796 % Qaplib/lp_nug08
	    260 % HB/well1033
	    261 % HB/well1850
	    230 % HB/plsk1919
	    649 % LPnetlib/lp_pds_02
	    660 % LPnetlib/lp_qap12
	    609 % LPnetlib/lp_cre_a
	    619 % LPnetlib/lp_dfl001
	    661 % LPnetlib/lp_qap15
	    650 % LPnetlib/lp_pds_06
	    379 % Cote/vibrobox
	    638 % LPnetlib/lp_ken_11
	    799 % Qaplib/lp_nug20
	    ]' ;
    else
	nmat = max (0,nmat) ;
	nmat = min (nmat, length (f)) ;
	f = f (1:nmat) ;
    end
end

skip = [
    811, ...		% Simon/appu, which is a random matrix
    937:939, ...	% large ND/ problems
    1157:1168 ...	% duplicates
    799			% rather large: Qaplib/lp_nug20
    ] ;

tlimit = 0.1 ;
fprintf ('test22: chol and chol2 are repeated so each take >= %g sec\n',tlimit);

klimit = 1 ;

% warmup, for more accurate timing
[R,p] = chol (sparse (1)) ;						    %#ok
[R,p] = chol2 (sparse (1)) ;						    %#ok
clear R p

for i = f

    if (any (i == skip))
	continue ;
    end

    Problem = UFget (i) ;
    A = Problem.A ;
    [m n] = size (A) ;
    fprintf ('\n================== %4d: Problem: %s  m: %d n: %d nnz: %d', ...
	i, Problem.name, m, n, nnz (A)) ;
    clear Problem ;

    try	% create a symmetric version of the matrix
	if (m == n)
	    if (nnz (A-A') > 0)
		A = A+A' ;
	    end
	else
	    A = A*A' ;
	end
    catch
	fprintf ('skip\n') ;
	continue
    end

    fprintf (' %d\n', nnz (A)) ;

    p = amd2 (A) ;
    A = A (p,p) ;
    anorm = norm (A,1) ;

    % Run each code for at least 'tlimit' seconds

    % MATLAB
    k = 0 ;
    t1 = 0 ;
    while (t1 < tlimit & k < klimit)					    %#ok
	tic ;
	[R1,p1] = chol (A) ;
	t = toc ;
	t1 = t1 + t ;
	k = k + 1 ;
    end
    t1 = t1 / k ;

    % CHOLMOD
    k = 0 ;
    t2 = 0 ;
    while (t2 < tlimit & k < klimit)					    %#ok
	tic ;
	[R2,p2] = chol2 (A) ;
	t = toc ;
	t2 = t2 + t ;
	k = k + 1 ;
    end
    t2 = t2 / k ;

    if (klimit == 1)
	rmin = full (min (abs (diag (R2)))) ;
	rmax = full (max (abs (diag (R2)))) ;
	if (p2 ~= 0 | isnan (rmin) | isnan (rmax) | rmax == 0)		    %#ok
	    rcond = 0 ;
	else
	    rcond = rmin / rmax ;
	end
	fprintf ('rcond: %30.20e\n', rcond) ;
    end

    if (p1 == 1)
	% MATLAB does not follow its own definitions.  If p is 1, then R is
	% supposed to be 0-by-n, not 0-by-0.  CHOLMOD fixes this bug.
	% Here, A is m-by-m
	R1 = sparse (0,m) ;
    end

    kerr = 0 ;
    if (p1 ~= p2)
	% MATLAB and CHOLMOD don't agree.  See if both are correct,
	% because differences in roundoff errors can make one go
	% a little farther than the other.

	% if p1 is zero, it means MATLAB was fully successful
	k1 = p1 ;
	if (k1 == 0)
	    k1 = n ;
	end

	% if p2 is zero, it means CHOLMOD was fully successful
	k2 = p2 ;
	if (k2 == 0)
	    k2 = n ;
	end

	if (k1 > k2)
	    % MATLAB went further than CHOLMOD. This is OK if MATLAB found
	    % a small entry where CHOLMOD stopped.
	    k = k2 ;
	    kerr = R1 (k,k) ;
	    % now reduce R1 in size, to compare with R2
	    R1 = R1 (1:k2-1,:) ;
	else
	    % CHOLMOD went further than MATLAB. This is OK if CHOLMOD found
	    % a small entry where MATLAB stopped.
	    k = k1 ;
	    kerr = R2 (k,k) ;
	    % now reduce R2 in size, to compare with R1
	    R2 = R2 (1:k1-1,:) ;
	end
    end

    err = norm (R1-R2,1) / max (anorm,1) ;
    fprintf ('p: %6d %6d MATLAB: %10.4f CHOLMOD: %10.4f speedup %6.2f err:', ...
	p1, p2, t1, t2, t1/t2) ;

    if (err == 0)
	fprintf ('      0') ;
    else
	fprintf (' %6.0e', err) ;
    end

    if (kerr == 0)
	fprintf ('      0\n') ;
    else
	fprintf (' %6.0e\n', kerr) ;
    end

%    if (err > 1e-6)
%	error ('!') ;
%    end

% pause
    clear R1 R2 p1 p2 p A

end

fprintf ('test22: all tests passed\n') ;

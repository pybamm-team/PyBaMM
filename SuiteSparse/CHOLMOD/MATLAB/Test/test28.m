function test28
%TEST28 test nesdis
% Example:
%   test28
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

index = UFget ;

[ignore f] = sort (index.nnz) ;

% f = find (index.nrows < index.ncols) ;
% [ignore i] = sort (index.nnz (f)) ;
% f = f (i) ;

f = f (1:100) ;

for i = f

    try

	Prob = UFget (i, index) ;
	A = spones (Prob.A) ;
	[m n] = size (A) ;

	if (m < n)
	    A = A*A' ;
	elseif (m > n) ;
	    A = A'*A ;
	else
	    A = A+A' ;
	end

	% default: do not split connected components
	[p1 cp1 cmem1] = nesdis (A) ;					    %#ok

	% order connected components separately
	[p2 cp2 cmem2] = nesdis (A, 'sym', [200 1]) ;			    %#ok
	c1 = symbfact (A (p1,p1)) ;
	c2 = symbfact (A (p2,p2)) ;
	lnz1 = sum (c1) ;
	lnz2 = sum (c2) ;
	fprintf ('%35s %8d %8d ', Prob.name, lnz1, lnz2)
	if (lnz1 == lnz2)
	    fprintf ('        1\n') ;
	else
	    fprintf (' %8.3f\n', lnz1/lnz2) ;
	end

	subplot (2,3,1) ; spy (A) ;
	subplot (2,3,2) ; spy (A (p1,p1)) ;
	subplot (2,3,3) ; treeplot (cp1) ;
	subplot (2,3,5) ; spy (A (p2,p2)) ;
	subplot (2,3,6) ; treeplot (cp2) ;
	drawnow

    catch
	fprintf ('%4d failed\n', i) ;
    end
end

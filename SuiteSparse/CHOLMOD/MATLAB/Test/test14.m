function test14 (nmat)
%TEST14 test metis, symbfact2, and etree2
% Example:
%   test14(nmat)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test14: test metis, symbfact2, and etree2\n') ;

index = UFget ;

[ignore f] = sort (max (index.nrows, index.ncols)) ;

% f1 = find (max (index.nrows (f), index.ncols (f)) > 55500) ;
% f1 = f1 (1) ;
% f = f (f1:end) ;

% These bugs show up when Common->metis_memory is set to zero:
% skip = [ 1298 ] ; % runs out of memory in metis(A,'row')
skip =  1257  ;   %#ok	% GHS_psdef/crankseg_1: segfault in metis(A,'row') ;
skip =  850  ;    %#ok	% Chen/pkustk04: segfault in metis(A,'row') ;
skip = [ ] ;	  %#ok

if (nargin > 0)
    nmat = max (0,nmat) ;
    nmat = min (nmat, length (f)) ;
    f = f (1:nmat) ;
end

for i = f

    fprintf ('%d:\n', i) ;
    if (any (skip == i))
	fprintf ('skip %s / %s\n', index.Group {i}, index.Name {i}) ;
	continue
    end

    % try

	Prob = UFget (i)						    %#ok
	A = Prob.A ;
	[m n] = size (A) ;

	if (m == n)
	    S = spones (A) ; 
	else
	    n = min (m,n) ;
	    S = spones (A (1:n,1:n)) ;
	end

	try % compute nnz(S*S')
	    nzaat = nnz (S*S') ;
	catch
	    nzaat = -1 ;
	end
	try % compute nnz(S'*S)
	    nzata = nnz (S'*S) ;
	catch
	    nzata = -1 ;
	end
	S = S | S' ;

	fprintf ('nnz(A)    %d\n', nnz (A)) ;
	fprintf ('nnz(S)    %d\n', nnz (S)) ;
	fprintf ('nnz(A*A'') %d\n', nzaat) ;
	fprintf ('nnz(A''*A) %d\n', nzata) ;

	fprintf ('metis (S):\n') ;     p1 = metis (S) ;
	fprintf ('metis (A,row):\n') ; p2 = metis (A, 'row') ;
	fprintf ('metis (A,col):\n') ; p3 = metis (A, 'col') ;

	fprintf ('turning off postorder:\n') ;
	fprintf ('metis (S):\n') ;     n1 = metis (S, 'sym', 'no postorder') ;
	fprintf ('metis (A,row):\n') ; n2 = metis (A, 'row', 'no postorder') ;
	fprintf ('metis (A,col):\n') ; n3 = metis (A, 'col', 'no postorder') ;

	fprintf ('analyzing results:\n') ;

	[pa1 po1] = etree2 (S (n1,n1)) ;
	[pa2 po2] = etree2 (A (n2,:), 'row') ;
	[pa3 po3] = etree2 (A (:,n3), 'col') ;

	q1 = n1 (po1) ;
	q2 = n2 (po2) ;
	q3 = n3 (po3) ;

	if (any (p1 ~= q1))
	    error ('1!') ;
	end

	if (any (p2 ~= q2))
	    error ('2!') ;
	end

	if (any (p3 ~= q3))
	    error ('3!') ;
	end

	s1 = symbfact2 (S (p1,p1)) ;
	s2 = symbfact2 (A (p2,:), 'row') ;
	s3 = symbfact2 (A (:,p3), 'col') ;

	t1 = symbfact2 (S (n1,n1)) ;
	t2 = symbfact2 (A (n2,:), 'row') ;
	t3 = symbfact2 (A (:,n3), 'col') ;

	if (any (s1 ~= t1 (po1)))
	    error ('s1!') ;
	end

	if (any (s2 ~= t2 (po2)))
	    error ('s2!') ;
	end

	if (any (s3 ~= t3 (po3)))
	    error ('s3!') ;
	end

    % catch
    %	fprintf ('%d failed\n') ;
    % end
end


fprintf ('test14 passed\n') ;

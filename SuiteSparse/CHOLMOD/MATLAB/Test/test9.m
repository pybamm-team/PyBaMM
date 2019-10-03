function test9
%TEST9 test metis, etree, bisect, nesdis
% Example:
%   test9
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test9: test metis, etree, bisect, nesdis\n') ;

% Prob = UFget ('LPnetlib/lp_qap15') ;
Prob = UFget ('HB/bcsstk15')						    %#ok
A = Prob.A ;
C = A'*A ;
R = A*A' ;

fprintf ('\nmetis:\n') ;
tic ; p0 = metis (R) ; toc						    %#ok
% [pa po] = etree2 (R (p0,p0)) ; sparse (po - (1:size(R,1)))

tic ; p1 = metis (C) ; toc						    %#ok
% [pa po] = etree2 (C (p1,p1)) ; sparse (po - (1:size(C,1)))

tic ; p2 = metis (C, 'sym') ; toc					    %#ok
% [pa po] = etree2 (C (p1,p1)) ; sparse (po - (1:size(C,1)))

tic ; p3 = metis (A, 'row') ; toc					    %#ok
% [pa po] = etree2 (A (p1,:), 'row') ; sparse (po - (1:size(A,1)))

tic ; p4 = metis (A, 'col') ; toc					    %#ok
% [pa po] = etree2 (A (:,p1), 'col') ; sparse (po - (1:size(A,2)))

fprintf ('\nmetis(A):\n') ;
[m n] = size(A) ;
if (m == n)
    if (nnz (A-A') == 0)
	tic ; p5 = metis (A) ; toc
	% spy (A (p5,p5)) ;
	[ignore q] = etree (A(p5,p5)) ;
	p5post = p5 (q) ;						    %#ok
	% spy (A (p5post,p5post)) ;
	lnz0 = sum (symbfact (A (p5,p5)))				    %#ok
    end
end

fprintf ('\namd:\n') ;
if (m == n)
    if (nnz (A-A') == 0)
	tic ; z0 = amd2 (A) ; toc					    %#ok
	lnz = sum (symbfact (A (z0,z0)))				    %#ok
    end
end

fprintf ('\nbisect:\n') ;
tic ; s0 = bisect (R) ; toc						    %#ok
tic ; s1 = bisect (C) ; toc						    %#ok
tic ; s2 = bisect (C, 'sym') ; toc					    %#ok
tic ; s3 = bisect (A, 'row') ; toc					    %#ok
tic ; s4 = bisect (A, 'col') ; toc					    %#ok


fprintf ('\nnested dissection:\n') ;
tic ; [c0 cp0 cmem0] = nesdis (R) ; toc					    %#ok
tic ; [c1 cp1 cmem1] = nesdis (C) ; toc					    %#ok
tic ; [c2 cp2 cmem2] = nesdis (C, 'sym') ; toc				    %#ok
tic ; [c3 cp3 cmem3] = nesdis (A, 'row') ; toc				    %#ok
tic ; [c4 cp4 cmem4] = nesdis (A, 'col') ; toc				    %#ok

fprintf ('\nnested_dissection(A):\n') ;
if (m == n)
    if (nnz (A-A') == 0)
	tic ; c5 = nesdis (A) ; toc					    %#ok
	lnz1 = sum (symbfact (A (c5,c5)))				    %#ok
    end
end

fprintf ('test9 passed\n') ;

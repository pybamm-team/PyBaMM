function test10 (nmat)
%TEST10 test cholmod2's backslash on real and complex matrices
% Example:
%   test10(nmat)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test10: test cholmod2''s backslash\n') ;

rand ('state',0) ;
randn ('state',0) ;

index = UFget ;
f = find (index.posdef) ;
[ignore i] = sort (index.nrows (f)) ;
f = f (i) ;

% start after nd6k
% f = f ((find (f == 937) + 1):end) ;

skip = [937:939 1202:1211] ;
if (nargin > 0)
    nmat = max (0,nmat) ;
    nmat = min (nmat, length (f)) ;
    f = f (1:nmat) ;
end

fprintf ('test matrices sorted by dimension:\n') ;
for i = f
    if (any (i == skip))
	continue
    end
    fprintf ('%4d: %-20s %-20s %12d %d\n', i,  ...
	index.Group {i}, index.Name {i}, index.nrows (i), index.posdef (i)) ;
end


for nn = f
    % for nn = 23

    if (any (nn == skip))
	continue
    end

    % try

	for complexity = 0:1

	    if nn < 0
		n = -nn ;
		A = rand (n) + (complexity * rand(n) * 1i) ;
		A=A*A' ;
		full (A)
		A = sparse (A) ;

	    elseif (nn == 0)

		i = 1i ;
		A = [ 11  4-i 1+i 2+2*i 
		      4+i  22  0   0
		      1-i  0  33   0
		      2-2*i 0 0   44 ] ;
		A = sparse (A) ;
		p = [4 3 2 1] ;			%#ok
		full (A)
		A = sparse (A) ;

	    else

		if (~complexity)
		    nn					%#ok
		    Prob = UFget (nn)			%#ok
		end
		A = Prob.A ;
		if (complexity)
		    A = A / norm(A,1) ;
		    Z = .1 * sprandn (A) * 1i ;
		    Z = Z+Z' ;
		    A = A + Z ;
		    A = A + norm(A,1) * speye (size(A,1)) ;
		end
		n = size (A,1) ;
	    end

	    for sparsity = 0:1

		if (sparsity)
		    b = sprandn (n,4,0.1) ;
		else
		    b = rand (n,4) ;
		end

		b1 = b (:,1) ;

		[x1,x2,e1,e2] = testsolve (A,b1) ;	%#ok
		[x1,x2,e1,e2] = testsolve (A,b) ;	%#ok

		if (sparsity)
		    b = sprandn (n,9,0.1) ;
		else
		    b = rand (n,9) ;
		end

		[x1,x2,e1,e2] = testsolve (A,b) ;	%#ok

		if (sparsity)
		    b = sprandn (n,9,0.1) + sprandn (n,9,0.1)*1i ;
		else
		    b = rand (n,9) + rand(n,9)*1i ;
		end

		b1 = b (:,1) ;

		[x1,x2,e1,e2] = testsolve (A,b1) ;	%#ok
		[x1,x2,e1,e2] = testsolve (A,b) ;	%#ok

	    end
	end
    % catch
    % 	fprintf (' failed\n') ;
    % end
end

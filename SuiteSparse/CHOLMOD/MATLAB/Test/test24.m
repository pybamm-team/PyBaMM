function test24
%TEST24 test sdmult
% Example:
%   test24
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test24: test sdmult\n') ;

rand ('state', 0) ;
randn ('state', 0) ;
maxerr = 0 ;

for trials = 1:1000

    sm = fix (20 * rand (1)) ;
    sn = fix (20 * rand (1)) ;
    fn = fix (20 * rand (1)) ;

    for complexity = 0:1
	for transpose = 0:1
	    if (transpose)
		fm = sm ;
	    else
		fm = sn ;
	    end
	    S = sprand (sm,sn,0.5) ;
	    F = rand (fm,fn) ;

	    if (complexity)
		S = S + 1i * sprand (S) ;
		F = F + 1i * rand (fm,fn) ;
	    end

	    % MATLAB does not support empty complex matrices
	    if (isempty (S) | isempty (F))				    %#ok
		S = sparse (real (S)) ;
		F = real (F) ;
	    end

	    C = sdmult (S,F,transpose) ;

	    if (transpose)
		D = S'*F ;
	    else
		D = S*F ;
	    end

	    err = norm (C-D,1) ;
	    maxerr = max (err, maxerr) ;
	    if (err > 1e-13)
		error ('!') ;
	    end
	end
    end
end

fprintf ('test 24 passed, maxerr %g\n', maxerr) ;

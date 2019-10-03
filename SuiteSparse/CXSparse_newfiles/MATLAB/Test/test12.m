function test12
%TEST12 test cs_qr and compare with svd
%
% Example:
%   test12
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('test 12\n') ;
rand ('state',0) ;
% A = rand (3,4)

for trial = 1:100
    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    d = .1 * rand (1) ;
    A = sprandn (m,n,d) ;
    if (m < n)
        continue ;
    end
    if (m == 0 | n == 0)                                                    %#ok
        continue ;
    end

    for cmplex = 0:double(~ispc)
        if (cmplex)
            A = A + 1i * sprand (A) ;
        end

        fprintf ('m %d n %d nnz %d\n', m, n, nnz(A)) ;
        [V,Beta,p,R] = cs_qr (A) ;

        s1 = svd (full (A)) ;
        s2 = svd (full (R)) ;
        s2 = s2 (1:length(s1)) ;
        err = norm (s1-s2) ; 
        if (length (s1) > 1)
            err = err / s1 (1) ;
        end
        fprintf ('err %g\n', err) ;
        if (err > 1e-12)
            error ('!') ;
        end
    end
end

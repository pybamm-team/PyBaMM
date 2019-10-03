function err = test_all_cod
%TEST_ALL_COD test the COD factorization
%
% Example
%   test_all_cod
%
% See also test_cod

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

err = 0 ;
for m = 0:10
    for n = 0:10
        for sp = 0:1
            for im = 0:1
                A = sprand (m, n, 0.2) ;
                if (im)
                    A = A + 1i*sprand (m, n, 0.1) ;
                end
                if (~sp)
                    A = full (A) ;
                end
                err = max (err, test_cod (A)) ;
                err = max (err, test_cod (A, 1e-12)) ;
            end
        end
    end
end

fprintf ('test COD, COD_SPARSE, and RQ: error %g\n', err) ;

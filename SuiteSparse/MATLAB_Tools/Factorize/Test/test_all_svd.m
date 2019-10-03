function err = test_all_svd
%TEST_ALL_SVD tests the svd factorization method for a range of problems.
%
% Example
%   test_all_svd
%
% See also test_all.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

reset_rand ;
err = test_svd ;

err = max (err, test_svd (ones (3))) ;
A = zeros (3) ;
F = factorize (A, 'svd') ;
nrm = norm (F) ;
err = max (err, nrm) ;
nrm = norm (inverse (F)) ;
c = cond (F,1) ;
if (~isequal (nrm, inf) || ~isequal (c, inf))
    error ('svd test failure') ;
end

for sp = 0:1
    for n = 0:6
        for m = 0:6
            for im = 0:1
                A = rand (m,n) ;
                if (im)
                    A = A + 1i*rand (m,n) ;
                end
                if (sp)
                    A = sparse (A) ;
                end
                err = max (err, test_svd (A)) ;
            end
        end
        fprintf ('\n') ;
    end
end

for n = 2:7
    A = gallery ('chow', n, pi, 2) ;
    err = max (err, test_svd (A)) ;
    err = max (err, test_svd (A')) ;
    err = max (err, test_svd (gallery ('clement', n, 0))) ;
    err = max (err, test_svd (gallery ('clement', n, 1))) ;
    A = sprandn (n, 2*n, 0.2) ;
    err = max (err, test_svd (A)) ;
    err = max (err, test_svd (A')) ;
end
fprintf ('\n') ;
err = max (err, test_svd (gallery ('condex', 4, 1))) ;
err = max (err, test_svd (gallery ('condex', 3, 2))) ;
err = max (err, test_svd (gallery ('condex', 7, 3))) ;
err = max (err, test_svd (gallery ('condex', 8, 4))) ;
err = max (err, test_svd (gallery ('dorr', 20))) ;
err = max (err, test_svd (gallery ('frank', 20))) ;
err = max (err, test_svd (gallery ('gearmat', 20))) ;
err = max (err, test_svd (gallery ('lauchli', 20))) ;
err = max (err, test_svd (gallery ('neumann', 6^2))) ;
fprintf ('\ntest_all_svd error so far: %g\n', err) ;

if (err > 1e-6)
    error ('error too high') ;
end

fprintf ('Testing on gallery (''randsvd'',50) matrices:\n') ;
err = max (err, test_svd (gallery ('randsvd', 50))) ;
err = max (err, test_svd (gallery ('randsvd', 50, 10/eps))) ;
fprintf ('\nFinal test_all_svd error: %g\n', err) ;

if (err > 1e-6)
    error ('error too high') ;
end


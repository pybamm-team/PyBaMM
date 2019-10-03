function err = test_functions
%TEST_FUNCTIONS test various functions applied to a factorize object
% on a set of matrices
%
% Example:
%   test_functions
%
% See also test_all, factorize, inverse, mldivide

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\n----- Test functions:\n') ;
reset_rand ;
err = 0 ;

err = max (err, test_function ([ ])) ;
err = max (err, test_function ([ ], 'ldl', 1)) ;
err = max (err, test_function (eye (4))) ;
err = max (err, test_function (eye (4,3))) ;
err = max (err, test_function (eye (3,4))) ;

err = max (err, test_function (inf*eye (4))) ;
err = max (err, test_function (inf*eye (4,3))) ;
err = max (err, test_function (inf*eye (3,4))) ;

err = max (err, test_function (nan*eye (4))) ;
err = max (err, test_function (nan*eye (4,3))) ;
err = max (err, test_function (nan*eye (3,4))) ;

A = rand (3) ;
err = max (err, test_function (A'*A, [ ], 1)) ;
err = max (err, test_function (A, 'svd', 1)) ;
err = max (err, test_function) ;

A = rand (10) ;
A = A' + A + 20*eye(10) ;
err = max (err, test_function (A, 'svd', 1)) ;
err = max (err, test_function (A, 'chol', 1)) ;

for imaginary = 0:1
    for m = 1:6
        for n = 1:6
            fprintf ('.') ;
            A = rand (m,n) ;
            if (imaginary)
                A = A + 1i * rand (m,n) ;
            end
            err = max (err, test_function (A)) ;
            A = sparse (A) ;
            err = max (err, test_function (A)) ;
            if (m < n)
                A = A*A'  ;
            else
                A = A'*A  ;
            end
            err = max (err, test_function (A)) ;
            A = full (A) ;
            err = max (err, test_function (A)) ;
        end
    end
end

fprintf ('\ntest_functions, max error: %g\n', err) ;

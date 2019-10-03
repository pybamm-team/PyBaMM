function test_all (performance)
%TEST_ALL test the Factorize package (factorize, inverse, and related)
%
% If you have editted the Factorize package, type "clear classes" before
% running any tests.
%
% Example
%   test_all                % run all tests
%   test_all (0) ;          % do not run performance tests
%
% See also factorize, inverse, test_performance, test_accuracy, test_disp,
% test_errors

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    performance = 1 ;
end

help test_all
reset_rand ;
test_disp ;                 % test disp(F)
test_errors ;               % test error handling for invalid matrices
err1 = test_functions ;     % functionality tests
err2 = test_accuracy ;      % test accuracy on a range of problems
err3 = test_all_svd ;       % test SVD factorization
err4 = test_all_cod ;       % test COD, COD_SPARSE, and RQ factorizations
err = max ([err1 err2 err3 err4]) ;
if (performance)
    err = max (err, test_performance) ;         % performance tests
end
fprintf ('\nAll tests passed, maximum error OK: %g\n', err) ;

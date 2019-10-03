% Test code for the FACTORIZE object.
%
% To run all the tests, use test_all.  The SPQR mexFunction from SuiteSparse
% is required.  The FACTORIZE method works without SPQR, but it will not use
% COD for sparse matrices in that case (which this test relies upon).  The
% output of this test in MATLAB R2011a is given in test_all.txt.
%
% Files
%   test_accuracy    - test the accuracy of the factorize object
%   test_all         - test the Factorize package (factorize, inverse, and related)
%   test_all_cod     - test the COD factorization
%   test_all_svd     - tests the svd factorization method for a range of problems.
%   test_cod         - test the COD, COD_SPARSE and RQ functions
%   test_disp        - test the display method of the factorize object
%   test_errors      - tests error handling for the factorize object methods
%   test_factorize   - test the accuracy of the factorization object
%   test_function    - test various functions applied to a factorize object
%   test_functions   - test various functions applied to a factorize object
%   test_performance - compare performance of factorization/solve methods.
%   test_svd         - test factorize(A,'svd') and factorize(A,'cod') for a given matrix

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

% SPQR_RANK
%
% For sparse, rank deficient matrices the SPQR_RANK package provides
% utilities that are useful for finding solutions to
%
%                min || b - A x ||
%
% and to determine the numerical structure of A, including the numerical
% rank of A as well as bases for the numerical null spaces of A and of A
% transpose.  The utilities reliably determine the numerical rank in the sense
% that in almost all cases the numerical rank is accurately determined when a
% warning flag returned by the utilities indicates that the numerical rank
% should be correct. Reliable determination of numerical rank is often
% critical to calculations with rank deficient matrices.
%
% See "Algorithm xxx: Reliable Calculation of Numerical Rank, Null Space Bases,
% Pseudoinverse Solutions and  Basic Solutions using SuiteSparseQR" by Leslie
% Foster and Timothy Davis, submitted ACM Transactions on Mathematical
% Software, 2011, for detailed discussion of the package.
%
% Files
%   spqr_basic          - approximate basic solution to min(norm(B-A*x))
%   spqr_cod            - approximate pseudoinverse solution to min(norm(B-A*x)
%   spqr_null           - finds an orthonormal basis for numerical null space of a matrix
%   spqr_pinv           - approx pseudoinverse solution to min(norm(B-A*X))
%   spqr_ssi            - block power method or subspace iteration applied to inv(R)
%   spqr_ssp            - block power method or subspace iteration applied to A or A*N
%   spqr_null_mult      - multiplies a matrix by numerical null space from spqr_rank methods
%   spqr_explicit_basis - converts a null space basis to an explicit matrix
%   spqr_rank_opts      - sets and prints the default options for spqr_rank
%   spqr_rank_stats     - prints the statistics from spqr_rank functions
%
%   quickdemo_spqr_rank - quick demo of the spqr_rank package
%   demo_spqr_rank      - lengthy demo for spqr_rank functions (requires SJget)
%   test_spqr_rank      - extensive functionality test of spqr_rank functions
%   test_spqr_coverage  - statement coverage test of spqr_rank functions
%
% To install this package, simply install all of SuiteSparse from
% http://www.suitesparse.com, the spqr_rank package
% in SuiteSparse/MATLAB_TOOLS/spqr_rank will be installed, along with
% SuiteSparseQR (spqr) and all its dependent packages.

% Copyright 2012, Leslie Foster and Timothy A Davis.

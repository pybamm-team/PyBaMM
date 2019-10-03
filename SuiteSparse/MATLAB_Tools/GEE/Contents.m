% GEE_ITS_SIMPLE: Gaussian Elimination Example (Gee! It's Simple!)
% A simple illustration of Guassian elimination with partial pivoting.
% This package is not meant for production use, just as a teaching tool.
% It also illustrates how a proper MATLAB package should be documented and
% written (with comments, error checking, exhaustive test code, simple code
% as opposed to cumbersome for loops, etc).
%
% The gee_its_simple functions are fully commented.  The gee_its_short function
% is the same as gee_its_simple except that it has no comments or error
% checking so you can see how few lines it takes to solve this problem (without
% using backslash, of course!).  It is just 15 lines of code excluding comments
% (with pivoting).  Without pivoting, gee_its_too_short lives up to its name
% (it is only 11 lines of code, but it's inaccurate).  For the actual
% implementation of backslash in MATLAB 7.5, I would give a rough estimate of
% about 250,000 lines of code (it includes LAPACK, the BLAS, UMFPACK, CHOLMOD,
% MA57, AMD, COLAMD, COLMMD, a sparse Givens-based QR, specialized banded
% solvers, and a left-looking sparse LU much like cs_lu in CSparse).  This
% estimated line count excludes x=A\b for symbolic variables, which would use
% the Symbolic Toolbox.  Compared with gee_its_simple, however, backslash is
% about 25 times faster for a dense matrix of order 1000, and it is
% "infinitely" faster for the sparse case (depending on the matrix).
%
% Example:
%   x = gee_its_simple (A,b) ;  % x = A\b using Gaussian elimination
%   x = gee_its_short (A,b) ;   % same as gee_its_simple, just shorter
%
% For production use:
%   Use x=A\b instead of x = gee_its_simple (A,b)
%   Use x=A\b instead of x = gee_its_short (A,b)
%   Use x=L\b instead of x = gee_its_simple_forwardsolve (L,b)
%   Use x=U\b instead of x = gee_its_simple_backsolve (U,b)
%   Use [L,U,p]=lu(A) and rcond(A) for [LU,p,rcnd] = gee_its_simple_factorize(A)
%
% Primary Files:
%   gee_its_short               - x=A\b, no comments or error checking (just for line count)
%   gee_its_simple              - solves A*x=b using a Gaussian Elimination Example (it's simple!)
%
% Secondary Files:
%   gee_its_simple_factorize    - Gaussian Elimination Example, with partial pivoting
%   gee_its_simple_forwardsolve - computes x=L\b where L is unit lower triangular
%   gee_its_simple_backsolve    - x=U\b where U is upper triangular
%   gee_its_simple_resid        - compute the relative residual for x=A\b
%   gee_its_simple_test         - tests the "Gee! It's Simple!" package
%
% Just for fun:
%   gee_its_sweet               - solves Ax=b with just x=A\b; it doesn't get sweeter than this
%   gee_its_too_short           - x=A\b, no pivoting (thus unstable!), just bare bones
%
% See also: lu, mldivide, rcond

% Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

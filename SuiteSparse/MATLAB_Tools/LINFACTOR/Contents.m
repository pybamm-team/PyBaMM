% LINFACTOR
%
% This simple MATLAB function shows you how to use LU or CHOL to factor a matrix
% and then solve a linear system A*x=b.
%
% Files
%   linfactor - factorize a matrix, or use the factors to solve Ax=b.
%   lintests  - test linfactor with many different kinds of systems.
%   lintest   - test A*x=b, using linfactor, x=A\b, and (ack!) the explicit inv(A).
%
% Example:
%   F = linfactor (A) ;     % factor A, returning an object F
%   x = linfactor (F,b) ;   % solve Ax=b using F (the factorization of A)
%   lintests ;              % test linfactor with various kinds of systems

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

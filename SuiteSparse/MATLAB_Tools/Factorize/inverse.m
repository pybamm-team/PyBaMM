function S = inverse (A, varargin)
%INVERSE factorized representation of inv(A) or pinv(A).
% INVERSE is a fast and accurate replacement for INV or PINV when you want to
% solve a linear system or least squares problem, or when you want to multiply
% something by the inverse of A.  The inverse itself is NOT computed, UNLESS
% the factorized form of the inverse is converted into a matrix via
% double(inverse(A)).  If A is rectangular and has full rank, or rank deficient
% and COD is able to accurately estimate the rank, then inverse(A) is a
% factorized form of the pseudo-inverse of A, pinv(A).
%
% Example
%
%   x = inv(A)*b ;      % slow and inaccurate way to solve A*x=b
%   x = inverse(A)*b ;  % fast an accurate way to solve A*x=b (uses x=A\b)
%   x = A\b ;           % same as inverse(A)*b
%
%   x1 = A\b1 ; x2 = A\b2 ;                        % accurate but slow
%   S = inverse(A) ; x1 = S*b1 ; x2 = S*b1 ;       % fast and accurate
%   S = inv(A)     ; x1 = S*b1 ; x2 = S*b1 ;       % slow and inaccurate
%
%   Z = double(inverse(A)) ; % same as Z=inv(A), computes the inverse of
%                            % A, returning Z as a matrix, not an object.
%
%   F = factorize(A) ;  % computes the factorization of A
%   S = inverse(F) ;    % no flops, flags S as a factorized form of inv(A)
%
% An optional 2nd input selects the strategy used to factorize the matrix,
% and an optional 3rd input tells the function to display how it factorizes
% the matrix.  See the 'strategy' and 'burble' of the factorize function.
%
% Never use inv to multiply the inverse of a matrix A by another matrix.
% There are rare uses for the explicit inv(A), but never do inv(A)*B or
% B*inv(A).  Never do Z=A\eye(n), which is just the same thing as Z=inv(A).
%
% "Don't let that inv go past your eyes; to solve that system, factorize!"
%
% See also factorize, slash, inv, pinv.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

% This function is only called when A is a matrix.  If A is a factorize
% object, then factorize.inverse is called instead.

S = inverse (factorize (A, varargin {:})) ;

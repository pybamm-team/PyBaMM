function [A, Z, title, key, mtype] = RBreade (filename)
%RBREADE read a symmetric finite-element matrix from a R/B file
% Usage:
%   [A Z title key mtype] = RBreade (filename)
%
% The file must contain a Rutherford/Boeing matrix of type *se or *he, where *
% can be r, p, i, or c.  See RBread for a description of the outputs.
%
% If CHOLMOD is installed, this function is faster and uses less memory.
%
% Example:
%
%   [A Z title key mtype] = RBreade ('lap_25.pse') ;
%
% See also RBread, RBraw, sparse2.

% Optionally uses the CHOLMOD sparse2 mexFunction.

% Copyright 2009, Timothy A. Davis

%-------------------------------------------------------------------------------
% read in the raw contents of the Rutherford/Boeing file
%-------------------------------------------------------------------------------

[mtype Ap Ai Ax title key n] = RBraw (filename) ;
mtype = lower (mtype) ;
if (~(mtype (2) == 's' | mtype (2) == 'h') | (mtype (3) ~= 'e'))	    %#ok
    error ('RBreade is only for symmetric unassembled finite-element matrices');
end

%-------------------------------------------------------------------------------
% determine dimension, number of elements, and convert numerical entries
%-------------------------------------------------------------------------------

Ap = double (Ap) ;
Ai = double (Ai) ;

% number of elements
ne = length (Ap) - 1 ;

% dimension.
if (max (Ai) > n)
    error ('invalid dimension') ;
end

% determine number of numerical entries
nz = 0 ;
for e = 1:ne
    p1 = Ap (e) ;
    p2 = Ap (e+1) - 1 ;
    nu = p2 - p1 + 1 ; 
    nz = nz + (nu * (nu+1)/2) ;
end

% Ax can be empty, for a pse matrix
if (~isempty (Ax))
    if (mtype (1) == 'c')
	% Ax is real, with real/imaginary parts interleaved
	if (2 * nz ~= length (Ax))
	    error ('invalid matrix (wrong number of complex values)') ;
	end
	Ax = Ax (1:2:end) + (1i * Ax (2:2:end)) ;
    elseif (mtype (1) == 'i')
	Ax = double (Ax) ;
    end
    % numerical values must be of the right size
    if (nz ~= length (Ax))
	error ('invalid matrix (wrong number of values)') ;
    end
end

%-------------------------------------------------------------------------------
% create triplet form
%-------------------------------------------------------------------------------

% row and column indices for triplet form of the matrix
ii = zeros (nz, 1) ;
jj = zeros (nz, 1) ;

nx = 0 ;

% create triplet row and column indices from finite-element pattern
for e = 1:ne
    p1 = Ap (e) ;
    p2 = Ap (e+1) - 1 ;
    for p = p1:p2
	j = Ai (p) ;
	for pp = p:p2
	    i = Ai (pp) ;
	    nx = nx + 1 ;
	    ii (nx) = max (i,j) ;
	    jj (nx) = min (i,j) ;
	end
    end
end

%-------------------------------------------------------------------------------
% create the sparse matrix form
%-------------------------------------------------------------------------------

if (exist ('sparse2') == 3)						    %#ok
    % Use sparse2 in CHOLMOD.  It's faster, allows integer Ai and Aj, and
    % returns the Z matrix as the 2nd output argument.
    if (isempty (Ax))
	Ax = 1 ;
    end
    % numerical matrix
    [A Z] = sparse2 (ii, jj, Ax, n, n) ;
else
    % stick with MATLAB, without CHOLMOD.  This is slower and takes more memory.
    if (isempty (Ax))
	% pattern-only matrix
	A = spones (sparse (ii, jj, 1, n, n)) ;
	Z = sparse (n, n) ;
    else
	% numerical matrix
	A = sparse (ii, jj, Ax, n, n) ;
	% determine the pattern of explicit zero entries
	S = spones (sparse (ii, jj, 1, n, n)) ;
	Z = S - spones (A) ;
    end
end

% add the upper triangular part
if (mtype (2) == 's')
    A = A + tril (A,-1).' ;
elseif (mtype (2) == 'h')
    A = A + tril (A,-1)' ;
end
Z = Z + tril (Z,-1)' ;

% remove duplicates created from triplet form for a pattern-only matrix
if (mtype (1) == 'p')
    A = spones (A) ;
end


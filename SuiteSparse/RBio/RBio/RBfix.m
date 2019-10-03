function [A, Z, title, key, mtype] = RBfix (filename)
%RBFIX read a possibly corrupted matrix from a R/B file
% (assembled format only).  Usage:
%
% [A Z title key mtype] = RBfix (filename)
%
% The Rutherford/Boeing format stores a sparse matrix in a file in compressed-
% column form, using 3 arrays: Ap, Ai, and Ax.  The row indices of entries in
% A(:,j) are in Ai(p1:p2) and the corresponding numerical values are Ax(p1:p2),
% where p1 = Ap(j) and p2 = Ap(j+1)-1.  The row indices ought to be sorted, and
% no duplicates should appear, but this function ignores that requirement.
% Duplicate entries are summed if they exist, and A is returned with sorted
% columns.  Symmetric matrices are stored with just their lower triangular
% parts in the file.  Normally, it is an error if entries are present in the
% upper triangular part of a matrix that is declared in the file to be
% symmetric.  This function simply ignores those entries.
%
% If CHOLMOD is installed, this function is faster and uses less memory.
%
% Example:
%
%   load west0479
%   RBwrite ('mywest', west0479, [ ], 'My west0479 file', 'west0479') ;
%   [A Z title key mtype] = RBfix ('mywest') ;
%   isequal (A, west0479)
%   title, key, mtype
%
% See also mread, RBread, RBwrite, RBreade, sparse2.

% Optionally uses the CHOLMOD sparse2 mexFunction.

% Copyright 2009, Timothy A. Davis

%-------------------------------------------------------------------------------
% read in the raw contents of the Rutherford/Boeing file
%-------------------------------------------------------------------------------

[mtype Ap Ai Ax title key nrow] = RBraw (filename) ;
mtype = lower (mtype) ;

%-------------------------------------------------------------------------------
% determine dimension, number of entries, and convert numerical entries
%-------------------------------------------------------------------------------

% number of columns
ncol = length (Ap) - 1 ;

% number of entries
nz = length (Ai) ;

% check column pointers
if (any (Ap ~= sort (Ap)) | (Ap (1) ~= 1) | (Ap (ncol+1) - 1 ~= nz))	    %#ok
    error ('invalid column pointers') ;
end

% check row indices
if ((double (max (Ai)) > nrow) | double (min (Ai)) < 1)			    %#ok
    error ('invalid row indices') ;
end

% Ax can be empty, for a p*a matrix
if (~isempty (Ax))
    if (mtype (1) == 'c')
	% Ax is real, with real/imaginary parts interleaved
	if (2 * nz ~= length (Ax))
	    error ('invalid matrix') ;
	end
	Ax = Ax (1:2:end) + (1i * Ax (2:2:end)) ;
    elseif (mtype (1) == 'i')
	Ax = double (Ax) ;
    end
    % numerical values must be of the right size
    if (nz ~= length (Ax))
	error ('invalid matrix') ;
    end
end

%-------------------------------------------------------------------------------
% create the triplet form
%-------------------------------------------------------------------------------

% construct column indices
Aj = zeros (nz,1) ;
for j = 1:ncol
    p1 = Ap (j) ;
    p2 = Ap (j+1) - 1 ;
    Aj (p1:p2) = j ;
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
    [A Z] = sparse2 (Ai, Aj, Ax, nrow, ncol) ;
else
    % stick with MATLAB, without CHOLMOD.  This is slower and takes more memory.
    Ai = double (Ai) ;
    Aj = double (Aj) ;
    if (isempty (Ax))
	% pattern-only matrix
	A = spones (sparse (Ai, Aj, 1, nrow, ncol)) ;
	Z = sparse (nrow, ncol) ;
    else
	% numerical matrix
	A = sparse (Ai, Aj, Ax, nrow, ncol) ;
	% determine the pattern of explicit zero entries
	S = spones (sparse (Ai, Aj, 1, nrow, ncol)) ;
	Z = S - spones (A) ;
    end
end

% check for entries in upper part
if (any (mtype (2) == 'shz') & nnz (triu (A,1) > 0))			    %#ok
    fprintf ('entries in upper triangular part of %s matrix ignored\n', mtype);
end

% add the upper triangular part
if (mtype (2) == 's')
    A = A + tril (A,-1).' ;
    Z = Z + tril (Z,-1)' ;
elseif (mtype (2) == 'h')
    A = A + tril (A,-1)' ;
    Z = Z + tril (Z,-1)' ;
elseif (mtype (2) == 'z')
    A = A - tril (A,-1).' ;
    Z = Z + tril (Z,-1)' ;
end


function A = meshsparse (G, stencil)
%MESHSPARSE convert a 2D or 3D mesh into a sparse matrix matrix.
%
% Example:
% A = meshsparse (G)
% A = meshsparse (G,5)		    % 2D 5-point stencil (default for 2D case)
% A = meshsparse (G,9)		    % 2D 9-point stencil
% A = meshsparse (G,7)		    % 3D 7-point stencil (default for 3D case)
% A = meshsparse (G,27)		    % 3D 27-point stencil
% A = meshsparse (G,stencil)	    % user-provided stencil
%
% To create a sparse matrix for an m-by-n 2D mesh or m-by-n-by-k 3D mesh, use
%
% A = meshsparse (meshnd (m,n)) ;
% A = meshsparse (meshnd (m,n,k)) ;
%
% G is an m-by-n-by-k matrix, with entries numbered 1 to m*n*k (with k=1 for
% the 2D case).  The entries in G can appear in any order, but no duplicate
% entries can appear.  That is sort(G(:))' must equal 1:m*n*k. A is returned as
% a sparse matrix with m*n*k rows and columns whose pattern depends on the
% stencil.  The number of nonzeros in most rows/columns of A is equal to the
% number of points in the stencil.  For examples on how to specify your own
% stencil, see the contents of meshsparse.m.
%
% See also meshnd.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 2)
    [m n k] = size (G) ;
    if (k == 1)
	stencil = 5 ;	% 2D default is a 5-point stencil
    else
	stencil = 7 ;	% 3D default is a 7-point stencil
    end
end

if (numel (stencil) == 1)

    % create the stencil

    if (stencil == 5)

	% 5-point stencil (2D)
	stencil = [
	    -1  0   	% north
	     1  0   	% south
	     0  1   	% east
	     0 -1   ] ;	% west

    elseif (stencil == 9)

	% 9-point stencil (2D)
	stencil = [
	    -1  0   	% north
	     1  0   	% south
	     0  1   	% east
	     0 -1   	% west
	    -1 -1   	% north-west
	    -1  1   	% north-east
	     1 -1   	% south-west
	     1  1   ] ;	% south-east

    elseif (stencil == 7)

	% 7-point stencil (3D)
	stencil = [
	    -1  0  0	% north
	     1  0  0	% south
	     0  1  0	% east
	     0 -1  0	% west
	     0  0 -1    % up
	     0  0  1] ; % down

    elseif (stencil == 27)

	% 27-point stencil (3D)
	stencil = zeros (26, 3) ;
	t = 0 ;
	for i = -1:1
	    for j = -1:1
		for k = -1:1
		    if (~(i == 0 & j == 0 & k == 0))    %#ok
			t = t + 1 ;
			stencil (t,:) = [i j k] ;
		    end
		end
	    end
	end
    end
end

stencil = fix (stencil) ;
[npoints d] = size (stencil) ;
if (d == 2)
    % append zeros onto a 2D stencil to make it "3D"
    stencil = [stencil zeros(npoints,1)] ;
end
[npoints d] = size (stencil) ;
if (d ~= 3)
    error ('invalid stencil') ;
end

[m n k] = size (G) ;
i1 = 1:m ;
j1 = 1:n ;
k1 = 1:k ;

Ti = zeros (npoints*m*n*k, 1) ;
Tj = zeros (npoints*m*n*k, 1) ;
nz = 0 ;

for point = 1:npoints

    % find the overlapping rows of G
    idelta = stencil (point,1) ;
    i2 = i1 + idelta ;
    ki = find (i2 >= 1 & i2 <= m) ;

    % find the overlapping columns of G
    jdelta = stencil (point,2) ;
    j2 = j1 + jdelta ;
    kj = find (j2 >= 1 & j2 <= n) ;

    % find the overlapping slices of G
    kdelta = stencil (point,3) ;
    k2 = k1 + kdelta ;
    kk = find (k2 >= 1 & k2 <= k) ;

    % find the nodes in G the shifted G that touch
    g2 = G (i2 (ki), j2 (kj), k2 (kk)) ;    % shifted mesh
    g1 = G (i1 (ki), j1 (kj), k1 (kk)) ;    % unshifted mesh

    % place the edges in the triplet list
    e = numel (g1) ;
    Ti ((nz+1):(nz+e)) = g1 (:) ;
    Tj ((nz+1):(nz+e)) = g2 (:) ;
    nz = nz + e ;
end

% convert the triplets into a sparse matrix
Ti = Ti (1:nz) ;
Tj = Tj (1:nz) ;
A = npoints * speye (m*n*k) - sparse (Ti, Tj, 1, m*n*k, m*n*k) ;

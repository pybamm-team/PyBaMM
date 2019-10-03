function [G, p, pinv, Gnew] = meshnd (arg1,n,k)
%MESHND creation and nested dissection of a regular 2D or 3D mesh.
% [G p pinv Gnew] = meshnd (m,n) constructs an m-by-n 2D mesh G, and then finds
% a permuted mesh Gnew where Gnew = pinv(G) and G = p(Gnew).  meshnd(m,n,k)
% creates an m-by-n-by-k 3D mesh.
%
% [G p pinv Gnew] = meshnd (G) does not construct G, but uses the mesh G as
% given on input instead.
%
% Example:
% [G p pinv Gnew] = meshnd (4,5) ;
%
% returns
%    Gnew =
%        1     2    17     9    10
%        7     8    18    15    16
%        3     5    19    11    13
%        4     6    20    12    14
%    G =
%        1     2     3     4     5
%        6     7     8     9    10
%       11    12    13    14    15
%       16    17    18    19    20
%
% With no inputs, a few example meshes are generated and plotted.
%
% See also nested, numgrid.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

% get the inputs and create the mesh if not provided on input
if (nargin == 0)

    % run a simple example
    meshnd_example ;

elseif (nargin == 1)

    % the mesh is provided on input
    G = arg1 ;
    [m n k] = size (G) ;

elseif (nargin == 2)

    % create the m-by-n-by-k mesh in "natural" (row-major) order.  This is how
    % a typical 2D mesh is ordered.  A column-major order would be better, since
    % in that case G(:) would equal 1:(m*n) ... but let's stick with tradition.
    m = arg1 ;
    k = 1 ;
    G = reshape (1:(m*n*k), n, m, k)' ;

elseif (nargin == 3)

    % create the m-by-n-by-k mesh in column-major order.  The first m-by-n-by-1
    % slice is in column-major order, followed by all the other slices 2 to k.
    m = arg1 ;
    G = reshape (1:(m*n*k), m, n, k) ;

else

    error ('Usage: [G p pinv Gnew] = meshnd(G), meshnd(m,n) or meshnd(m,n,k)') ;

end

if (nargout > 1)
    p = nd2 (G)' ;	    % order the mesh
end

if (nargout > 2)
    pinv (p) = 1:(m*n*k) ;  % find the inverse permutation
end

if (nargout > 3)
    Gnew = pinv (G) ;	    % find the permuted mesh
end

%-------------------------------------------------------------------------------

function p = nd2 (G)
%ND2 p = nd2 (G) permutes a 2D or 3D mesh G.
% Compare with nestdiss which uses p as a scalar offset and returns a modified
% mesh G that corresponds to Gnew in meshnd.  Here, the scalar offset p in
% nestdiss is not needed.  Instead, p is a permutation, and the modified mesh
% Gnew is not returned.

[m n k] = size (G) ;

if (max ([m n k]) <= 2)

    % G is small; do not cut it
    p = G (:) ;

elseif k >= max (m,n)

    % cut G along the middle slice, cutting k in half
    s = ceil (k/2) ;
    middle = G (:,:,s) ;
    p = [(nd2 (G (:,:,1:s-1))) ; (nd2 (G (:,:,s+1:k))) ; middle(:)] ;

elseif n >= max (m,k)

    % cut G along the middle column, cutting n in half
    s = ceil (n/2) ;
    middle = G (:,s,:) ;
    p = [(nd2 (G (:,1:s-1,:))) ; (nd2 (G (:,s+1:n,:))) ; middle(:)] ;

else   

    % cut G along the middle row, cutting m in half
    s = ceil (m/2) ;
    middle = G (s,:,:) ;
    p = [(nd2 (G (1:s-1,:,:))) ; (nd2 (G (s+1:m,:,:))) ; middle(:)] ;

end

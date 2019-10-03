%MESHND: creation and nested dissection of regular 2D and 3D meshes. 
%
%   meshnd         - creation and nested dissection of a regular 2D or 3D mesh.
%   meshnd_quality - test the ordering quality computed by meshnd.
%   meshsparse     - convert a 2D or 3D mesh into a sparse matrix matrix.
%   meshnd_example - example usage of meshnd and meshsparse.
%
% The outputs of the meshnd example and meshnd_quality are in meshd.png,
% meshnd_quality_out.txt, and meshnd_quality.png.
%
% Example:
%   % with no inputs or outputs, meshnd runs a demo:
%   meshnd	    
%
%   % create the sparse matrix for a 7-by-5-by-2 mesh:
%   A = meshsparse (meshnd (7,5,2)) ;
%
%   % create a 7-by-5-by-2 mesh and find the nested dissection ordering:
%   [G p] = meshnd (7,5,2) ;
%   A = meshsparse (G) ;
%   subplot (1,2,1) ; spy (A) ;
%   subplot (1,2,2) ; spy (A (p,p)) ;
%

% Copyright 2009, Timothy A. Davis, http://www.suitesparse.com

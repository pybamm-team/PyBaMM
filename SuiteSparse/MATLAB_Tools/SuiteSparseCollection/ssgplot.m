function ssgplot (A, xyz, directed, nodename)
%SSGPLOT draw a plot of the graph of a sparse matrix
% Usage:
%   ssgplot (A, xyz, directed, nodename)
%
% A: a square sparse matrix of size n-by-n.
% xyz: an n-by-2 or n-by-3 array of the XY or XYZ coordinates of each node.
% directed: 1 if A is directed, 0 otherwise.  0 if not present.
% nodename: a char array with n rows, or empty, containing the name of each
%   node.  Not used if not present.
%
% Example:
%
%   Problem = ssget ('Pajek/football') ;
%   ssgplot (Problem.A, Problem.aux.coord, 1, Problem.aux.nodename) ;
%
% See also gplot, ssget.

% Copyright 2006-2018, Timothy A. Davis

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

[m n] = size (A) ;
if (m ~= n)
    error ('A must be square') ;
end

if (nargin < 4)
    nodename = [ ] ;
end
if (nargin < 3)
    directed = 0 ;
end

%-------------------------------------------------------------------------------
% determine if 2D or 3D
%-------------------------------------------------------------------------------

if (size (xyz, 2) == 2)

    %---------------------------------------------------------------------------
    % no Z coordinates, draw a 2D graph
    %---------------------------------------------------------------------------

    xy = xyz ;
    if (directed)
	ti = '2D directed graph' ;
    else
	ti = '2D undirected graph' ;
    end

else

    %---------------------------------------------------------------------------
    % a 3D graph; rotate for better viewing
    %---------------------------------------------------------------------------

    dx = -45 ;
    dy = 45 ;
    dz = 45 ;
    a = dx * (2*pi/360) ;
    xrotation = [
	1      0      0
	0  cos(a) sin(a)
	0 -sin(a) cos(a) ] ;
    b = dy * (2*pi/360) ;
    yrotation = [
	cos(b)  0 -sin(b)
	    0   1      0
	sin(b)  0  cos(b) ] ;
    c = dz * (2*pi/360) ;
    zrotation = [
	 cos(c) sin(c) 0
	-sin(c) cos(c) 0
	     0      0  1 ] ;
    r = xrotation * yrotation * zrotation ;
    xy = xyz * r ;
    xy = xy (:,1:2) ;
    if (directed)
	ti = '3D directed graph' ;
    else
	ti = '3D undirected graph' ;
    end
end

%-------------------------------------------------------------------------------
% draw the graph
%-------------------------------------------------------------------------------

[X,Y] = gplot (A, xy) ;
if (n < 100)
    msize = 12 ;
else
    msize = 6 ;
end

if (n < 200)
    if (directed)
	plot (X, Y, 'mo', 'MarkerEdgeColor', 'k', ...
	    'MarkerFaceColor', [.49 1 .63], 'MarkerSize', msize) ;
	hold on
	axis equal
	axis off
	for k = 0:(length(X) / 3)-1
	    [x1 y1] = dsxy2figxy (gca, X (3*k+1), Y (3*k+1)) ;
	    [x2 y2] = dsxy2figxy (gca, X (3*k+2), Y (3*k+2)) ;
	    annotation ('arrow', [x1 x2], [y1 y2], 'Color', [1 0 1], ...
		'HeadWidth', 4, 'HeadLength', 8) ;
	end
    else
	plot (X, Y, '-mo', 'MarkerEdgeColor', 'k', ...
	    'MarkerFaceColor', [.49 1 .63], 'MarkerSize', msize) ;
	hold on
	axis equal
	axis off
    end
    if (~isempty (nodename) && n < 100)
	for k = 1:n
	    text (xy (k,1), xy (k,2), ['  ' nodename(k,:)], ...
		'Interpreter', 'none') ;
	end
    end
else
    plot (X, Y, '-m.', 'MarkerEdgeColor', 'k') ;
    axis equal
    axis off
end

title (ti) ;

hold off

function pagerankdemo (steps)
% PAGERANKDEMO draw a 6-node web and compute its pagerank
%
% PAGERANKDEMO draws the 6-node "tiny web" in Section 2.11 of "Numerical
% Computing with MATLAB", by Cleve Moler, SIAM, 2004.  It then simulates the
% computation of Google's PageRank algorithm, by randomly selecting links to
% traverse.  If a link is traversed, the edge and the target node are displayed
% in red.  If the "random surfer" jumps to an arbitrary page, the target node
% is displayed in blue.  The number of hits at each node, and the page rank
% (in %) are displayed % on each node.  Note that after a large number of
% steps, the PageRanks (in percentages) converge to the values given in Section
% 2.11 of Moler (alpha: .321, sigma: .2007, beta: .1705, delta: .1368,
% gamma: .1066, rho: .0643).  See http://www.mathworks.com/moler for more
% details (the pagerank M-file, in particular).
%
% Note that this method is NOT how the PageRank is actually computed.  Instead
% the eigenvalue problem A*x=x is solved for x, where A is the Markov
% transition matrix, A = p*G*D + e*z', where G is the binary matrix used here.
% The method here is a simplistic random-hopping demonstration of the Markov
% process, to motivate the A*x=x formulation of the problem.  In this example,
% A does control how the transitions are made, but the matrix A is not formed
% explicitly.
%
% This demo only operates on a single graph.  It is meant as a simple demo
% only, suitable for in-class use.  To compute the PageRanks for an arbitrary
% graph, use pagerank.m, or the power method (repeat x=A*x until convergence,
% where A is the Markov transition matrix of the web).
%
% Example:
%   pagerankdemo
%   pagerankdemo (1000)   % run 1000 steps with no user input, then quit
%
% See also pagerank
%
% I suggest single-stepping a dozen times or so to see the link traversal in
% process, and then type "1000".  Hit control-C to quit.
%
% Copyright 2007, Tim Davis

% Initial graph
Graph = graphinit ;
rand ('state', 0) ;
n = size (Graph.G, 1) ;

help pagerankdemo

% initialize the page counts
hits = zeros (1,n) ;
oldwhere = 1 ;
where = 1 ;
hits (where) = 1 ;
set (Graph.node (where), 'FaceColor', [0 0 1]) ;

p = 0.85 ;		% probability a link will be followed
c = sum (Graph.G) ;	% outgoing degree

links = cell (1,n) ;
for k = 1:n
    links {k} = find (Graph.G (:,k)) ;
end

follow_link = 0 ;
if (nargin < 1)
    input ('hit enter to start at node alpha: ') ;
end

% write the stats to the figure
set (Graph.nodelabel (where), 'string', ...
	sprintf ('%s %d (%3.1f%%)', Graph.nodes {where}, hits (where), ...
	100 * hits (where) / sum (hits))) ;

if (nargin < 1)
    input ('hit enter to take one step: ') ;
    steps = 1 ;
end

% repeat
while (1)

    % clear the old color and old arrow
    set (Graph.node (where), 'FaceColor', [0 1 0]) ;
    if (follow_link)
	set (Graph.arrows (where,oldwhere), 'LineWidth', 2) ;
	set (Graph.arrows (where,oldwhere), 'Color', [0 0 0]) ;
    end

    % determine where to go to next
    oldwhere = where ;
    if (c (where) == 0 || rand > p)
	% no outgoing links, or ignore the links
	follow_link = 0 ;
	where = floor (n * rand + 1) ;
	set (Graph.node (where), 'FaceColor', [0 0 1]) ;
    else
	% move along the link
	follow_link = 1 ;
	where = links{where}(floor (c (where) * rand + 1)) ;
	set (Graph.node (where), 'FaceColor', [1 0 0]) ;
	set (Graph.arrows (where,oldwhere), 'LineWidth', 5) ;
	set (Graph.arrows (where,oldwhere), 'Color', [1 0 0]) ;
    end

    % increment the hit count
    hits (where) = hits (where) + 1 ;

    % write the stats to the figure
    for k = 1:n
	set (Graph.nodelabel (k), 'string', ...
	sprintf ('%s %d (%3.1f%%)', Graph.nodes {k}, hits (k), ...
	100 * hits (k) / sum (hits))) ;
    end

    drawnow

    % go the next step
    steps = steps - 1 ;
    if (steps <= 0)
        if (nargin > 0)
            break ;
        end
        steps = input ...
	    ('number of steps to make (default 1, control-C to quit): ') ;
	if (steps == 0)
	    break ;
	end
	if (isempty (steps))
	    steps = 1 ;
	end
    end

end

%-------------------------------------------------------------------------------

function Graph = graphinit
% GRAPHINIT create the tiny-web example in Moler, section 2.11, and draw it.
% Example
%   G = graphinit ;

figure (1)
clf

nodes = { 'alpha', 'beta', 'gamma', 'delta', 'rho', 'sigma' } ;

xy = [
0 4
1 3
1 2
2 4
2 0
0 0
] ;

x = xy (:,1) ;
y = xy (:,2) ;

% scale x and y to be in the range 0.1 to 0.9
x = 0.8 * x / 2 + .1 ;
y = 0.8 * y / 4 + .1 ;
xy = [x y] ;

xy_delta = [
 .08 .04 0
-.03 -.02 -1
 .04  0   0
-.05  .04 -1
-.03  0  -1
 .03  0   0
] ;

xd = xy_delta (:,1) ;
yd = xy_delta (:,2) ;
tjust = xy_delta (:,3) ;

G = [
0 0 0 1 0 1
1 0 0 0 0 0
0 1 0 0 0 0
0 1 1 0 0 0
0 0 1 0 0 0
1 0 1 0 0 0 ] ;

clf

n = size (G,1) ;

axes ('Position', [0 0 1 1], 'Visible', 'off') ;

node = zeros (n,1) ;
nodelabel = zeros (n,1) ;
for k = 1:n
    node (k) = annotation ('ellipse', [x(k)-.025 y(k)-.025 .05 .05]) ;
    set (node (k), 'LineWidth', 2) ;
    set (node (k), 'FaceColor', [0 1 0]) ;
    nodelabel (k) = text (x (k) + xd (k), y (k) + yd (k), nodes {k}, ...
	'Units', 'normalized', 'FontSize', 16) ;
    if (tjust (k) < 0) 
	set (nodelabel (k), 'HorizontalAlignment', 'right') ;
    end
end


axis off

% Yes, I realize that this is overkill; arrows should be sparse.
% This example is not meant for large graphs.
arrows = zeros (n,n) ;

[i j] = find (G) ;
for k = 1:length (i)
    % get the center of the two nodes
    figx = [x(j(k)) x(i(k))] ;
    figy = [y(j(k)) y(i(k))] ;
%   [figx figy] = dsxy2figxy (gca, axx, axy);
    % shorten the arrows by s units at each end
    s = 0.03 ;
    len = sqrt (diff (figx)^2 + diff (figy)^2) ;
    fy (1) = diff (figy) * (s/len) + figy(1) ;
    fy (2) = diff (figy) * (1-s/len) + figy(1) ;
    fx (1) = diff (figx) * (s/len) + figx(1) ;
    fx (2) = diff (figx) * (1-s/len) + figx(1) ;
    arrows (i(k),j(k)) = annotation ('arrow', fx, fy) ;
    set (arrows (i(k),j(k)), 'LineWidth', 2) ;
    set (arrows (i(k),j(k)), 'HeadLength', 20) ;
    set (arrows (i(k),j(k)), 'HeadWidth', 20) ;
end

Graph.G = G ;
Graph.nodes = nodes ;
Graph.node = node ;
Graph.xy = xy ;
Graph.xy_delta = xy_delta ;
Graph.nodelabel = nodelabel ;
Graph.arrows = arrows ;

function find_components_example(example, dopause)
%FIND_COMPONENTS_EXAMPLE gives an example usage of find_components.
%
% Example:
%   find_components_example(0)  % a small example, with lots of printing
%   find_components_example(1)  % Doug's example, with lots of printing
%   find_components_example(2)  % a large example, just plotting
%   find_components_example(A)  % use the matrix A for the example
%
% See http://blogs.mathworks.com/pick/2008/08/18 for Doug Hull's description of
% the problem this m-file solves.  With no inputs, Doug's example is used.
%
% See also FIND_COMPONENTS, LARGEST_COMPONENT, DMPERM, GPLOT

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% construct an image
%-------------------------------------------------------------------------------

if (nargin < 1)
    example = 1 ;
end
if (example == 0)
    A = [ 1 2 2 3
          1 1 2 3
          0 0 1 2
          0 1 3 3 ] ;
elseif (example == 1)
    A = [ 2 2 1 1 2
          3 0 1 0 1
          3 2 2 2 1
          1 2 2 1 2
          0 3 2 0 1 ] ;
elseif (example == 2)
    A = round (rand (30) * 2) ;
else
    A = example ;
end
[m n] = size (A) ;

%-------------------------------------------------------------------------------
% find all of its components
%-------------------------------------------------------------------------------

tic
[p r nc G xy] = find_components (A,1) ;
t = toc ;
fprintf ('Image size: %d-by-%d, time taken: %g seconds\n', m, n, t) ;

%-------------------------------------------------------------------------------
% walk through the components, plotting and printing them.
%-------------------------------------------------------------------------------

prompt = 'hit enter to single-step, ''a'' to show all, ''q'' to quit: ' ;
small = (max (m,n) <= 10) ;
if (nargin < 2)
    dopause = 1 ;
end

for k = 1:nc

    % get the nodes of the kth component
    nodes = p (r (k) : r (k+1)-1) ;

    % for large graphs, do not show components of size 1
    if (~small && length (nodes) == 1)
        continue
    end

    % plot the graph with the kth component highlighted
    hold off
    gplot (G, xy, '-') ;
    hold on
    [X,Y] = gplot (G * sparse (nodes, nodes, 1, m*n, m*n), xy, 'r-') ;
    plot (X, Y, 'r-', 'LineWidth', 3) ;
    axis ([0 n+1 0 m+1]) ;
    a = A (p (r (k))) ;
    siz = length (nodes) ;
    Title = sprintf ('Graph component %d, size %d, value %g', k, siz, a) ;
    title (Title, 'FontSize', 20) ;
    label_nodes (xy, A, small, nodes) ;
    drawnow

    % print the image and the kth component, if the image is small
    if (small)
        fprintf ('\n%s\n', Title) ;
        C = nan (m,n) ;
        C (nodes) =  a ;
        fprintf ('A = \n') ; disp (A) ;
        fprintf ('the component = \n') ; disp (C) ;
    end

    % pause, or prompt the user
    if (dopause && (k < nc))
        s = input (prompt, 's') ;
        dopause = isempty (s) ;
        if (~dopause && s (1) == 'q')
            break ;
        end
    else
        pause (0.5)
    end
end

%-------------------------------------------------------------------------------
% plot the whole graph, no components highlighted
%-------------------------------------------------------------------------------

hold off
gplot (G, xy, '-') ;
title (sprintf ('%d connected components', nc), 'FontSize', 20) ;
axis ([0 n+1 0 m+1]) ;
label_nodes (xy, A, small)

%-------------------------------------------------------------------------------

function label_nodes (xy, A, small, nodes)
%LABEL_NODES label all the nodes in the plot
if (small)
    [m n] = size (A) ;
    for i = 1:m*n
        text (xy (i,1), xy (i,2), sprintf ('%g', A (i)), 'FontSize', 20) ;
    end
    if (nargin == 4)
        for i = nodes
            text (xy (i,1), xy (i,2), sprintf ('%g', A (i)), ...
                'FontSize', 20, 'Color', 'r') ;
        end
    end
end

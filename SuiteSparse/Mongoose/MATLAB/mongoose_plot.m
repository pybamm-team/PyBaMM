function mongoose_plot (G, x_vec, y_vec, plot_name, colorstyles, bgcolor, grad) %#ok
%MONGOOSE_PLOT use graphvis to create a plot of a graph.
%   mongoose_plot (A, left, right) draws the graph A (a sparse matrix) via
%   graphvis. If A is rectangular, [0 A ; A' 0] is drawn.  left and right
%   are vectors of size n where the matrix A or [0  A ;A' 0] is n-by-n.
%   
%   left(i)=1 if node i is on the left.
%   right(i)=1 if node i is on the right.
%   If both left(i) and right(i) are 0, then node is is in the node separator
%
% Example:
%
%   A = sparse (gallery ('gcdmat', 20) > 2) ;
%   mongoose_plot (A) ;                 % no colors on the nodes
%   subplot (1,2,1) ;
%   imshow (imread ('separator_plot.png')) ;
%   subplot (1,2,2) ;
%   n = size (A,1) ;
%   k = floor (n / 2) ;
%   left  = [ones(1,k) zeros(1,n-k)] ;
%   right = 1-left ;
%   mongoose_plot (A, left, right) ;    % color the partitions
%   imshow (imread ('separator_plot.png')) ;
%
% The plot is placed in the file 'separator_plot.png'.
% A fourth input argument specifies an alternate filename
%
% See also spy, mongoose_test.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

fprintf ('Using graphvis by Yifan Hu to draw the graph\n') ;
fprintf ('(ignore any error message about "remove_overlap")\n') ;

DEBUG = 0;
do_png = 1 ;

if (nargin < 4)
    plot_name = 'separator_plot' ;
end

penwidth = 0 ;

if (nargin < 5 || isempty (colorstyles))
    colorstyles = 0 ;
end

if (nargin < 6 || isempty (bgcolor))
    bgcolor = [0 0 0 255];
end

minsizes = 0 ;
% trim1 = 0 ;
do_svg = 0 ;
do_smooth = 0 ;
resolution = 300 ;
margin = 0 ;
% nodekind = 0 ;
edgesfirst = 0 ;

[m, n] = size (G) ;
if (m ~= n)
    % error ('graph must be square') ;
    G = [sparse(m,m) G ; G' sparse(n,n)] ;
end

if (nargin < 3)
    x_vec = zeros (1,n) ;
    y_vec = zeros (1,n) ;
end

gname = plot_name ;
slash = find (gname == '/') ;
if (~isempty (slash))
    gname = gname ((slash(end)+1):end) ;
end
if (DEBUG) 
    fprintf ('Gplot %s\n', gname) ; %#ok
end

if (nnz (G-G') > 0)
    warning ('graph is not symmetric') ;
    G = (G + G') / 2 ;
end

% norig = n ;
[~, n] = size (G) ;
if (DEBUG)
    fprintf ('n: %d\n', n) ; %#ok
end

if (penwidth == 0)
    if (n < 10)
        penwidth = 10 ;
    elseif (n < 100)
        penwidth = 5 ;
    elseif (n < 1000)
        penwidth = 2 ;
    else
        penwidth = 1 ;
    end
elseif (penwidth == -1)
    if (n < 10)
        penwidth = 10 ;
    elseif (n < 100)
        penwidth = 5 ;
    else
        penwidth = 2 ;
    end
end

if (DEBUG)
    fprintf ('penwidth: %d\n', penwidth) ; %#ok
end

% rsize = sort (rsize, 'descend') ;
% if (DEBUG)
%     fprintf ('singletons: %d\n', length (find (rsize == 1))) ;
% end
% nonsingleton = find (rsize ~= 1) ;
% if (DEBUG)
%     fprintf ('non singletons: %d\n', length (nonsingleton)) ;
% end
% if (~isempty (nonsingleton) && DEBUG)
%     fprintf ('non single graphs of size:\n') ;
%     fprintf ('%d ', rsize (nonsingleton)) ;
%     fprintf ('\n') ;
% end

[i, j, x] = find (tril (G, -1)) ;

% change any non-recognized edge into 89
efix = find (x < 1 | x > 88 | x ~= fix (x)) ;
x (efix) = 89 ; %#ok

% gvdir = 'gv/'
gvdir = '/tmp/' ;

gv1file = [gvdir gname '.gv'] ;
f = fopen (gv1file, 'w') ;
fprintf (f, 'graph %s {\n', gname) ;
fprintf (f, 'size="11!"; bgcolor=black; resolution="%g"; dpi="%g"; ordering=out; outputorder=nodesfirst\n', resolution, resolution) ;
fprintf (f, 'node [label="", fixedsize=1, style="invisible", shape="point", height=0, width=0, color="#00000000"];\n') ;
if (penwidth > 0)
    fprintf (f, 'edge [penwidth=%g];\n', penwidth) ;
end
nz = length (i) ;
for k = 1:nz
    fprintf (f, '%d--%d\n', i (k), j (k)) ;
end
fprintf (f, '}\n') ;
fclose (f) ;

if (ismac)
    where = '/usr/local/bin/' ;
else
    where = '/usr/bin/' ;
end

% get the node positions
s = '"\-\-"' ;
gv2file = [gvdir gname '_sfdp.gv'] ;
posfile = [gvdir gname '.pos'] ;
system (sprintf ('%ssfdp %s > %s', where, gv1file, gv2file)) ;
system (sprintf ('grep pos %s | grep -v %s > %s', gv2file, s, posfile)) ;
delete (gv1file) ;
%delete (gv2file) ;

% read them back in
f = fopen (posfile, 'r') ;
tline = fgetl (f) ;
pos = zeros (n,2) ;
while (ischar (tline))
    node = sscanf (tline, '%d', 1) ;
    matches = strfind (tline, 'pos=') ;
    if (length (matches) == 1)
        tline = tline (matches(1):end) ;
        xy = sscanf (tline, 'pos="%g,%g"', 2)' ;
        pos (node, :) = xy ;
    end
    tline = fgetl (f) ;
end
fclose (f) ;
delete (posfile) ;

if (DEBUG)
    fprintf ('pos x: %g to %g, y: %g to %g\n', ...
        min (pos (:,1)), max (pos (:,1)), min (pos (:,2)), max (pos (:,2))) ; %#ok
end


for minsize = minsizes
    if (DEBUG)
        fprintf ('minsize: %d\n', minsize) ; %#ok
    end
    if (minsize > 0)
        % make sure the smaller dimension is scaled up to minsize
        xmax = max (pos (:,1)) ;
        ymax = max (pos (:,2)) ;
        minhw = min (xmax, ymax) ;
        pos = ( pos / minhw ) * minsize ;
        if (DEBUG)
            fprintf ('original x %g y %g\n', xmax, ymax) ; %#ok
            fprintf ('minsize %g: new pos x: %g to %g, y: %g to %g\n', minsize, ...
            min (pos (:,1)), max (pos (:,1)), min (pos (:,2)), max (pos (:,2))) ;
        end
    end

    for ccc = colorstyles

        %bgcolor = [0 0 0 255 ] ;        % black
        if (iscell (ccc))
            colorstyle = ccc {1} ;
        else
            colorstyle = ccc ;
        end

        if (numel (colorstyle) > 1)

            % c must be an 89-by-4 array with int values in range 0 to 255
            c = round (colorstyle) ;
            if (size (c, 1) ~= 89 || size (c,2) ~= 4)
                size (c)
                error ('wrong size') ;
            end
            if (min(min(c)) < 0 || max (max (c)) > 255)
                error ('colorstyle out of range') ;
            end
            bgcolor = c (89,:) ;
            c = c (1:88,:) ;

        elseif (colorstyle == 0)

            % original color style
            c = round (255 * hsv (256)) ;
            c = c (256:-1:1,:) ;
            % no alpha
            c (:,4) = 255 ;

        elseif (colorstyle == 1)

            keys = (1:88)-4 ;
            hue  = mod (keys, 12) / 12 ;
            alf  = min (1, linspace (1.6, 0.3, 88)) ;
            val  = min (1, linspace (0.4, 2.0, 88)) ;
            sat  = ones (1,88) ;
            hsvmap = [hue ; sat ; val]' ;
            rgbmap = hsv2rgb (hsvmap) ;
            c = round (255 * [rgbmap alf']) ;

        elseif (colorstyle == 2)

            c = jet (88) ;
            c = c (end:-1:1, :) ;
            alf = min (1, linspace (1.0, 0.1, 88)) ;
            c = round (255 * [c alf']) ;

        elseif (colorstyle == 3 || colorstyle == 4)

            c = jet (88) ;
            %%% c = c (end:-1:1, :) ;
            alf = min (1, linspace (1.0, 0.1, 88)) ;
            c = round (255 * [c alf']) ;

        elseif (colorstyle == 5)

            c = jet (88) ;
            alf = min (1, linspace (1.0, 0.1, 88)) ;
            c = round (255 * [c alf']) ;
            c (76:end, 4) = 10 ;

        elseif (colorstyle == 6)

            r = [ 0 0 0 linspace(0,0,36) linspace(0,1,22) linspace(1,1,26) 1] ;
            g = [ 0 0 0 linspace(0,1,28) linspace(1,1,28) linspace(1,0,28) 0] ;
            b = [ 1 1 1 linspace(1,1,26) linspace(1,0,22) linspace(0,0,36) 0] ;

            a = [ 1 1 1 linspace(1,1,28) linspace(1,1,28) linspace(1,1,28) 1] ;
            c = [ r' g' b' a'] ;

%           if (0)
%               figure (4) 
%               colormap (c (:,1:3)) ;
%               image (1:88)
%               figure (5)
%               plot ( 1:88, c (:,1), 'ro', 1:88, c (:,2), 'go', ...
%                   1:88, c (:,3), 'bo') ;
%               pause
%           end

            c = round (255 * c) ;

        elseif (colorstyle == 7)

            c = hsv (88) ;
            c = c (end:-1:1,:) ;
            % no alpha
            c (:,4) = 1 ;

%           if (0)
%               figure (4)
%               colormap (c (:,1:3)) ;
%               image (1:88)
%               figure (5)
%               plot ( 1:88, c (:,1), 'ro', 1:88, c (:,2), 'go', ...
%                   1:88, c (:,3), 'bo') ;
%               pause
%           end

            c = round (255 * c) ;

        end

        % add an invisible edge at the end of the color list
        % c (89, :) = 255 * [1 1 1 1] ;  % white for now
          c (89, :) = [0 0 0 0] ;  % black, invisible

        alf256 = round (255 * min (1, linspace (1.5, 0.2, 256))) ;

        edgelen = (pos (i,:) - pos (j,:)) ;
        edgelen = sqrt (edgelen (:,1).^2 + edgelen (:,2).^2) ;
        maxlen = max (edgelen) + 1 ;
        minlen = min (edgelen) ;

        % map [minlen,maxlen] to 1:256:
        edgelen = 1 + 255 * ((edgelen - minlen) / (maxlen - minlen)) ;
        edgelen = floor (edgelen) ;

        % redo the plot, with colored edges
        gv3file = [gvdir gname '_color.gv'] ;
        f = fopen (gv3file, 'w') ;
        fprintf (f, 'graph %s {\n', gname) ;
        fprintf (f, 'size="11!"; bgcolor="#%02x%02x%02x%02x"; resolution="%g"; dpi="%g"; ordering=out\n', ...
            bgcolor, resolution, resolution) ;
        if (margin > 0)
            fprintf ('pad="%g"\n', margin) ;
            fprintf (f, 'pad="%g"\n', margin) ;
        end
        if (edgesfirst)
            fprintf (f, 'outputorder=edgesfirst\n') ;   %#ok
        else
            fprintf (f, 'outputorder=nodesfirst\n') ;
        end

        fprintf (f, 'node [label="", fontcolor="#ffffffff", fixedsize=10, shape="point", height=0.1, width=0.1];\n') ;

        fprintf (f, 'edge [') ;
        if (do_smooth)
            fprintf (f, 'headclip=false, tailclip=false, ') ;   %#ok
        end
        if (penwidth > 0)
            fprintf (f, 'penwidth=%g', penwidth) ;
        end
        fprintf (f, '];\n') ;
        
        node_color = zeros(n,3);
        for k = 1:n
            if x_vec(k) == 0 && y_vec(k) == 0
                node_color(k,:) = [255 0 0];
            elseif x_vec(k) == 1
                node_color(k,:) = [0 255 0];
            else
                node_color(k,:) = [0 0 255];
            end
        end
        
        for k = 1:n
            %fprintf (f, '%d [pos="%.2f,%.2f", color="#%02x%02x%02x%02x", xlabel="%.2f",];\n', k, pos (k,:), node_color(k, 1:3), 255, grad(k)) ;
            %fprintf('Node %d: x = %d, y = %d, color = #%02x%02x%02x%02x\n', k, x_vec(k), 
            fprintf (f, '%d [pos="%.2f,%.2f", color="#%02x%02x%02x%02x"];\n', k, pos (k,:), node_color(k, 1:3), 255) ;
        end
        nz = length (i) ;

        if (numel (colorstyle) > 1)
            colorkind = -1 ;
        else
            colorkind = colorstyle ;
        end
        
        if (colorkind == 0)
            % original
            for k = 1:nz
                if (x (k) == 89)
                    cedge = c (89, :) ; %#ok
                else
                    cedge = c (edgelen (k), :) ;    %#ok
                end
                %fprintf('k = %d: (%d, %d)\n', k, i(k), j(k));
                fprintf (f, '%d--%d [color="#%02x%02x%02x%02x"];\n', ...
                    i (k), j (k), [200, 200, 200, 255]) ;
                %edgelen(k)
                %c(edgelen(k),:)
            end

        elseif (colorkind == 1)
            for k = 1:nz
                cedge = c (x (k), :) ;
                % cedge (4) = alf256 (edgelen (k)) ;
                fprintf (f, '%d--%d [color="#%02x%02x%02x%02x"];\n', ...
                    i (k), j (k), cedge) ;
            end

        elseif (colorkind == 2 || colorkind == 3 || ...
            colorkind == 6 || colorkind == 7)
            for k = 1:nz
                cedge = c (x (k), :) ;
                if (x (k) ~= 89)
                    cedge (4) = alf256 (edgelen (k)) ;
                end
                fprintf (f, '%d--%d [color="#%02x%02x%02x%02x"];\n', ...
                    i (k), j (k), cedge) ;
            end

        elseif (colorkind == 4 || colorkind == -1)
            for k = 1:nz
                cedge = c (x (k), :) ;
                if (x (k) ~= 89)
                    cedge (4) = 255 ;
                end
                fprintf (f, '%d--%d [color="#%02x%02x%02x%02x"];\n', ...
                    i (k), j (k), cedge) ;
            end

        elseif (colorkind == 5)
            for k = 1:nz
                cedge = c (x (k), :) ;
                fprintf (f, '%d--%d [color="#%02x%02x%02x%02x"];\n', ...
                    i (k), j (k), cedge) ;
            end

        end

        fprintf (f, '}\n') ;
        fclose (f) ;

        % remove nodes
        if (ismac)
            filter = './svgfilter_mac' ;    %#ok
        else
            filter = './svgfilter_linux' ;  %#ok
        end
        
        if (DEBUG)
            fprintf ('plot: %s\n', plot_name) ; %#ok
        end
        
        cmd = 'neato -n' ;

        if (do_png)
            % create png
            if (DEBUG) 
                fprintf ('creating %s.png\n', plot_name) ; %#ok
            end
            tempout = [tempname '.png'] ;
            if (DEBUG) 
                fprintf ('%s%s %s -Tpng > %s', where, cmd, gv3file, tempout) ; %#ok
            end
            system (sprintf ('%s%s %s -Tpng > %s', where, cmd, gv3file, tempout)) ;
            movefile (tempout, [plot_name '.png'], 'f') ;
        end
        if (do_svg)
            % create svg
            if (DEBUG) %#ok
                fprintf ('creating %s.svg\n', plot_name) ;
            end
            tempout = [tempname '.svg'] ;
            % system (sprintf ('%s%s %s -Tsvg | %s > %s', where, cmd, gv3file, filter, tempout)) ;
            system (sprintf ('%s%s %s -Tsvg > %s', where, cmd, gv3file, tempout)) ;
            movefile (tempout, [plot_name '.svg'], 'f') ;
        end

        %delete (gv3file) ;
    end
end


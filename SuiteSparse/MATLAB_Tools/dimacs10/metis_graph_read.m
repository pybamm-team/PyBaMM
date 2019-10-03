function [A w fmt] = metis_graph_read (file)
%METIS_GRAPH_READ reads a graph file in METIS format into a MATLAB sparse matrix.
%
%   [A w fmt] = metis_graph_read (file)
%
% Returns a symmetric n-by-n sparse matrix A.  w is an array of node weights of
% size n-by-k where there are k weights for each node.  If k = 0 then the graph
% has no node weights.  Normally, the diagonal of A is all zero (except for
% the fmt=100 case, added for the DIMACS10 set).
%
% METIS defines 4 kinds of graphs, as determined by the 'fmt' code in the first
% line of the file.  The DIMACS10 data set adds a fifth kind:
%
%   0: no node or edge weights.  w is n-by-0.  A is binary.
%   1: no node weights, has edge weights.  w is n-by-0.  A is non-binary.
%  10: has node weights, no edge weights.  w is n-by-k with k > 0.  A is binary.
%  11: both node and edge weights.  w is n-by-k with k > 0.  A non-binary.
% 100: no edge or node weights.  w is n-by-0.  A is a representation of
%       a multigraph, where A(i,j) is the number of edges (i,j).  This
%       format is a DIMACS10 extension of the METIS format.
%
% If present, edge weights are > 0 but need not be integers.
% Node weights, if present, are integers >= 0.
%
% Example
%
%   % in the dimacs10/ directory:
%   [A w fmt] = metis_graph_read ('fig8d.graph')
%
% See also sprand, gallery

% Copyright 2011, Tim Davis

if (nargin ~= 1 || nargout > 3)
    error ('metis_graph:usage', ...
            'usage: [A w fmt] = metis_graph_read (filename)') ;
end

% read the file
[i j x w fmt] = metis_graph_read_mex (file) ;
n = size (w, 1) ;

is_symmetric = 1 ;
nneg = 0 ;
nself = 0 ;
ndupl = 0 ;

% make sure the edges weights are valid
try
    bad = find (x <= 0) ;
    nneg = length (bad) ;
    if (nneg > 0)
        fprintf ('\n    %d edge weights <= 0\n', nneg) ;
        for t = 1:min (nneg, 20)
            fprintf ('    A(%d,%d): %g\n', i(bad(t)), j(bad(t)), x(bad(t))) ;
        end
        if (nneg > 20)
            fprintf ('    ...\n') ;
        end
        % fix the bad edges, and continue looking for errors
        x (bad) = 1 ;
    end
catch me
    warning ('metis_graph:errorcheck', 'unable to check for errors ...') ;
    disp (me.message) ;
end

% convert from triplet format to MATLAB sparse matrix format
A = sparse (i, j, x, n, n) ;

try

    % the normal METIS formats (0, 1, 10, 11) do not allow multiple edges
    % or self-edges.  DIMACS10 allows for this.
    if (fmt ~= 100)

        % make sure there are no self-edges
        i2 = find (diag (A)) ;
        nself = length (i2) ;
        if (nself > 0)
            fprintf ('\n    %d self edges, fmt = %d\n', nself, fmt) ;
            for t = 1:min (nself, 20)
                fprintf ('    A(%d,%d): %g\n', ...
                    i2(t), i2(t), full(A(i2(t),i2(t)))) ;
            end
            if (nself > 20)
                fprintf ('    ...\n') ;
            end
        end
        clear i2

        % make sure the graph has no duplicate entries
        ndupl = 0 ;
        if (length (i) ~= nnz (A))
            A1 = sparse (i, j, 1, n, n) ;
            [i2 j2] = find (A1 > 1) ;
            ndupl = length (i2) ;
            fprintf ('\n    %d duplicate edges, fmt = %d\n', ndupl, fmt) ;
            for t = 1:min (ndupl, 20)
                fprintf ('    A(%d,%d): %g\n', ...
                    i2(t), j2(t), full (A(i2(t),j2(t)))) ;
            end
            if (ndupl > 20)
                fprintf ('    ...\n') ;
            end
        end
        clear A1 i2 j2
    end

    clear i j x

    % make sure the graph is symmetric
    is_symmetric = isequal (A, A') ;

catch me
    warning ('metis_graph:errorcheck', 'unable to check for errors ...') ;
    disp (me.message) ;
end

if (ndupl > 0 || nself > 0 || nneg > 0)
     fmt = 100 ;
     fprintf ('forcing fmt = 100.  Edge weights <= 0 set to 1.\n') ;
     warning ('metis_graph:invalid_edges', ...
        '%d duplicate edges, %d self-edges, %d edge weights <= 0', ...
        ndupl, nself, nneg) ;
end

% make sure the graph is symmetric
if (~is_symmetric)
    error ('metis_graph:unsymmetric', 'graph must be symmetric') ;
end

% make sure the node weights are valid
if (any (any (w < 0)) || any (any (w ~= fix (w))))
    error ('metis_graph:invalid_node_weights', ...
        'node weights must be integers >= 0') ;
end

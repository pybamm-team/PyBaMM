function metis_graph_test
%METIS_GRAPH_TEST tests the metis_graph installation
% Your current directory must be dimacs10/
%
% Example
%
%   metis_graph_test
%
% See also metis_graph_read, metis_graph_install

% Copyright 2011, Tim Davis

%-------------------------------------------------------------------------------
% tests with valid graphs
%-------------------------------------------------------------------------------

graphs = {
'fig8a.graph'
'fig8b.graph'
'fig8c.graph'
'fig8d.graph'
'adjnoun.graph'
'ilp_test.graph'
'multi.graph'
} ;

clf ;

for i = 1:length (graphs)
    fprintf ('\n----------------------------METIS graph %s :\n', graphs {i}) ;
    [A w fmt] = metis_graph_read (graphs {i}) ;
    k = size (w, 2) ;
    fprintf ('fmt: %3d : ', fmt) ;
    switch fmt
        case {0, 100}
            fprintf ('no node weights, no edge weights.\n') ;
        case 1
            fprintf ('no node weights, has edge weights.\n') ;
        case 10
            fprintf ('has %d node weight(s) per node, no edge weights.\n', k) ;
        case 11
            fprintf ('has %d node weight(s) per node, has edge weights.\n', k) ;
    end
    if (fmt == 100)
        fprintf ('DIMACS10 extension: may have self-edges and multiple\n') ;
        fprintf ('edges.  A(i,j) is the # of edges (i,j)\n') ;
    else
        fprintf ('No self-edges and no multiple edges present in the graph.\n');
    end
    subplot (2,4,i) ;
    spy (A) ;
    if (size (A,1) < 10)
        A = full (A) ;
        display (A) ;
    end
    title (graphs {i}) ;
    if (size (w,2) > 0)
        display (w) ;
    end
end

%-------------------------------------------------------------------------------
% error testing with invalid graphs or invalid usage
%-------------------------------------------------------------------------------

err = 0 ;
bad_graphs = {
'nosuchfile.graph'
'bad1.graph'
'bad2.graph'
'bad3.graph'
'bad4.graph'
'bad5.graph'
'bad6.graph'
'bad7.graph'
'bad8.graph'
'bad9.graph'
} ;

fprintf ('\nTesting error handling (errors and warnings are expected):\n') ;
lastwarn ('') ;
for i = 1:length (bad_graphs)
    fprintf ('%-20s : ', bad_graphs {i}) ;
    try
        [A w fmt] = metis_graph_read (bad_graphs {i}) ;                     %#ok
        if (isempty (lastwarn))
            err = err + 1 ;
        end
    catch me
        fprintf ('expected error: %s\n', me.message) ;
    end
end

fprintf ('invalid usage        : ') ;
try
    % too few input arguments
    [A w fmt] = metis_graph_read ;                                          %#ok
    err = err + 1 ;
catch me
    fprintf ('expected error: %s\n', me.message) ;
end

fprintf ('invalid usage        : ') ;
try
    % too many output arguments
    [i j x w fmt gunk] = metis_graph_read_mex ('fig8a.graph') ;             %#ok
    err = err + 1 ;
catch me
    fprintf ('expected error: %s\n', me.message) ;
end

fprintf ('invalid usage        : ') ;
try
    % invalid input
    [i j x w fmt] = metis_graph_read_mex (0) ;                              %#ok
    err = err + 1 ;
catch me
    fprintf ('expected error: %s\n', me.message) ;
end

if (err > 0)
    error ('%d errors not caught!', err) ;
end

fprintf ('\nAll tests passed.\n') ;

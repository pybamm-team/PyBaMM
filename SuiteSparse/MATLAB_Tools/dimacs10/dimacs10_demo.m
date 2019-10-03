function dimacs10_demo (arg)
%DIMACS10_DEMO reads in the DIMACS10 graphs.
% Reads them in and displays them in increasing order of nnz(S).
% Can also read in just a subset.
%
% Example
%   dimacs10_demo
%   dimacs10_demo (-10) ;           % just the smallest 10 graphs
%   dimacs10_demo ([1 3 5 8]) ;     % read just those four graphs
%
% See also dimacs10, ssget

% Copyright 2011, Timothy A Davis

lastwarn ('') ;
index = dimacs10 ;
[~,list] = sort (index.nnz) ;
if (nargin > 0)
    if (isscalar (arg) && arg < 0)
        list = list (1:(-arg)) ;
    else
        list = arg ;
    end
end
if (length (list) > 1)
    fprintf ('\ndimacs10_demo: testing %d graphs\n', length (list)) ;
end
for id = list (:)'
    [S name kind] = dimacs10 (id) ;
    fprintf ('%3d %s : %s : n: %d nnz %d\n', id, name, kind, size(S,1), nnz(S));
    ssweb (index.ssname {id}) ;
    clf
    spy (S) ;
    drawnow
end

S = dimacs10 ('adjnoun') ;
spy (S)
title ('clustering/adjnoun') ;

% test error handling
fail = 0 ;
try
    S = dimacs10 (0) ;                                                      %#ok
    fail = 1 ;
catch me                                                                    %#ok
end
try
    S = dimacs10 ('crud') ;                                                 %#ok
    fail = 1 ;
catch me                                                                    %#ok
end
try
    S = dimacs10 (1,1) ;                                                    %#ok
    fail = 1 ;
catch me                                                                    %#ok
end

if (fail)
    error ('test failed') ;
end

if (length (list) > 1)
    fprintf ('\ndimacs10: all tests passed\n') ;
end

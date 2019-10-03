function dimacs10_install (demo)
%DIMACS10_INSTALL compiles and installs dimacs10 for use in MATLAB.
% Your current working directory must be the one containing this
% file (dimacs10/dimacs10_install.m).  Also runs a few short tests.
%
% Example
%
%   dimacs10_install
%
% See also dimacs10, metis_graph_read, ssget.

% Copyright 2011, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    demo = 1 ;
end

% add dimacs10 to the path
addpath (pwd) ;

% compile the mexFunctions
if (~isempty (strfind (computer, '64')))
    fprintf ('Compiling dimacs10 (64-bit)\n') ;
    mex -largeArrayDims dimacs10_convert_to_graph.c
    mex -largeArrayDims metis_graph_read_mex.c
else
    fprintf ('Compiling dimacs10 (32-bit)\n') ;
    mex dimacs10_convert_to_graph.c
    mex metis_graph_read_mex.c
end

% run some tests
if (demo)
    metis_graph_test ;
    dimacs10_demo (-1) ;
    dimacs10_demo ([7 15 5 6 109 23 82 57 24]) ;
end


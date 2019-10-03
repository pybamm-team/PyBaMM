% MATLAB interface to the Mongoose partitioning library
%
% Files
%   coarsen         - coarsen a graph unsafely but quickly.
%   edgecut         - find an edge separator in a graph.
%   edgecut_options - create a struct of default options for edge cuts.
%   mongoose_demo   - a simple demo of Mongoose graph partitioner.
%   mongoose_make   - compiles the Mongoose mexFunctions.
%   mongoose_plot   - use graphvis to create a plot of a graph.
%   mongoose_test   - a simple test of Mongoose.
%   safe_coarsen    - coarsen a graph after attempting to sanitize it.
%   safe_edgecut    - sanitizes and computes an edge cut of a graph.
%   sanitize        - sanitize a sparse adjacency matrix for graph partitioning.
%
%  Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager
%  http://suitesparse.com

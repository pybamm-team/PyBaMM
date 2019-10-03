function [G_coarse, A_coarse, map] = coarsen (G, O, A)      %#ok
%COARSEN coarsen a graph unsafely but quickly.
%   coarsen(G) computes a matching of vertices in the graph G and then coarsens
%   the graph by combining all matched vertices into supervertices. It assumes
%   that the matrix G provided has an all zero diagonal, is symmetric, and has
%   all positive edge weights. With no option struct specified, the coarsening
%   is done using a combination of heavy-edge matching and other more
%   aggressive techniques to avoid stalling. An optional vertex weight vector A
%   can also be specified.  Note that mongoose_coarsen_mex does NOT check to
%   see if the supplied matrix is of the correct form, and may provide
%   erroneous results if used incorrectly.
%
%   [G_coarse, A_coarse, map] = coarsen(G) coarsens a graph represented with
%   sparse adjacency matrix G. G_coarse is the coarsened adjacency matrix,
%   A_coarse is the coarsened array of vertex weights, and map is a mapping of
%   original vertices to coarsened vertices.
%
%   [G_coarse, A_coarse, map] = coarsen(G, O) uses the option struct O to
%   specify coarsening options (e.g. matching strategies).
%
%   [G_coarse, A_coarse, map] = coarsen(G, O, A) uses the array A as vertex
%   weights, such that A(i) is the vertex weight of vertex i. If A is not
%   specified, A is assumed to be an array of all ones (all weights are one).
%
%   Example:
%       Prob = ssget('DNVS/troll'); A = Prob.A;
%       G = sanitize(A);
%       G_coarse = coarsen(G);
%       subplot(1,2,1); spy(G); subplot(1,2,2); spy(G_coarse);
%
%   See also SAFE_COARSEN, EDGECUT_OPTIONS.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

error ('coarsen mexFunction not found') ;

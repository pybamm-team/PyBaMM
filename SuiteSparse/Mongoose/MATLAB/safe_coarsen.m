function [G_coarse, A_coarse, map] = safe_coarsen(G, O, A)
%SAFE_COARSEN coarsen a graph after attempting to sanitize it.
%   safe_coarsen(G) computes a matching of vertices in the graph G
%   and then coarsens the graph by combining all matched vertices into
%   supervertices. With no option struct specified, the coarsening is done 
%   using a combination of heavy-edge matching and other more aggressive
%   techniques to avoid stalling. An optional vertex weight vector A can 
%   also be specified, and a fine-to-coarse mapping of vertices can also
%   be obtained (e.g. map(vertex_fine) = vertex_coarse).
%
%   [G_coarse, A_coarse, map] = safe_coarsen(G) sanitizes and then coarsens a
%   graph represented with sparse adjacency matrix G. G_coarse is the coarsened
%   adjacency matrix, A_coarse is the coarsened array of vertex weights, and
%   map is a mapping of original vertices to coarsened vertices.
%
%   [G_coarse, A_coarse, map] = safe_coarsen(G, O) uses the option struct O to
%   specify coarsening options (e.g. matching strategies).
%
%   [G_coarse, A_coarse, map] = safe_coarsen(G, O, A) uses the array A as
%   vertex weights, such that A(i) is the vertex weight of vertex i. If A is
%   not specified, A is assumed to be an array of all ones (all weights are 1).
%
%   Example:
%       Prob = ssget('DNVS/troll'); G = Prob.A;
%       G_coarse = safe_coarsen(G);
%       spy(G_coarse);
%
%   See also COARSEN, EDGECUT_OPTIONS.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

G_safe = sanitize(G);

if (nargin == 1)
    [G_coarse, A_coarse, map] = coarsen(G_safe);
elseif (nargin == 2)
    [G_coarse, A_coarse, map] = coarsen(G_safe, O);
else
    [G_coarse, A_coarse, map] = coarsen(G_safe, O, A);
end

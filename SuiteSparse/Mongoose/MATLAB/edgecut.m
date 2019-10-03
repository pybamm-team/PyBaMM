function partition = edgecut(G, O, A)   %#ok
%EDGECUT find an edge separator in a graph.
%   partition = edgecut(G) uses a multilevel hybrid combinatoric and quadratic
%   programming algorithm to compute a partitioning of the graph G. With no
%   option struct specified, the target is for each part to contain 50% of the
%   graph's vertices, and the coarsening is done using a combination of
%   heavy-edge matching and other more aggressive techniques to avoid stalling.
%
%   partition = EDGECUT(G) assumes default options and no vertex weights (i.e.
%   all vertex weights are 1). The partition is returned as a binary array.
%
%   partition = EDGECUT(G, O) uses the options struct to define how the edge
%   cut algorithms are run.
%
%   partition = EDGECUT(G, O, A) initializes the graph with vertex weights
%   provided in the array A such that A(i) is the vertex weight of vertex i.
%
%   Example:
%       Prob = ssget('HB/494_bus'); A = Prob.A;
%       A = sanitize(A);
%       part = edgecut(A);
%       perm = [find(part) find(1-part)];
%       A_perm = A(perm, perm); % Permute the matrix
%       spy(A_perm);
%
%   See also EDGECUT_OPTIONS, SAFE_EDGECUT.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

error ('edgecut mexFunction not found') ;

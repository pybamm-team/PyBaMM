function partition = safe_edgecut(G,O,A)
%SAFE_EDGECUT sanitizes and computes an edge cut of a graph.
%   safe_edgecut sanitizes the graph adjacency matrix by removing its diagonal
%   and attempting to form a symmetric matrix from the input. After
%   sanitization, an edge cut of the graph is computed and returned as a binary
%   array.
%
%   partition = SAFE_EDGECUT(G) assumes default options and no vertex weights
%   (i.e. all vertex weights are 1). The partition is returned as a binary
%   array.
%
%   partition = SAFE_EDGECUT(G, O) uses the options struct to define how the
%   edge cut algorithms are run.
%
%   partition = SAFE_EDGECUT(G, O, A) initializes the graph with vertex weights
%   provided in the array A such that A(i) is the vertex weight of vertex i.
%
%   Example:
%       Prob = ssget('HB/494_bus'); A = Prob.A;
%       part = safe_edgecut(A);
%       perm = [find(part) find(1-part)];
%       A_perm = A(perm, perm); % Permute the matrix
%       spy(A_perm);
%
%   See also EDGECUT, EDGECUT_OPTIONS.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

G_safe = sanitize(G);

if nargin == 1
    partition = edgecut(G_safe);
elseif nargin == 2
    partition = edgecut(G_safe,O);
else
    partition = edgecut(G_safe,O,A);
end

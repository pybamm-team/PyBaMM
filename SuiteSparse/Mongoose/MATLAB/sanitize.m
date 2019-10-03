function A_safe = sanitize (A)  %#ok
%SANITIZE sanitize a sparse adjacency matrix for graph partitioning.
%   A_safe = sanitize(A) sanitizes an adjacency matrix by removing its diagonal
%   and attempting to form a symmetric matrix (i.e. an undirected graph). If
%   the input matrix is not symmetric, a symmetric matrix is formed using
%   A_safe = (A + A')/2.
%
%   Example:
%       Prob = ssget('HB/west0479'); A = Prob.A;
%       A_safe = sanitize(A);
%       subplot(1,2,1); spy(A); subplot(1,2,2); spy(A_safe);
%
%   See also SAFE_EDGECUT, SAFE_COARSEN.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

error ('sanitize mexFunction not found') ;

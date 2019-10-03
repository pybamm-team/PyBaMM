function mongoose_demo
%MONGOOSE_DEMO a simple demo of Mongoose graph partitioner.
%
% A simple demo to demonstrate Mongoose. Reads in a matrix, sanitizes it,
% and partitions it several different ways.
%
% Example:
%   mongoose_demo
%
% See also mongoose_test, mongoose_make

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

% Obtain the adjacency matrix
matfile_data = matfile('494_bus.mat');
Prob = matfile_data.Problem;
A = Prob.A;
[m, ~] = size(A);

% Sanitize the adjacency matrix: remove diagonal elements, make edge weights 
% positive, and make sure it is symmetric. If the matrix is not symmetric 
% or square, a symmetric matrix (A+A')/2 is built.
A = sanitize(A);

% Create a vertex weight vector and create a heavy vertex
V = ones(1,m);
V(10) = 300;

% Create a set of default options and modify the target balance
O = edgecut_options();
O.target_split = 0.3;

% Run Mongoose to partition the graph with edge and vertex weights.
partVert = edgecut(A, O, V);

fprintf('\n\nPartitioning graph with edge and vertex weights\n\n');
fprintf('=== Cut Info ===\n');
fprintf('Cut Size:   %d\n', full(sum(partVert .* sum(sign(A)))));
fprintf('Cut Weight: %d\n\n', full(sum(partVert .* sum(A))));
fprintf('=== Balance Info ===\n');
fprintf('Target Split:     %g\n', O.target_split);
fprintf('Actual Split:     %1.4f\n', sum(partVert .* V) / sum(V));
fprintf('Unweighted Split: %1.4f\n', sum(partVert) / m);

% Run Mongoose to partition the graph with no vertex weights.
partEdge = edgecut(A, O);

fprintf('\n\nPartitioning graph with only edge weights\n\n');
fprintf('=== Cut Info ===\n');
fprintf('Cut Size:   %d\n', full(sum(partEdge .* sum(sign(A)))));
fprintf('Cut Weight: %d\n\n', full(sum(partEdge .* sum(A))));
fprintf('=== Balance Info ===\n');
fprintf('Target Split:     %g\n', O.target_split);
fprintf('Actual Split:     %1.4f\n', sum(partEdge) / m);

% Remove edge weights
A = sanitize(A, 1);

% Run Mongoose to partition the graph with no edge weights.
% Note that only the graph is passed as an argument, so default
% options are assumed.
partPattern = edgecut(A);

fprintf('\n\nPartitioning graph with only edge weights\n\n');
fprintf('=== Cut Info ===\n');
fprintf('Cut Size:   %d\n', full(sum(partPattern .* sum(sign(A)))));
fprintf('Cut Weight: %d\n\n', full(sum(partPattern .* sum(A))));
fprintf('=== Balance Info ===\n');
fprintf('Target Split:     0.5 (default)\n');
fprintf('Actual Split:     %1.4f\n', sum(partPattern) / m);

f = gcf ;
clf ;
f.Position = [100, 100, 1000, 400] ;

% Plot the original matrix before permutation
subplot(1, 3, 1);
spy(A)
title('HB/494\_bus: Before Partitioning')

% Plot the matrix after the permutation
subplot(1, 3, 2);
perm = [find(partPattern) find(1-partPattern)];
A_perm = A(perm, perm); % Permute the matrix
spy(A_perm)
hold on
nleft = length (find (partPattern)) ;
plot ([1 m], [nleft nleft], 'g') ;
plot ([nleft nleft], [1 m], 'g') ;
hold off
title('HB/494\_bus: After (50/50) Partitioning')

% Plot the matrix after the 30/70 permutation
subplot(1, 3, 3);
perm = [find(partEdge) find(1-partEdge)];
A_perm = A(perm, perm); % Permute the matrix
spy(A_perm)
hold on
nleft = length (find (partEdge)) ;
plot ([1 m], [nleft nleft], 'g') ;
plot ([nleft nleft], [1 m], 'g') ;
hold off
title('HB/494\_bus: After (30/70) Partitioning')

end

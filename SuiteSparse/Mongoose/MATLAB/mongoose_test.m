function mongoose_test
%MONGOOSE_TEST a simple test of Mongoose.
%
% Example:
%   mongoose_test
%
% See also mongoose_demo.

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

% A simple demo to demonstrate and test Mongoose. Reads in a matrix, 
% sanitizes it, partitions it, and displays the results.

% Obtain the matrix
matfile_data = matfile('494_bus.mat');
Prob = matfile_data.Problem;
A = Prob.A;

% Sanitize the matrix: remove diagonal elements, check for positive edge
% weights, and make sure it is symmetric.
A = sanitize(A);

% Run Mongoose to partition the graph.
part = edgecut(A);

% Create a Graphviz plot of the graph and solution
if (has_graphviz)
    viz = 1;
else
    viz = 0;
end

f = gcf ;
clf ;
f.Position = [100, 100, 1000, 400] ;

if (viz)
    plotname = sanitize_plotname(Prob.name);
    mongoose_plot(A, part, 1-part, plotname);
    subplot(1, 2+viz, 1);
    img = imread([plotname '.png']);
    imshow(img)
    title('HB/494\_bus Graphviz Visualization')
end

% Plot the original matrix before permutation
subplot(1, 2+viz, 1+viz);
spy(A)
title('HB/494\_bus Before Partitioning')

% Plot the matrix after the permutation
subplot(1, 2+viz, 2+viz);
perm = [find(part) find(1-part)];
A_perm = A(perm, perm); % Permute the matrix
spy(A_perm)
hold on
m = size (A,1) ;
nleft = length (find (part)) ;
plot ([1 m], [nleft nleft], 'g') ;
plot ([nleft nleft], [1 m], 'g') ;
hold off
title('HB/494\_bus After Partitioning')

end

% Sanitize the plot name - neato does not like slashes or dashes.
function new_plotname = sanitize_plotname(old_plotname)
    new_plotname = strrep(old_plotname, '/', '_');
    new_plotname = strrep(new_plotname, '-', '_');
end

% Check if Graphviz (specifically neato) is installed.
function bool = has_graphviz
    if (ismac)
        where = '/usr/local/bin/' ;
    else
        where = '/usr/bin/' ;
    end
    command = 'neato -V';
    [status, ~] = system(sprintf('%s%s', where, command));
    bool = (status == 0);
end


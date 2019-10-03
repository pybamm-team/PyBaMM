function [p, r, nc, G, xy] = find_components (A,sorted)
%FIND_COMPONENTS finds all connected components of an image.
% Two pixels in an image are in the same connected component if and only if
% they are adjacent and have the same value.  "Adjacent" in this context means
% north/south or east/west, not diagonally.  That is, A(2,3) is adjacent to
% A(2,2), A(2,4), A(1,3), and A(3,3) only, and is not adjacent to A(3,4).
%
% Let [p r] = find_components(A) where A is m-by-n.  The result is a permutation
% p and component boundaries r.  p is a permutation of 1:m*n, and refers to the
% linear indexing of A.  That is, A(i,j) is refered to as A(i+j*m) in the list
% p.  The kth connected component consists of A (p (r(k) : r(k+1)-1)).
% The number of connected components is nc = length (r) - 1.  The components
% are ordered by the smallest linear index in each component (assuming you have
% MATLAB 7.5 or later, which uses DMPERM from CSparse; otherwise the ordering
% is not defined).
%
% With a single output argument, c = find_components (A) just returns a list
% of the nodes in the largest component (the component with the largest value
% if ties, and if there are 2 components still tied, return the one containing
% the smallest node index).
%
% The are no restrictions on the image A except that it must be a 2D matrix,
% and the operator "==" must be defined on its entries.  If sorting of the
% components by size is requested or if the largest component is requested,
% double(A) must be computable.
%
% Usage:
%   c = find_components (A) ;       % just return nodes in the largest component
%   [p r nc G xy] = find_components (A) ;       % sorted by least node number
%   [p r nc G xy] = find_components (A, 1) ;    % sorted by size, ties by value
%   find_components (A) ;                       % just plot the graph
%
% Example:
%
%   A = [ 1 2 2 3
%         1 1 2 3
%         0 0 1 2
%         0 1 3 3 ]
%   [p r nc] = find_components (A,1)
%   [m n] = size (A) ;
%   for k = 1:nc
%       a = A (p (r (k))) ;
%       fprintf ('\ncomponent %d, size %d, value %d\n', k, r(k+1)-r(k), a) ;
%       C = nan (m,n) ;
%       C (p (r (k) : r (k+1)-1)) =  a ;
%       fprintf ('A = \n') ; disp (A) ;
%       fprintf ('the component = \n') ; disp (C) ;
%       input (': ', 's') ;
%   end
%
% The optional outputs G and xy give a graph representation of the problem
% which can be viewed with gplot(G,xy).
%
% See also LARGEST_COMPONENT, FIND_COMPONENTS_EXAMPLE, DMPERM, GPLOT

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% number the nodes
%-------------------------------------------------------------------------------

% K is a matrix containing the linear index into A.  For the example above,
%
% K = [ 1 5  9 13
%       2 6 10 14
%       3 7 11 15
%       4 8 12 16 ]
%
% The "nodes" of A are simply their linear indices.  For example, A(2,3)
% is called node 10.

[m n] = size (A) ;
N = m*n ;                       % N = number of nodes in A
K = reshape (1:N, m, n) ;

%-------------------------------------------------------------------------------
% look to the east
%-------------------------------------------------------------------------------

% If A(i,j) == A(i,j+1), then East(i,j) is set to K(i,j+1).  That is, East(i,j)
% gives the node number of the Eastern neighbor of the node A(i,j).
%
% In this example, East = [
%   0 9  0  0
%   6 0  0  0
%   7 0  0  0
%   0 0 16  0 ]
%
% because (for example) node 5 has an Eastern neighbor, node 9 (A(5) == A(9)).

East = [(K (:,2:n) .* (A (:,1:n-1) == A (:,2:n))) zeros(m,1)] ;

% E gives the node numbers of all nodes with Eastern neighbors.  For example,
% E = [2 3 5 12]' ;

E = find (East) ;

%-------------------------------------------------------------------------------
% look to the south
%-------------------------------------------------------------------------------

% If A(i,j) == A(i+1,j), then South(i,j) is set to K(i+1,j).  That is,
% South(i,j) gives the node number of the Southern neighbor of the node A(i,j).
%
% In this example, South = [
%   2 0 10 14
%   0 0  0  0
%   4 0  0  0
%   0 0  0  0 ]
%
% because (for example) node 4 has a Southern neighbor, node 4 (A(3) == A(4)).
% Then S gives the node numbers of all nodes with Southern neighbors.

South = [(K (2:m,:) .* (A (1:m-1,:) == A (2:m,:))) ; zeros(1,n)] ;
S = find (South) ;

%-------------------------------------------------------------------------------
% create the graph G
%-------------------------------------------------------------------------------

% The graph G is N-by-N for an image A of size m-by-n with N=m*n nodes.
% There is an edge between two nodes if they are neighbors (that is, if the
% two nodes are adjacent and have the same value).  A diagonal is added to
% the graph so that DMPERM knows that the matrix G does not need to first be
% permuted to reveal a maximum matching ... that step is skipped.
%
% Ignoring the diagonal entries, in this example, G = 
%
%      (2,1)        1
%      (1,2)        1
%      (6,2)        1
%      (4,3)        1
%      (7,3)        1
%      (3,4)        1
%      (9,5)        1
%      (2,6)        1
%      (3,7)        1
%      (5,9)        1
%     (10,9)        1
%      (9,10)       1
%     (16,12)       1
%     (14,13)       1
%     (13,14)       1
%     (12,16)       1
%
% If drawn as a 4-by-4 mesh, with edges between neighbors, G looks like this:
%
%       1   5 -  9   13
%       |        |    |
%       2 - 6   10   14
%
%       3 - 7   11   15
%       |
%       4   8   12 - 16
%
%  If the nodes are labeled according the value of A, the graph looks like this:
%
%       1   2 -  2    3
%       |        |    |
%       1 - 1    2    3
%
%       0 - 0    1   15
%       |
%       0   1    3 -  3

G = sparse ([K(E) ; K(S)], [East(E) ; South(S)], 1, N, N) ;
clear K East E South S      % free up some space in case the problem is large
G = G + G' + speye (N) ;

%-------------------------------------------------------------------------------
% find the connected components of G
%-------------------------------------------------------------------------------

% Note that p==q and r==s, since the matrix G is square with zero-free diagonal.
% nc gives the number of connected components.

[p q r s] = dmperm (G) ;            %#ok  (s is unused, present for comments)
nc = length (r) - 1 ;

%-------------------------------------------------------------------------------
% sort the components by size, if requested (the rest is all optional)
%-------------------------------------------------------------------------------

if (nargin < 2)
    % the default is not to sort the components
    sorted = 0 ;
end

if (nargout == 1)
    % largest component is requested, so we must sort
    sorted = 1 ;
end

if (sorted)

    [ignore i] = sortrows ([diff(r)' double(A(p(r(1:end-1)))')], [-1 -2]) ;

    if (nargout == 1)

        % just return the largest component
        c = i (1) ;
        p = p (r (c) : r (c + 1) - 1) ;

    else

        % sort all the components (this can be costly)
        p2 = zeros (1,N) ;
        r2 = zeros (1,nc+1) ;
        k2 = 0 ;
        for k = 1:nc
            % get the nodes and the size of the kth largest component, c
            c = i (k) ;
            nodes = p (r (c) : r (c + 1) - 1) ;
            csize = length (nodes) ;
            % place the nodes in the new output permutation
            p2 (k2+1 : k2+csize) = nodes ;
            r2 (k) = k2+1 ;
            k2 = k2 + csize ;
        end
        r2 (nc+1) = N+1 ;
        p = p2 ;
        r = r2 ;
    end
end

%-------------------------------------------------------------------------------
% return the XY coordinates, if requested or if the graph is to be plotted
%-------------------------------------------------------------------------------

if (nargout >= 5 || nargout == 0)
    x = repmat (1:n, n, 1) ;
    y = repmat (m:-1:1, 1, m) ;
    xy = [x(:) y(:)] ;
end

%-------------------------------------------------------------------------------
% plot the graph if no outputs requested
%-------------------------------------------------------------------------------

if (nargout == 0)
    if (N < 100)
        gplot (G, xy, 'o-') ;
    else
        gplot (G, xy) ;
    end
    axis ([0 n+1 0 m+1]) ;
    title (sprintf ('%d connected components', nc)) ;
end


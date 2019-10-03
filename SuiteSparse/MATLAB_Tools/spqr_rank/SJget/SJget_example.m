%SJGET_EXAMPLE a demo for SJget.
%   This example script gets the index file of the SJSU Singular matrix collection,
%   and then loads in all symmetric non-binary matrices, in increasing order of
%   number of rows in the matrix.
%
%   Example:
%       type SJget_example ; % to see an example of how to use SJget
%
%   See also SJget, SJweb, SJgrep.

%   Derived from the ssget toolbox on March 18, 2008.
%   Copyright 2007, Tim Davis, University of Florida.

%   modified by L. Foster 09/17/2008

type SJget_example ;

index = SJget ;
% find all symmetric matrices that are not binary,
% have at least 4000 column and have a gap in the singular
% values at the numerical rank of at least 1000
f = find (index.numerical_symmetry == 1 & ~index.isBinary & ...
    index.ncols >= 4000 & index.gap >= 1000 );
% sort by the dimension of the numerical null space
[y, j] = sort (index.ncols (f) - index.numrank (f) ) ;
f = f (j) ;

for i = f
    fprintf ('Loading %s%s%s, please wait ...\n', ...
	index.Group {i}, filesep, index.Name {i}) ;
    Problem = SJget (i,index) ;
    %display the problem structure
    disp (Problem) ;
    % display a plot of the singular values
    SJplot(Problem) ;
    shg
    disp(' ')
    disp(['dim. of numerical null space = ', ...
        num2str(index.ncols (i) - index.numrank (i))]);
    disp(' ')
    pause(1)
    %display the web page with matrix details
    SJweb (i) ;
    pause(1)
end


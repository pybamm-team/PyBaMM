function sspages (list)
%SSPAGES create images for each matrix in SuiteSparse Matrix Collection
% Usage: sspages (list)
%
% list: list of matrix id's to process.  All matrices in the collection are
%   processed if not present.
%
% Example:
%
%   sspages             % create all the pages
%   sspages (1:10)      % create pages for just matrices 1 to 10
%
% See also sspage, ssget.

% Copyright 2006-2018, Timothy A. Davis

index = ssget ;
if (nargin < 1)
    list = 1:length (index.nrows) ;
end
for i = list
    sspage (i, index) ;
end


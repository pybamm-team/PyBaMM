function klu_test (nmat)
%klu_test KLU test
% Example:
%   klu_test
%
% See also klu

% Copyright 2004-2012, University of Florida

if (nargin < 1)
    nmat = 200 ;
end

test1 (nmat) ;
test2 (nmat) ;
test3 ;
test4 (nmat) ;
test5  ;

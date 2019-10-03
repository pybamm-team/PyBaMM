function reset_rand
%RESET_RAND resets the state of rand
%
% Example
%   reset_rand
%
% See also RandStream, rand, rng

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (verLessThan ('matlab', '7.12'))
    rand ('seed', 0) ;                                                      %#ok
else
    rng ('default') ;
end

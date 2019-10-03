function test3
%TEST3 test sparse on int8, int16, and logical
% Example:
%   test3
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test3: test sparse on int8, int16, and logical\n') ;

clear all
c =  ['a' 'b' 0 'd']							    %#ok

fprintf ('---- char:\n') ;
sparse2(c)
nzmax(ans)  %#ok
whos

fprintf ('---- char transposed:\n') ;
sparse2(c')
nzmax(ans)  %#ok
whos

fprintf ('---- int8:\n') ;
sparse2(int8(c))
nzmax(ans)  %#ok
whos

fprintf ('---- int16:\n') ;
sparse2 (int16(c))
nzmax(ans)  %#ok
whos

fprintf ('---- logical (using the MATLAB "sparse"):\n') ;
s = logical(rand(4) > .5)						    %#ok
sparse (s)
nzmax(ans)  %#ok
whos

fprintf ('---- logical (using sparse2):\n') ;
sparse2(s)
nzmax(ans)  %#ok
whos

fprintf ('---- double (using the MATLAB "sparse"):\n') ;
x = rand(4)								    %#ok
sparse (x > .5)								    %#ok
nzmax(ans)  %#ok
whos

fprintf ('---- double (using sparse2):\n') ;
sparse2 (x > .5)							    %#ok
nzmax(ans)  %#ok
whos

fprintf ('test3 passed\n') ;

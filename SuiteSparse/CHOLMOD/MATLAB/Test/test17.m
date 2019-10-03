function test17
%TEST17 test lchol on a few large matrices
% Example:
%   test17
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test17: test lchol on a few large matrices\n') ;

rand ('state',1) ;
randn ('state',1) ;

Prob = UFget (887)							%#ok
A = Prob.A ;
[L,s,p] = lchol (A) ;							%#ok
norm (L,1)

clear all

Prob = UFget (936)							%#ok
A = Prob.A ;
[L,s,p] = lchol (A) ;							%#ok
norm (L,1)								%#ok

clear all

Prob = UFget (887)							%#ok
A = Prob.A ;
[L,s,p] = lchol (A) ;							%#ok
norm (L,1)								%#ok

function test16
%TEST16 test cholmod2 on a large matrix
% Example:
%   test16
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test16: test cholmod2 on a large matrix\n') ;

rand ('state',1) ;
randn ('state',1) ;

Prob = UFget (936)							    %#ok
A = Prob.A ;
% tic
% [L,s,p] = lchol (A) ;
% toc
% norm (L,1)

n = size (A,1) ;
b = rand (n,1) ;
tic
x = cholmod2(A,b) ;
t = toc ;
fprintf ('time %g\n', t) ;
err = norm (A*x-b) ;

if (err > 1e-5)
    error ('!') ;
end

fprintf ('test16 passed\n') ;

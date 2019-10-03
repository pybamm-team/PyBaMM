function test19
%TEST19 look for NaN's from lchol (caused by Intel MKL 7.x bug)
% Example:
%   test19
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test19: look for NaN''s from lchol (caused by Intel MKL 7.x bug)\n') ;

Prob = UFget (936)							    %#ok
A = Prob.A ;
[p count] = analyze (A) ;
A = A (p,p) ;
tic
L = lchol (A) ;
t = toc ;
fl = sum (count.^2) ;
fprintf ('mflop rate: %8.2f\n', 1e-6*fl/t) ;
n = size (L,1) ;
for k = 1:n
    if (any (isnan (L (:,k))))
	k								    %#ok
	error ('!') ;
    end
end

fprintf ('test19 passed; you have a NaN-free BLAS (must not be MKL 7.x...)\n') ;

function test5
%TEST5 test sparse2
% Example:
%   test5
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test5: test sparse2\n') ;

randn ('state', 0) ;
rand  ('state', 0) ;

A = sprandn (10,20,0.2) ;
[m n ] = size (A) ;

[i j x] = find (A) ;
% [i2 j2 x2] = cholmod_find (A) ;
% if (any (i ~= i2))
%    error ('i!') ;
%end
%if (any (j ~= j2))
%    error ('j!') ;
%end
%if (any (x ~= x2))
%    error ('x!') ;
%end

% full (sum (spones (A')))

C = sparse (i,j,x,m,n) ;
B = sparse2 (i,j,x,m,n) ;
err = norm(A-B,1) ;
if (err > 0)
    error ('dtri 1') ;
end
err = norm(C-B,1) ;
if (err > 0)
    error ('dtri 1b') ;
end

nz = length (x) ;
p = randperm (nz) ;

i2 = i(p) ;
j2 = j(p) ;
x2 = x(p) ;								    %#ok

B = sparse2 (i,j,x,m,n) ;
err = norm(A-B,1) ;
if (err > 0)
    error ('dtri 2') ;
end

ii = [i2 ; i2] ;
jj = [j2 ; j2] ;
xx = rand (2*nz,1) ;

C = sparse (ii,jj,xx,m,n) ;
D = sparse2 (ii,jj,xx,m,n) ;
err = norm (C-D,1) ;
if (err > 0)
    error ('dtri 3') ;
end

% E = sparse2 (ii,jj,xx,n,n,+1) ;
E = sparse (min(ii,jj), max(ii,jj), xx, n, n) ;
F = sparse (min(ii,jj), max(ii,jj), xx, n, n) ;
err = norm (E-F,1) ;
if (err > 0)
    error ('dtri 4') ;
end

% E = sparse2 (ii,jj,xx,n,n,-1) ;
E = sparse (max(ii,jj), min(ii,jj), xx, n, n) ;
F = sparse (max(ii,jj), min(ii,jj), xx, n, n) ;
err = norm (E-F,1) ;
if (err > 0)
    error ('dtri 5') ;
end

[i1 j1 x1] = find (F) ;							    %#ok
% [i2 j2 x2] = cholmod_find (F) ;
% if (any (i1 ~= i2))
%     error ('i!') ;
% end
% if (any (j1 ~= j2))
%     error ('j!') ;
% end
% if (any (x1 ~= x2))
%     error ('x!') ;
% end

fprintf ('test5 passed\n') ;

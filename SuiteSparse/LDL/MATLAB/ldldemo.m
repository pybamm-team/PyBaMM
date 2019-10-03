function ldldemo
%LDLDEMO demo program for LDL
%
% Example:
%   ldldemo
%
% See also ldlsparse.

% Copyright 2006-2007 by Timothy A. Davis, http://www.suitesparse.com

% compile the LDLSPARSE and LDLSYMBOL mexFunctions
help ldlsparse

fprintf ('\nTesting ldlsparse and ldlsymbol:\n') ;

% create a small random symmetric positive definite sparse matrix
n = 100 ;
d = 0.03 ;
rand ('state', 0) ;
randn ('state', 0) ;
A = sprandn (n, n, d) ;
A = speye (n) + A*A' ;
b = randn (n, 1) ;

clf
subplot (2,2,1) ;
spy (A) ;
title ('original matrix') ;

% permute for sparsity
p = symamd (A) ;
C = A (p,p) ;

subplot (2,2,2) ;
spy (C) ;
title ('permuted matrix') ;
drawnow

% factorize, without using ldlsparse's internal permutation
[L, D, Parent, fl] = ldlsparse (C) ;
L = L + speye (n) ;
err = norm (L*D*L' - C, 1) ;
fprintf ('norm (LDL''-PAP'') = %g\n', err) ;

% solve Ax=b
x = L' \ (D \ (L \ (b (p)))) ;
x (p) = x ;
resid = norm (A*x-b) ;
fprintf ('residual %g for ldlsparse, flops %10.1f\n', resid, fl) ;

% solve Ax=b with one call to ldlsparse
x = ldlsparse (C, [ ], b (p)) ;
x (p) = x ;
resid = norm (A*x-b) ;
fprintf ('residual %g for ldlsparse solve\n', resid) ;

subplot (2,2,3) ;
spy (L + D + L') ;
title ('L+D+L''') ;

subplot (2,2,4) ;
treeplot (Parent)
title ('elimination tree') ;

% try ldlrow (this will be slow)
[L, D] = ldlrow (C) ;
x = L' \ (D \ (L \ (b (p)))) ;
x (p) = x ;
resid = norm (A*x-b) ;
fprintf ('residual %g for ldlrow.m\n', resid) ;

% factorize, using ldlsparse's internal permutation
[L, D, Parent, fl] = ldlsparse (A, p) ;
L = L + speye (n) ;
err = norm (L*D*L' - C, 1) ;
fprintf ('norm (LDL''-PAP'') = %g\n', err) ;

% solve Ax=b
x = L' \ (D \ (L \ (b (p)))) ;
x (p) = x ;
resid = norm (A*x-b) ;
fprintf ('residual %g for ldlsparse, flops %10.1f\n', resid, fl) ;

% solve Ax=b with one call to ldlsparse
x = ldlsparse (A, p, b) ;
resid = norm (A*x-b) ;
fprintf ('residual %g for ldlsparse solve\n\n', resid) ;

% compare ldlsymbol and symbfact
[Lnz, Parent, fl] = ldlsymbol (A) ;
fprintf ('Original matrix: nz in L: %5d  flop count: %g\n', sum (Lnz), fl) ;

Lnz2 = symbfact (A) - 1 ;
Parent2 = etree (A) ;
fl2 = sum (Lnz2 .* (Lnz2 + 2)) ;
if (any (Lnz ~= Lnz2))
    error ('Lnz mismatch') ;
end
if (any (Parent ~= Parent2))
    error ('Parent mismatch') ;
end
if (fl ~= fl2)
    error ('fl mismatch') ;
end

[Lnz, Parent, fl] = ldlsymbol (A, p) ;
fprintf ('Permuted matrix: nz in L: %5d  flop count: %g\n', sum (Lnz), fl) ;

Lnz2 = symbfact (A (p,p)) - 1 ;
Parent2 = etree (A (p,p)) ;
fl2 = sum (Lnz2 .* (Lnz2 + 2)) ;
if (any (Lnz ~= Lnz2))
    error ('Lnz mismatch') ;
end
if (any (Parent ~= Parent2))
    error ('Parent mismatch') ;
end
if (fl ~= fl2)
    error ('fl mismatch') ;
end


fprintf ('\nldldemo: all tests passed\n') ;

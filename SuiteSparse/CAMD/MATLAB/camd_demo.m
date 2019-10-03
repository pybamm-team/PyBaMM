function camd_demo
%CAMD_DEMO a demo of camd, using the can_24 matrix
%
% A demo of CAMD for MATLAB.
%
% Example:
%   camd_demo
%
% See also: camd, camd_make

% Copyright 1994-2007, Tim Davis, Patrick R. Amestoy, Iain S. Duff, and Y. Chen.

% This orders the same matrix as the ANSI C demo, camd_demo.c.  It includes an
% additional analysis of the matrix via MATLAB's symbfact routine.

% First, print the help information for CAMD
help camd

% Get the Harwell/Boeing can_24 matrix.

load can_24
A = spconvert (can_24) ;

n = size (A,1) ;

rand ('state', 0) ;
C = irand (6, n) ;

clf
hold off
subplot (2,2,1) ;
spy (A)
title ('HB/can24 matrix') ;

% print the details during CAMD ordering and SYMBFACT
% spparms ('spumoni', 1) ;

% order the matrix.  Note that the Info argument is optional.
fprintf ('\nIf the next step fails, then you have\n') ;
fprintf ('not yet compiled the CAMD mexFunction.\n') ;
[p, Info] = camd (A) ;          %#ok

% order again, but this time print some statistics
[p, camd_Info] = camd (A, [10 1 1], C) ;

fprintf ('Permutation vector:\n') ;
fprintf (' %2d', p) ;

fprintf ('\n\n') ;
fprintf ('Corresponding constraint sets:\n') ;

if (any (sort (C (p)) ~= C (p)))
    error ('Error!') ;
end

for j = 1:n
    fprintf (' %2d', C (p (j))) ;
end
fprintf ('\n\n\n') ;

subplot (2,2,2) ;
spy (A (p,p)) ;
title ('Permuted matrix') ;

% The camd_demo.c program stops here.

fprintf ('Analyze A(p,p) with MATLAB symbfact routine:\n') ;
[cn, height, parent, post, R] = symbfact (A(p,p)) ;

subplot (2,2,3) ;
spy (R') ; 
title ('Cholesky factor L') ;

subplot (2,2,4) ;
treeplot (parent) ;
title ('etree') ;

% results from symbfact
lnz = sum (cn) ;                % number of nonzeros in L, incl. diagonal
cn = cn - 1 ;                   % get the count of off-diagonal entries
fl = n + sum (cn.^2 + 2*cn) ;   % flop count for chol (A (p,p)
fprintf ('number of nonzeros in L (including diagonal):      %d\n', lnz) ;
fprintf ('floating point operation count for chol (A (p,p)): %d\n', fl) ;

% approximations from camd:
lnz2 = n + camd_Info (10) ;
fl2 = n + camd_Info (11) + 2 * camd_Info (12) ;
fprintf ('\nResults from CAMD''s approximate analysis:\n') ;
fprintf ('number of nonzeros in L (including diagonal):      %d\n', lnz2) ;
fprintf ('floating point operation count for chol (A (p,p)): %d\n\n', fl2) ;

fprintf ('\nNote that the ordering quality is not as good as p=amd(A).\n') ;
fprintf ('This is only because the ordering constraints, C, have been\n') ;
fprintf ('randomly selected.\n') ;

if (lnz2 ~= lnz | fl ~= fl2)						    %#ok
    fprintf ('Note that the nonzero and flop counts from CAMD are slight\n') ;
    fprintf ('upper bounds.  This is due to the approximate minimum degree\n');
    fprintf ('method used, in conjunction with "mass elimination".\n') ;
    fprintf ('See the discussion about mass elimination in camd.h and\n') ;
    fprintf ('camd_2.c for more details.\n') ;
end

% turn off diagnostic output in MATLAB's sparse matrix routines
% spparms ('spumoni', 0) ;

%-------------------------------------------------------------------------------

function i = irand (n,s)
% irand: return a random vector of size s, with values between 1 and n
if (nargin == 1)
    s = 1 ;
end
i = min (n, 1 + floor (rand (1,s) * n)) ;

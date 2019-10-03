% KLU:  a "Clark Kent" LU factorization algorithm
%
%   klu         - sparse left-looking LU factorization, using a block triangular form.
%   klu_install - compiles and installs the KLU, BTF, AMD, and COLAMD mexFunctions
%   klu_demo    - KLU demo
%   klu_make    - compiles the KLU mexFunctions
%
% Example:
%   
%   LU = klu (A) ;
%   x = klu (A, '\', b) ;
%   x = klu (LU, '\', b) ;
%
% Copyright 2004-2009, Univ. of Florida
% KLU Version 1.0.


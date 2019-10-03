% LDL package: simple sparse LDL factorization
%
% Primary routines:
% 
%   ldlsparse   - LDL' factorization of a real, sparse, symmetric matrix
%   ldlsymbol   - symbolic Cholesky factorization
%
% Helper routines:
%
%   ldldemo     - demo program for LDL
%   ldlrow      - an m-file description of the algorithm used by LDL
%   ldltest     - test program for LDL
%   ldlmain2    - compiles and runs a longer test program for LDL
%   ldl_install - compile and install the LDL package for use in MATLAB.
%   ldl_make    - compile LDL
%
% Example:
%
%       [L, D, Parent, fl] = ldlsparse (A)

% Copyright 2006-2007 by Timothy A. Davis, http://www.suitesparse.com

% LDL License:  see LDL/Doc/License.txt
%
% Acknowledgements:
%
%   This work was supported by the National Science Foundation, under
%   grant CCR-0203270.
%
%   Portions of this work were done while on sabbatical at Stanford University
%   and Lawrence Berkeley National Laboratory (with funding from the SciDAC
%   program).  I would like to thank Gene Golub, Esmond Ng, and Horst Simon
%   for making this sabbatical possible.

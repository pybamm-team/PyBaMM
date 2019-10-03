%Contents of the UMFPACK sparse matrix toolbox:
%
% umfpack         - computes x=A\b, x=A/b, or lu (A) for a sparse matrix A
% umfpack_make    - to compile umfpack for use in MATLAB
% umfpack_install - to compile and install umfpack and amd2 for use in MATLAB
% umfpack_details - details on all the options for using umfpack in MATLAB
% umfpack_report  - prints optional control settings and statistics
% umfpack_demo    - a lenghty demo
% umfpack_simple  - a simple demo
% umfpack_btf     - factorize A using a block triangular form
% umfpack_solve   - x = A\b or x = b/A
% lu_normest      - estimates norm (L*U-A, 1) without forming L*U-A
% luflop          - given L and U, computes # of flops required to compute them
% umfpack_test    - for testing umfpack (requires ssget)
%
% Example:
%   x = umfpack (A, '\', b) ;   % same as x = A\b, if A square and unsymmetric
%
% See also these built-in functions:
% x=A\b             uses the built-in version of UMFPACK 
% amd               symmetric minimum degree ordering
% colamd            unsymmetric column approx minimum degree ordering
% symamd            symmetric approx minimum degree ordering, based on colamd
%
% Copyright 1995-2009 by Timothy A. Davis.
% All Rights Reserved.  See UMFPACK/Doc/License.txt for License.

help Contents

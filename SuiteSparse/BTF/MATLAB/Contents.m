% BTF ordering toolbox:
%
% Primary functions:
%
%   btf        - permute a square sparse matrix into upper block triangular form
%   maxtrans   - permute the columns of a sparse matrix so it has a zero-free diagonal
%   strongcomp - symmetric permutation to upper block triangular form
%
% Other:
%   btf_install - compile and install BTF for use in MATLAB.
%   btf_demo    - demo for BTF
%   drawbtf     - plot the BTF form of a matrix
%   btf_make    - compile BTF for use in MATLAB
%
% Example:
%   q = maxtrans (A)
%   [p,q,r] = btf (A)
%   [p,r] = strongcomp (A)

% Copyright 2004-2007, University of Florida

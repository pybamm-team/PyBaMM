% RBio: MATLAB toolbox for reading/writing sparse matrices in the Rutherford/
%   Boeing format, and for reading/writing problems in the UF Sparse Matrix
%   Collection from/to a set of files in a directory.
%
%   RBread    - read a sparse matrix from a Rutherford/Boeing file
%   RBreade   - read a symmetric finite-element matrix from a R/B file
%   RBtype    - determine the Rutherford/Boeing type of a sparse matrix
%   RBwrite   - write a sparse matrix to a Rutherford/Boeing file
%   RBraw     - read the raw contents of a Rutherford/Boeing file
%   RBfix     - read a possibly corrupted matrix from a R/B file
%   RBinstall - install the RBio toolbox for use in MATLAB
%   RBmake    - compile the RBio toolbox for use in MATLAB
%
% Example:
%
%   load west0479
%   C = west0479 ;
%   RBwrite ('mywest', C, 'WEST0479 chemical eng. problem', 'west0479')
%   A = RBread ('mywest') ;
%   norm (A-C,1)
%
% See also UFget, mread, mwrite.

% Copyright 2009, Timothy A. Davis

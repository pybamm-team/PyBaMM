function mtype = RBwrite (filename, A, Z, title, key)			    %#ok
%RBWRITE write a sparse matrix to a Rutherford/Boeing file
% Usage:
%   mtype = RBwrite (filename, A, Z, title, key)
%
%   filename: name of the file to create
%   A: a sparse matrix
%   Z: binary pattern of explicit zero entries to include in the
%       Rutherford/Boeing file.  This always has the same size as A, and is
%       always sparse.  Not used if empty ([ ]), or if nnz(Z) is 0.
%   title: title for 1st line of  Rutherford/Boeing file, up to 72 characters
%   key: matrix key, up to 8 characters, for 1st line of the file
%
% Z is optional.  RBwrite (filename, A) uses a default title and key, and does
% not include any explicit zeros.  RBwrite (filname, A, 'title...', 'key') uses
% the given title and key.  A must be sparse.  Z must be empty, or sparse.
%
% mtype is a 3-character string with the Rutherford/Boeing type used:
%   mtype(1):  r: real, p: pattern, c: complex, i: integer
%   mtype(2):  r: rectangular, u: unsymmetric, s: symmetric,
%              h: Hermitian, Z: skew symmetric
%   mtype(3):  a: assembled matrix, e: finite-element (not used by RBwrite)
%
% Example:
%   load west0479
%   C = west0479 ;
%   RBwrite ('west0479', C, 'WEST0479 chemical eng. problem', 'west0479')
%   A = RBread ('west0479') ;
%   norm (A-C,1)
%
% See also RBread, RBtype.

% Copyright 2009-2011, Timothy A. Davis, http://www.suitesparse.com

error ('RBwrite mexFunction not found') ;


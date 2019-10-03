function mtype = mwrite (filename, A, Z, comments_filename)		    %#ok
%MWRITE write a matrix to a file in Matrix Market form.
%
%   Example:
%        mtype = mwrite (filename, A, Z, comments_filename)
%
% A can be sparse or full.
%
% If present and non-empty, A and Z must have the same dimension.  Z contains
% the explicit zero entries in the matrix (which MATLAB drops).  The entries
% of Z appear as explicit zeros in the output file.  Z is optional.  If it is
% an empty matrix it is ignored.  Z must be sparse or empty, if present.
% It is ignored if A is full.
%
% filename is the name of the output file.  comments_filename is the file
% whose contents are include after the Matrix Market header and before the
% first data line.  Ignored if an empty string or not present.
%
% See also mread.

% Copyright 2006-2007, Timothy A. Davis

error ('mwrite mexFunction not found') ;

function ssfull_write (filename, A)					    %#ok
%SSFULL_WRITE write a full matrix using a subset of Matrix Market format
% Usage:
%
%   ssfull_write (filename, A)
%
% A small subset of the Matrix Market format is used.  The first line is one of:
%
%    %%MatrixMarket matrix array real general
%    %%MatrixMarket matrix array complex general
% 
% The second line contains two numbers: m and n, where A is m-by-n.  The next
% m*n lines contain the numerical values (one per line if real, two per line
% if complex, containing the real and imaginary parts).  The values are listed
% in column-major order.  The resulting file can be read by any Matrix Market
% reader, or by ssfull_read.  No comments or blank lines are used.
%
% Example:
%   x = rand (8)
%   ssfull_write ('xfile', x)
%   y = ssfull_read ('xfile')
%   norm (x-y)
%
% See also mread, mwrite, RBwrite, RBread.

% Copyright 2006-2018, Timothy A. Davis

error ('ssfull_write mexFunction not found') ;


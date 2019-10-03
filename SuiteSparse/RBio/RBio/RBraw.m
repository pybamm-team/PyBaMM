function [mtype, Ap, Ai, Ax, title, key, nrow] = RBraw (filename)	    %#ok
%RBRAW read the raw contents of a Rutherford/Boeing file
%
%   [mtype Ap Ai Ax title key nrow] = RBraw (filename)
%
%   mtype: Rutherford/Boeing matrix type (psa, rua, rsa, rse, ...)
%   Ap: column pointers (1-based)
%   Ai: row indices (1-based)
%   Ax: numerical values (real, complex, or integer).  Empty for p*a matrices.
%       A complex matrix is read in as a single double array Ax, where the kth
%       entry has real value Ax(2*k-1) and imaginary value Ax(2*k).
%   title: a string containing the title from the first line of the R/B file
%   key: a string containing the 8-character key, from the 1st line of the file
%   nrow: number of rows in the matrix
%
% This function works for both assembled and unassembled (finite-element)
% matrices.  It is also useful for checking the contents of a Rutherford/Boeing
% file in detail, in case the file has invalid column pointers, unsorted
% columns, duplicate entries, entries in the upper triangular part of the file 
% for a symmetric matrix, etc.
%
% Example:
%
%   load west0479
%   RBwrite ('mywest', west0479, [ ], 'My west0479 file', 'west0479') ;
%   [mtype Ap Ai Ax title key nrow] = RBraw ('mywest') ;
%
% See also RBfix, RBread, RBreade.

% Copyright 2009, Timothy A. Davis

error ('RBraw mexFunction not found') ;

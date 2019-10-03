function [A, Z, title, key, mtype] = RBread (filename)			    %#ok
%RBREAD read a sparse matrix from a Rutherford/Boeing file
% Usage:
%   [A Z title key mtype] = RBread (filename)
%
%   A: a sparse matrix (no explicit zero entries)
%   Z: binary pattern of explicit zero entries in Rutherford/Boeing file.
%       This always has the same size as A, and is always sparse.
%   title: the 72-character title string in the file
%   key: the 8-character matrix name in the file
%   mtype: the Rutherford/Boeing type (see RBwrite for a description).
%       This function does not support finite-element matrices (use RBreade
%       instead).
%
% Example:
%   load west0479
%   C = west0479 ;
%   RBwrite ('mywest', C, 'WEST0479 chemical eng. problem', 'west0479')
%   A = RBread ('mywest') ;
%   norm (A-C,1)
%
% See also RBwrite, RBreade, RBtype.

% Copyright 2009, Timothy A. Davis

error ('RBread mexFunction not found') ;


function s = ssmultsym (A,B)                                                %#ok
%SSMULTSYM computes nnz(C), memory, and flops to compute C=A*B; A and B sparse.
% s = ssmultsym (A,B) returns a struct s with the following fields:
%
%   s.nz            nnz (A*B), assuming no numerical cancelation
%   s.flops         flops required to compute C=A*B
%   s.memory        memory required to compute C=A*B, including C itself.
%
% Either A or B, or both, can be complex.  Only matrices of class "double" are
% supported.  If i is the size of an integer (4 bytes on 32-bit MATLAB, 8 bytes
% on 64-bit MATLAB) and x is the size of an entry (8 bytes if real, 16 if
% complex), and [m n]=size(C), then the memory usage of SSMULT is
% (i+x)*nnz(C) + i*(n+1) for C itself, and (i+x)*m for temporary workspace.
% SSMULTSYM itself does not compute C, and uses only i*m workspace.
%
% Example:
%   load west0479
%   A = west0479 ;
%   B = sprand (west0479) ;
%   C = A*B ;
%   D = ssmult (A,B) ;
%   C-D
%   ssmultsym (A,B)
%   nnz (C)
%   whos ('C')
%   [m n] = size (C)
%   mem = 12*nnz(C) + 4*(n+1) + (12*m)          % assuming real, 32-bit MATLAB
%
% This function can also compute the statistics for any of the 64 combinations
% of C = op (op(A) * op(B)) where op(A) is A, A', A.', or conj(A).  The general
% form is
%
%   C = ssmultsym (A,B, at,ac, bt,bc, ct,cc)
%
% See ssmult for a description of the at,ac, bt,bc, and ct,cc arguments.
%
% See also ssmult, mtimes.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

error ('ssmultsym mexFunction not found') ;

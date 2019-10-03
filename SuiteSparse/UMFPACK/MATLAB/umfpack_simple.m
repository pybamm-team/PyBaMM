%UMFPACK_SIMPLE a simple demo
%
% Example:
%   umfpack_simple
%
% Copyright 1995-2007 by Timothy A. Davis, http://www.suitesparse.com
%
% UMFPACK License:  See UMFPACK/Doc/License.txt.
%
% See also: umfpack, umfpack_details

help umfpack_simple

format short

A = [
 2  3  0  0  0
 3  0  4  0  6
 0 -1 -3  2  0
 0  0  1  0  0
 0  4  2  0  1
] ;
fprintf ('A = \n') ; disp (A) ;

A = sparse (A) ;

b = [8 45 -3 3 19]' ;
fprintf ('b = \n') ; disp (b) ;

fprintf ('Solution to Ax=b via UMFPACK:\n') ;
fprintf ('x1 = umfpack (A, ''\\'', b)\n') ;

x1 = umfpack (A, '\', b) ;
fprintf ('x1 = \n') ; disp (x1) ;

fprintf ('Solution to Ax=b via MATLAB:\n') ;
fprintf ('x2 = A\\b\n') ;

x2 = A\b ;
fprintf ('x2 = \n') ; disp (x2) ;

fprintf ('norm (x1-x2) should be small: %g\n', norm (x1-x2)) ;

fprintf ('Type ''umfpack_demo'' for a full demo of UMFPACK\n') ;

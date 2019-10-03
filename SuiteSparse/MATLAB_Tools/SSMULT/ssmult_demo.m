function ssmult_demo
%SSMULT_DEMO simple demo for ssmult.
%
% Example:
%   ssmult_demo
%
% See also ssmult, ssmult_unsorted, ssmultsym, sstest, sstest2.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

type ssmult_demo
load west0479
A = west0479 ;
B = sprand (A) ;
C = A*B ;
D = ssmult (A,B) ;
err = norm (C-D,1) / norm (C,1) ;
fprintf ('ssmult west0479 error: %g\n', err) ;

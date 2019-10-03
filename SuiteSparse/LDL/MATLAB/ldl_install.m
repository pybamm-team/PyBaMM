function ldl_install
%LDL_INSTALL compile and install the LDL package for use in MATLAB.
% Your current working directory must be LDL for this function to work.
%
% Example:
%   ldl_install
%
% See also ldlsparse, ldlsymbol

% Copyright 2006-2007 by Timothy A. Davis, http://www.suitesparse.com

ldl_make
addpath (pwd) ;
fprintf ('LDL has been compiled and installed.  The path:\n') ;
disp (pwd) ;
fprintf ('has been added to your path.  Use pathtool to add it permanently.\n');
ldldemo

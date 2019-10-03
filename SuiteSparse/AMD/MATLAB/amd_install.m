function amd_install
%AMD_INSTALL compile and install amd2 for use in MATLAB
%   Your current directory must be AMD/MATLAB for this function to work.
%
% Example:
%   amd_install
%
% See also amd, amd2.

% Copyright 1994-2007, Tim Davis, Patrick R. Amestoy, and Iain S. Duff. 

% This orders the same matrix as the ANSI C demo, amd_demo.c.  It includes an

amd_make
addpath (pwd)
fprintf ('\nThe following path has been added.  You may wish to add it\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n\n', pwd) ;
amd_demo

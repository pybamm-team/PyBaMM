function RBinstall (quiet)
%RBINSTALL install the RBio toolbox for use in MATLAB
% Compiles the Fortran mexFunctions RBread, RBwrite, RBtype, and RBraw, and
% the C mexFunction UFfull_write, and adds the current directory to the MATLAB
% path.
%
% Example:
%
%   RBinstall
%
% See also RBread, RBwrite, RBtype, RBraw.
%
% Copyright 2009, Timothy A. Davis

if (nargin < 1)
    quiet = 0 ;
end

if (~quiet)
    help RBio
end

RBmake

s = pwd ;
addpath (s) ;

cd private
testRB1
if (exist ('UFget') == 2) %#ok<EXIST>
    testRB2
end
cd (s)

if (~quiet)
    fprintf ('\nRBio is ready to use.  Your path has been modified for\n') ;
    fprintf ('this session, by adding the following path:\n') ;
    fprintf ('%s\n', s) ;
    fprintf ('Use the pathtool to modify your path permanently.\n') ;
end


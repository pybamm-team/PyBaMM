function btf_install
%BTF_INSTALL compile and install BTF for use in MATLAB.
% Your current working directory must be BTF/MATLAB for this function to work.
%
% Example:
%   btf_install
%
% See also btf, maxtrans, stroncomp, dmperm.

% Copyright 2004-2007, University of Florida

btf_make
addpath (pwd) ;
fprintf ('BTF has been compiled and installed.  The path:\n') ;
disp (pwd) ;
fprintf ('has been added to your path.  Use pathtool to add it permanently.\n');
btf_demo

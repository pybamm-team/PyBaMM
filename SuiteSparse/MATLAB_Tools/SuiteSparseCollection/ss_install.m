function ss_install (nlargefile)
%SS_INSTALL install the SuiteSparseCollection toolbox
%
% Example:
%   ss_install
%
% See also ssget.

% Copyright 2006-2018, Timothy A. Davis

if (nargin < 1)
    % try with large-file I/O
    nlargefile = 0 ;
end

if (nlargefile)
    fprintf ('Trying to compile without large file support...\n') ;
    mex -DNLARGEFILE ssfull_write.c
else
    try
	mex ssfull_write.c
    catch
	fprintf ('Trying to compile without large file support...\n') ;
	mex --DNLARGEFILE ssfull_write.c
    end
end

addpath (pwd) ;
fprintf ('sscollection toolbox successfully compiled.\n') ;

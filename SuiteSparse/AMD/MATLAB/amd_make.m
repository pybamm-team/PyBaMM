function amd_make
%AMD_MAKE to compile amd2 for use in MATLAB
%
% Example:
%   amd_make
%
% See also amd, amd2.

% Copyright 1994-2007, Tim Davis, Patrick R. Amestoy, and Iain S. Duff. 

details = 0 ;	    % 1 if details of each command are to be printed

d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end

% MATLAB 8.3.0 now has a -silent option to keep 'mex' from burbling too much
if (~verLessThan ('matlab', '8.3.0'))
    d = ['-silent ' d] ;
end

i = sprintf ('-I../Include -I../../SuiteSparse_config') ;
cmd = sprintf ('mex -O %s -DDLONG -output amd2 %s amd_mex.c %s', d, i, ...
    '../../SuiteSparse_config/SuiteSparse_config.c') ;
files = {'amd_order', 'amd_dump', 'amd_postorder', 'amd_post_tree', ...
    'amd_aat', 'amd_2', 'amd_1', 'amd_defaults', 'amd_control', ...
    'amd_info', 'amd_valid', 'amd_preprocess' } ;
for i = 1 : length (files)
    cmd = sprintf ('%s ../Source/%s.c', cmd, files {i}) ;
end
if (details)
    fprintf ('%s\n', cmd) ;
end

if (~(ispc || ismac))
    % for POSIX timing routine
    cmd = [cmd ' -lrt'] ;
end

eval (cmd) ;

fprintf ('AMD successfully compiled.\n') ;

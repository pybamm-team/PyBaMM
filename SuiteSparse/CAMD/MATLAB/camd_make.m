function camd_make
%CAMD_MAKE to compile camd for use in MATLAB
%
% Example:
%   camd_make
%
% See also camd.

% Copyright 1994-2007, Tim Davis, Patrick R. Amestoy, Iain S. Duff, and Y. Chen.

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
cmd = sprintf ('mex -O %s -DDLONG -output camd %s camd_mex.c %s', d, i, ...
    '../../SuiteSparse_config/SuiteSparse_config.c') ;
files = {'camd_order', 'camd_dump', 'camd_postorder', ...
    'camd_aat', 'camd_2', 'camd_1', 'camd_defaults', 'camd_control', ...
    'camd_info', 'camd_valid', 'camd_preprocess' } ;
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

fprintf ('CAMD successfully compiled.\n') ;

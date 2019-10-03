function ldlmain2
%LDLMAIN2 compiles and runs a longer test program for LDL
%
% Example:
%   ldlmain2
%
% See also ldlsparse.

% Copyright 2006-2007 by Timothy A. Davis, http://www.suitesparse.com

help ldlmain2

detail = 0 ;

ldl_make

d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end

mx = sprintf (...
    'mex -O %s -DLDL_LONG -DDLONG -I../../SuiteSparse_config -I../Include', d) ;

% compile ldlmain without AMD
cmd = sprintf ('%s ../Demo/ldlmain.c ../Source/ldl.c', mx) ;
if (detail)
    fprintf ('%s\n', cmd) ;
end
eval (cmd) ;
ldlmain

% compile ldlamd (ldlmain with AMD)
cmd = sprintf ('%s -I../../AMD/Include', mx) ;

cmd = [cmd ' ../../SuiteSparse_config/SuiteSparse_config.c' ] ;

files = {'amd_order', 'amd_dump', 'amd_postorder', 'amd_post_tree', ...
    'amd_aat', 'amd_2', 'amd_1', 'amd_defaults', 'amd_control', ...
    'amd_info', 'amd_valid', 'amd_preprocess' } ;
for i = 1 : length (files)
    cmd = sprintf ('%s ../../AMD/Source/%s.c', cmd, files {i}) ;
end

if (~(ispc || ismac))
    % for POSIX timing routine
    cmd = [cmd ' -lrt'] ;
end

cmd = [cmd ' -DUSE_AMD -output ldlamd ../Demo/ldlmain.c ../Source/ldl.c'] ;
if (detail)
    fprintf ('%s\n', cmd) ;
end
eval (cmd) ;
ldlamd


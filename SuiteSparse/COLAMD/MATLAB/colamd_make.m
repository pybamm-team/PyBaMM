function colamd_make
%COLAMD_MAKE compiles COLAMD2 and SYMAMD2 for MATLAB
%
% Example:
%   colamd_make
%
% See also colamd, symamd

%    Copyright 1998-2007, Timothy A. Davis, and Stefan Larimore
%    Developed in collaboration with J. Gilbert and E. Ng.

details = 0 ;	    % 1 if details of each command are to be printed
d = '' ;
if (~isempty (strfind (computer, '64')))
    d = '-largeArrayDims' ;
end

% MATLAB 8.3.0 now has a -silent option to keep 'mex' from burbling too much
if (~verLessThan ('matlab', '8.3.0'))
    d = ['-silent ' d] ;
end

src = '../Source/colamd.c ../../SuiteSparse_config/SuiteSparse_config.c' ;
cmd = sprintf ( ...
    'mex -DDLONG -O %s -I../../SuiteSparse_config -I../Include -output ', d) ;
s = [cmd 'colamd2mex colamdmex.c ' src] ;

if (~(ispc || ismac))
    % for POSIX timing routine
    s = [s ' -lrt'] ;
end

if (details)
    fprintf ('%s\n', s) ;
end
eval (s) ;
s = [cmd 'symamd2mex symamdmex.c ' src] ;

if (~(ispc || ismac))
    % for POSIX timing routine
    s = [s ' -lrt'] ;
end

if (details)
    fprintf ('%s\n', s) ;
end
eval (s) ;
fprintf ('COLAMD2 and SYMAMD2 successfully compiled.\n') ;

function RBmake
%RBMAKE compile the RBio toolbox for use in MATLAB
% Compiles the Fortran mexFunctions RBread, RBwrite, RBtype, and RBraw.
%
% Example:
%
%   RBmake
%
% See also RBread, RBwrite, RBtype, RBraw, RBinstall.
%
% Copyright 2009, Timothy A. Davis

mexcmd = ['mex -O %s %s RBerror.c ../Source/RBio.c ' ...
    '../../SuiteSparse_config/SuiteSparse_config.c ' ...
    '-I../../SuiteSparse_config -I../Include'] ;

try
    % ispc does not appear in MATLAB 5.3
    pc = ispc ;
    mac = ismac ;
catch
    % if ispc fails, assume we are on a Windows PC if it's not unix
    pc = ~isunix ;
    mac = 0 ;
end

if (~(pc || mac))
    % for POSIX timing routine
    mexcmd = [mexcmd ' -lrt'] ;
end

% MATLAB 8.3.0 now has a -silent option to keep 'mex' from burbling too much
if (~verLessThan ('matlab', '8.3.0'))
    mexcmd = [mexcmd ' -silent '] ;
end

files = { 'RBread.c', 'RBwrite.c', 'RBraw.c', 'RBtype.c' } ;
n = length (files) ;

if (~isempty (strfind (computer, '64')))
    try
        % try with -largeArrayDims (will fail on old MATLAB versions)
        for k = 1:n
            eval (sprintf (mexcmd, '-largeArrayDims', files {k})) ;
        end
    catch %#ok<CTCH>
        % try without -largeArrayDims (will fail on recent MATLAB versions)
        for k = 1:n
            eval (sprintf (mexcmd, '', files {k})) ;
        end
    end
else
    for k = 1:n
        eval (sprintf (mexcmd, '', files {k})) ;
    end
end

fprintf ('RBio successfully compiled.\n') ;

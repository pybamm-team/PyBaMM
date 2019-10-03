function [objfiles, timestamp] = cs_make (f, docomplex)
%CS_MAKE compiles CXSparse for use in MATLAB.
%   Usage:
%       cs_make
%       [objfiles, timestamp] = cs_make (f, docomplex)
%
%   With no input arguments, or with f=0, only those files needing to be
%   compiled are compiled (like the "make" command, but not
%   requiring "make").  If f is a nonzero number, all files are compiled.
%   If f is a string, only that mexFunction is compiled.  For example,
%   cs_make ('cs_add') just compiles the cs_add mexFunction.  This option is
%   useful when developing a single new mexFunction.  This function can only be
%   used if the current directory is CXSparse/MATLAB/CSparse.  Returns a list of
%   the object files in CXSparse, and the latest modification time of any source
%   codes.
%
%   NOTE: if your compiler does not support the ANSI C99 complex type (most
%   notably Microsoft Windows), the CXSparse mexFunctions will not support
%   complex sparse matrices.  The complex case is not attempted if docomplex is
%   zero.
%
%   To add a new function and its MATLAB mexFunction to CXSparse:
%
%       (1) Create a source code file CXSparse/Source/cs_mynewfunc.c.
%       (2) Create a help file, CXSparse/MATLAB/CSparse/cs_mynewfunc.m.
%           This is very useful, but not strictly required.
%       (3) Add the prototype of cs_mynewfunc to CXSparse/Include/cs.h.
%       (4) Create its MATLAB mexFunction, CXSparse/MATLAB/cs_mynewfunc_mex.c.
%       (5) Edit cs_make.m, and add 'cs_mynewfunc' to the 'cs' and 'csm' lists.
%       (6) Type 'cs_make' in the CXSparse/MATLAB/CSparse directory.
%           If all goes well, your new function is ready for use in MATLAB.
%
%       (7) Optionally add 'cs_mynewfunc' to CXSparse/Source/Makefile
%           and CXSparse/MATLAB/CSparse/Makefile, if you want to use the
%           make command instead of cs_make.m.  See where
%           'cs_add' and 'cs_add_mex' appear in those files, and add
%           'cs_mynewfunc' accordingly.
%       (8) Optionally add 'cs_mynewfunc' to Tcov/Makefile, and add additional
%           test code to cs_test.c, and add MATLAB test code to MATLAB/Test/*.
%
%   Example:
%       cs_make                  % compile everything
%       cs_make ('cs_chol') ;    % just compile cs_chol mexFunction
%
%   See also MEX.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    f = 0 ;
end
if (nargin < 2)
    docomplex = 1 ;
end

try
    % ispc does not appear in MATLAB 5.3
    pc = ispc ;
catch
    % if ispc fails, assume we are on a Windows PC if it's not unix
    pc = ~isunix ;
end

if (pc)
    docomplex = 0 ;
end

if (docomplex == 0)
    % do not attempt to compile with complex matrices
    [objfiles, timestamp] = cs_make_helper (f, 0) ;
else
    try
        % try with complex support
        [objfiles, timestamp] = cs_make_helper (f, 1) ;
    catch me
        % oops - that failed, try without complex support
        disp (me.message)
        fprintf ('retrying without complex matrix support\n') ;
        [objfiles, timestamp] = cs_make_helper (f, 0) ;
    end
end

if (f > 0)
    fprintf ('CXSparse successfully installed.\n') ;
end


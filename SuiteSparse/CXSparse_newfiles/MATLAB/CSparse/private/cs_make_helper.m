function [objfiles, timestamp_out] = cs_make_helper (f, docomplex)
%CS_MAKE_HELPER compiles CXSparse for use in MATLAB.
%   Usage:
%       [objfiles, timestamp] = cs_make (f, docomplex)
%
%   With f=0, only those files needing to be
%   compiled are compiled (like the "make" command, but not
%   requiring "make").  If f is a nonzero number, all files are compiled.
%   If f is a string, only that mexFunction is compiled.  For example,
%   cs_make ('cs_add') just compiles the cs_add mexFunction.  This option is
%   useful when developing a single new mexFunction.  This function can only be
%   used if the current directory is CXSparse/MATLAB/CSparse.  Returns a list of
%   the object files in CXSparse, and the latest modification time of any source
%   codes.
%
%   NOTE: if your compiler does not support the ANSI C99 complex type, the
%   CXSparse mexFunctions will not support complex sparse matrices.
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
%       cs_make_helper (1,1) ;      % compile everything
%       cs_make ('cs_chol', 1) ;    % just compile cs_chol mexFunction
%
%   See also MEX.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

mexcmd = 'mex -DCS_LONG -I../../../SuiteSparse_config' ;
if (~isempty (strfind (computer, '64')))
    mexcmd = [mexcmd ' -largeArrayDims'] ;
end

% MATLAB 8.3.0 now has a -silent option to keep 'mex' from burbling too much
if (~verLessThan ('matlab', '8.3.0'))
    mexcmd = [mexcmd ' -silent'] ;
end

if (nargin < 2)
    docomplex = 1 ;
end

if (~docomplex)
    mexcmd = [mexcmd ' -DNCOMPLEX'] ;
end

% CSparse source files, in ../../Source, such as ../../Source/cs_add.c.
% Note that not all CSparse source files have their own mexFunction.
cs = { 'cs_add', 'cs_amd', 'cs_chol', 'cs_cholsol', 'cs_counts', ...
    'cs_cumsum', 'cs_dfs', 'cs_dmperm', 'cs_droptol', 'cs_dropzeros', ...
    'cs_dupl', 'cs_entry', 'cs_etree', 'cs_fkeep', 'cs_gaxpy', 'cs_happly', ...
    'cs_house', 'cs_ipvec', 'cs_load', 'cs_lsolve', 'cs_ltsolve', 'cs_lu', ...
    'cs_lusol', 'cs_malloc', 'cs_maxtrans', 'cs_multiply', 'cs_norm', ...
    'cs_permute', 'cs_pinv', 'cs_post', 'cs_print', 'cs_pvec', 'cs_qr', ...
    'cs_qrsol', 'cs_scatter', 'cs_scc', 'cs_schol', 'cs_sqr', 'cs_symperm', ...
    'cs_tdfs', 'cs_transpose', 'cs_compress', 'cs_updown', 'cs_usolve', ...
    'cs_utsolve', 'cs_util', 'cs_reach', 'cs_spsolve', 'cs_ereach', ...
    'cs_leaf', 'cs_randperm' } ;
    % add cs_mynewfunc to the above list

details = 0 ;
kk = 0 ;
csm = { } ;
if (nargin == 0)
    force = 0 ;
elseif (ischar (f))
    fprintf ('cs_make: compiling ../../Source files and %s_mex.c\n', f) ;
    force = 0 ;
    csm = {f} ;
else
    force = f ;
    details = details | (force > 1) ;                                       %#ok
    if (force & details)                                                    %#ok
        fprintf ('cs_make: re-compiling everything\n') ;
    end
end

if (isempty (csm))
    % mexFunctions, of the form cs_add_mex.c, etc, in this directory
    csm = { 'cs_add', 'cs_amd', 'cs_chol', 'cs_cholsol', 'cs_counts', ...
        'cs_dmperm', 'cs_droptol', 'cs_etree', 'cs_gaxpy', 'cs_lsolve', ...
        'cs_ltsolve', 'cs_lu', 'cs_lusol', 'cs_multiply', 'cs_permute', ...
        'cs_print', 'cs_qr', 'cs_qrsol', 'cs_scc', 'cs_symperm', 'cs_thumb', ...
        'cs_transpose', 'cs_sparse', 'cs_updown', 'cs_usolve', ...
        'cs_utsolve', 'cs_randperm', 'cs_sqr' } ;
        % add cs_mynewfunc to the above list
end


try
    % ispc does not appear in MATLAB 5.3
    pc = ispc ;
catch
    % if ispc fails, assume we are on a Windows PC if it's not unix
    pc = ~isunix ;
end

if (pc)
    obj = '.obj' ;
else
    obj = '.o' ;
end

srcdir = '../../Source/' ;
hfile = '../../Include/cs.h' ;

% compile each CSparse source file
[anysrc timestamp kk] = compile_source ('', 'cs_mex', obj, hfile, force, ...
    mexcmd, kk, details) ;
CS = ['cs_mex' obj] ;
if (nargout > 0)
    objfiles = ['../CSparse/cs_mex' obj] ;
end
for i = 1:length (cs)

    [s t kk] = compile_source (srcdir, cs{i}, obj, hfile, force, mexcmd, ...
        kk, details) ;
    timestamp = max (timestamp, t) ;
    anysrc = anysrc | s ;                                                   %#ok
    CS = [CS ' ' cs{i} obj] ;                                               %#ok
    if (nargout > 0)
        objfiles = [objfiles ' ../CSparse/' cs{i} obj] ;   %#ok
    end

    % complex version:
    if (docomplex)
        csrc = cs {i} ;
        csrc = [ 'cs_cl_' csrc(4:end) ] ;
        CS = [CS ' ' csrc obj] ;            %#ok
        if (nargout > 0)
            objfiles = [objfiles ' ../CSparse/' csrc obj] ;%#ok
        end
        if (s)
            cpfile (['../../Source/' cs{i} '.c'], [csrc '.c']) ;
            if (details)
                fprintf ('%s\n', ['cp -f ../../Source/' cs{i} '.c ' csrc '.c']);
            end
            cmd = sprintf ('%s -DCS_COMPLEX -O -c -I../../Include %s.c\n', ...
                mexcmd, csrc) ;
            kk = do_cmd (cmd, kk, details) ;
        end
    end

end

% compile each CSparse mexFunction
obj = ['.' mexext] ;
for i = 1:length (csm)
    [s t] = cs_must_compile ('', csm{i}, '_mex', obj, hfile, force) ;
    timestamp = max (timestamp, t) ;
    if (anysrc | s)                                                         %#ok
        cmd = sprintf ('%s -O -I../../Include %s_mex.c %s -output %s\n', ...
            mexcmd, csm{i}, CS, csm{i}) ;
        kk = do_cmd (cmd, kk, details) ;
    end
end

if (nargout > 1)
    timestamp_out = timestamp ;
end

fprintf ('\n') ;

%-------------------------------------------------------------------------------
function [s,t,kk] = compile_source (srcdir, f, obj, hfile, force, mexcmd, ...
    kk, details)
% compile a source code file in ../../Source, leaving object file in
% this directory.
[s t] = cs_must_compile (srcdir, f, '', obj, hfile, force) ;
if (s)
    cmd = sprintf ('%s -O -c -I../../Include %s%s.c\n', mexcmd, srcdir, f) ;
    kk = do_cmd (cmd, kk, details) ;
end

%-------------------------------------------------------------------------------
function kk = do_cmd (s, kk, details)
%DO_CMD: evaluate a command, and either print it or print a "."
if (details)
    fprintf ('%s', s) ;
else
    if (mod (kk, 60) == 0)
        fprintf ('\n') ;
    end
    kk = kk + 1 ;
    fprintf ('.') ;
end
eval (s) ;

%-------------------------------------------------------------------------------
function rmfile (file)
% rmfile:  delete a file, but only if it exists
if (length (dir (file)) > 0)                                                %#ok
    delete (file) ;
end

%-------------------------------------------------------------------------------
function cpfile (src, dst)
% cpfile:  copy the src file to the filename dst, overwriting dst if it exists
%% fprintf ('cp %s %s\n', src, dst) ; return ;
rmfile (dst)
if (length (dir (src)) == 0)	%#ok
    fprintf ('File does not exist: %s\n', src) ;
    error ('File does not exist') ;
end
try
    copyfile (src, dst) ;
catch ME
    % ignore errors of the form "cp: preserving permissions: ...
    % Operation not supported".  rethrow all other errors.
    if (isempty (strfind (ME.message, 'Operation not supported')))
        rethrow (ME) ;
    end
end

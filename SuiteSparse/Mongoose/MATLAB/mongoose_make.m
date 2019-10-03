function mongoose_make (run_test)
%MONGOOSE_MAKE compiles the Mongoose mexFunctions.
%
% Example:
%   mongoose_make ;     % compile and test
%   mongoose_make (0) ; % compile but do not test
%
% See also mongoose_test, mongoose_make

%   Copyright (c) 2018, N. Yeralan, S. Kolodziej, T. Davis, W. Hager

if (nargin < 1)
    run_test = 1;
end

details = 0 ;	    % 1 if details of each command are to be printed

% v = getversion ;

flags = '' ;
is64 = (~isempty (strfind (computer, '64'))) ;  %#ok
if (is64)
    % 64-bit MATLAB
    flags = ' -largeArrayDims' ;
end

include = '-I. -I../Include -I../External/Include -I../SuiteSparse_config' ;

% Linux/Unix require these flags for large file support
flags = [flags ' -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE'] ;

% We're compiling this from within a mex function.
flags = [flags ' -DGP_MEX_FUNCTION'] ;

% Append optimization and 64-bit flags
flags = [flags ' -DDLONG -O -silent COPTIMFLAGS="-O3 -fwrapv"'];

cpp_flags = '' ;
lib = '';
if (isunix)
    if(~ismac)
        % Mac doesn't need librt
        lib = [lib ' -lrt'];
    end
end

% Fix the include & library path.
include = strrep (include, '/', filesep) ;
lib = strrep (lib, '/', filesep) ;

%-------------------------------------------------------------------------------

config_src = {
    '../SuiteSparse_config/SuiteSparse_config' };

mongoose_src = {
    '../Source/Mongoose_BoundaryHeap', ...
    '../Source/Mongoose_Coarsening', ...
    '../Source/Mongoose_CSparse', ...
    '../Source/Mongoose_EdgeCut', ...
    '../Source/Mongoose_EdgeCutOptions', ...
    '../Source/Mongoose_EdgeCutProblem', ...
    '../Source/Mongoose_Graph', ...
    '../Source/Mongoose_GuessCut', ...
    '../Source/Mongoose_ImproveFM', ...
    '../Source/Mongoose_ImproveQP', ...
    '../Source/Mongoose_Logger', ...
    '../Source/Mongoose_Matching', ...
    '../Source/Mongoose_QPBoundary', ...
    '../Source/Mongoose_QPDelta', ...
    '../Source/Mongoose_QPGradProj', ...
    '../Source/Mongoose_QPLinks', ...
    '../Source/Mongoose_QPMinHeap', ...
    '../Source/Mongoose_QPMaxHeap', ...
    '../Source/Mongoose_QPNapDown', ...
    '../Source/Mongoose_QPNapUp', ...
    '../Source/Mongoose_QPNapsack', ...
    '../Source/Mongoose_Random', ...
    '../Source/Mongoose_Refinement', ...
    '../Source/Mongoose_Sanitize', ...
    '../Source/Mongoose_Waterdance' };

mex_util_src = {
    './mex_util/mex_get_graph', ...
    './mex_util/mex_get_options', ...
    './mex_util/mex_getput_vector', ...
    './mex_util/mex_put_options', ...
    './mex_util/mex_struct_util' } ;

mongoose_mex_src = { 
    'edgecut_options', ...
    'edgecut', ...
    'coarsen', ...
    'sanitize' } ;

% Keep track of object files
obj_list = '' ;

% Build SuiteSparse config
% fprintf('\n\nBuilding SuiteSparse_config');
obj_files = mex_compile(config_src, 'c', flags, include, details);
obj_list = [obj_list obj_files];

% Build Mongoose
% Check if library is built already
% fprintf('\n\nSearching for Mongoose...');
location = fileparts(mfilename('fullpath'));
if (exist([location '/../Lib/libmongoose.a'], 'file') == 2)
    % fprintf('\nMongoose static library found! Using static linking.\n');
    lib = [lib ' -L../Lib -lmongoose'];
else
    % fprintf('\nMongoose static library not found! Compiling Mongoose using mex.\n');

    % Compile Mongoose
    % fprintf('\n\nBuilding Mongoose');
    obj_files = mex_compile(mongoose_src, 'cpp', [cpp_flags flags], include, details);
    obj_list = [obj_list obj_files];
end
    

% fprintf('\nBuilding MEX Utilities') ;

obj_files = mex_compile(mex_util_src, 'cpp', [cpp_flags flags], include, details);
obj_list = [obj_list obj_files];

% fprintf('\nBuilding Mongoose MEX functions');
kk = 1 ;
for f = mongoose_mex_src
    s = sprintf ('mex %s %s %s.cpp', [cpp_flags flags], include, f{1}) ;
    s = [s obj_list ' ' lib] ;  %#ok
    kk = do_cmd (s, kk, details) ;
end

% Clean up
% fprintf('\nCleaning up');
s = ['delete ' obj_list] ;
do_cmd (s, 1, details) ;
% fprintf ('\nMongoose successfully compiled\n') ;

% Run the demo if needed
if (run_test)
    fprintf ('\nRunning Mongoose test...\n') ;
    mongoose_test
    fprintf ('\nMongoose test completed successfully\n') ;
end

%-------------------------------------------------------------------------------
function obj_files = mex_compile (files, ext, flags, include, details)
%MEX_COMPILE: compile C/C++ files using mex and return list of obj files
kk = 1;
obj_files = '';
for f = files
    ff = strrep (f{1}, '/', filesep) ;
    slash = strfind (ff, filesep) ;
    if (isempty (slash))
        slash = 1 ;
    else
        slash = slash (end) + 1 ;
    end
    o = ff (slash:end) ;
    obj_files = [obj_files ' ' o '.o'] ;        %#ok
    s = sprintf ('mex %s %s -c %s.%s', flags, include, ff, ext) ;
    kk = do_cmd (s, kk, details) ;
end

%-------------------------------------------------------------------------------
function kk = do_cmd (s, kk, details)
%DO_CMD: evaluate a command, and either print it or print a "."

if (details)
    fprintf ('%s\n', s) ;
else
    if (mod (kk, 60) == 0)
	fprintf ('\n') ;
    end
    kk = kk + 1 ;
    fprintf ('.') ;
end
eval (s) ;

%-------------------------------------------------------------------------------
% function v = getversion
% determine the MATLAB version, and return it as a double.
% v = sscanf (version, '%d.%d.%d') ;
% v = 10.^(0:-1:-(length(v)-1)) * v ;

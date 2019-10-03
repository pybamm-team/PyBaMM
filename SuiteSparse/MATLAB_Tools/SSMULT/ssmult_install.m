function ssmult_install (dotests)
%SSMULT_INSTALL compiles, installs, and tests ssmult.
% Note that the "lcc" compiler provided with MATLAB for Windows can generate
% slow code; use another compiler if possible.  Your current directory must be
% SSMULT for ssmult_install to work properly.  If you use Linux/Unix/Mac, I
% recommend that you use COPTIMFLAGS='-O3 -DNDEBUG' in your mexopts.sh file.
%
% Example:
%   ssmult_install          % compile and install, do not test
%   ssmult_install (1)      % compile, install, and test
%
% See also ssmult, ssmultsym, sstest, sstest2, mtimes.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

fprintf ('Compiling SSMULT:\n') ;

%-------------------------------------------------------------------------------
% compile ssmult and add it to the path
%-------------------------------------------------------------------------------

d = '' ;
if (~isempty (strfind (computer, '64')))
    % 64-bit MATLAB
    d = ' -largeArrayDims -DIS64' ;
end

if (nargin < 1)
    dotests = 0 ;
end

if (verLessThan ('matlab', '6.5'))
    % mxIsDouble is false for a double sparse matrix in MATLAB 6.1 or earlier
    d = [d ' -DMATLAB_6p1_OR_EARLIER'] ;
end

% MATLAB 8.3.0 now has a -silent option to keep 'mex' from burbling too much
if (~verLessThan ('matlab', '8.3.0'))
    d = [' -silent ' d] ;
end

cmd = sprintf ('mex -O%s ssmult.c ssmult_mex.c ssmult_saxpy.c ssmult_dot.c ssmult_transpose.c', d) ;
if (dotests)
    disp (cmd) ;
end
eval (cmd) ;

cmd = sprintf ('mex -O%s ssmultsym.c', d) ;
if (dotests)
    disp (cmd) ;
end
eval (cmd) ;

cmd = sprintf ('mex -O%s sptranspose.c ssmult_transpose.c', d) ;
if (dotests)
    disp (cmd) ;
end
eval (cmd) ;

addpath (pwd) ;
if (dotests)
fprintf ('\nssmult has been compiled, and the following directory has been\n') ;
fprintf ('added to your MATLAB path.  Use pathtool to add it permanently:\n') ;
fprintf ('\n%s\n\n', pwd) ;
fprintf ('If you cannot save your path with pathtool, add the following\n') ;
fprintf ('to your MATLAB startup.m file (type "doc startup" for help):\n') ;
fprintf ('\naddpath (''%s'') ;\n\n', pwd) ;
else
fprintf ('SSMULT successfully installed\n') ;
end

%-------------------------------------------------------------------------------
% test ssmult and ssmultsym
%-------------------------------------------------------------------------------

if (dotests)
    fprintf ('Please wait while your new ssmult function is tested ...\n') ;
    ssmult_test ;
end

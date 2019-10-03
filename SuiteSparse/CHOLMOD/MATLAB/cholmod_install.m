function cholmod_install (metis_path)
%CHOLMOD_INSTALL compile and install CHOLMOD, AMD, COLAMD, CCOLAMD, CAMD
%
% Example:
%   cholmod_install                 % compiles using METIS
%   cholmod_install ('/my/metis')   % using non-default path to METIS
%   cholmod_install ('no metis')    % do not use METIS at all
%
% CHOLMOD relies on AMD and COLAMD, for its ordering options, and can
% optionally use CCOLAMD, CAMD, and METIS as well.  By default, CCOLAMD, CAMD,
% and METIS are used.
%
% See http://www-users.cs.umn.edu/~karypis/metis for a copy of METIS 4.0.1.
%
% You can only use cholmod_install while in the CHOLMOD/MATLAB directory.
%
% See also analyze, bisect, chol2, cholmod2, etree2, lchol, ldlchol, ldlsolve,
%   ldlupdate, metis, spsym, nesdis, septree, resymbol, sdmult, sparse2,
%   symbfact2, mread, mwrite, amd2, colamd2, camd, ccolamd, ldlrowmod

%   Copyright 2006-2015, Timothy A. Davis

if (nargin < 1)
    metis_path = '../../metis-5.1.0' ;
end

% compile CHOLMOD and add to the path
cholmod_make (metis_path) ;
cholmod_path = pwd ;
addpath (cholmod_path)

fprintf ('\nNow compiling the AMD, COLAMD, CCOLAMD, and CAMD mexFunctions:\n') ;

% compile AMD and add to the path
cd ../../AMD/MATLAB
amd_make
amd_path = pwd ;
addpath (amd_path)

% compile COLAMD and add to the path
cd ../../COLAMD/MATLAB
colamd_make
colamd_path = pwd ;
addpath (colamd_path)

% compile CCOLAMD and add to the path
cd ../../CCOLAMD/MATLAB
ccolamd_make
ccolamd_path = pwd ;
addpath (ccolamd_path)

% compile CAMD and add to the path
cd ../../CAMD/MATLAB
camd_make
camd_path = pwd ;
addpath (camd_path)

cd (cholmod_path)

fprintf ('\nThe following paths have been added.  You may wish to add them\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n', cholmod_path) ;
fprintf ('%s\n', amd_path) ;
fprintf ('%s\n', colamd_path) ;
fprintf ('%s\n', ccolamd_path) ;
fprintf ('%s\n', camd_path) ;

fprintf ('\nTo try your new mexFunctions, cut-and-paste this command:\n') ;
fprintf ('amd_demo, colamd_demo, ccolamd_demo, camd_demo, graph_demo, cholmod_demo\n') ;


function spqr_install (tbb)
%SPQR_INSTALL compile and install SuiteSparseQR
%
% Example:
%   spqr_install                        % compiles using METIS, no TBB
%   spqr_install ('tbb')                % compiles with TBB
%
% SuiteSparseQR relies on CHOLMOD, AMD, and COLAMD, and can optionally use
% CCOLAMD, CAMD, and METIS as well.  By default, CCOLAMD, CAMD, and METIS are
% used.  METIS is assumed to be in the ../../metis-5.1.0 directory.  If not
% present there, it is not used.
%
% You can only use spqr_install while in the SuiteSparseQR/MATLAB directory.
%
% Multithreading based on Intel Threading Building Blocks (TBB) is optionally
% used. It is not enabled by default since it conflicts with multithreading in
% the BLAS.
%
% See also spqr, spqr_solve, spqr_qmult.

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    tbb = 0 ;
end

% compile SuiteSparseQR and add to the path
spqr_make (tbb) ;
spqr_path = pwd ;
addpath (spqr_path)

fprintf ('\nThe following path has been added.  You may wish to add it\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n', spqr_path) ;

fprintf ('\nTo try your new mexFunctions, try this command:\n') ;
fprintf ('spqr_demo\n') ;



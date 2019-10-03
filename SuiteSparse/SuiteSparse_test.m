function SuiteSparse_test
% SuiteSparse_test exhaustive test of all SuiteSparse packages
%
% Your current directory must be SuiteSparse for this function to work.
% SuiteSparse_install must be run prior to running this test.  Warning:
% this test takes a *** long **** time.
%
% Example:
%   SuiteSparse_test
%
% See also SuiteSparse_install, SuiteSparse_demo.

% Copyright 1990-2015, Timothy A. Davis, http://www.suitesparse.com.

help SuiteSparse_test

npackages = 19 ;
h = waitbar (0, 'SuiteSparse test:') ;
SuiteSparse = pwd ;
package = 0 ;

try

    %---------------------------------------------------------------------------
    % CSparse
    %---------------------------------------------------------------------------

    % compile and install CSparse (not installed by SuiteSparse_install)
    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: CSparse') ;
    addpath ([SuiteSparse '/CSparse/MATLAB/CSparse']) ;
    addpath ([SuiteSparse '/CSparse/MATLAB/Demo']) ;
    cd ([SuiteSparse '/CSparse/MATLAB/CSparse']) ;
    cs_make ;
    % test CSparse
    cd ([SuiteSparse '/CSparse/MATLAB/Test']) ;
    testall ;
    % uninstall CSparse by removing it from path
    rmpath ([SuiteSparse '/CSparse/MATLAB/CSparse']) ;
    rmpath ([SuiteSparse '/CSparse/MATLAB/Demo']) ;
    rmpath ([SuiteSparse '/CSparse/MATLAB/ssget']) ;

    %---------------------------------------------------------------------------
    % CXSparse
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: CXSparse') ;
    cd ([SuiteSparse '/CXSparse/MATLAB/Test']) ;
    testall ;

    %---------------------------------------------------------------------------
    % COLAMD
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: COLAMD') ;
    cd ([SuiteSparse '/COLAMD/MATLAB']) ;
    colamd_test ;

    %---------------------------------------------------------------------------
    % CCOLAMD
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: CCOLAMD') ;
    cd ([SuiteSparse '/CCOLAMD/MATLAB']) ;
    ccolamd_test ;

    %---------------------------------------------------------------------------
    % UMFPACK
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: UMFPACK') ;
    cd ([SuiteSparse '/UMFPACK/MATLAB']) ;
    umfpack_test (100) ;

    %---------------------------------------------------------------------------
    % CHOLMOD
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: CHOLMOD') ;
    cd ([SuiteSparse '/CHOLMOD/MATLAB/Test']) ;
    cholmod_test ;

    %---------------------------------------------------------------------------
    % BTF
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: BTF') ;
    cd ([SuiteSparse '/BTF/MATLAB/Test']) ;
    btf_test ;

    %---------------------------------------------------------------------------
    % KLU
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: KLU') ;
    cd ([SuiteSparse '/KLU/MATLAB/Test']) ;
    klu_test (100) ;

    %---------------------------------------------------------------------------
    % LDL
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: LDL') ;
    cd ([SuiteSparse '/LDL/MATLAB']) ;
    ldlmain2 ;
    ldltest ;

    %---------------------------------------------------------------------------
    % LINFACTOR:  MATLAB 7.3 (R2006b) or later required
    %---------------------------------------------------------------------------

    package = package + 1 ;
    if (verLessThan ('matlab', '7.3'))
        % skip test of LINFACTOR
    else
        waitbar (package/(npackages+1), h, 'SuiteSparse test: LINFACTOR') ;
        cd ([SuiteSparse '/MATLAB_Tools/LINFACTOR']) ;
        lintests ;
    end

    %---------------------------------------------------------------------------
    % MESHND
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: MESHND') ;
    cd ([SuiteSparse '/MATLAB_Tools/MESHND']) ;
    meshnd_quality ;

    %---------------------------------------------------------------------------
    % SSMULT
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: SSMULT') ;
    cd ([SuiteSparse '/MATLAB_Tools/SSMULT']) ;
    sstest3 ;

    %---------------------------------------------------------------------------
    % other MATLAB_Tools
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: MATLAB Tools') ;
    cd ([SuiteSparse '/MATLAB_Tools']) ;
    fprintf ('getversion: %g\n', getversion) ;
    seashell ;
    shellgui ;
    cd ([SuiteSparse '/MATLAB_Tools/waitmex']) ;
    waitmex ;
    url = 'http://www.suitesparse.com' ;
    fprintf ('<a href="%s">Click here for more details</a>\n', url) ;
    hprintf ('or see <a href="%s">\n', url) ;
    cd ([SuiteSparse '/MATLAB_Tools/find_components']) ;
    find_components_example (1, 0) ;
    cd ([SuiteSparse '/MATLAB_Tools/spok']) ;
    spok_test ;

    %---------------------------------------------------------------------------
    % FACTORIZE
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: FACTORIZE') ;
    cd ([SuiteSparse '/MATLAB_Tools/Factorize/Test']) ;
    test_all ;

    %---------------------------------------------------------------------------
    % SPARSEINV
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: SPARSEINV') ;
    cd ([SuiteSparse '/MATLAB_Tools/sparseinv']) ;
    sparseinv_test

    %---------------------------------------------------------------------------
    % SPQR_RANK
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: spqr_rank') ;
    cd ([SuiteSparse '/MATLAB_Tools/spqr_rank']) ;
    demo_spqr_rank ;

    %---------------------------------------------------------------------------
    % Mongoose
    %---------------------------------------------------------------------------

    package = package + 1 ;
    waitbar (package/(npackages+1), h, 'SuiteSparse test: mongoose') ;
    cd ([SuiteSparse '/Mongoose/MATLAB']) ;
    mongoose_test ;

    %---------------------------------------------------------------------------
    % PIRO_BAND
    %---------------------------------------------------------------------------

%   package = package + 1 ;
%   waitbar (package/(npackages+1), h, 'SuiteSparse test: PIRO_BAND') ;
%   cd ([SuiteSparse '/PIRO_BAND/MATLAB/Test']) ;
%   demo_spqr_rank ;

    %---------------------------------------------------------------------------
    % AMD, CAMD, SuiteSparseCollection, ssget
    %---------------------------------------------------------------------------

    % no exhaustive tests; tested via other packages

catch

    %---------------------------------------------------------------------------
    % test failure
    %---------------------------------------------------------------------------

    cd (SuiteSparse) ;
    disp (lasterr) ;                                                        %#ok
    fprintf ('SuiteSparse test: FAILED\n') ;
    return

end

%-------------------------------------------------------------------------------
% test OK
%-------------------------------------------------------------------------------

close (h) ;
fprintf ('SuiteSparse test: OK\n') ;
cd (SuiteSparse) ;

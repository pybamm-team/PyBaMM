function SuiteSparse_demo (matrixpath, dopause)
%SUITESPARSE_DEMO a demo of all packages in SuiteSparse
%
% Example:
%   SuiteSparse_demo
%
% See also umfpack, cholmod, amd, camd, colamd, ccolamd, btf, klu, spqr,
%   CSparse, CXSparse, ldlsparse, mongoose

% Copyright 2016, Timothy A. Davis, http://www.suitesparse.com.

if (nargin < 1 || isempty (matrixpath) || ~ischar (matrixpath))
    try
	% older versions of MATLAB do not have an input argument to mfilename
	p = mfilename ('fullpath') ;
	t = strfind (p, '/') ;
	matrixpath = [ p(1:t(end)) 'CXSparse/Matrix' ] ;
    catch me    %#ok
	% mfilename failed, assume we're in the SuiteSparse directory
	matrixpath = 'CXSparse/Matrix' ;
    end
end

if (nargin < 2)
    dopause = false ;
end

if (dopause)
    input ('Hit enter to run the CXSparse demo: ', 's') ;
end
try
    cs_demo (0, matrixpath)
catch me
    disp (me.message) ;
    fprintf ('\nIf you have an older version of MATLAB, you must run the\n') ;
    fprintf ('SuiteSparse_demo while in the SuiteSparse directory.\n\n') ;
    fprintf ('CXSparse demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the UMFPACK demo: ', 's') ;
end
try
    umfpack_demo (1)
catch me
    disp (me.message) ;
    fprintf ('UMFPACK demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the CHOLMOD demo: ', 's') ;
end
try
    cholmod_demo
catch me
    disp (me.message) ;
    fprintf ('CHOLMOD demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the CHOLMOD graph partitioning demo: ', 's') ;
end
try
    graph_demo
catch me
    disp (me.message) ;
    fprintf ('graph_demo failed, probably because METIS not installed\n') ;
end

if (dopause)
    input ('Hit enter to run the AMD demo: ', 's') ;
end
try
    amd_demo
catch me
    disp (me.message) ;
    fprintf ('AMD demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the CAMD demo: ', 's') ;
end
try
    camd_demo
catch me
    disp (me.message) ;
    fprintf ('CAMD demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the COLAMD demo: ', 's') ;
end
try
    colamd_demo
catch me
    disp (me.message) ;
    fprintf ('COLAMD demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the CCOLAMD demo: ', 's') ;
end
try
    ccolamd_demo
catch me
    disp (me.message) ;
    fprintf ('CCOLAMD demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the BTF demo: ', 's') ;
end
try
    btf_demo
catch me
    disp (me.message) ;
    fprintf ('BTF demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the KLU demo: ', 's') ;
end
try
    klu_demo
catch me
    disp (me.message) ;
    fprintf ('KLU demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the LDL demo: ', 's') ;
end
try
    ldldemo
catch me
    disp (me.message) ;
    fprintf ('LDL demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the SSMULT demo: ', 's') ;
end
try
    ssmult_demo
catch me
    disp (me.message) ;
    fprintf ('SSMULT demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the MESHND demo: ', 's') ;
end
try
    meshnd_example
catch me
    disp (me.message) ;
    fprintf ('MESHND demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the SPARSEINV demo: ', 's') ;
end
try
    sparseinv_test
catch me
    disp (me.message) ;
    fprintf ('SPARSEINV demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the SuiteSparseQR demo: ', 's') ;
end
try
    spqr_demo
catch me
    disp (me.message) ;
    fprintf ('SuiteSparseQR demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the quick spqr_rank demo: ', 's') ;
end
try
    quickdemo_spqr_rank
catch me
    disp (me.message) ;
    fprintf ('spqr_rank demo failed\n' )
end

if (dopause)
    input ('Hit enter to run the Mongoose demo: ', 's') ;
end
try
    mongoose_demo
catch me
    disp (me.message) ;
    fprintf ('Mongoose demo failed\n' )
end

%{
if (dopause)
    input ('Hit enter to run the piro_band demo: ', 's') ;
end
try
    piro_band_demo
catch me
    disp (me.message) ;
    fprintf ('piro_band_demo failed\n' )
end
%}

fprintf ('\n\n---- SuiteSparse demos complete\n') ;

function SuiteSparse_install (do_demo)
%SuiteSparse_install: compiles and installs all of SuiteSparse
% A Suite of Sparse matrix packages, authored or co-authored by Tim Davis.
%
% Packages in SuiteSparse:
%
% GraphBLAS      graph algorithms via sparse linear algebra (graphblas.org)
%                (does not yet have a MATLAB interface)
% Mongoose       graph partitioner
% UMFPACK        sparse LU factorization (multifrontal)
% CHOLMOD        sparse Cholesky factorization, and many other operations
% AMD            sparse symmetric approximate minimum degree ordering
% COLAMD         sparse column approximate minimum degree ordering
% CAMD           constrained AMD
% CCOLAMD        constrained COLAMD
% CSparse        a Concise Sparse matrix package (32-bit or 64-bit, real only)
% CXSparse       extended version of CSparse (32-bit/64-bit/real/complex)
% ssget          interface to SuiteSparse Matrix Collection
% KLU            sparse LU factorization (left-looking)
% BTF            permutation to block triangular form (like dmperm)
% LDL            sparse LDL' factorization
% SuiteSparseCollection   tools for managing the SuiteSparse Matrix Collection
% RBio           read/write Rutherford/Boeing files
% SSMULT         sparse matrix times sparse matrix
% MESHND         2D and 3D regular mesh generation and nested dissection
% FACTORIZE      an object-oriented solver for x=A\b
% SPARSEINV      sparse inverse subset; computes entries of inv(sparse(A))
% MATLAB_Tools   various simple m-files and demos
% SuiteSparseQR  sparse QR factorization
% spqr_rank      MATLAB toolbox for sparse rank deficient matrices
%
% Example:
%    SuiteSparse_install        % compile and prompt to run each package's demo
%    SuiteSparse_install(0)     % compile but do not run the demo
%    SuiteSparse_install(1)     % compile and run the demos with no prompts
%    help SuiteSparse           % for more details
%
% See also AMD, COLAMD, CAMD, CCOLAMD, CHOLMOD, UMFPACK, CSPARSE, CXSPARSE,
%      ssget, RBio, SuiteSparseCollection, KLU, BTF, MESHND, SSMULT, LINFACTOR,
%      SPOK, SPQR_RANK, SuiteSparse, SPQR, PATHTOOL, PATH, FACTORIZE,
%      SPARSEINV, Mongoose.
%
% This script installs the full-featured CXSparse rather than CSparse.
%
% Copyright 1990-2018, Timothy A. Davis, http://www.suitesparse.com.
% In collaboration with (in alphabetical order): Patrick Amestoy, David
% Bateman, Yanqing Chen, Iain Duff, Les Foster, William Hager, Scott Kolodziej,
% Stefan Larimore, Ekanathan Palamadai Natarajan, Sivasankaran Rajamanickam,
% Sanjay Ranka, Wissam Sid-Lakhdar, and Nuri Yeralan.

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

paths = { } ;
SuiteSparse = pwd ;

% determine the MATLAB version (6.1, 6.5, 7.0, ...)
v = version ;
pc = ispc ;

% print the introduction
help SuiteSparse_install

fprintf ('\nInstalling SuiteSparse for MATLAB version %s\n\n', v) ;

% add SuiteSparse to the path
paths = add_to_path (paths, SuiteSparse) ;

%-------------------------------------------------------------------------------
% compile and install the packages
%-------------------------------------------------------------------------------

% compile and install UMFPACK
try
    paths = add_to_path (paths, [SuiteSparse '/UMFPACK/MATLAB']) ;
    umfpack_make ;
catch me
    disp (me.message) ;
    fprintf ('UMFPACK not installed\n') ;
end

% compile and install CHOLMOD
try
    paths = add_to_path (paths, [SuiteSparse '/CHOLMOD/MATLAB']) ;
    cholmod_make ;
catch me
    disp (me.message) ;
    fprintf ('CHOLMOD not installed\n') ;
end

% compile and install AMD
try
    paths = add_to_path (paths, [SuiteSparse '/AMD/MATLAB']) ;
    amd_make ;
catch me
    disp (me.message) ;
    fprintf ('AMD not installed\n') ;
end

% compile and install COLAMD
try
    paths = add_to_path (paths, [SuiteSparse '/COLAMD/MATLAB']) ;
    colamd_make ;
catch me
    disp (me.message) ;
    fprintf ('COLAMD not installed\n') ;
end

% compile and install CCOLAMD
try
    paths = add_to_path (paths, [SuiteSparse '/CCOLAMD/MATLAB']) ;
    ccolamd_make ;
catch me
    disp (me.message) ;
    fprintf ('CCOLAMD not installed\n') ;
end

% compile and install CAMD
try
    paths = add_to_path (paths, [SuiteSparse '/CAMD/MATLAB']) ;
    camd_make ;
catch me
    disp (me.message) ;
    fprintf ('CAMD not installed\n') ;
end

% install ssget, unless it's already in the path
try
    % if this fails, then ssget is not yet installed
    index = ssget ;
    fprintf ('ssget already installed:\n') ;
    which ssget
catch
    index = [ ] ;
end
if (isempty (index))
    % ssget is not installed.  Use SuiteSparse/ssget
    fprintf ('Installing SuiteSparse/ssget\n') ;
    try
        paths = add_to_path (paths, [SuiteSparse '/ssget']) ;
    catch me
        disp (me.message) ;
        fprintf ('ssget not installed\n') ;
    end
end

% compile and install CXSparse
try
    paths = add_to_path (paths, [SuiteSparse '/CXSparse/MATLAB/Demo']) ;
    paths = add_to_path (paths, [SuiteSparse '/CXSparse/MATLAB/CSparse']) ;
    fprintf ('Compiling CXSparse:\n') ;
    if (pc)
	% Windows does not support ANSI C99 complex, which CXSparse requires
	cs_make (1, 0) ;
    else
	cs_make (1) ;
    end
catch me
    disp (me.message) ;
    fprintf ('CXSparse not installed\n') ;
end

% compile and install LDL
try
    paths = add_to_path (paths, [SuiteSparse '/LDL/MATLAB']) ;
    ldl_make ;
catch me
    disp (me.message) ;
    fprintf ('LDL not installed\n') ;
end

% compile and install BTF
try
    paths = add_to_path (paths, [SuiteSparse '/BTF/MATLAB']) ;
    btf_make ;
catch me
    disp (me.message) ;
    fprintf ('BTF not installed\n') ;
end

% compile and install KLU
try
    paths = add_to_path (paths, [SuiteSparse '/KLU/MATLAB']) ;
    klu_make ;
catch me
    disp (me.message) ;
    fprintf ('KLU not installed\n') ;
end

% compile and install SuiteSparseQR
try
    if (pc)
        fprintf ('Note that SuiteSparseQR will not compile with the lcc\n') ;
        fprintf ('compiler provided with MATLAB on Windows\n') ;
    end
    paths = add_to_path (paths, [SuiteSparse '/SPQR/MATLAB']) ;
    spqr_make ;
catch me
    disp (me.message) ;
    fprintf ('SuiteSparseQR not installed\n') ;
end

% compile and install RBio
try
    paths = add_to_path (paths, [SuiteSparse '/RBio/RBio']) ;
    RBmake ;
catch me
    disp (me.message) ;
    fprintf ('RBio not installed.\n') ;
end

% install MATLAB_Tools/*
try
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/Factorize']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/MESHND']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/LINFACTOR']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/find_components']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/GEE']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/shellgui']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/waitmex']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/spqr_rank']) ;
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/spqr_rank/SJget']) ;
    fprintf ('MATLAB_Tools installed\n') ;
catch me
    disp (me.message) ;
    fprintf ('MATLAB_Tools not installed\n') ;
end

% compile and install SuiteSparseCollection
try
    % do not try to compile with large-file I/O for MATLAB 6.5 or earlier
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/SuiteSparseCollection']) ;
    ss_install (verLessThan ('matlab', '7.0')) ;
catch me
    disp (me.message) ;
    fprintf ('SuiteSparseCollection not installed\n') ;
end

% compile and install SSMULT
try
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/SSMULT']) ;
    ssmult_install ;
catch me
    disp (me.message) ;
    fprintf ('SSMULT not installed\n') ;
end

% compile and install dimacs10
try
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/dimacs10']) ;
    dimacs10_install (0) ;
catch me
    disp (me.message) ;
    fprintf ('MATLAB_Tools/dimacs10 not installed\n') ;
end

% compile and install spok
try
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/spok']) ;
    spok_install ;
catch me
    disp (me.message) ;
    fprintf ('MATLAB_Tools/spok not installed\n') ;
end

%{
% compile and install PIRO_BAND
try
    paths = add_to_path (paths, [SuiteSparse '/PIRO_BAND/MATLAB']) ;
    piro_band_make ;
catch me
    disp (me.message) ;
    fprintf ('PIRO_BAND not installed\n') ;
end
%}

% compile and install sparsinv
try
    paths = add_to_path (paths, [SuiteSparse '/MATLAB_Tools/sparseinv']) ;
    sparseinv_install ;
catch me
    disp (me.message) ;
    fprintf ('MATLAB_Tools/sparseinv not installed\n') ;
end

% compile and install Mongoose
try
    paths = add_to_path (paths, [SuiteSparse '/Mongoose/MATLAB']) ;
    mongoose_make (0) ;
catch me
    disp (me.message) ;
    fprintf ('Mongoose not installed\n') ;
end

%-------------------------------------------------------------------------------
% post-install wrapup
%-------------------------------------------------------------------------------

cd (SuiteSparse)
fprintf ('SuiteSparse is now installed.\n\n') ;

% run the demo, if requested
if (nargin < 1)
    % ask if demo should be run
    y = input ('Hit enter to run the SuiteSparse demo (or "n" to quit): ', 's');
    if (isempty (y))
        y = 'y' ;
    end
    do_demo = (y (1) ~= 'n') ;
    do_pause = true ;
else
    % run the demo without pausing
    do_pause = false ;
end
if (do_demo)
    try
	SuiteSparse_demo ([ ], do_pause) ;
    catch me
        disp (me.message) ;
	fprintf ('SuiteSparse demo failed\n') ;
    end
end

% print the list of new directories added to the path
fprintf ('\nSuiteSparse installation is complete.  The following paths\n') ;
fprintf ('have been added for this session.  Use pathtool to add them\n') ;
fprintf ('permanently.  If you cannot save the new path because of file\n');
fprintf ('permissions, then add these commands to your startup.m file.\n') ;
fprintf ('Type "doc startup" and "doc pathtool" for more information.\n\n') ;
for k = 1:length (paths)
    fprintf ('addpath %s\n', paths {k}) ;
end
cd (SuiteSparse)

fprintf ('\nSuiteSparse for MATLAB %s installation complete\n', v) ;

%-------------------------------------------------------------------------------
function paths = add_to_path (paths, newpath)
% add a path
cd (newpath) ;
addpath (newpath) ;
paths = [paths { newpath } ] ;

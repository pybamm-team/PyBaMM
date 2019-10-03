% CHOLMOD: a sparse supernodal Cholesky update/downdate package
%
%   cholmod2        - supernodal sparse Cholesky backslash, x = A\b
%   chol2           - sparse Cholesky factorization, A=R'R.
%   lchol           - sparse A=L*L' factorization.
%   ldlchol         - sparse A=LDL' factorization
%   ldlupdate       - multiple-rank update or downdate of a sparse LDL' factorization.
%   ldlrowmod       - add/delete row from a sparse LDL' factorization.
%   resymbol        - recomputes the symbolic Cholesky factorization of the matrix A.
%   ldlsolve        - solve LDL'x=b using a sparse LDL' factorization
%   ldlsplit        - split an LDL' factorization into L and D.
%   metis           - nested dissection ordering via METIS_NodeND.
%   nesdis          - nested dissection ordering via CHOLMOD's nested dissection.
%   septree         - prune a separator tree.
%   bisect          - computes a node separator based on METIS_ComputeVertexSeparator.
%   analyze         - order and analyze a matrix using CHOLMOD's best-effort ordering.
%   etree2          - sparse elimination tree.
%   sparse2         - replacement for SPARSE
%   symbfact2       - symbolic factorization
%   sdmult          - sparse matrix times dense matrix
%   mread           - read a sparse matrix from a file in Matrix Market format.
%   mwrite          - write a matrix to a file in Matrix Market form.
%   spsym           - determine if a sparse matrix is symmetric, Hermitian, or skew-symmetric.
%   ldl_normest     - estimate the 1-norm of A-L*D*L' without computing L*D*L'
%   cholmod_demo    - a demo for CHOLMOD
%   cholmod_install - compile and install CHOLMOD, AMD, COLAMD, CCOLAMD, CAMD
%   cholmod_make    - compiles the CHOLMOD mexFunctions
%   graph_demo      - graph partitioning demo
%   cholmod_updown_demo - a demo of update/downdate and row add/delete functions
%
%
% Example:
%   x = cholmod2(A,b)

% Note: cholmod has been renamed cholmod2, so as not to conflict with itself
% (the MATLAB built-in cholmod function).

%   Copyright 2006-2015, Timothy A. Davis, http://www.suitesparse.com

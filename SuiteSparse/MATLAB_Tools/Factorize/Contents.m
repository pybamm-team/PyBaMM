%FACTORIZE:  an object-oriented method for solving linear systems and least
% squares problems.  The method provides an efficient way of computing
% mathematical expressions involving the inverse, without actually
% computing the inverse.  For example, S=A-B*inverse(D)*C computes the
% Schur complement by computing S=A-B*(D\C) instead.
%
%   factorize  - an object-oriented method for solving linear systems
%   inverse    - factorized representation of inv(A) or pinv(A).
%
% The package also includes methods for complete orthogonal decomposition
% of full and sparse matrices:
%
%   cod        - complete orthogonal decomposition of a full matrix A = U*R*V'
%   rq         - economy RQ or QL factorization of a full matrix A.
%   cod_sparse - complete orthogonal decomposition of a sparse matrix A = U*R*V'
%   cod_qmult  - computes Q'*X, Q*X, X*Q', or X*Q with Q from COD_SPARSE.
%
% Example
%   cd Demo ; fdemo       % run the demo
%
% "Don't let that INV go past your eyes; to solve that system, FACTORIZE!"
%
% See also chol, lu, ldl, qr, svd.

% Installation and testing:
%
% To install this package, type "pathtool" in the MATLAB command window.  Add
% the directory that contains this Factorize/Contents.m file to the path.  Save
% the path for future use.  Alternatively, type these commands while in this
% directory:
%
%   addpath(pwd)
%   savepath
%
% If you do not have the proper file permissions to save your path, create a
% startup.m file that includes the command "addpath(here)" where "here" is the
% directory containing this file.  Type "help startup" for more information.
%
% The cod function for sparse matrices requires the SPQR mexFunction from the
% SuiteSparse library.  The simplest way to get this is to install all of
% SuiteSparse from http://www.suitesparse.com.
%
% The Test/ subdirectory contains functions that test this package.
%
% The Doc/ subdirectory contains a document that illustrates how to use
% the package (the output of fdemo).

% Object-oriented methods, not meant to be user-callable:
%
%   factorization             - a generic matrix factorization object
%   factorization_chol_dense  - A = R'*R where A is full and symmetric pos. def.
%   factorization_chol_sparse - P'*A*P = L*L' where A is sparse and sym. pos. def.
%   factorization_cod_dense   - complete orthogonal factorization: A = U*R*V' where A is full.
%   factorization_cod_sparse  - complete orthogonal factorization: A = U*R*V' where A is sparse.
%   factorization_ldl_dense   - A(p,p) = L*D*L' where A is sparse and full
%   factorization_ldl_sparse  - P'*A*P = L*D*L' where A is sparse and symmetric
%   factorization_lu_dense    - A(p,:) = L*U where A is square and full.
%   factorization_lu_sparse   - P*A*Q = L*U where A is square and sparse.
%   factorization_qr_dense    - A = Q*R where A is full.
%   factorization_qr_sparse   - (A*P)'*(A*P) = R'*R where A is sparse.
%   factorization_qrt_dense   - A' = Q*R where A is full.
%   factorization_qrt_sparse  - (P*A)*(P*A)'=R'*R where A is sparse.
%   factorization_svd         - A = U*S*V'

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

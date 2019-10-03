function test_disp
%TEST_DISP test the display method of the factorize object
%
% Example
%   test_disp
%
% See also factorize, test_all.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

reset_rand ;
tol = 1e-10 ;
err = 0 ;

%-------------------------------------------------------------------------------
% dense LU
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense LU factorization:\n') ;
A = rand (3) ;
[err,F] = test_factorization (A, tol, err, [ ], 'factorization_lu_dense') ;

fprintf ('\nDense LU With an imaginary F.alpha: ') ;
alpha = (pi + 2i) ;
F = alpha*F ;
display (F) ;
b = rand (3,1) ;
x = F\b ;
y = (alpha*A)\b ;
err = norm (x-y) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
fprintf ('error %g\n', err) ;

%-------------------------------------------------------------------------------
% sparse LU
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse LU factorization:\n') ;
A = sparse (A) ;
err = test_factorization (A, tol, err, [ ], 'factorization_lu_sparse') ;

%-------------------------------------------------------------------------------
% dense Cholesky
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense Cholesky factorization:\n') ;
A = A*A' + eye (3) ;
err = test_factorization (A, tol, err, [ ], 'factorization_chol_dense') ;

%-------------------------------------------------------------------------------
% sparse Cholesky
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse Cholesky factorization:\n') ;
A = sparse (A) ;
err = test_factorization (A, tol, err, [ ], 'factorization_chol_sparse') ;

%-------------------------------------------------------------------------------
% dense QR of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense QR factorization:\n') ;
A = rand (3,2) ;
err = test_factorization (A, tol, err, 'qr', 'factorization_qr_dense') ;

%-------------------------------------------------------------------------------
% dense COD of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense COD factorization:\n') ;
err = test_factorization (A, tol, err, [ ], 'factorization_cod_dense') ;

%-------------------------------------------------------------------------------
% sparse COD of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse COD factorization:\n') ;
A = sparse (A) ;
err = test_factorization (A, tol, err, 'cod', 'factorization_cod_sparse') ;

%-------------------------------------------------------------------------------
% dense QR of A'
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense QR factorization of A'':\n') ;
A = full (A) ;
err = test_factorization (A', tol, err, 'qr', 'factorization_qrt_dense') ;

%-------------------------------------------------------------------------------
% sparse QR of A
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse QR factorization:\n') ;
A = sparse (A) ;
err = test_factorization (A, tol, err, [ ], 'factorization_qr_sparse') ;

%-------------------------------------------------------------------------------
% sparse QR of A'
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse QR factorization of A'':\n') ;
err = test_factorization (A', tol, err, [ ], 'factorization_qrt_sparse') ;

%-------------------------------------------------------------------------------
% svd
%-------------------------------------------------------------------------------

fprintf ('\n----------SVD factorization:\n') ;
err = test_factorization (A, tol, err, 'svd', 'factorization_svd') ;

%-------------------------------------------------------------------------------
% dense LDL
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense LDL factorization:\n') ;
A = rand (3) ;
A = [zeros(3) A ; A' zeros(3)] ;
err = test_factorization (A, tol, err, 'ldl', 'factorization_ldl_dense') ;

%-------------------------------------------------------------------------------
% sparse LDL
%-------------------------------------------------------------------------------

fprintf ('\n----------Sparse LDL factorization:\n') ;
A = sparse (A) ;
err = test_factorization (A, tol, err, 'ldl', 'factorization_ldl_sparse') ;

%-------------------------------------------------------------------------------
% test QR and QR' with scalar A and sparse right-hand side
%-------------------------------------------------------------------------------

fprintf ('\n----------Dense QR and QR'' with scalar A and sparse b:\n') ;
b = sparse ([1 2]) ;
A = pi ;
F = factorization_qr_dense (A,0) ;
display (F) ;
x = F\b ;
err = max (err, norm (A\b - x)) ;
x = b'/F ;
err = max (err, norm (b'/A - x)) ;
F = factorization_qrt_dense (A,0) ;
display (F) ;
x = F\b ;
err = max (err, norm (A\b - x)) ;
x = b'/F ;
err = max (err, norm (b'/A - x)) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end

fprintf ('\nAll disp tests passed, max error: %g\n', err) ;

%-------------------------------------------------------------------------------

function [err, F] = test_factorization (A, tol, err, option, kind)
%TEST_FACTORIZATION factorize a matrix and check its kind and error norm
F = factorize (A, option, 1) ;
display (F) ;
S = inverse (F) ;
display (S) ;
err2 = error_check (F) ;
fprintf ('error: %g\n', err2) ;
err = max (err, err2) ;
if (err > tol)
    error ('error too high: %g\n', err) ;
end
if (F.is_inverse || ~isa (F, kind))
    error ('invalid contents') ;
end
if (~(S.is_inverse) || ~isa (S, kind))
    error ('invalid contents') ;
end

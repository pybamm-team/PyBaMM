function quickdemo_spqr_rank
%QUICKDEMO_SPQR_RANK quick demo of the spqr_rank package
%
% Example:
%   quickdemo_spqr_rank
%
% See also spqr_basic, spqr_cod, spqr_null, spqr_pinv, spqr.

% Copyright 2012, Leslie Foster and Timothy A Davis.

A = sparse(gallery('kahan',100));
B = randn(100,1); B = B / norm(B);

%-------------------------------------------------------------------------------
% spqr_basic test / demo
%-------------------------------------------------------------------------------

fprintf ('\nSPQR_BASIC approximate basic solution to min(norm(B-A*x)):\n') ;
fprintf ('x = spqr_basic(A,B)\n') ;

x = spqr_basic(A,B);
norm_x = norm(x) ;
% compare with
x2 = spqr_solve(A,B);
norm_x2 = norm(x2) ;
% or
[x,stats,NT]=spqr_basic(A,B);                                               %#ok
norm_NT_transpose_times_A = norm(full(spqr_null_mult(NT,A,0))) ;
% or
opts = struct('tol',1.e-5) ;
[x,stats,NT]=spqr_basic(A,B,opts);                                          %#ok
display (stats) ;
fprintf ('norm of x: %g\nnorm of x with just SPQR: %g\n', norm_x, norm_x2) ;

if (norm_NT_transpose_times_A > 1e-12)
    error ('test failure') ;
end

%-------------------------------------------------------------------------------
% spqr_cod test / demo
%-------------------------------------------------------------------------------

fprintf ('\nSPQR_COD approximate pseudoinverse solution to min(norm(B-A*x)\n') ;
fprintf ('x = spqr_cod(A,B)\n') ;

x = spqr_cod(A,B);
x_pinv = pinv(full(A))*B;
rel_error_in_x = norm(x - x_pinv) / norm(x_pinv) ;
fprintf ('relative error in x: %g\n', rel_error_in_x) ;
% or
[x,stats,N,NT]=spqr_cod(A,B);                                               %#ok
norm_A_times_N = norm(full(spqr_null_mult(N,A,3))) ;
norm_A_transpose_times_NT = norm(full(spqr_null_mult(NT,A,0))) ;
% or
opts = struct('tol',1.e-5) ;
[x,stats]=spqr_cod(A,B,opts);                                               %#ok
display (stats) ;

if (max ([rel_error_in_x norm_A_times_N norm_A_transpose_times_NT]) > 1e-12)
    error ('test failure') ;
end

%-------------------------------------------------------------------------------
% spqr_null test / demo
%-------------------------------------------------------------------------------

fprintf ('\nSPQR_NULL orthonormal basis for numerical null space\n') ;
fprintf ('N = spqr_null(A)\n') ;

N = spqr_null(A) ;
display (N) ;
norm_A_times_N = norm(full(spqr_null_mult(N,A,3))) ;
% or
opts = struct('tol',1.e-5,'get_details',2);
[N,stats]=spqr_null(A,opts);                                                %#ok
rank_spqr_null = stats.rank ;
rank_spqr = stats.rank_spqr ;
rank_svd = rank(full(A)) ;

fprintf ('rank with spqr: %g with svd: %g with spqr_null: %g\n', ...
    rank_spqr, rank_svd, rank_spqr_null) ;

if (rank_spqr_null ~= rank_svd || norm_A_times_N > 1e-12)
    error ('test failure') ;
end

%-------------------------------------------------------------------------------
% spqr_pinv test / demo
%-------------------------------------------------------------------------------

fprintf ('\nSPQR_PINV approx pseudoinverse solution to min(norm(B-A*X))\n') ;
fprintf ('x = spqr_pinv(A,B)\n') ;

x = spqr_pinv(A,B) ;
x_pinv = pinv(full(A))*B ;
rel_error_in_x = norm (x - x_pinv) / norm (x_pinv) ;
fprintf ('relative error in x: %g\n', rel_error_in_x) ;
% or
[x,stats,N,NT] = spqr_pinv (A,B) ;                                          %#ok
display (N) ;
display (NT) ;
norm_A_times_N = norm (full(spqr_null_mult(N,A,3))) ;
norm_N_transpose_times_A = norm (full(spqr_null_mult(NT,A,0))) ;
% or
opts = struct('tol',1.e-5) ;
[x,stats] = spqr_pinv (A,B,opts) ;                                          %#ok
display (stats) ;

if (max ([rel_error_in_x norm_A_times_N norm_N_transpose_times_A ]) > 1e-12)
    error ('test failure') ;
end

fprintf ('quickdemo_spqr_rank: all tests passed\n') ;

function [N,stats] = spqr_null (A, varargin)
%SPQR_NULL finds an orthonormal basis for numerical null space of a matrix
%
% [N,stats] = spqr_null(A,opts)
%
% Returns an orthonormal basis, N, for the numerical null space of the
% m by n matrix A for tolerance tol. An orthonormal basis for the numerical
% null space is an  n by k matrix, stored explicitly or implicitly (see
% below), with orthonormal columns such that
%                 || A * N || <= tol                    (1)
% and no matrix with more than k orthonormal columns satisfies (1).
% This routine, when the error flag is 0, ensures that
%          an estimate of || A * N || satisfies (1).
% Also, optionally, the routine returns the numerical rank of the matrix
% A for tolerance tol (i.e. the number of singular values > tol) and
% additional statistics described below.
%
% Input:  A -- an m by n matrix
%         opts (optional) -- type 'help spqr_rank_opts' for details.
% Output:
%         N - orthonormal basis for the numerical null space of A:
%             an estimate of || A * N || <= opts.tol, when stats.flag is 0.
%
%  Examples:
%     A=sparse(gallery('kahan',100));
%     N = spqr_null(A);
%     norm_A_times_N = norm(full(spqr_null_mult(N,A,3)))
%     % or
%     opts = struct('tol',1.e-5,'get_details',2);
%     [N,stats]=spqr_null(A,opts);
%     rank_spqr_null = stats.rank
%     rank_spqr = stats.rank_spqr
%     rank_svd = rank(full(A))
%
% See also spqr_basic, spqr_null, spqr_pinv, spqr_cod.

% Copyright 2012, Leslie Foster and Timothy A Davis.

% Algorithm:  First spqr is used to construct a QR factorization of the
%    n by m matrix A': A'*P1 = Q1*R where R' = [ R1' 0 ] + E1, R1 is a
%    k by n upper trapezoidal matrix and E1 is a small error matrix.
%    Let R1  = [ R11 R12] where R11 is k by k and R12 is k by n-k.
%    Subspace iteration, using the routine spqr_ssi, is applied to R11 to
%    determine if the rank returned by spqr, k, is correct. If k is correct
%    then an orthogonal basis for the null space of A is
%                          [ 0 ]
%                 N = Q1 * [ I ]                    (1)
%    where I is an n-k by n-k identity matrix and 0 is k by n-k.  Suppose
%    that the numerical rank of R11 is r < k.  The routine spqr_ssi constructs
%    an orthogonal basis, U, for the numerical null space of R11'.  Then
%    the candidate for the orthogonal basis for the null space of A is
%                          [ U 0 ]
%                 N = Q1 * [ 0 I ]                  (2)
%
%    where I is n-k by n-k and U is k by k-r.  N is used to estimate
%    upper bounds on the smallest n-r singular values of A to confirm
%    (or not) that N is indeed an orthogonal basis for the numerical null
%    space of A.  If opts.implicit_null_space_basis is 1 (the default) then
%    N is stored implicitly by saving the factors in (1) or (2) and storing
%    Q1 using its Householder factors.

%-------------------------------------------------------------------------------
% get opts: tolerance and number of singular values to estimate
%-------------------------------------------------------------------------------

[ignore,opts,stats,start_tic,ok] = spqr_rank_get_inputs (A,0,varargin {:}); %#ok
clear ignore

if (~ok || nargout > 2)
    error ('usage: [N,stats] = spqr_null (A,opts)') ;
end

if (opts.get_details == 1)
    % the only thing needed from stats, above, is stats.time_initialize
    % and stats.normest_A, if calculated
    t = stats.time_initialize ;
    if ( isfield(stats,'normest_A') )
       normest_A = stats.normest_A ;
    end
end

%-------------------------------------------------------------------------------
% use spqr_basic on A' (with no B) to find the null space
%-------------------------------------------------------------------------------

% In spqr_basic, input B, internal variable C, and output x will all be empty.
[ignore,stats,N] = spqr_basic (A', [ ], opts) ;                             %#ok
clear ignore

if (opts.get_details == 1)
    stats.time_initialize = t ;
    if exist('normest_A','var')
        stats.normest_A = normest_A;
    end
end

%-------------------------------------------------------------------------------
% fix the stats to reflect N instead of NT
%-------------------------------------------------------------------------------

% spqr_basic might return early, so we need to check if they exist.
if (isfield (stats, 'est_norm_A_transpose_times_NT'))
    stats.est_norm_A_times_N = stats.est_norm_A_transpose_times_NT  ;
    stats = rmfield (stats, 'est_norm_A_transpose_times_NT')  ;
end

if (isfield (stats, 'est_err_bound_norm_A_transpose_times_NT'))
    stats.est_err_bound_norm_A_times_N = ...
        stats.est_err_bound_norm_A_transpose_times_NT ;
    stats = rmfield (stats, 'est_err_bound_norm_A_transpose_times_NT') ;
end

if (isfield (stats, 'stats_ssp_NT'))
    stats.stats_ssp_N = stats.stats_ssp_NT;
    stats = rmfield (stats,'stats_ssp_NT') ;
end

if opts.get_details == 1
    % order the fields of stats in a convenient order (the fields when
    %    get_details is 0 or 2 are already in a good order)
    stats = spqr_rank_order_fields(stats);
end

if (opts.get_details == 1)
    stats.time = toc (start_tic) ;
end


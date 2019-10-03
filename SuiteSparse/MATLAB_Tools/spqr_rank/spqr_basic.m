function [x,stats,NT] = spqr_basic (A, varargin)
%SPQR_BASIC approximate basic solution to min(norm(B-A*x))
% for a rank deficient matrix A.
%
% [x,stats,NT] = spqr_basic (A,B,opts)
%
% This function returns an approximate basic solution to
%       min || B - A x ||                     (1)
% for rank deficient matrices A.
%
% Optionally returns statistics including the numerical rank of the matrix A
% for tolerance tol (i.e. the number of singular values > tol), and
% an orthnormal basis for the numerical null space of A'.
%
% The solution is approximate since the algorithm allows small perturbations in
% A (columns of A may be changed by no more than opts.tol).
%
% Input:
%   A -- an m by n matrix
%   B -- an m by p right hand side matrix
%   opts (optional) -- type 'help spqr_rank_opts' for details.
%
% Output:
%   x -- this n by p matrix contains basic solutions to (1).  Each column of x
%       has at most k nonzero entries where k is the approximate numerical rank
%       returned by spqr.  The magnitude of x(:,j) is bounded by
%       norm(B(:,j))/s, where s = stats.est_sval_lower_bounds(nlarge_svals) is
%       an estimated lower bound on (stats.rank)th singular value of
%       A.
%   stats -- statistics, type 'help spqr_rank_stats' for details.
%   NT -- orthonormal basis for numerical null space of A'.
%
% Example:
%
%     A = sparse(gallery('kahan',100));
%     B = randn(100,1); B = B / norm(B);
%     x = spqr_basic(A,B);
%     norm_x = norm(x)
%     % note compare with
%     x2 = spqr_solve(A,B);
%     norm_x2 = norm(x2)
%     % or
%     [x,stats,NT]=spqr_basic(A,B);
%     norm_NT_transpose_times_A = norm(full(spqr_null_mult(NT,A,0)))
%     % or
%     opts = struct('tol',1.e-5) ;
%     [x,stats,NT]=spqr_basic(A,B,opts);
%     stats
%
% See also spqr_cod, spqr_null, spqr_pinv, spqr_ssi, spqr_ssp

% Copyright 2012, Leslie Foster and Timothy A Davis.

% Algorithm:  First spqr is used to construct a QR factorization of the
%    m by n matrix A: A*P1 = Q1*R where R' = [ R1' 0 ] + E1, R1 is a
%    k by n upper trapezoidal matrix and E1 is a small error matrix.
%    Let R11 be the leading k by k submatrix of R1. Subspace iteration,
%    using the routine spqr_ssi,  is applied to R11 to determine if the rank
%    returned by spqr is correct and also, often, to determine the correct
%    numerical rank.  If k is the correct then the basic solution is
%    x = R11 \ ch where ch is the first k entries in Q1'*B. If k is not
%    the correct numerical rank, deflation (see SIAM SISC, 11:519-530,
%    1990) is used in the calculation of a basic solution.

%-------------------------------------------------------------------------------
% get opts: tolerance and number of singular values to estimate
%-------------------------------------------------------------------------------

[B,opts,stats,start_tic,ok] = spqr_rank_get_inputs (A, 1, varargin {:}) ;

if (~ok || nargout > 3)
    error ('usage: [x,stats,NT] = spqr_basic (A,B,opts)') ;
end

% get the options
tol = opts.tol ;
nsvals_small = opts.nsvals_small ;
nsvals_large = opts.nsvals_large ;
get_details = opts.get_details ;

% set the order of the stats fields
%     stats.flag, stats.rank, stats.rank_spqr, stats.rank_spqr (if get_details
%     >= 1), stats.tol, stats.tol_alt, stats.normest_A (if calculated),
%     stats.est_sval_upper_bounds, stats.est_sval_lower_bounds, and
%     stats.sval_numbers_for_bounds already initialized in spqr_rank_get_inputs
if nargout == 3
   stats.est_norm_A_transpose_times_NT = -1 ;
end
if get_details == 2
    stats.stats_ssi = -1 ;
end
% order for the additional stats fields for case where get_details is 1 will be
%     set using spqr_rank_order_fields, called from spqr_rank_assign_stats

if (get_details == 1)
    stats.opts_used = opts ;
    stats.time_basis = 0 ;
end

[m,n] = size (A) ;

%-------------------------------------------------------------------------------
% QR factorization of A, and initial estimate of numerical rank, via spqr
%-------------------------------------------------------------------------------

% compute Q*R = A(:,p) and C=Q'*B.  Keep Q if NT is requested; else discard.
if nargout <= 2
    [Q,R,C,p,info_spqr1] = spqr_wrapper (A, B, tol, 'discard Q', get_details) ;
else
    [Q,R,C,p,info_spqr1] = spqr_wrapper (A, B, tol, 'keep Q', get_details) ;
end

% the next line is equivalent to: rank_spqr = size (R,1) ;
rank_spqr = info_spqr1.rank_A_estimate;
norm_E_fro = info_spqr1.norm_E_fro ;

% save the stats
if (get_details == 1 || get_details == 2)
    stats.rank_spqr = rank_spqr ;
end
if (get_details == 1)
    stats.info_spqr1 = info_spqr1 ;
end

%-------------------------------------------------------------------------------
% use spqr_ssi to check and adjust numerical rank from spqr
%-------------------------------------------------------------------------------

R11 = R (:, 1:rank_spqr) ;
if get_details == 0
    % opts2 used to include a few extra statistics in stats_ssi
    opts2 = opts;
    opts2.get_details = 2;
else
    opts2 = opts;
end
[U,S,V,stats_ssi] = spqr_ssi (R11, opts2) ;

if (get_details == 1 || get_details == 2)
    stats.stats_ssi = stats_ssi ;
end
if (get_details == 1)
    stats.time_svd = stats_ssi.time_svd ;
end

%-------------------------------------------------------------------------------
% check for early return
%-------------------------------------------------------------------------------

if stats_ssi.flag == 4
    % overflow occurred during the inverse power method in ssi
    [stats x NT] = spqr_failure (4, stats, get_details, start_tic) ;
    return
end

%-------------------------------------------------------------------------------
% Estimate lower bounds on the singular values of A
%-------------------------------------------------------------------------------

% In spqr the leading rank_spqr column of A*P are unmodified. Therefore
% by the interleave theorem for singular values the singular values of
% R11 = R(:,1:rank_spqr) are lower bounds for the singular
% values of A.  In spqr_ssi estimates for the errors in calculating the
% singular values of R are in stats_ssi.est_error_bounds.  Therefore,
% for i = 1:k, where S is k by k, estimated lower bounds on singular
% values number (rank_spqr - k + i) of A are in est_sval_lower_bounds:
%
est_sval_lower_bounds = max (diag(S)' - stats_ssi.est_error_bounds,0) ;

% lower bounds on the remaining singular values of A are zero
est_sval_lower_bounds (length(S)+1:length(S)+min(m,n)-rank_spqr) = 0 ;

numerical_rank = stats_ssi.rank ;

% limit nsvals_small and nsvals_large due to number of singular values
%     available and calculated by spqr_ssi
nsvals_small = min ([nsvals_small, min(m,n) - numerical_rank]) ;

nsvals_large = min (nsvals_large, rank_spqr) ;
nsvals_large = min ([nsvals_large, numerical_rank, ...
    numerical_rank - rank_spqr + stats_ssi.ssi_max_block_used]) ;

% return nsvals_large + nsvals_small of the estimates
est_sval_lower_bounds = est_sval_lower_bounds (1:nsvals_large+nsvals_small) ;

%-------------------------------------------------------------------------------
% Estimate upper bounds on the singular values of A
%-------------------------------------------------------------------------------

% By the minimax theorem for singular values, for any rank_spqr by k
% matrix U with orthonormal columns, for i = 1:k singular value i
% of U'*R is an upper bound on singular values rank_spqr - k + i of
% the rank_spqr by n matrix R.  Therefore we have upper bounds on the
% singular values number rank_spqr - k + i, i = 1:k, of R:

if (get_details == 1)
    t = tic ;
end

s = svd (full (U'*R))' ;

if (get_details == 1)
    stats.time_svd = stats.time_svd + toc (t);
end

% Since the Frobenius norm of A*P - Q*R is norm_E_fro, the singular
% values of A and R differ by at most norm_E_fro.  Therefore we have
% the following upper bounds on the singular values of A:
est_sval_upper_bounds = s + norm_E_fro;

% upper bounds on the remaining singular values of A are norm_E_fro
est_sval_upper_bounds(length(S)+1:length(S)+min(m,n)-rank_spqr) = norm_E_fro;

% return nsvals_large + nsvals_small components of the estimates
est_sval_upper_bounds = est_sval_upper_bounds(1:nsvals_large+nsvals_small);

%-------------------------------------------------------------------------------
% if requested, calculate orthonormal basis for null space of A'
%-------------------------------------------------------------------------------

if nargout == 3

    call_from = 1;
    [NT, stats, stats_ssp_NT, est_sval_upper_bounds] = spqr_rank_form_basis(...
        call_from, A, U, V ,Q, rank_spqr, numerical_rank, stats, opts, ...
        est_sval_upper_bounds, nsvals_small, nsvals_large) ;

end

%-------------------------------------------------------------------------------
% find solution R11 * wh = ch where R11 = R(:,1:rank_spqr)
%-------------------------------------------------------------------------------

call_from = 1;
R = R (:, 1:rank_spqr) ;            % discard R12, keep R11 only
x = spqr_rank_deflation(call_from, R, U, V, C, m, n, rank_spqr, ...
        numerical_rank, nsvals_large, opts, p);

%-------------------------------------------------------------------------------
% determine flag which indicates accuracy of the estimated numerical rank
%    and return statistics
%-------------------------------------------------------------------------------

call_from = 1;
stats_ssp_N = [];
if nargout < 3
    stats_ssp_NT = [ ] ;
end
stats  =  spqr_rank_assign_stats(...
   call_from, est_sval_upper_bounds, est_sval_lower_bounds, tol, ...
   numerical_rank, nsvals_small, nsvals_large, stats, ...
   stats_ssi, opts, nargout, stats_ssp_N, stats_ssp_NT, start_tic) ;


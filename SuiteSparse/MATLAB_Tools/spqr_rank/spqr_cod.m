function [x,stats,N,NT]= spqr_cod(A,varargin)
%SPQR_COD approximate pseudoinverse solution to min(norm(B-A*x)
% for a rank deficient matrix A.
%
% [x,stats,N,NT] = spqr_cod (A,B,opts)
%
% This function returns an approximate pseudoinverse solution to
%       min || B - A x ||                     (1)
% for rank deficient matrices A.
% The psuedoinverse solution is the min norm solution to the
% least squares problem (1).
%
% Optionally returns statistics including the numerical rank of the matrix A
% for tolerance tol (i.e. the number of singular values > tol), and
% orthnormal bases for the numerical null spaces of A and A'.
% The solution is approximate since the algorithm allows small perturbations in
% A (columns of A may be changed by no more than opts.tol).
%
% This routine calculates an approximate complete orthogonal decomposition of
% A. The routine can be more accurate -- and more expensive -- than spqr_pinv.
%
% Input:
%   A -- an m by n matrix
%   B -- an m by p right hand side matrix
%   opts (optional) -- type 'help spqr_rank_opts' for details.
%
% Output:
%   x -- this n by p matrix contains psuedoinverse solutions to (1).
%   stats -- statistics, type 'help spqr_rank_stats' for details.
%   N -- orthonormal basis for numerical null space of A.
%   NT -- orthonormal basis for numerical null space of A'.
%
% Example:
%
%     A = sparse(gallery('kahan',100));
%     B = randn(100,1); B = B / norm(B);
%     x = spqr_cod(A,B);
%     x_pinv = pinv(full(A))*B;
%     rel_error_in_x = norm(x - x_pinv) / norm(x_pinv)
%     % or
%     [x,stats,N,NT]=spqr_cod(A,B);
%     norm_A_times_N = norm(full(spqr_null_mult(N,A,3)))
%     norm_A_transpose_times_NT = norm(full(spqr_null_mult(NT,A,0)))
%     % or
%     opts = struct('tol',1.e-5) ;
%     [x,stats]=spqr_cod(A,B,opts);
%     stats
%
% See also spqr_basic, spqr_null, spqr_pinv, spqr_ssi, spqr_ssp

% Copyright 2012, Leslie Foster and Timothy A Davis.

% Algorithm:  First spqr is used to construct a QR factorization
%   of A:
%       m by n matrix A: A*P1 = Q1*R where R' = [ R1' 0 ] + E1,
%       R1 is a k by n upper trapezoidal matrix
%   or A':
%       n by m matrix A': A'*P1 = Q1*R where R' = [ R1' 0 ] + E1,
%       R1 is a k by m upper trapezoidal matrix
%   and where E1 is a small error matrix.
%
%    Next a QR factorization of R1' is calculated:
%    R1' * P2 = Q2 * R2 where R2' = [ T' 0 ] and T is k by k, upper
%    triangular. This determines an approximate complete orthogonal
%    decomposition of A:
%                  [ P2  0 ]    [ T'  0 ]
%        A = Q1 *  [ 0   I ] *  [ 0   0 ] * Q2' * P1' + (E1*P1').    (2)
%   or of A':
%                      [ T  0 ]  [ P2'  0 ]
%        A = P1 *Q2 *  [ 0  0 ] *[ 0    I ]  * Q1' + (P1*E1').    (2)
%
%    This is then used to calculate an approximate pseudoinverse solution
%    to (1) and an orthogonal basis for the left and right null spaces of
%    A, when they are requested.  Subspace iteration, using the routine
%    spqr_ssi,  is applied to T to determine if the rank returned by spqr is
%    correct and also, often, to determine the correct numerical
%    rank.  When the two ranks are different deflation (see SIAM SISC,
%    11:519-530, 1990.) is used in the caculation of the psuedoinverse
%    solution.
%
%    When opts.start_with_A_transpose is 1 (the default value is 0) then
%    initially spqr constructs a QR factorization of A'.  By default, A
%    is factorized.

%-------------------------------------------------------------------------------
% get opts: tolerance and number of singular values to estimate
%-------------------------------------------------------------------------------

[B,opts,stats,start_tic,ok] = spqr_rank_get_inputs (A, 1, varargin {:}) ;

if (~ok || nargout > 4)
    error ('usage: [x,stats,N,NT] = spqr_cod (A,B,opts)') ;
end

% get the options
tol = opts.tol ;
nsvals_small = opts.nsvals_small ;
nsvals_large = opts.nsvals_large ;
get_details = opts.get_details ;
start_with_A_transpose = opts.start_with_A_transpose ;

% set the order of the stats fields
%     stats.flag, stats.rank, stats.rank_spqr, stats.rank_spqr (if get_details
%     >= 1), stats.tol, stats.tol_alt, stats.normest_A (if calculated),
%     stats.est_sval_upper_bounds, stats.est_sval_lower_bounds, and
%     stats.sval_numbers_for_bounds already initialized in spqr_rank_get_inputs
if nargout >= 3
    stats.est_norm_A_times_N = -1 ;
end
if nargout == 4
   stats.est_norm_A_transpose_times_NT = -1 ;
end
if get_details == 2
    stats.stats_ssi = -1 ;
end
% order for the additional stats fields for case where get_details is 1 will be
%     set using spqr_rank_order_fields at the end of the routine


if (get_details == 1)
    stats.opts_used = opts ;
    stats.time_basis = 0 ;
end

[m,n] = size (A) ;

%-------------------------------------------------------------------------------
% first QR factorization of A or A', and initial estimate of num. rank, via spqr
%-------------------------------------------------------------------------------

if (start_with_A_transpose)
    % Compute Q1*R = A(p1,:)', do not compute C, and keep Q1.
    [Q1,R,C,p1,info_spqr1] = ...
        spqr_wrapper (A', [ ], tol, 'keep Q', get_details) ;
    B = B(p1,:);   % or:  p=1:m; p(p1) = 1:length(p1); B=B(p,:);
else
    % Compute Q1*R = A(:,p1), C=Q1'*B.  Keep Q only if needed for NT
    if nargout <= 3
        [Q1,R,C,p1,info_spqr1] = ...
            spqr_wrapper (A, B, tol, 'discard Q', get_details) ;
    else
        [Q1,R,C,p1,info_spqr1] = ...
            spqr_wrapper (A, B, tol, 'keep Q', get_details) ;
    end
end

% the next line is the same as: rank_spqr = size (R,1) ;
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
% second QR factorization of R', with zero tolerance
%-------------------------------------------------------------------------------

if (start_with_A_transpose)
    % Compute Q2*R2 = R(p2,:)', C=Q2'*B, overwrite R with R2.  Keep Q for NT
    if nargout <= 3
        [Q2,R,C,p2,info_spqr2] = ...
            spqr_wrapper (R', B, 0, 'discard Q', get_details) ;
    else
        [Q2,R,C,p2,info_spqr2] = ...
            spqr_wrapper (R', B, 0, 'keep Q', get_details) ;
    end
else
    % Compute Q2*R2 = R(p2,:)', do not compute C, keep Q2, overwrite R with R2.
    [Q2,R,ignore,p2,info_spqr2] = spqr_wrapper (R', [ ], 0, 'keep Q', ...
        get_details) ;                                                      %#ok
    clear ignore
end

if (get_details == 1)
    stats.info_spqr2 = info_spqr2 ;
end

%-------------------------------------------------------------------------------
% check if the numerical rank is consistent between the two QR factorizations
%-------------------------------------------------------------------------------

if rank_spqr ~= size (R,1)
    % Approximate rank from two sparse QR factorizations are inconsistent.
    % This should "never" happen.  We know of no matrix that triggers this
    % condition, and so the following line of code is untested.
    error ('spqr_rank:inconsistent', 'inconsistent rank estimates') ; % untested
    % rather than returning an error, we could do the following instead,
    % but the code would be still untestable:
    %   warning ('spqr_rank:inconsistent', 'inconsistent rank estimates') ;
    %   [stats x N NT] = spqr_failure (5, stats, get_details, start_tic) ;
    %   return
end

% R is now square and has dimension rank_spqr

%-------------------------------------------------------------------------------
% use spqr_ssi to check and adjust numerical rank from spqr
%-------------------------------------------------------------------------------

R11 = R ;
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
    % overflow occurred during the inverse power method in spqr_ssi
    [stats x N NT] = spqr_failure (4, stats, get_details, start_tic) ;
    return
end

%-------------------------------------------------------------------------------
% Estimate lower bounds on the singular values of A
%-------------------------------------------------------------------------------

% In equation (2) the Frobenius norm of E1*P1' is equal to norm_E_fro =
% info_spqr1.norm_E_fro and therefore || E1*P1' || <= norm_E_fro.
% By the pertubation theorem for singular values, for i = 1, 2, ...,
% rank_spqr, singular value i of A differs at most by norm_E_fro
% from singular value i of R.  The routine spqr_ssi returns estimates of the
% singular values of R in S and stats_ssi.est_error_bounds contains
% estimates of error bounds on the entries in S. Therefore,
% for i = 1:k, where S is k by k, estimated lower bounds on singular
% values number (rank_spqr - k + i) of A are in est_sval_lower_bounds:
%
est_sval_lower_bounds = ...
    max (diag(S)' - stats_ssi.est_error_bounds - norm_E_fro, 0) ;

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

% Again, by the pertubation theorem for singular values, for i = 1, 2,...,
% rank_spqr, singular value i of A differs at most by norm_E_fro
% from singular value i of R.  The routine spqr_ssi returns estimates of the
% singular values of R in S and stats_ssi.est_error_bounds contains
% estimates of error bounds on the entries in S. Therefore,
% for i = 1:k, where S is k by k, estimated lower bounds on singular
% values number (rank_spqr - k + i) of A are in est_sval_upper_bounds:
est_sval_upper_bounds = diag(S)' + stats_ssi.est_error_bounds + norm_E_fro;

% upper bounds on the remaining singular values of A are norm_E_fro
est_sval_upper_bounds(length(S)+1:length(S)+min(m,n)-rank_spqr) = norm_E_fro;

% return nsvals_large + nsvals_small components of the estimates
est_sval_upper_bounds = est_sval_upper_bounds(1:nsvals_large+nsvals_small);

%-------------------------------------------------------------------------------
% calculate orthonormal basis for null space of A
%-------------------------------------------------------------------------------

% always construct null space basis for A since this can produce better
%    estimated upper bounds on the singular values and should not require
%    significantly more work or memory

call_from = 2;
[N, stats, stats_ssp_N, est_sval_upper_bounds] = spqr_rank_form_basis(...
    call_from, A, U, V ,Q1, rank_spqr, numerical_rank, stats, opts, ...
    est_sval_upper_bounds, nsvals_small, nsvals_large, p1, Q2, p2) ;

%-------------------------------------------------------------------------------
% if requested, form null space basis of A'
%-------------------------------------------------------------------------------

if nargout == 4

    call_from = 3;
    [NT, stats, stats_ssp_NT, est_sval_upper_bounds] = spqr_rank_form_basis(...
        call_from, A, U, V ,Q1, rank_spqr, numerical_rank, stats, opts, ...
        est_sval_upper_bounds, nsvals_small, nsvals_large, p1, Q2, p2) ;

end

%-------------------------------------------------------------------------------
% find psuedoinverse solution to (1):
%-------------------------------------------------------------------------------

call_from = 2 ;
x = spqr_rank_deflation(call_from, R, U, V, C, m, n, rank_spqr, ...
        numerical_rank, nsvals_large, opts, p1, p2, N, Q1, Q2) ;

%-------------------------------------------------------------------------------
% determine flag which indicates accuracy of the estimated numerical rank
%    and return statistics
%-------------------------------------------------------------------------------

call_from = 2;
if nargout < 4
    stats_ssp_NT = [ ];
end
stats  =  spqr_rank_assign_stats(...
   call_from, est_sval_upper_bounds, est_sval_lower_bounds, tol, ...
   numerical_rank, nsvals_small, nsvals_large, stats, ...
   stats_ssi, opts, nargout, stats_ssp_N, stats_ssp_NT, start_tic) ;


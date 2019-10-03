function [x,stats,N,NT] = spqr_pinv (A, varargin)
%SPQR_PINV approx pseudoinverse solution to min(norm(B-A*X))
%
% usage: [x,stats,N,NT] = spqr_pinv (A,B,opts) ;
%
% This function returns an approximate psuedoinverse solution to
%                 min || B - A x ||                     (1)
% for rank deficient matrices A.  The psuedoinverse solution is the minimum
% norm solution to the least squares problem (1).  Also, optionally, the
% routine returns statistics including the numerical rank of the matrix A for
% tolerance tol (i.e. the number of singular values > tol) and other statistics
% (see below), as well as, if requested, an orthonormal basis for the numerical
% null space to A and an orthonormal basis for the null space of A transpose.
% The psuedoinverse solution is approximate since the algorithm allows small
% perturbations in A (columns of A may be changed by no more than a user
% defined value in opts.tol).
%
% Input:  A -- an m by n matrix
%         B -- an m by p right hand side matrix
%         opts (optional) -- see below
% Output: x -- this n by p matrix contains psuedoinverse solutions to (1).
%              If B is empty then x will also be empty.
%         stats (optional) -- statistics including:
%
%  Examples:
%     A = sparse (gallery('kahan',100)) ;
%     B = randn (100,1) ; B = B / norm(B) ;
%     x = spqr_pinv(A,B) ;
%     x_pinv = pinv(full(A))*B ;
%     rel_error_in_x = norm (x - x_pinv) / norm (x_pinv)
%     % or
%     [x,stats,N,NT] = spqr_pinv (A,B) ;
%     norm_A_times_N = norm (full(spqr_null_mult(N,A,3)))
%     norm_N_transpose_times_A = norm (full(spqr_null_mult(NT,A,0)))
%     % or
%     opts = struct('tol',1.e-5) ;
%     [x,stats] = spqr_pinv (A,B,opts) ;
%     stats
%
% See also spqr_basic, spqr_null, spqr_pinv, spqr_cod.

% Copyright 2012, Leslie Foster and Timothy A Davis.

% Algorithm:  a basic solution is calculated using spqr_basic. Following
%    this an orthogonal basis, stored in N, for the numerical null space
%    of A is calculated using spqr_null. The psuedoinverse solution is
%    then x - N*(N'*x) which is calculated using the routine spqr_null_mult.

%-------------------------------------------------------------------------------
% set tolerance and number of singular values to estimate
%-------------------------------------------------------------------------------

[B,opts,stats,start_tic,ok] = spqr_rank_get_inputs (A, 1, varargin {:}) ;

if (~ok || nargout > 4)
    error ('usage: [x,stats,N,NT] = spqr_pinv (A,B,opts)') ;
end

% get the options
get_details = opts.get_details ;

if (get_details == 1)
    stats.opts_used = opts ;
end

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
% order for the additional stats fields needed when get_details is 1 will be
%     set using spqr_rank_order_fields at the end of the routine

%-------------------------------------------------------------------------------
% find basic solution
%-------------------------------------------------------------------------------

if get_details == 0
    % opts2 used to include a few extra statistics in stats_ssi
    opts2 = opts;
    opts2.get_details = 2;
else
    opts2 = opts;
end
if nargout == 4
    % save basis for null space of A', if there are four output
    %   parameters.  This will require more memory.
    [x,stats_spqr_basic,NT] = spqr_basic (A, B, opts2) ;
else
    % do not save the basis for the null space of A'. Saves memory.
    [x,stats_spqr_basic] = spqr_basic (A, B, opts2) ;
end

stats.rank = stats_spqr_basic.rank ;

% save the stats
if (get_details == 1 || get_details == 2)
    stats.rank_spqr = stats_spqr_basic.rank_spqr ;
end

%-------------------------------------------------------------------------------
% check for early return
%-------------------------------------------------------------------------------

if stats_spqr_basic.flag == 4
    % overflow in spqr_ssi, called by spqr_basic.
    % spqr_basic has already issued a warning regarding overflow
    [stats x N NT] = spqr_failure (4, stats, get_details, start_tic) ;
    return
end

%-------------------------------------------------------------------------------
% calculate basis for the null space of A
%-------------------------------------------------------------------------------

% spqr_null calls spqr_ssi.  Set the block size in spqr_ssi based on the
% difference in the numerical rank calculated by spqr_basic and by spqr.  Note
% that this overwrites the user-provided value of opts.ssi_min_block, if any.
opts.ssi_min_block = max (2, stats_spqr_basic.rank_spqr - stats.rank + 1);

opts.ssi_min_block = ...
    min (opts.ssi_min_block, stats_spqr_basic.stats_ssi.ssi_max_block_used) ;

[N,stats_spqr_null] = spqr_null (A, opts2) ;

if (get_details == 1)
    stats.stats_spqr_basic = stats_spqr_basic ;
    stats.stats_spqr_null = stats_spqr_null ;
end

%-------------------------------------------------------------------------------
% check for early return
%-------------------------------------------------------------------------------

% spqr_basic and spqr_null both return error flags.  Since both must be correct
% to calculate the psuedoinverse solution, choose the largest (worst) error.

stats.flag = max (stats_spqr_basic.flag, stats_spqr_null.flag) ;

if stats_spqr_basic.flag == 0 && stats_spqr_null.flag == 0  && ...
    stats_spqr_basic.rank ~= stats_spqr_null.rank
    % Rank from spqr_basic and spqr_null differ.  This is so rare that we know
    % of no matrices that trigger this condition.  This block of code is thus
    % untested.
    error ('spqr_rank:inconsistent', 'inconsistent rank estimates') ; % untested
    % an alternative, which would cause a return in the code below.
    %   stats.flag = 5 ;
end

if stats.flag >= 4
    % early return: overflow in inverse power method in ssi,
    % or inconsistent rank estimates by spqr_basic and spqr_null.
    [stats x N NT] = spqr_failure (stats.flag, stats, get_details, start_tic) ;
    return
end

%-------------------------------------------------------------------------------
% calculate the psuedoinverse solution
%-------------------------------------------------------------------------------

x = x - spqr_null_mult (N, spqr_null_mult (N,x,0), 1) ;

%-------------------------------------------------------------------------------
% select from two estimates of numerical rank and two sets of bounds
%-------------------------------------------------------------------------------

% Strategy -- the choice of stats.flag above was the
%     max (stats_spqr_basic.flag, stats_spqr_null.flag)
% We will choose the bounds and rank corresponding to this choice of flag.
% When the flags are the same we will use spqr_basic results for
% the bounds, except when the flags are both 1 we will choose the bounds
% corresponding to the maximum of stats_spqr_basic.tol_alt and
% stats_spqr_null.tol_alt (the conservative choice).

if stats_spqr_basic.flag == 1 && stats_spqr_null.flag == 1
    if stats_spqr_basic.tol_alt >= stats_spqr_null.tol_alt
        st = stats_spqr_basic ;
    else
        st = stats_spqr_null ;
    end
elseif stats_spqr_basic.flag >= stats_spqr_null.flag
        st = stats_spqr_basic ;
else
        st = stats_spqr_null ;
end
stats.rank = st.rank ;
if isfield(st,'tol_alt')
    stats.tol_alt = st.tol_alt ;
end
stats.sval_numbers_for_bounds = st.sval_numbers_for_bounds ;
stats.est_sval_upper_bounds  = st.est_sval_upper_bounds ;
stats.est_sval_lower_bounds  = st.est_sval_lower_bounds ;

%-------------------------------------------------------------------------------
% return statistics
%-------------------------------------------------------------------------------

if (get_details == 1 || nargout >= 3 )
   stats.est_norm_A_times_N = stats_spqr_null.est_norm_A_times_N ;
end
if (get_details == 1)
    stats.est_err_bound_norm_A_times_N = ...
        stats_spqr_null.est_err_bound_norm_A_times_N ;
end
if nargout == 4
    stats.est_norm_A_transpose_times_NT = ...
        stats_spqr_basic.est_norm_A_transpose_times_NT ;
    if (get_details == 1)
        stats.est_err_bound_norm_A_transpose_times_NT = ...
            stats_spqr_basic.est_err_bound_norm_A_transpose_times_NT ;
    end
end

if (get_details == 1)
    stats.time_svd = stats_spqr_basic.time_svd + ...
                     stats_spqr_basic.stats_ssi.time_svd + ...
                     stats_spqr_null.time_svd + ...
                     stats_spqr_null.stats_ssi.time_svd + ...
                     stats_spqr_null.stats_ssp_N.time_svd ;
    if nargout == 4
        stats.time_svd = stats.time_svd+stats_spqr_basic.stats_ssp_NT.time_svd ;
    end
    stats.time_basis = stats_spqr_basic.time_basis + stats_spqr_null.time_basis;
end

if stats.tol_alt == -1
    stats = rmfield(stats, 'tol_alt') ;
end

if get_details == 1
    % order the fields of stats in a convenient order (the fields when
    %    get_details is 0 or 2 are already in a good order)
    stats = spqr_rank_order_fields(stats);
end

if (get_details == 1)
    stats.time = toc (start_tic) ;
end


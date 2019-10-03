function [ stats ] =  spqr_rank_assign_stats(...
    call_from, est_sval_upper_bounds, est_sval_lower_bounds, tol, ...
    numerical_rank, nsvals_small, nsvals_large, stats, stats_ssi, ...
    opts, nargout_call, stats_ssp_N, stats_ssp_NT, start_tic)
%SPQR_RANK_ASSIGN_STATS set flag and other statistics.
%
% Called by spqr_basic and spqr_cod as well as, indirectly, by spqr_pinv and
% spqr_null after these methods have determined estimated bounds on the
% singular values of A. Set the flag which indicates the success of
% and other statistics for spqr_basic, spqr_null, spqr_pinv, spqr_cod.
% Not user callable.
%
% Output:
%    stats - statistics for spqr_basic, spqr_null, spqr_pinv and spqr_cod.
%       including the following fields:
%
%       stats.flag which indicates whether the method has determined the
%          correct numerical rank:
%       stats.flag is 0 if it is likely, although not
%          guaranteed, that stats.rank is the correct numerical rank for
%          tolerance tol (i.e. agrees with the numerical rank
%          determined by the singular values of R).
%       stats.flag is 1 if the calculated numerical rank stats.rank may
%           be correct for the tolerance stats.tol but the estimated error
%           bounds are too large to confirm this.  However stats.rank appears
%           to be correct for an alternate tolerance tol_alt.  More
%           generally stats.rank appears to be correct for any tolerance
%           between stats.est_sval_lower_bounds(nsvals_large) and
%           stats.est_sval_upper_bounds(nsvals_large+1).
%       stats.flag is 2 if the calculated numerical rank stats.numerical
%           may be correct but estimated error bounds are too large to confirm
%           this.  The conditions for stats.flag to be 0 or 1 are not
%           satisfied.
%       stats.flag is 3 if is likely that the numerical rank returned,
%          stats.rank, is too large.
%
%       stats.tol_alt is an alternate tolerance that corresponds to the
%          calculated numerical rank when stats.flag is 1.
%
%       other fields of statistics
%
% Input:
%    call_from ==
%        call_from = 1 indicates a call from spqr_basic
%        call_from = 2 indicates a call from spqr_cod
%    est_sval_upper_bounds -- est_sval_upper_bounds(i) is an
%        estimate of an upper bound on singular value number
%        stats.sval_numbers_for_bounds(i) of A.
%    est_sval_lower_bounds -- stats.est_sval_lower_bounds(i) is an
%        estimate of an lower bound on singular value number
%        sval_numbers_for_bounds(i) of A.
%    tol - the tolerance defining the numerical rank.  The true
%        numerical rank is the number of singular values larger than tol.
%    numerical_rank -- the estimated numerical rank of A
%    nsvals_small -- the number of estimated singular values, from
%        the set that have been estimated, that appear to be smaller
%        than or equal to tol.
%    nsvals_large -- the number of estimated singular values, from
%        the set that have been estimated, that appear to be larger
%        than tol.
%    stats -- the statistics returned by spqr_basic, spqr_null, spqr_pinv
%        and spqr_cod.
%    stats_ssi -- the statistics returned by spqr_ssi.
%    opts -- options for the calls to spqr_function.
%    nargout_call -- the value of nargout in the calling function
%    stats_ssp_N -- stats from spqr_ssp applied to A * N
%    stats_ssp_NT -- stats from spqr_ssp applied to A' * NT
%    start_tic -- tic time for start of calling routine

% Copyright 2012, Leslie Foster and Timothy A Davis.

get_details = opts.get_details ;

%-------------------------------------------------------------------------------
% determine flag which indicates accuracy of the estimated numerical rank
%-------------------------------------------------------------------------------

if (  ( nsvals_small == 0 || ...
        est_sval_upper_bounds(nsvals_large+1) <= tol ) ) && ...
       ( nsvals_large > 0 && est_sval_lower_bounds(nsvals_large) > tol )
    % numerical rank is correct, assuming estimated error bounds are correct
    flag = 0;
elseif ( nsvals_small > 0 && nsvals_large > 0 ) && ...
       ( est_sval_lower_bounds(nsvals_large) > ...
             est_sval_upper_bounds(nsvals_large + 1) && ...
             est_sval_upper_bounds(nsvals_large + 1) > tol )
    % in this case, assuming that the estimated error bounds are correct,
    % then the numerical rank is correct with a modified tolerance
    flag = 1;
    stats.tol_alt = est_sval_upper_bounds(nsvals_large + 1);
    % Note: satisfactory values of tol_alt are in the range
    %    est_sval_lower_bounds(nsvals_large) > tol_alt
    %    >= est_sval_upper_bounds(nsvals_large + 1)
elseif stats_ssi.flag == 3
    % in this case ssi failed and it is often the case that then
    % calculated numerical rank is too high
    flag = 3;
else
    % in this case, assuming that the estimated error bounds are correct,
    % the errors in the error bounds are too large to determine the
    % numerical rank
    flag = 2;
end


%-------------------------------------------------------------------------------
% return statistics
%-------------------------------------------------------------------------------

stats.flag = flag ;
stats.rank = numerical_rank ;
stats.est_sval_upper_bounds = est_sval_upper_bounds;
stats.est_sval_lower_bounds = est_sval_lower_bounds;
stats.sval_numbers_for_bounds = ...
    numerical_rank - nsvals_large + 1 : ...
    numerical_rank + nsvals_small ;

if call_from == 2
    if ( get_details == 1 || nargout_call >= 3 )
        stats.est_norm_A_times_N = stats_ssp_N.est_svals(1);
    end
    if (get_details == 1)
        stats.est_err_bound_norm_A_times_N = stats_ssp_N.est_error_bounds(1);
    end
end


if ( ( call_from == 1 && nargout_call == 3 ) || ...
        ( call_from == 2 && nargout_call == 4 ) )
    % include estimated norm of A transpose time NT from call to ssp
    stats.est_norm_A_transpose_times_NT = stats_ssp_NT.est_svals(1);
    if get_details == 1
        stats.est_err_bound_norm_A_transpose_times_NT = ...
           stats_ssp_NT.est_error_bounds(1);
    end
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


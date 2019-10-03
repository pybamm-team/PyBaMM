function [U, S, V, stats] = spqr_ssi (R, varargin)
%SPQR_SSI block power method or subspace iteration applied to inv(R)
% to estimate rank, smallest singular values, and left/right singular vectors.
%
% [U,S,V,stats] = spqr_ssi (R,opts);
%
% Uses the block power method or subspace iteration applied to the inverse of
% the triangular matrix R (implicitly) to estimate the numerical rank, smallest
% singular values, and left and right singular vectors R.  The algorithm will
% be efficient in the case the nullity of R is relatively low and inefficient
% otherwise.
%
% Input:
%
%   R -- an n by n nonsingular triangular matrix
%   opts (optional) -- see 'help spqr_rank_opts' for details.
%
% Output:
%
%   U  -- n by k matrix containing estimates of the left singular vectors of R
%       corresponding to the singular values in S. When stats.flag is 0 (and
%       when the optional parameter opts.nsvals_large has its default value 1),
%       r = n - k + 1 is the estimated rank of R. Also NT=U(:,2:k) is an
%       orthonormal basis for the numerical null space of R'.
%   S -- A k by k diagonal matrix whose diagonal entries are the estimated
%       smallest k singular values of R, with k as described above.  For i=1:k,
%       S(i,i) is an an estimate of singular value (r + i - 1) of R.
%       Note that, unless stats.flag = 3 (see below), S(1,1)) > tol and for
%       i =2:k, S(i,i) <= tol.
%   V  -- n by k matrix containing estimates of the right singular vectors of R
%       corresponding to the singular values in S.   When stats.flag is 0,
%       N=V(:,2:k) is an orthonormal basis for the numerical null space of R.
%   stats -- statistics, type 'help spqr_rank_stats' for details.
%
% Note that U' * R = S * V' (in exact arithmetic) and that R * V is
% approximately equal to U * S.
%
% output (for one or two output arguments): [s,stats] = spqr_ssi (...)
%    s -- diag(S), with S as above.
%
% Example:
%    R = sparse (gallery ('kahan',100)) ;
%    [U,S,V] = spqr_ssi (R) ;
%    norm_of_residual = norm( U' * R - S * V' )   % should be near zero
%    % or
%    [U,S,V,stats] = spqr_ssi (R) ;
%    N = V(:,2:end);   % orthonormal basis for numerical null space of R
%    NT = U(:,2:end);  % orthonormal basis for numerical null space of R'
%    norm_R_times_N = norm(R*N)
%    norm_R_transpose_times_NT = norm(R'*NT)
%    % or
%    opts = struct('tol',1.e-5,'nsvals_large',3) ;
%    [s,stats] = spqr_ssi (R,opts);
%    stats  % information about several singular values
%
% See also spqr_basic, spqr_null, spqr_pinv, spqr_cod.

% Copyright 2012, Leslie Foster and Timothy A Davis.

% Outline of algorithm:
%    let b = initial block size
%    U = n by b random matrix with orthonormal columns
%    repeat
%       V1 = R \ U
%       [V,D1,X1]= svd(V1,0) = the compact svd of V1
%       U1 = R' \ V
%       [U,D2,X2] = svd(U1,0) = the compact svd of U1
%       sdot = diag(D2)
%       s = 1 ./ sdot = estimates of singular values of R
%       if s(end) <= tol
%          increase the number of columns of U and repeat loop
%       else
%          repeat loop until stopping criteria (see code) is met
%       end
%    end
%    k = smallest i with  sdot(i) > tol
%    V = V*X2 (see code)
%    V = first k columns of V in reverse order
%    U = first k columns of U in reverse order
%    s = 1 ./ (first k entries in sdot in reverse order)
%    S = diag(s)

%    Note for i = 1, 2, ..., k, S(i,i) is an upper bound on singular
%    value (r + i - 1) of R in exact arithmetic but not necessarily
%    in finite precision arithmetic.

% start block power method to estimate the smallest singular
%     values and the corresponding right singular vectors (in the
%     r by nblock matrix V) and left singular vectors (in the r by
%     nblock matrix U) of the n by n matrix R

% disable nearly-singular matrix warnings, and save the previous state
warning_state = warning ('off', 'MATLAB:nearlySingularMatrix') ;

%-------------------------------------------------------------------------------
% get input options
%-------------------------------------------------------------------------------

[ignore,opts,stats,start_tic,ok] = spqr_rank_get_inputs (R,0,varargin{:}) ; %#ok
clear ignore

if (~ok || nargout > 4)
    error ('usage: [U,S,V,stats] = spqr_ssi (R,opts)') ;
end

[m,n] = size (R) ;
if m ~= n
    error('R must be square')
end

% see spqr_rank_get_inputs.m for defaults:
tol = opts.ssi_tol ;                                        % default opts.tol
min_block = opts.ssi_min_block ;                            % default 3
max_block = opts.ssi_max_block ;                            % default 10
min_iters = opts.ssi_min_iters ;                            % default 3
max_iters = opts.ssi_max_iters ;                            % default 100
nblock_increment = opts.ssi_nblock_increment ;              % default 5
convergence_factor = opts.ssi_convergence_factor ;          % default 0.1
nsvals_large = opts.nsvals_large ;                          % default 1
get_details = opts.get_details ;                            % default false
repeatable = opts.repeatable ;                              % default false

private_stream = spqr_repeatable (repeatable) ;

%-------------------------------------------------------------------------------
% adjust the options to reasonable ranges
%-------------------------------------------------------------------------------

% cannot compute more than n large singular values, where R is n-by-n
nsvals_large = min (nsvals_large, n) ;

% make the block size large enough so that there is a possiblity of
% calculating all nsvals_large singular values
max_block = max (max_block, nsvals_large) ;

% max_block cannot be larger than n
max_block = min (max_block, n) ;

% min_block cannot be larger than n
min_block = min (min_block, n) ;

% start with nblock = min_block;
nblock = min_block ;

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

% stats.flag and stats.rank initialized in spqr_rank_get_inputs
% set the order of the remaining stats fields
stats.tol = tol ;
stats.tol_alt = -1 ;   % removed later if remains -1
if get_details == 1 && isfield(stats,'normest_A')
    normest_A = stats.normest_A ;
    stats = rmfield( stats,'normest_A' ) ;  % to place in proper order
    stats.normest_R = normest_A ;
end
stats.est_svals_of_R = 0 ;
stats.est_error_bounds = -1 ;
stats.sval_numbers_for_bounds = -1 ;
if get_details == 2
    stats.ssi_max_block_used = -1 ;
    stats.ssi_min_block_used = -1 ;
end
if (get_details == 1)
    stats.norm_R_times_N = -1 ;
    stats.norm_R_transpose_times_NT = -1 ;
    stats.iters = -1 ;
    stats.nsvals_large_found = 0 ;
    stats.final_blocksize = -1 ;
    stats.ssi_max_block_used = -1 ;
    stats.ssi_min_block_used = -1 ;
    stats.opts_used = opts ;
    stats.time = 0 ;
    time_initialize = stats.time_initialize ;
    stats = rmfield(stats,'time_initialize') ; % to place in proper order
    stats.time_initialize = time_initialize ;
    stats.time_iters = 0 ;
    stats.time_est_error_bounds = 0 ;
    stats.time_svd = 0 ;
end


stats.rank = [ ] ;
stats.tol = tol ;
if (get_details == 1 || get_details == 2)
    stats.ssi_max_block_used = max_block ;
    stats.ssi_min_block_used = nblock ;
end

if (get_details == 1)
    start_iters_tic = tic ;
end

if (~isempty (private_stream))
    U = randn (private_stream, n, nblock) ;
else
    U = randn (n, nblock) ;
end
[U,ignore] = qr (U,0) ;                                                     %#ok
clear ignore
% est_error_bound_calculated = 0 ;      % set to 1 later if bound calculated
flag_overflow = 0 ;                     % set to 1 later if overflow occurs

%-------------------------------------------------------------------------------
% iterations
%-------------------------------------------------------------------------------

for iters = 1:max_iters

    U0= U;
    V1 = R \ U ;
    if min(min(isfinite(V1))) == 0
        flag_overflow = 1;
        break   % *************>>>>> early exit from for loop
    end

    if (get_details == 1), time_svd = tic ; end

    [V,D1,X1] = svd (V1,0) ;

    if (get_details == 1)
        stats.time_svd = stats.time_svd + toc( time_svd ) ;
    end

    % d1 = diag(D1)';
    U1 = ( V' / R )';  % uses less memory than U1 = R' \ V;
    % Note: with the inverse power method overflow is a potential concern
    %     in extreme cases (SJid = 137 or UFid = 1239 is close)
    if min (min (isfinite (U1))) == 0
        % *************>>>>> early exit from for loop
        % We know of no matrix that triggers this condition, so the next
        % two lines are untested.
        flag_overflow = 1;      % untested
        break                   % untested
    end

    if (get_details == 1), time_svd = tic ; end

    [U,D2,X2] = svd (U1,0) ;

    if (get_details == 1)
        stats.time_svd = stats.time_svd + toc(time_svd) ;
    end

    d2 = diag (D2)' ;
    k = find(d2*tol<1);

    if isempty(k) && (nblock == n)
        % success since the numerical rank of R is zero
        break   % *************>>>>> early exit from for loop
    end

    if ~isempty(k)
        k = k(1);  % k equals the calculated nullity + 1, k corresponds to
                   % the first singular value of R that is greater than tol
        % reduce nsvals_large if necessary due to max_block limit:
        nsvals_large_old = nsvals_large;
        nsvals_large = min(nsvals_large,max_block - k + 1);

        if nblock >= k + nsvals_large - 1
            % estimate error bound for singular value n - k + 1 of R:
            est_error_bound =  norm( ...
                U0*(X1*(D1 \ X2(:,k))) - U(:,k)/D2(k,k) ) / sqrt(2);
            % est_error_bound_calculated = 1;
            k2 = k + nsvals_large - 1;
            % When nsvals_large is 1, k and k2 are the same.  When
            % nsvals_large > 1, k2 corresponds to the largest singular
            % value of R that is returned.
            if k2 ~= k
                % estimate error bound for singular value n - k2 + 1 of R:
                est_error_bound2 =  norm( ...
                    U0*(X1*(D1 \ X2(:,k2))) - U(:,k2)/D2(k2,k2) ) / sqrt(2);
            else
                est_error_bound2 = est_error_bound;
            end
        end

        % Note that
        %     [ 0     R ] [  U   ]  =    [ U0 * X1 * ( D1 \ X2 ) ]
        %     [ R'    0 ] [ V*X2 ]       [     V * ( X2 / D2 )   ]
        % It follows that, by Demmel's Applied Numerical Linear Algebra,
        % Theorem 5.5, some eigenvalue of
        %     B  =  [ 0    R ]
        %           [ R'   0 ]
        % will be within a distance of est_error_bound of 1 / d2(k), where s(1)
        % = 1 / d2(k) is our estimate of singular value n - k + 1 of R.
        % Typically the eigenvalue of B is singular value number n - k + 1 of
        % R, although this is not guaranteed.  est_error_bound is our estimate
        % of a bound on the error in using s(1) to approximate singular value
        % number n - k + 1 of R.

        if nblock >= k + nsvals_large - 1 && ...
            est_error_bound <= convergence_factor*abs( 1/d2(k)-tol ) && ...
            est_error_bound2 <= convergence_factor*abs( 1/d2(k2) ) && ...
            iters >= min_iters
            % Motivation for the tests:
            % The first test in the if statement is an attempt to insure
            % that nsvals_large singular values of R larger than tol are
            % calculated.
            %
            % Goal of the power method is to increase nblock until we find
            % sigma = (singular value n - k + 1 of R) is > tol.  If
            % this is true it is guaranteed that n - k + 1 is the
            % correct numerical rank of R.  If we let s(1) = 1 / d2(k)
            % then s(1) > tol.  However s(1) is only an estimate of sigma.
            % However, in most cases
            %     | s(1) - sigma | <= est_error_bound                   (1)
            % and to be conservative assume only that
            %  |s(1) - sigma|<= (1/convergence_factor)*est_error_bound  (2)
            % where convergence_factor<=1. By the second test in the if
            % statement
            %  est_error_bound <= convergence_factor * | s(1) - tol |   (3)
            % Equations (2) and (3) imply that
            %      | s(1) - sigma | <= | s(1) - tol |
            % This result and s(1) > tol imply that sigma > tol, as
            % desired.  Thus the second test in the if statement attempts
            % to carry out enough iterations to insure that the calculated
            % numerical rank is correct.
            %
            % The third test in the if statement checks on the accuracy of
            % the estimate for singular values n - k2 + 1.  Let sigma2 be
            % singular value n - k2 + 1 of R.  Usually it is true
            % that
            %     | s( k2 ) - sigma2 | <= est_error_bound2.             (4)
            % Assuming (4) and the third test in the if statement it
            % follows that the estimated relative
            % error in s(k2),  as measured by est_error_bound2 / s( k2) ,
            % is less that or equal to convergence_factor.  Therefore
            % the third test in the if statement attempts to insure that
            % the largest singular value returned by ssi has a relative
            % error <= convergence_factor.
            %
            % SUCCESS!!!, found singular value or R larger than tol
            break  % *************>>>>> early exit from for loop
        end
        nsvals_large = nsvals_large_old;  % restore original value
    end

    if nblock == max_block && iters >= min_iters && isempty(k)
        % reached max_block block size without encountering any
        %     singular values of R larger than tolerance
        break    % *************>>>>> early exit from for loop
    end

    if (1 <= iters && iters < max_iters) && ( isempty(k)  || ...
            ( ~isempty(k) && nblock < k(1) + nsvals_large - 1) )
        % increase block size
        nblock_prev = nblock;
        if isempty(k)
            nblock = min(nblock + nblock_increment, max_block);
        else
            nblock = min( k(1) + nsvals_large - 1, max_block );
        end
        if (nblock > nblock_prev)
            if (~isempty (private_stream))
                Y = randn (private_stream, n, nblock-nblock_prev) ;
            else
                Y = randn (n, nblock-nblock_prev) ;
            end
            Y = Y - U*(U'*Y);
            [Y,ignore]=qr(Y,0) ;                                            %#ok
            clear ignore
            U = [U, Y];      %#ok
        end
    end
end

if (get_details == 1)
    stats.final_blocksize = nblock ;    % final block size
    stats.iters = iters ;               % number of iterations taken in ssi
    stats.time_iters = toc (start_iters_tic) ;
end

%-------------------------------------------------------------------------------
% check for early return
%-------------------------------------------------------------------------------

if flag_overflow == 1
    warning ('spqr_rank:overflow', 'overflow in inverse power method') ;
    warning (warning_state) ;
    [stats U S V] = spqr_failure (4, stats, get_details, start_tic) ;
    if (nargout == 2)
        S = stats ;
    end
    return
end

%-------------------------------------------------------------------------------
% determine estimated singular values of R
%-------------------------------------------------------------------------------

est_error_bounds = [ ] ;

if ~isempty(k)
    % Note: in this case nullity = k - 1 and rank = n - k + 1
    s = 1.0 ./ d2(min(nblock,k+nsvals_large-1):-1:1);
    est_error_bounds (nsvals_large) = est_error_bound;
    numerical_rank = n - k + 1;
else
    k = nblock;
    if nblock == n
        numerical_rank = 0;
        % success since numerical rank is 0
    else
        % in this case rank not successfully determined
        % Note: In this case k is a lower bound on the nullity and
        %       n - k is an upper bound on the rank
        numerical_rank = n - k;  %upper bound on numerical rank
        nsvals_large = 0 ; % calculated no singular values > tol
    end
    s = 1.0 ./ d2(nblock:-1:1);
end

stats.rank = numerical_rank ;

nkeep = length (s) ;   % number of cols kept in U and V
S = diag (s) ;

if (get_details == 1)
    stats.nsvals_large_found = nsvals_large ;
end


%-------------------------------------------------------------------------------
% adjust V so that R'*U = V*S (in exact arithmetic)
%-------------------------------------------------------------------------------

V = V*X2;
% reverse order of U and V and keep only nkeep singular vectors
V = V(:,nkeep:-1:1);
U = U(:,nkeep:-1:1);

if (get_details == 1)
    t = tic ;
end

if nsvals_large > 0
    % this recalculates est_error_bounds(nsvals_large)
    U0 = R * V(:,1:nsvals_large) - ...
        U(:,1:nsvals_large) * S(1:nsvals_large,1:nsvals_large) ;
    U0 = [ U0;
           R' * U(:,1:nsvals_large) - V(:,1:nsvals_large)* ...
             S(1:nsvals_large,1:nsvals_large)];
    est_error_bounds (1:nsvals_large) = sqrt(sum(U0 .* conj(U0) )) / sqrt(2) ;
end

% this code calculates estimated error bounds for singular values
%    nsvals_large+1 to nkeep
ibegin = nsvals_large+1;
U0 = R * V(:,ibegin:nkeep) - U(:,ibegin:nkeep)* ...
         S(ibegin:nkeep,ibegin:nkeep);
U0 = [ U0;
       R' * U(:,ibegin:nkeep) - V(:,ibegin:nkeep)* ...
         S(ibegin:nkeep,ibegin:nkeep)];
est_error_bounds (ibegin:nkeep) = sqrt(sum(U0 .* conj(U0) )) /sqrt(2);
% Note that
%    [ 0      R ]  [ U ]   -    [ U ] * S = [ R * V - U * S ]
%    [ R'     0 ]  [ V ]   -    [ V ]       [      0        ].
% It follows, by Demmel's Applied Numerical Linear Algebra,
% Theorem 5.5, for i = 2, . . .,  k, some eigenvalue of
%     B  =  [ 0    R ]
%           [ R'   0 ]
% will be within a distance of the norm of the ith column of
% [ R * V - U * S ; R'*U - V*S] / sqrt(2) from S(i,i). Typically this
% eigenvalue will be singular value number n - k + i of R,
% although it is not guaranteed that the singular value number
% is correct.  est_error_bounds(i) is the norm of the ith column
% of [ R * V - U * S; R' * U - V * S ] / sqrt(2).


if (get_details == 1)
    % Note that stats.time_est_error_bounds includes the time for the error
    %   bound calculations done outside of the subspace iterations loop
    stats.time_est_error_bounds = toc (t) ;
end

stats.sval_numbers_for_bounds = ...
    numerical_rank - nsvals_large + 1 : ...
    numerical_rank - nsvals_large + length (est_error_bounds) ;

%-------------------------------------------------------------------------------
% compute norm R*N and R'*NT
%-------------------------------------------------------------------------------

% compute norm R*N where N = V(:,nsvals_large+1:end) is the approximate
%         null space of R and
%         norm R'*NT where NT = U(:,nsvals_large+1:end) is the approximate
%         null space of R'

if (get_details == 1), t = tic ; end

% svals_R_times_N = svd(R*V(:,nsvals_large+1:end))';
norm_R_times_N = norm (R * V(:,nsvals_large+1:end)) ;
% svals_R_transpose_times_NT = svd(R'*U(:,nsvals_large+1:end))';
norm_R_transpose_times_NT = norm (R' * U(:,nsvals_large+1:end)) ;

if (get_details == 1)
    stats.norm_R_times_N = norm_R_times_N;
    stats.norm_R_transpose_times_NT = norm_R_transpose_times_NT;
    stats.time_svd = stats.time_svd + toc (t) ;
end

% Note: norm_R_times_N is an upper bound on sing. val. rank1+1 of R
% and norm_R_transpose_times_NT is also an upper bound on sing. val.
%      rank1+1 of R
max_norm_RN_RTNT = max (norm_R_times_N,norm_R_transpose_times_NT);
% choose max here to insure that both null spaces are good

%-------------------------------------------------------------------------------
% determine flag indicating the accuracy of the rank calculation
%-------------------------------------------------------------------------------

if numerical_rank == 0
    % numerical rank is 0 in this case
    stats.flag = 0;
elseif (numerical_rank == n  || max_norm_RN_RTNT <= tol) && ...
       (nsvals_large > 0 && ...
       s(nsvals_large) - est_error_bounds(nsvals_large) > tol )
    % in this case, assuming est_error_bounds(nsvals_large) is a true
    % error bound, then the numerical rank is correct.  Also
    % N = V(:,nsvals_large+1:end) and NT = U(:,nsvals_large+1:end)
    % are bases for the numerical null space or R and R', respectively
    stats.flag = 0;
elseif ( nsvals_large > 0 && numerical_rank == n ) || ...
        ( nsvals_large > 0 && ...
        s(nsvals_large) - est_error_bounds(nsvals_large) > ...
        max_norm_RN_RTNT )
    % in this case, assuming est_error_bounds(nsvals_large) is a true
    % error bound, then the numerical rank is correct with a modified
    % tolerance.  This is a rare case.
    stats.flag = 1;
    tol_alt = ( s(nsvals_large) - est_error_bounds(nsvals_large) );
    tol_alt = tol_alt - eps(tol_alt);  % so that tol_alt satisfies the >
                                       % portion of the inequality below
    % tol_alt = max_norm_RN_RTNT;
    % Note: satisfactory values of tol_alt are in the range
    %    s(nsvals_large) - est_error_bounds(nsvals_large) > tol_alt
    %    >= max_norm_RN_RTNT
    stats.tol_alt = tol_alt;
elseif  nsvals_large > 0 && s(nsvals_large) > tol && ...
        max_norm_RN_RTNT <= tol && ...
        s(nsvals_large) - est_error_bounds(nsvals_large) ...
        <= max_norm_RN_RTNT
    % in this case, assuming est_error_bounds(nsvals_large) is a true
    % error bound, the error bound is too large to determine the
    % numerical rank.  This case is very rare.
    stats.flag = 2;
else
    % in this case all the calculated singular values are
    % smaller than tol or either N is not a basis for the numerical
    % null space of R with tolerance tol or NT is not such a basis for R'.
    % stats.rank is an upper bound on the numerical rank.
    stats.flag = 3;
end

%-------------------------------------------------------------------------------
% return results
%-------------------------------------------------------------------------------

% restore the warnings
warning (warning_state) ;

stats.est_svals_of_R = diag (S)' ;
stats.est_error_bounds = est_error_bounds;

if (get_details == 1)
    stats.time = toc (start_tic) ;
end

if stats.tol_alt == -1
    stats = rmfield(stats, 'tol_alt') ;
end

if (get_details == 1)
    stats.time = toc (start_tic) ;
end

if (nargout <= 2)
    U = diag (S) ;
    S = stats ;
end


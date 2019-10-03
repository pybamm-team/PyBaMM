function [U, S, V, stats] = spqr_ssp (A, varargin)
%SPQR_SSP block power method or subspace iteration applied to A or A*N
%
% [U,S,V,stats] = spqr_ssp (A, N, k, opts) ;
%
% Uses the block power method or subspace iteration to estimate the largest k
% singular values, and left and right singular vectors an m-by-n matrix A or,
% if the n-by-p matrix N is included in the input, of A*N.
%
% Alternate input syntaxes (opts are set to defaults if not present):
%
%       spqr_ssp (A)         k = 1, N is [ ]
%       spqr_ssp (A,k)       k is a scalar
%       spqr_ssp (A,N)       N is a matrix or struct
%       spqr_ssp (A,N,k)     k is a scalar
%       spqr_ssp (A,N,opts)  opts is a struct
%
% Input:
%   A -- an m-by-n matrix
%   N (optional) -- an n-by-p matrix representing the null space of A, stored
%       either explicitly as a matrix or implicitly as a structure, and
%       returned by spqr_basic, spqr_null, spqr_cod, or spqr_pinv.
%       N is ignored if [ ].
%   k (optional) -- the number of singular values to calculate. Also the block
%       size in the block power method.  Also can be specified as opts.k.
%       Default: 1.
%   opts (optional) -- type 'help spqr_rank_opts' for details
%
% Output:
%   U  -- m-by-k matrix containing estimates of the left singular vectors of A
%       (or A*N if N is included in the input) corresponding to the singular
%       values in S.
%   S -- k-by-k diagonal matrix whose diagonal entries are the estimated
%       singular values of A (or A*N). Also for i = 1:nsval, S(i,i) is a lower
%       bound on singular value i of A (or A*N).
%   V -- n-by-k matrix containing estimates of the right singular vectors of A
%       corresponding to the singular values in S.
%   stats -- type 'help spqr_rank_stats'
%
% Note that A*V = U*S (or A*N*V = U*S) and that U'*A (or U'*A*N)
% is approximately equal to S*V'.
%
% Output (for one or two output arguments): [s,stats]= spqr_ssp (...)
%   s -- diag(S), with S as above
%
% Example:
%    A = sparse (gallery ('kahan',100)) ;
%    [U,S,V] = spqr_ssp (A) ;   % singular triple for largest singular value
%    % or
%    [U,S,V,stats] = spqr_ssp (A,4) ;  % same for largest 4 singular values
%    % or
%    [s,stats] = spqr_ssp (A,4) ;
%    stats     % stats has information for largest 4 singular values
%    % or
%    N = spqr_null ( A ) ;   % N has a basis for the null space of A
%    s = spqr_ssp (A,N) ;
%    s     % s has estimate of largest singular value (or norm) of A*N
%
% See also spqr_basic, spqr_null, spqr_pinv, spqr_cod.

% Copyright 2012, Leslie Foster and Timothy A Davis.

% Outline of algorithm:
%    Let B = A or B = A*N, if the optional input N is included
%    where A is m-by-n and N is n-by-p. Note that the code is written so
%    that B is not explicitly formed.  Let n = p if N is not input.
%    Let k = block size = min(k,m,p)
%    U = m-by-k random matrix with orthonormal columns
%    repeat
%       V1 = B' * U
%       [V,D1,X1]= svd(V1,0) = the compact svd of V1
%       U1 = B * V
%       [U,D2,X2] = svd(U1,0) = the compact svd of U1
%       s = diag(D2)
%       kth_est_error_bound = norm(B'*U(:,k) - V*(X2(:,k)*D2(k,k))) / sqrt(2);
%       if kth_est_error_bound <= convergence_factor*s(k) && iters >= min_iters
%            flag = 0
%            exit loop
%       elseif number of iteration > max number of iterations
%            flag = 1
%            exit loop
%       else
%            continue loop
%       end
%    end
%    V = V*X2; % adjust V so that A*V = U*S
%    calculate estimated error bounds for singular values 1:k-1
%       as discussed in the code
%    S = diag(s);

% start block power method to estimate the largest singular values and the
% corresponding right singular vectors (in the n-by-k matrix V) and left
% singular vectors (in the m-by-k matrix U) of the m-by-n matrix A

%-------------------------------------------------------------------------------
% get input options
%-------------------------------------------------------------------------------
[N,opts,stats,start_tic,ok] = spqr_rank_get_inputs (A, 2, varargin {:}) ;

if (~ok || nargout > 4)
    error ('usage: [U,S,V,stats] = spqr_ssp (A,N,k,opts)') ;
end

k = max (opts.k, 0) ;
min_iters = opts.ssp_min_iters ;
max_iters = opts.ssp_max_iters ;
convergence_factor = opts.ssp_convergence_factor ;
get_details = opts.get_details ;
repeatable = opts.repeatable ;

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

% the order of all the fields of stats are set in spqr_rank_get_inputs

[m,n] = size (A) ;

if (isempty (N))
    % do not use N
    use_N = 0 ;
    p = n ;
elseif (isstruct (N))
    % N is a struct
    use_N = 2 ;
    p = size (N.X, 2) ;
else
    % N is a matrix
    use_N = 1 ;
    p = size (N, 2) ;
end

k = min ([k m p]) ;     % number of singular values to compute

stats.flag = 1 ;
stats.est_error_bounds = zeros (1, max (k,1)) ;
stats.sval_numbers_for_bounds = 1:k ;

if (k <= 0)
    % quick return.  This is not an error condition.
    stats.flag = 0 ;
    stats.est_svals = 0 ;
    stats.est_error_bounds = 0 ;
    stats.sval_numbers_for_bounds = 0 ;
    U = zeros (m, 0) ;
    S = 0 ;
    V = zeros (n, 0) ;
    if (nargout == 1)
        % usage: s = spqr_ssp ( ... )
        S = 0 ;
    elseif (nargout == 2)
        % usage: [s,stats] = spqr_ssp ( ... )
        U = 0 ;
        S = stats ;
    end
    return
end

private_stream = spqr_repeatable (repeatable) ;

if (~isempty (private_stream))
    U = randn (private_stream, m, k) ;
else
    U = randn (m, k) ;
end

[U, ignore] = qr (U, 0) ;                                                   %#ok
clear ignore

%-------------------------------------------------------------------------------
% block power iterations
%-------------------------------------------------------------------------------

if (get_details == 1)
    t = tic ;
end

for iters = 1:max_iters

    if use_N == 0
        V1 = A' * U ;
    elseif use_N == 1
        V1 = N' * (A' * U) ;
    else
        V1 = spqr_null_mult (N, (A' * U), 0) ;
    end

    if issparse (V1)
        V1 = full (V1) ;        % needed if U is 1-by-1
    end

    if (get_details == 1), time_svd = tic ; end
    % [V,D1,X1] = svd (V1, 0) ;
    [V,ignore1,ignore2] = svd (V1, 0) ;                                     %#ok
    clear ignore1 ignore2
    if (get_details == 1)
        stats.time_svd = stats.time_svd + toc(time_svd) ;
    end

    % d1 = diag(D1)';

    if use_N == 0
       U1 = A * V;
    elseif use_N == 1
       U1 = A * (N * V);
    else
       U1 = A * spqr_null_mult (N,V,1) ;
    end

    if issparse(U1)
        U1 = full (U1) ;        % needed if V is 1-by-1
    end

    if (get_details == 1), time_svd = tic ; end
    [U,D2,X2] = svd (U1, 0) ;
    if (get_details == 1)
        stats.time_svd = stats.time_svd + toc(time_svd) ;
    end

    d2 = diag (D2)' ;

    % estimate error bound on kth singular value
    if use_N == 0
        V1 = A' * U(:,k) ;
    elseif use_N == 1
        V1 = N' * (A' * U(:,k)) ;
    else
        V1 = spqr_null_mult(N,(A' * U(:,k)),0);
    end
    kth_est_error_bound = norm( V1 - V *( X2(:,k)*D2(k,k) ) ) / sqrt(2);

    % Note that
    %     [ 0     B ] [  U   ]  =    [       U * D2          ]
    %     [ B'    0 ] [ V*X2 ]       [     V * ( X2 * D2 )   ]
    % where B = A or B = A*N (if N is included in the input).  It follows that,
    % by Demmel's Applied Numerical Linear Algebra, Theorem 5.5, some
    % eigenvalue of
    %     C  =  [ 0    B ]
    %           [ B'   0 ]
    % will be within a distance of kth_est_error_bound of D2(k,k).  Typically
    % the eigenvalue of C is the kth singular value of B, although this is not
    % guaranteed.  kth_est_error_bound is our estimate of a bound on the error
    % in using D2(k,k) to approximate kth singular value of B.

    if kth_est_error_bound <= convergence_factor* d2(k) && iters >= min_iters
        % The test indicates that the estimated relative error in
        % the estimate, D2(k,k) for the kth singular value is smaller than
        % convergence_factor, which is the goal.
        stats.flag = 0 ;
        break
    end
end

if (get_details == 1)
    stats.time_iters = toc (t) ;
end

s = d2 (1:k);
stats.est_error_bounds (k) = kth_est_error_bound ;
S = diag (s) ;
stats.est_svals = diag (S)' ;


% adjust V so that A*V = U*S
V = V*X2 ;

%-------------------------------------------------------------------------------
% estimate error bounds for the 1st through (k-1)st singular values
%-------------------------------------------------------------------------------

if (get_details == 1)
    t = tic ;
end

if k > 1

    if use_N == 0
        V1 = A' * U(:,1:k-1) ;
    elseif use_N == 1
        V1 = N'* (A' * U(:,1:k-1)) ;
    else
        V1 = spqr_null_mult(N,(A' * U(:,1:k-1)),0);
    end

    U0 = V1 - V(:,1:k-1)*S(1:k-1,1:k-1);
    stats.est_error_bounds (1:k-1) = sqrt (sum (U0 .* conj(U0))) / sqrt (2) ;

    % Note (with V adjusted as above) that
    %    [ 0      B ]  [ U ]   -    [ U ] * S = [       0       ]
    %    [ B'     0 ]  [ V ]   -    [ V ]       [   A'*U - V*S  ]
    % where B = A or B = A*N (if N is included in the input).  It follows, by
    % Demmel's Applied Numerical Linear Algebra, Theorem 5.5, for i = 1:k, some
    % eigenvalue of
    %     C  =  [ 0    B ]
    %           [ B'   0 ]
    % will be within a distance of the norm of the ith column of [ B' * U - V *
    % S ] / sqrt(2) from S(i,i). Typically this eigenvalue will be the ith
    % singular value number of B, although it is not guaranteed that the
    % singular value number is correct.  stats.est_error_bounds(i) is the norm
    % of the ith column of [ B' * U - V * S ] / sqrt(2).

end

%-------------------------------------------------------------------------------
% return results
%-------------------------------------------------------------------------------

if (get_details == 1)
    stats.time_est_error_bounds = toc (t) ;
    stats.iters = iters ;
end

if (get_details == 1)
    stats.time = toc (start_tic) ;
end

if (nargout <= 2)
    % usage: [s,stats] = spqr_ssp ( ... )
    U = diag (S) ;
    S = stats ;
end


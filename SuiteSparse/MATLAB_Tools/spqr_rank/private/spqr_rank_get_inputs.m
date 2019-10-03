function [B,opts,stats,start_tic,ok] = spqr_rank_get_inputs (A,method,varargin)
%SPQR_RANK_GET_INPUTS get the inputs and set the default options.
% Not user-callable.  Handles the following input syntaxes for functions in
% the spqr_rank toolbox:
%
%   method = 0: spqr_null, spqr_ssi
%       [ ] = f (A)             % B is [ ], opts empty
%       [ ] = f (A, [ ])        % B is [ ], opts empty
%       [ ] = f (A, opts)       % B is [ ]
%
%   method = 1: spqr_basic, spqr_cod, spqr_pinv
%       [ ] = f (A)             % B is [ ], opts empty
%       [ ] = f (A, [ ])        % B is [ ], opts empty
%       [ ] = f (A, B)          %           opts empty
%       [ ] = f (A, B, [ ])     %           opts empty
%       [ ] = f (A, B, opts)    %
%       [ ] = f (A, B, tol)     %           opts.tol
%
%   method = 2: spqr_ssp
%       [ ] = f (A)             % N is [ ], k is 1, opts empty
%       [ ] = f (A, [ ])        % N is [ ], k is 1, opts empty
%       [ ] = f (A, k)          % N is [ ],         opts empty
%       [ ] = f (A, N)          %           k is 1, opts empty
%       [ ] = f (A, N, [ ])     %           k is 1, opts empty
%       [ ] = f (A, N, opts)    %           k is 1
%       [ ] = f (A, N, k)       %                   opts empty
%       [ ] = f (A, N, k, opts) %
%
%   method = 3: spqr_rank_opts, return opts for spqr_basic
%       [ ] = f (A, opts)       % B is [ ]
%
% The return values of this function are, in order:
%
%       B           the right-hand side, or N for spqr_ssp.  B = [ ] if not
%                   present.
%       opts        the opts struct (see below)
%       stats       a struct (stats.flag, and optionally stats.time_normest)
%       start_tic   empty, or the beginning tic if opts.get_vals > 1
%       ok          true if successful, false if inputs are invalid
%
% The opts struct is populated with default values, if not present on input.
%
% Note that the defaults are different for spqr_ssp than for the other
% functions.

% Copyright 2012, Leslie Foster and Timothy A. Davis

B = [ ] ;
opts = struct ;
stats = struct ;
start_tic = [ ] ;
ok = 1 ;

args = nargin - 1 ;     % # of input arguments to spqr_function

if (args == 1)

    % usage: [ ] = spqr_function (A)

elseif (args == 2)

    lastarg = varargin {1} ;
    is_null_basis_struct = ...
        isstruct(lastarg) && isfield(lastarg,'Q') && isfield(lastarg,'X') ;
    if ( isstruct(lastarg) && ~is_null_basis_struct )
        % usage: [ ] = spqr_function (A, opts)
        opts = lastarg;
        % since lastarg is a structure but not an implicit null space basis
    elseif (isempty (lastarg))
        % usage: [ ] = spqr_function (A, [ ])
    elseif (isreal (lastarg) && length (lastarg) == 1)
        % usage: [ ] = spqr_function (A, k)
        opts.k = round (lastarg) ;
    else
        % usage: [ ] = spqr_function (A, B)
        B = lastarg ;
    end

elseif (args == 3 && (method == 1 || method == 2))

    B = varargin {1} ;
    lastarg = varargin {2} ;
    if (isstruct (lastarg))
        % usage: [ ] = spqr_function (A, B, opts)
        opts = lastarg ;
    elseif (isempty (lastarg))
        % usage: [ ] = spqr_function (A, B, [ ])
    elseif (isreal (lastarg) && length (lastarg) == 1)
        if (method == 1)
            % usage: [ ] = spqr_function (A, B, tol)
            opts.tol = lastarg ;
        else
            % usage: [ ] = spqr_function (A, B, k)
            opts.k = round (lastarg) ;
        end
    else
        % invalid usage: last argument invalid
        ok = 0 ;
    end

elseif (args == 4 && method == 2)

    % usage: [ ] = spqr_ssp (A, N, k, opts)
    B = varargin {1} ;
    opts = varargin {3} ;
    opts.k = round (varargin {2}) ;

else

    % invalid usage: too few or too many arguments
    ok = 0 ;

end

%-------------------------------------------------------------------------------
% check B
%-------------------------------------------------------------------------------

[m,n] = size (A) ;
if (method == 1)
    if (isempty (B))
        B = zeros (m,0) ;
    elseif (size (B,1) ~= m)
        error ('A and B must have the same number of rows') ;
    end
end

if ~ok
    return
end

%-------------------------------------------------------------------------------
% options for all functions
%-------------------------------------------------------------------------------

if ~isfield (opts, 'get_details')
    % 0: basic statistics (the default)
    % 1: detailed statistics:  basic stats, plus input options, time taken by
    %   various phases, statistics from spqr and spqr_rank subfunctions called,
    %   and other details.  Normally of interest only to the developers.
    % 2: basic statistics and a few additional statistics.  Used internally
    %    by some routines to pass needed information.
    opts.get_details = 0 ;
end

if (opts.get_details == 1)
    start_tic = tic ;
end

if ~isfield (opts, 'repeatable')
    % by default, results are repeatable
    opts.repeatable = 1 ;
end

% spqr_pinv uses all opts

%-------------------------------------------------------------------------------
% options for spqr_basic, spqr_cod, and spqr_ssi
%-------------------------------------------------------------------------------

if (~isfield (opts, 'tol'))
    % a negative number means the default tolerance should be computed
    opts.tol = 'default' ;
end

if (~isfield (opts, 'tol_norm_type'))
    % 1: use norm (A,1) to compute the default tol
    % 2: use normest (A, 0.01).  This is the default.
    opts.tol_norm_type = 2 ;
end

if (method < 2 && tol_is_default (opts.tol))
    % compute the default tolerance, but not for spqr_ssp, which doesn't need it
    if (opts.tol_norm_type == 1)
        normest_A = norm (A,1) ;
    else
        % this is the default
        normest_A = normest(A,0.01);
    end
    opts.tol = max(m,n)*eps(normest_A) ;
end

if (~isfield (opts, 'nsvals_large'))
    % default number of large singular values to estimate
    opts.nsvals_large = 1 ;
end

%-------------------------------------------------------------------------------
% options for spqr_basic, spqr_null, spqr_cod, and spqr
%-------------------------------------------------------------------------------

if (~isfield (opts, 'nsvals_small'))
    % default number of small singular values to estimate
    opts.nsvals_small = 1 ;
end

if (~isfield (opts, 'implicit_null_space_basis'))
    opts.implicit_null_space_basis = 1 ;
end

%-------------------------------------------------------------------------------
% options for spqr_cod (only)
%-------------------------------------------------------------------------------

if (~isfield (opts, 'start_with_A_transpose'))
    opts.start_with_A_transpose = 0 ;
end

%-------------------------------------------------------------------------------
% options for spqr_ssi (called by spqr_basic, spqr_null, spqr_cod and spqr_pinv)
%-------------------------------------------------------------------------------

if (~isfield (opts, 'ssi_tol'))
    opts.ssi_tol = 'default' ;
end

if (method < 2 && tol_is_default (opts.ssi_tol))
    % default tolerance for spqr_ssi is the same as spqr_basic, spqr_cod,
    % spqr_pinv
    opts.ssi_tol = opts.tol ;
end

if (~isfield (opts, 'ssi_min_block'))
    opts.ssi_min_block = 3 ;
end

if (~isfield (opts, 'ssi_max_block'))
    opts.ssi_max_block = 10 ;
end

if (~isfield (opts, 'ssi_min_iters'))
    opts.ssi_min_iters = 3 ;
end

if (~isfield (opts, 'ssi_max_iters'))
    opts.ssi_max_iters = 100 ;
end

if (~isfield (opts, 'ssi_nblock_increment'))
    opts.ssi_nblock_increment = 5 ;
end

if (~isfield (opts, 'ssi_convergence_factor'))
    opts.ssi_convergence_factor = 0.1 ;     % also 0.25 is often safe
end

%-------------------------------------------------------------------------------
% options for spqr_ssi (called by spqr_basic, spqr_null, spqr_cod and spqr_pinv)
%-------------------------------------------------------------------------------

if (~isfield (opts, 'k'))
    % number of singular values to compute
    opts.k = 1 ;
end

if (~isfield (opts, 'ssp_min_iters'))
    opts.ssp_min_iters = 4 ;
end

if (~isfield (opts, 'ssp_max_iters'))
    opts.ssp_max_iters = 10 ;
end

if (~isfield (opts, 'ssp_convergence_factor'))
    opts.ssp_convergence_factor = 0.1 ;
end

%-------------------------------------------------------------------------------
% initialize stats
%-------------------------------------------------------------------------------

% initialize the order of stats, as much as possible
stats.flag = -1 ;       % -1 means 'not yet computed'
if method <= 1
   stats.rank = -1 ;   % -1 means not computed
end
if method == 0
    if ( opts.get_details == 1 && exist('normest_A','var')  )
        stats.normest_A = normest_A;
    end
end
% for spqr_null and spqr_ssi (method = 0) additional fields initialized
%      in spqr_null or spqr_ssi
if method == 1
    % spqr_basic, spqr_cod, spqr_pinv, initialize stats fields in common to all
    if opts.get_details == 2 ;
        stats.rank_spqr = -1 ;
    end
    stats.tol = opts.tol ;
    if ( opts.get_details == 1 && exist('normest_A','var')  )
        stats.normest_A = normest_A;
    end
    stats.tol_alt = -1 ;   % removed later if remains -1
    stats.est_sval_upper_bounds = -1 ;
    stats.est_sval_lower_bounds = -1 ;
    stats.sval_numbers_for_bounds = -1 ;
    % for spqr_basic, spqr_cod and spqr_pinv additional fields initialized
    %     in spqr_basic, spqr_cod or spqr_ssi
end
if method == 2
    % for spqr_ssp initialize order for all stats fields
    stats.est_svals = -1 ;   % -1 means not yet computed
    stats.est_error_bounds = -1 ;
    stats.sval_numbers_for_bounds = -1 ;
    if (opts.get_details == 1)
        stats.iters = 0 ;
        stats.opts_used = opts ;
        stats.time = 0 ;
        stats.time_initialize = 0 ;
        stats.time_iters = 0 ;
        stats.time_est_error_bounds = 0 ;
        stats.time_svd = 0 ;
    end
end

%-------------------------------------------------------------------------------
% return timings
%-------------------------------------------------------------------------------

if (opts.get_details == 1)
    % get the total time to initializations including to computing
    % normest(A,0.01), if called
    stats.time_initialize = toc (start_tic) ;
end


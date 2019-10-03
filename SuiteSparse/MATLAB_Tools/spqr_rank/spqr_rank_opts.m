function [opts] = spqr_rank_opts (opts, opts_for_ssp)
%SPQR_RANK_OPTS sets and prints the default options for spqr_rank
%
% Usage: opts = spqr_rank_opts (opts) ;
%
% With no input arguments or with an empty opts input, the default options are
% used.  If opts is provided on input, any missing options are filled in with
% default values.  With no output arguments, the opts are printed.
%
% Examples
%
%   opts = spqr_rank_opts ;         % returns the default opts
%   spqr_rank_opts (opts) ;         % prints the current opts
%   spqr_rank_opts                  % prints the default opts
%   opts = spqr_rank_opts (opts) ;  % adds all missing defaults to opts
%
% Specific options:
%
%   opts.get_details -- determine what statistics to return.
%       0: basic statistics
%       1: extensive statistics.
%       2: basic statistics and a few addional statistics.
%       See 'help spqr_rank_stats' for details.
%       Default: 0.
%
%   opts.tol -- estimate the numerical rank of A with tolerance tol = opts.tol
%       or, equivalently, estimate the number of singular values of A larger
%       than tol.  Due to  computer arithmetic errors the algorithm will
%       estimate the numerical rank of A + E where norm(E) = O(norm(A)*eps)
%       with eps = machine epsilon.  Therefore tol should be bigger than
%       O(norm(A)*eps),  for example, tol >= n * norm(A)*eps.  If the opts is a
%       number, not a structure, then tol = opts.  If a string 'default', or a
%       negative number, the default is used.
%       Default: max(m,n)*eps(normest(A,.01)).
%
%   opts.tol_norm_type -- if opts.tol is not specified calculate tol --
%       1:  tol = max( size(A) ) * eps( norm (A,1) )
%       2:  tol = max( size(A) ) * eps( normest (A,0.01) )
%       Default: 2.
%
%   opts.nsvals_large -- estimate and return upper and lower bounds on singular
%       values (stats.rank - nsvals_large + 1) to stats.rank, which are the
%       singular values just before the gap.
%       Default: 1.
%
%   opts.nsvals_small -- estimate and return upper and lower bounds on singular
%       values (stats.rank + 1) to (stats.rank + nsvals_small), which are
%       the singular values just after the gap.
%       Default: 1.
%
%   opts.implicit_null_space_basis --
%       0: null space basis of A and/or A' is returned as an explicit matrix.
%       1: null space basis of A and/or A' is returned implicitly,
%           as a product of Householder vectors.
%       Default: 1.
%
%   opts.start_with_A_transpose -- if false (0) then spqr_cod initially
%       calculates a QR factorization of A, if true (1) then it uses A'.
%       Default: 0.
%
%   opts.repeatable -- controls the random stream.
%       0: use the current random number stream.  The state of the stream will
%           be different when the method returns.  Repeated calls may generate
%           different results.  Faster for small matrices, however.
%       1: use a repeatable internal random number stream.  The current stream
%           will not be affected.  Repeated calls generate the same results.
%       Default: 1
%
% Options for spqr_ssi (used by spqr_basic, spqr_null, spqr_cod, and spqr_pinv):
%
%   opts.ssi_tol -- same as opts.tol, but for the spqr_ssi function (see above).
%
%   opts.ssi_min_block --  the initial block size.
%       Default: 3
%
%   opts.ssi_max_block -- the maximum block size in block inverse iteration.
%       The default value, 10, is usually sufficient to calculate the correct
%       numerical rank when spqr_ssi is called  by spqr_basic, spqr_null, 
%       spqr_pinv or   spqr_cod.  However if spqr_ssi is used directly to 
%       estimate singular values of a triangular matrix a larger value may be
%       desirable.
%       Default: 10.
%
%   opts.ssi_min_iters -- min # of iterations before checking convergence.
%       Default: 3.
%
%   opts.ssi_max_iters -- max # of iterations before stopping the iterations.
%       Default: 100.
%
%   opts.ssi_nblock_increment -- block size incremented by this amount if
%       convergence criteria not met with current block size.
%       Default: 5.
%
%   opts.ssi_convergence_factor -- continue power method iterations until an
%       estimated bound on the relative error in the approximation s(1), 
%       an estimate of the largest singular value returned by ssi, is <=
%       convergence_factor. See the code for additional description.  The
%       default = 0.1 appears to provides sufficient accuracy to correctly
%       determine the numerical rank in almost all cases, assuming that
%       stats.flag is returned as 0.  If the purpose for using spqr_basic,
%       spqr_null, spqr_pinv, spqr_cod or spqr_ssi is to estimate singular value
%       bounds a smaller value, for example 0.001 or smaller, may be useful.
%       Default: 0.1.
%
% Options for spqr_ssp (used by spqr_basic, spqr_null, spqr_cod, and spqr_pinv):
%
%   opts.k -- the # of singular values to compute.
%       Default 1.
%
%   opts.ssp_min_iters -- min # of iterations before checking convergence.
%       Default: 4
%
%   opts.ssp_max_iters -- max # of iterations before stopping the iterations.
%       The default value = 10 appears, with the default value of opts.tol,
%       to provide sufficient accuracy to correctly determine the numerical
%       rank when spqr_ssi is called by spqr_basic, spqr_null, spqr_pinv or
%       spqr_cod in almost all cases, assuming that stats.flag is 0. For
%       values of opts.tol larger than the default value, a larger value of
%       opts.ssp_max_iters, for example 100, may be useful.
%       Default: 10.
%
%   opts.ssp_convergence_factor -- continue power method iterations until an
%       estimated bound on the relative error in the approximation
%       S(k,k) to singular value number k of A is <= convergence_factor.
%       The default value = 0.1 appears, with the default value of
%       opts.tol, to provide sufficient accuracy to correctly determine
%       the numerical rank in spqr_basic, spqr_null, spqr_pinv or spqr_cod
%       in almost all cases, assuming that stats.flag is 0.  For values
%       of opts.tol larger than the default value, a smaller value of
%       opts.ssp_convergence_factor, for example 0.01, may be useful.
%       Default: 0.1
%
% See also spqr_basic, spqr_null, spqr_cod, spqr_pinv, spqr_ssp, spqr_ssi.

% Copyright 2012, Leslie Foster and Timothy A. Davis

% The second input parameter to spqr_rank_opts is no longer necessary
% but maintained for compatibility reasons.

%-------------------------------------------------------------------------------
% get the defaults, if not present on input, but do not compute tol
%-------------------------------------------------------------------------------

if (nargin < 1)
    opts = [ ] ;
end
if (nargin < 2)
    opts_for_ssp = 0 ;
end

% the ssp_* options  differ if spqr_ssp is called directly
%   -- in older versions but not so in the current version.
if (opts_for_ssp)
    [ignore, opts] = spqr_rank_get_inputs (1, 2, opts) ;                    %#ok
else
    [ignore, opts] = spqr_rank_get_inputs (1, 3, opts) ;                    %#ok
end
clear ignore

if (tol_is_default (opts.tol))
    opts.tol = 'default' ;
end

if (tol_is_default (opts.ssi_tol))
    opts.ssi_tol = 'default' ;
end

%-------------------------------------------------------------------------------
% print the defaults, if requested
%-------------------------------------------------------------------------------

if (nargout == 0)

    fprintf ('options for the spqr_rank functions:\n\n') ;

    %---------------------------------------------------------------------------
    % get_details
    %---------------------------------------------------------------------------

    fprintf ('  get_details : ') ;
    if (opts.get_details == 0)
        fprintf ('0 : basic statistics returned\n') ;
    elseif (opts.get_details == 1)
        fprintf ('1 : extensive statistics returned\n') ;
    elseif (opts.get_details == 2)
        fprintf (['2 : basic statistics and a few additional statistics' ...
            'returned\n']) ;
    end

    %---------------------------------------------------------------------------
    % tol
    %---------------------------------------------------------------------------

    if (ischar (opts.tol))
        if ( ~isfield(opts,'tol_norm_type') || isempty(opts.tol_norm_type) ...
                || opts.tol_norm_type ~= 1 )
           fprintf ('  tol : default : max(m,n)*eps(normest(A,0.01))\n') ;
        else
           fprintf ('  tol : max(m,n)*eps(norm(A,1))\n') ;
        end
    else
        fprintf ('  tol : %g\n', opts.tol) ;
    end
    
    %---------------------------------------------------------------------------
    % tol_norm_type
    %---------------------------------------------------------------------------

    if ~ischar ( opts.tol )
        fprintf(['  tol_norm_type: %d : not used since opts.tol ', ...
            'is specified'], opts.tol_norm_type) ;
    else
        if opts.tol_norm_type ~= 1
            fprintf(['  tol_norm_type: %d : tol = '...
               'max(m,n)*eps(normest(A,0.01))'], opts.tol_norm_type) ; 
        else
            fprintf(['  tol_norm_type: %d : tol = '...
               'max(m,n)*eps(norm(A,1))'], opts.tol_norm_type) ; 
        end
    end
    fprintf ('  \n') ;
    

    %---------------------------------------------------------------------------
    % nsvals_large
    %---------------------------------------------------------------------------

    fprintf ('  nsvals_large : %d : ', opts.nsvals_large) ;
    fprintf ('# of large singular values to estimate.\n');

    %---------------------------------------------------------------------------
    % nsvals_small
    %---------------------------------------------------------------------------

    fprintf ('  nsvals_small : %d : ', opts.nsvals_small) ;
    fprintf ('# of small singular values to estimate.\n');

    %---------------------------------------------------------------------------
    % implicit_null_space_basis
    %---------------------------------------------------------------------------

    fprintf ('  implicit_null_space_basis : ') ;
    if (opts.implicit_null_space_basis)
        fprintf ('true : N represented in Householder form.\n') ;
    else
        fprintf ('false : N represented as an explicit matrix.\n') ;
    end

    %---------------------------------------------------------------------------
    % start_with_A_transpose
    %---------------------------------------------------------------------------

    fprintf ('  start_with_A_transpose : ') ;
    if (opts.start_with_A_transpose)
        fprintf ('true : spqr_cod computes qr(A'').\n') ;
    else
        fprintf ('false : spqr_cod computes qr(A).\n') ;
    end

    %---------------------------------------------------------------------------
    % ssi_tol
    %---------------------------------------------------------------------------

    if (ischar (opts.ssi_tol))
        fprintf ('  ssi_tol : default : same as tol.\n') ;
    else
        fprintf ('  ssi_tol : %g\n', opts.ssi_tol) ;
    end

    %---------------------------------------------------------------------------
    % ssi_min_block
    %---------------------------------------------------------------------------

    fprintf ('  ssi_min_block : %d : ssi initial block size.\n', ...
        opts.ssi_min_block) ;

    %---------------------------------------------------------------------------
    % ssi_max_block
    %---------------------------------------------------------------------------

    fprintf ('  ssi_max_block : %d : spqr_ssi max block size.\n', ...
        opts.ssi_max_block) ;

    %---------------------------------------------------------------------------
    % ssi_min_iters
    %---------------------------------------------------------------------------

    fprintf (['  ssi_min_iters : %d : min # of iterations before checking ' ...
        'convergence\n'] , opts.ssi_min_iters) ;

    %---------------------------------------------------------------------------
    % ssi_max_iters
    %---------------------------------------------------------------------------

    fprintf (['  ssi_max_iters : %d : max # of iterations before stopping ' ...
        'spqr_ssi iterations\n'] , opts.ssi_max_iters) ;

    %---------------------------------------------------------------------------
    % ssi_nblock_increment
    %---------------------------------------------------------------------------

    fprintf (['  ssi_nblock_increment : %d : block size inc. if ' ...
        'convergence not met.\n'], opts.ssi_nblock_increment) ;

    %---------------------------------------------------------------------------
    % ssi_convergence_factor
    %---------------------------------------------------------------------------

    fprintf (['  ssi_convergence_factor : %g : spqr_ssi termination ' ...
        'criterion.\n'], opts.ssi_convergence_factor) ;

    %---------------------------------------------------------------------------
    % k
    %---------------------------------------------------------------------------

    fprintf (['  k : %d : number of singular values to compute in ' ...
        'spqr_ssp.\n]',  opts.k]) ;

    %---------------------------------------------------------------------------
    % ssp_min_iters
    %---------------------------------------------------------------------------

    fprintf (['  ssp_min_iters : %d : min # of iterations before checking ' ...
        'convergence\n'] , opts.ssp_min_iters) ;

    %---------------------------------------------------------------------------
    % ssp_max_iters
    %---------------------------------------------------------------------------

    fprintf (['  ssp_max_iters : %d : max # of ssp iterations before ' ...
        'stopping iterations\n'] , opts.ssp_max_iters) ;

    %---------------------------------------------------------------------------
    % ssp_convergence_factor
    %---------------------------------------------------------------------------

    fprintf (['  ssp_convergence_factor : %g ssp terminates when relative '...
        'error drops below this value.\n'], opts.ssp_convergence_factor) ;

    %---------------------------------------------------------------------------
    % repeatable
    %---------------------------------------------------------------------------

    fprintf ('  repeatable : ') ;
    if (opts.repeatable)
        fprintf (['true : internal random stream used to ' ...
            'guarantee repeatability\n']) ;
    else
        fprintf ('false : use whatever current random stream is in effect.\n') ;
    end

    % clearing opts makes the call "sqpr_rank_opts" (with no semicolon) more
    % readable, since nothing is returned, and thus no opts struct is printed.
    % Everything has already been printed above.
    clear opts ;
end


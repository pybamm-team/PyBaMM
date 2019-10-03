function spqr_rank_stats (stats, print_opts)
%SPQR_RANK_STATS prints the statistics from spqr_rank functions
%
% For a detailed description of the meaning of the basic statistic for
% spqr_basic, spqr_null, spqr_pinv or spqr_cod, just type
% 'spqr_rank_stats' with no inputs.  Type spqr_rank_stats('ssi') or
% spqr_rank_stats('ssp') for a description of the basic statistics for
% sqpr_ssi or spqr_ssp, respectively. spqr_rank_stats(1) also prints a
% detailed description of all statistics calculated by any spqr_function
% when opts.get_details is 1.  To print a summary of the stats struct
% returned by a spqr_rank function, use spqr_rank_stats(stats) or
% spqr_rank_stats(stats,1).
%
% Examples
%
% spqr_rank_stats ;          % prints a description of basic statistics
%                            % from spqr_basic, spqr_null, spqr_pinv, spqr_cod
% spqr_rank_stats('ssi') ;   % description of basic statistics from spqr_ssi
% spqr_rank_stats('ssp') ;   % description of basic statistics from spqr_ssp
% spqr_rank_stats ( 1 ) ;    % prints a verbose description of all statistics
% spqr_rank_stats (stats) ;  % prints a short summary of the stats struct
% spqr_rank_stats (stats,1); % prints options used, when opts.get_details is 1
%
% See also spqr_basic, spqr_cod, spqr_pinv, spqr_ssi, spqr_ssp

% Copyright 2012, Leslie Foster and Timothy A Davis

if (nargin == 0)
    stats = 0 ;
    get_details = 0 ;
    method = 1 ;   % spqr_basic, spqr_null, spqr_cod, spqr_pinv
end

if ( nargin == 1 )
    if ischar(stats)
        get_details = 0 ;
        if ( strcmp(stats,'spqr_ssi') || strcmp(stats,'ssi') )
            method = 2 ;   % spqr_ssp
        elseif ( strcmp(stats,'spqr_ssp') || strcmp(stats,'ssp') )
            method = 3 ;   % spqr_ssi
        else
            method = 1;    % spqr_basic, spqr_null, spqr_cod, spqr_pinv
        end
    elseif isreal(stats)
        get_details = stats;
        method = 4;    % any method
    end
    print_opts = 0 ;
end

if (isstruct (stats))

    %---------------------------------------------------------------------------
    % print the stats returned by the spqr_* function
    %---------------------------------------------------------------------------

    %---------------------------------------------------------------------------
    % flag
    %---------------------------------------------------------------------------

    flag = -1 ;
    if (isfield (stats, 'flag'))
        flag = stats.flag ;
    end

    if ~isfield( stats, 'est_svals')
        % description of flag for calls except for call to spqr_ssp
        if (flag == 0)
            fprintf ('\n  flag : %g : ', flag) ;
            fprintf ('ok.  stats.rank very likely to be correct.\n') ;
        elseif (flag <= 2)
            fprintf ('\n  flag : %g : ', flag) ;
            fprintf ('stats.rank may be correct for tolerance stats.tol,\n') ;
            fprintf ('      but error bounds are too high to confirm this.\n') ;
            if (flag == 1)
                fprintf (['      However, stats.rank appears to be correct' ...
                    ' for tolerance stats.tol_alt.\n']) ;
            end
        elseif (flag == 3)
            fprintf ('\n  flag : %g : ', flag) ;
            fprintf ('poor results.  stats.rank is likely too high.\n') ;
        elseif (flag == 4)
            fprintf ('\n  flag : %g : ', flag) ;
            fprintf ('failure.  Overflow during inverse power method.\n') ;
%       elseif (flag == 5)
%           % this code is disabled because stats.flag=5 is removed.
%           fprintf ('\n  flag : %g : ', flag) ;
%           fprintf ('failure.  Inconsistent rank estimates.\n') ;
        else
            error ('spqr_rank:invalid', 'invalid stats') ;
        end
    else
        % description of flag for calls to spqr_ssp
        fprintf ('\n  flag : %g : ', flag) ;
        if (flag == 0)
            fprintf (['ok. spqr_ssp converged with est. relative error ' ...
                '<= opts_ssp.convergence_factor.\n']) ;
        else
            fprintf (['spqr_ssp did not converge with est. relative error ' ...
                '<= opts_ssp.convergence_factor.\n']) ;
        end
    end

    %---------------------------------------------------------------------------
    % rank
    %---------------------------------------------------------------------------

    if (isfield (stats, 'rank'))
        fprintf ('  rank : %g : estimate of numerical rank.\n', stats.rank) ;
    end

    %---------------------------------------------------------------------------
    % rank_spqr
    %---------------------------------------------------------------------------

    if (isfield (stats, 'rank_spqr'))
        fprintf ('  rank_spqr : %g : ', stats.rank_spqr) ;
        fprintf ('estimate of numerical rank from spqr.\n') ;
        fprintf ('      This is normally an upper bound on the true rank.\n') ;
    end

    %---------------------------------------------------------------------------
    % tol
    %---------------------------------------------------------------------------

    if (isfield (stats, 'tol'))
        fprintf ('  tol : %g : numerical tolerance used.\n', stats.tol) ;
    end

    %---------------------------------------------------------------------------
    % tol_alt
    %---------------------------------------------------------------------------

    if (isfield (stats, 'tol_alt'))
        fprintf ('  tol_alt : %g : alternate numerical tolerance used.\n', ...
            stats.tol_alt) ;
    end

    %---------------------------------------------------------------------------
    % normest_A
    %---------------------------------------------------------------------------

    if (isfield (stats, 'normest_A'))
        fprintf ('  normest_A : %g : estimate of Euclidean norm of A.\n', ...
            stats.normest_A) ;
    end

    %---------------------------------------------------------------------------
    % normest_R
    %---------------------------------------------------------------------------

    if (isfield (stats, 'normest_R'))
        fprintf (['  normest_R : %g : estimate of Euclidean norm of R ' ...
            '(calculated for spqr_ssi).\n'], ...
            stats.normest_R) ;
    end

    %---------------------------------------------------------------------------
    % est_svals_upper_bounds
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_sval_upper_bounds'))
        fprintf ('  est_sval_upper_bounds : ') ;
        print_vector (stats.est_sval_upper_bounds) ;
        fprintf ('      : estimated upper bounds on singular value(s).\n') ;
    end

    %---------------------------------------------------------------------------
    % est_svals_lower_bounds
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_sval_lower_bounds'))
        fprintf ('  est_sval_lower_bounds : ') ;
        print_vector (stats.est_sval_lower_bounds) ;
        fprintf ('      : estimated lower bounds on singular value(s).\n') ;
    end

    %---------------------------------------------------------------------------
    % est_svals_of_R
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_svals_of_R'))
        fprintf ('  est_svals_of_R : ') ;
        print_vector (stats.est_svals_of_R) ;
        fprintf (['      : estimated singular value(s) of triangular ' ...
            'matrix R.\n']) ;
    end

    %---------------------------------------------------------------------------
    % est_svals
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_svals'))
        fprintf ('  est_svals : ') ;
        print_vector (stats.est_svals) ;
        fprintf (['      : estimated singular value(s) of A*N or A''*NT, ' ...
            'from spqr_ssp.\n']) ;
    end

    %---------------------------------------------------------------------------
    % est_error_bounds
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_error_bounds'))
        fprintf ('  est_error_bounds : ') ;
        print_vector (stats.est_error_bounds) ;
        fprintf ('      : error bounds for each singular value.\n') ;
    end

    %---------------------------------------------------------------------------
    % sval_numbers_for_bounds
    %---------------------------------------------------------------------------

    if (isfield (stats, 'sval_numbers_for_bounds'))
        fprintf ('  sval_numbers_for_bounds : ') ;
        print_vector (stats.sval_numbers_for_bounds) ;
        fprintf ('      : index of singular value(s), for bounds.\n') ;
    end

    %---------------------------------------------------------------------------
    % est_norm_A_times_N
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_norm_A_times_N'))
        fprintf (['  est_norm_A_times_N : %g : estimated' ...
            ' norm(A*N).\n'], stats.est_norm_A_times_N) ;
    end

    %---------------------------------------------------------------------------
    % est_err_bound_norm_A_times_N
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_err_bound_norm_A_times_N'))
        fprintf (['  est_err_bound_norm_A_times_N : %g : estimated error ' ...
            'bound for norm(A*N).\n'], stats.est_err_bound_norm_A_times_N) ;
    end

    %---------------------------------------------------------------------------
    % est_norm_A_transpose_times_NT
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_norm_A_transpose_times_NT'))
        fprintf (['  est_norm_A_transpose_times_NT : %g : estimated' ...
            ' norm(A''*NT).\n'], stats.est_norm_A_transpose_times_NT) ;
    end

    %---------------------------------------------------------------------------
    % est_err_bound_norm_A_transpose_times_NT
    %---------------------------------------------------------------------------

    if (isfield (stats, 'est_err_bound_norm_A_transpose_times_NT'))
        fprintf (['  est_err_bound_norm_A_transpose_times_NT : %g : ' ...
            ' estimated error bound for norm(A''*NT).\n'], ...
            stats.est_err_bound_norm_A_transpose_times_NT) ;
    end

    %---------------------------------------------------------------------------
    % norm_R_times_N
    %---------------------------------------------------------------------------

    if (isfield (stats, 'norm_R_times_N'))
        fprintf (['  norm_R_times_N : %g : norm of (R*N), from ' ...
            'spqr_ssi.\n'], stats.norm_R_times_N) ;
    end

    %---------------------------------------------------------------------------
    % norm_R_transpose_times_NT
    %---------------------------------------------------------------------------

    if (isfield (stats, 'norm_R_transpose_times_NT'))
        fprintf (['  norm_R_transpose_times_NT : %g : norm of (R''*NT), '...
            'from spqr_ssi.\n'], stats.norm_R_transpose_times_NT) ;
    end

    %---------------------------------------------------------------------------
    % iters
    %---------------------------------------------------------------------------

    if (isfield (stats, 'iters'))
        fprintf ('  iters : %g : iterations in spqr_ssi or spqr_ssp.\n', ...
            stats.iters) ;
    end

    %---------------------------------------------------------------------------
    % nsvals_large_found
    %---------------------------------------------------------------------------

    if (isfield (stats, 'nsvals_large_found'))
        fprintf (['  nsvals_large_found : %d : number of large singular ' ...
            'values found.\n'], stats.nsvals_large_found) ;
    end

    %---------------------------------------------------------------------------
    % final_blocksize
    %---------------------------------------------------------------------------

    if (isfield (stats, 'final_blocksize'))
        fprintf ('  final_blocksize : %d : final block size in spqr_ssi.\n', ...
            stats.final_blocksize) ;
    end

    %---------------------------------------------------------------------------
    % ssi_max_block_used
    %---------------------------------------------------------------------------

    if (isfield (stats, 'ssi_max_block_used'))
        fprintf (['  ssi_max_block_used : %d : max block size for ' ...
            'spqr_ssi.\n'], stats.ssi_max_block_used) ;
    end

    %---------------------------------------------------------------------------
    % ssi_min_block_used
    %---------------------------------------------------------------------------

    if (isfield (stats, 'ssi_min_block_used'))
        fprintf (['  ssi_min_block_used : %d : initial block size for ' ...
            'spqr_ssi.\n'], stats.ssi_min_block_used) ;
    end

    %---------------------------------------------------------------------------
    % time
    %---------------------------------------------------------------------------

    if (isfield (stats, 'time'))
        fprintf (['  time : %g : total time taken ' ...
            '(includes all timings below).\n'], stats.time) ;
    end

    %---------------------------------------------------------------------------
    % time_initialize
    %---------------------------------------------------------------------------

    if (isfield (stats, 'time_initialize'))
        fprintf ('  time_initialize : %g : time to initialize',...
            stats.time_initialize) ;
        if isfield (stats, 'normest_A') || isfield (stats, 'normest_R')
            fprintf (' including estimating the norm of A or R.\n')
        else
            fprintf('.\n')
        end
    end

    %---------------------------------------------------------------------------
    % time_svd
    %---------------------------------------------------------------------------

    if (isfield (stats, 'time_svd'))
        fprintf ('  time_svd : %g : total time taken by svd.\n', ...
            stats.time_svd) ;
    end

    %---------------------------------------------------------------------------
    % time_basis
    %---------------------------------------------------------------------------

    if (isfield (stats, 'time_basis'))
        fprintf ('  time_basis : %g : time to compute basis.\n', ...
            stats.time_basis) ;
    end

    %---------------------------------------------------------------------------
    % time_iters
    %---------------------------------------------------------------------------

    if (isfield (stats, 'time_iters'))
        fprintf ('  time_iters : %g : time for spqr_ssi iterations.\n', ...
            stats.time_iters) ;
    end

    %---------------------------------------------------------------------------
    % time_est_error_bounds
    %---------------------------------------------------------------------------

    if (isfield (stats, 'time_est_error_bounds'))
        fprintf (['  time_est_error_bounds : %g : time taken to estimate '...
            'error bounds in spqr_ssi.\n'], stats.time_est_error_bounds) ;
    end

    %---------------------------------------------------------------------------
    % opts_used
    %---------------------------------------------------------------------------

    if (print_opts && isfield (stats, 'opts_used'))
        fprintf ('\nopts_used : ') ;
        spqr_rank_opts (stats.opts_used) ;
    end

    %---------------------------------------------------------------------------
    % info_spqr1
    %---------------------------------------------------------------------------

    if (isfield (stats, 'info_spqr1'))
        fprintf ('\ninfo_spqr1 : statistics from first QR factorization.\n\n') ;
        disp (stats.info_spqr1) ;
    end

    %---------------------------------------------------------------------------
    % info_spqr2
    %---------------------------------------------------------------------------

    if (isfield (stats, 'info_spqr2'))
        fprintf ('\ninfo_spqr2 : statistics from second QR factorization.\n\n');
        disp (stats.info_spqr2) ;
    end

    %---------------------------------------------------------------------------
    % stats_ssi
    %---------------------------------------------------------------------------

    if (isfield (stats, 'stats_ssi'))
        % all opts used by spqr_ssp are the same stats.opts_used
        fprintf ('\nstats_ssi : statistics from spqr_ssi.\n') ;
        spqr_rank_stats (stats.stats_ssi, 0) ;
    end

    %---------------------------------------------------------------------------
    % stats_ssp_N
    %---------------------------------------------------------------------------

    if (isfield (stats, 'stats_ssp_N'))

        fprintf ('\nstats_ssp_N : statistics from spqr_ssp (A,N).\n') ;
        spqr_rank_stats (stats.stats_ssp_N, 0) ;
        % all other opts used by spqr_ssp are the same stats.opts_used
        if (isfield (stats.stats_ssp_N, 'opts_used') && ...
            isfield (stats.stats_ssp_N.opts_used, 'k') && print_opts)
                fprintf (['  stats_ssp_N.opts_used.k : %d : number of ' ...
                'singular values to compute in spqr_ssp(A,N).\n'], ...
                stats.stats_ssp_N.opts_used.k) ;
        end
    end

    %---------------------------------------------------------------------------
    % stats_ssp_NT
    %---------------------------------------------------------------------------

    if (isfield (stats, 'stats_ssp_NT'))
        fprintf ('\nstats_ssp_NT : statistics from spqr_ssp (A'',NT).\n') ;
        spqr_rank_stats (stats.stats_ssp_NT, 0) ;
        % all other opts used by spqr_ssp are the same stats.opts_used
        if (isfield (stats.stats_ssp_NT, 'opts_used') && ...
            isfield (stats.stats_ssp_NT.opts_used, 'k') && print_opts)
                fprintf (['  stats_ssp_NT.opts_used.k : %d : number of ' ...
                'singular values to compute in spqr_ssp(A'',NT).\n'], ...
                stats.stats_ssp_NT.opts_used.k) ;
        end
    end

    %---------------------------------------------------------------------------
    % stats_spqr_basic
    %---------------------------------------------------------------------------

    if (isfield (stats, 'stats_spqr_basic'))
        % all opts used by spqr_basic are the same stats.opts_used
        fprintf ('\nstats_spqr_basic : statistics from spqr_basic.\n') ;
        spqr_rank_stats (stats.stats_spqr_basic, 0) ;
    end

    %---------------------------------------------------------------------------
    % stats_spqr_null
    %---------------------------------------------------------------------------

    if (isfield (stats, 'stats_spqr_null'))
        fprintf ('\nstats_spqr_null : statistics from spqr_null.\n') ;
        % all other opts used by spqr_null are the same stats.opts_used
        if (isfield (stats.stats_spqr_null, 'opts_used') && ...
            isfield (stats.stats_spqr_null.opts_used, 'ssi_min_block'))
                fprintf (['\n  stats_spqr_null.opts_used.ssi_min_block : %d'...
                ' : initial block size in spqr_ssi as used by spqr_null.'], ...
                stats.stats_spqr_null.opts_used.ssi_min_block) ;
        end
        spqr_rank_stats (stats.stats_spqr_null, 0) ;

    end

else

    %---------------------------------------------------------------------------
    % describe each statistic
    %---------------------------------------------------------------------------

    fprintf ('\nDescription of stats returned by ') ;
    if method == 1
        fprintf ('spqr_basic, spqr_null,  spqr_pinv \nor spqr_cod:\n') ;
    elseif method == 2
        fprintf ('spqr_ssi:\n') ;
    elseif method == 3
        fprintf ('spqr_ssp:\n') ;
    elseif method == 4
        fprintf ('all spqr_functions:\n') ;
    end

    if ( method == 1  || method == 2 || get_details >= 1 )
        fprintf ([ ...
        '\nstats.flag (for all routines except spqr_ssp) -- \n' ...
        '   if stats.flag is 0 if it is likely, although not\n' ...
        '   guaranteed, that stats.rank is the correct numerical rank for\n' ...
        '   tolerance stats.tol (i.e. agrees with the numerical rank\n' ...
        '   determined by the singular values of R).\n' ...
        '   \n'...
        '   stats.flag is 1 if the calculated numerical rank stats.rank ' ...
            'may\n'...
        '   be correct for the tolerance stats.tol but the estimated error\n'...
        '   bounds are too large to confirm this.  However stats.rank '...
            'appears\n'...
        '   to be correct for an alternate tolerance stats.tol_alt.  More\n'...
        '   generally stats.rank appears to be correct for any tolerance\n'...
        '   between stats.est_sval_lower_bounds(nsvals_large) and\n'...
        '   stats.est_sval_upper_bounds(nsvals_large+1).\n' ...
        '   \n'...
        '   stats.flag is 2 if the calculated numerical rank ' ...
            'stats.numerical\n'...
        '   may be correct but estimated error bounds are too large to ' ...
            'confirm\n'...
        '   this.  The conditions for stats.flag to be 0 or 1 are not\n'...
        '   satisfied.\n' ...
        '   \n'...
        '   stats.flag is 3 if is likely that the numerical rank returned,\n'...
        '   stats.rank, is too large.\n'...
        '   \n'...
        '   stats.flag is 4 if overflow occurred during the inverse power\n'...
        '   method.  The method fails in this case, and all parameters ' ...
            'other\n'...
        '   stats are returned as empty ([ ]).\n' ...
        '   \n'...
        '   stats.flag is 5 if a catastrophic failure occurred.\n']) ;
    end

    if ( method == 3 || get_details >= 1 )
        fprintf ([ ...
        '\nstats.flag -- (for spqr_ssp) \n' ...
        '   stats.flag is 0 if spqr_ssp converged with estimated relative\n' ...
        '   error in singular value opts.k of A (or of A*N) <=\n' ...
        '   opts_ssp.convergence_factor. stats.flag is 1 if this is not ' ...
            'true.\n'])
    end

    if ( method == 1 || method == 2 || get_details >= 1 )
        fprintf ([ ...
        '\nstats.rank -- the estimated numerical rank when stats.flag is\n' ...
        '   0, 1 or 2.  stats.rank is typically an upper bound on the\n'...
        '   numerical rank when stats.flag is 3.  Note that stats.rank ' ...
            'is a\n'...
        '   correction to the rank returned by spqr (stats.rank_spqr) ' ...
            'in the\n'...
        '   case that the calculations in the routine inidicate that the ' ...
            'rank\n'...
        '   returned by spqr not correct.\n']) ;
    end

    if ( method == 1 || method == 2 || get_details >= 1 )
        fprintf ( ...
        '\nstats.tol -- the tolerance used to define the numerical rank.\n') ;
    end

    if ( method == 1 || get_details >= 1 )
        fprintf ([ ...
        '\nstat.tol_alt -- an alternate tolerance that corresponds to the\n' ...
        '   calculated numerical rank when stats.flag is 1.\n']) ;

        fprintf ([ ...
        '\nstats.est_sval_upper_bounds -- stats.est_sval_upper_bounds(i) ' ...
            'is an\n'...
        '   estimate of an upper bound on singular value number\n' ...
        '   stats.sval_numbers_for_bounds(i) of A.\n']) ;

        fprintf ([ ...
        '\nstats.est_sval_lower_bounds -- stats.est_sval_lower_bounds(i) ' ...
            'is an\n'...
        '   estimate of an lower bound on singular value number\n' ...
        '   stats.sval_numbers_for_bounds(i) of A.\n']) ;

        fprintf (['\n' ...
        '   Note that stats.est_sval_upper_bounds(i) is a rigorous upper '...
            'bound\n'...
        '   on some singular value of (A+E) where where E is ' ...
            'O(norm(A)*eps)\n'...
        '   Also stats.est_sval_lower_bounds(i) is a rigorous lower ' ...
            'bound on\n'...
        '   some singular value of (A+E).  In both cases the singular ' ...
            'value is\n'...
        '   normally singular value number sval_numbers_for_bounds(i) ' ...
            'of A,\n'...
        '   but the singular value number is not guaranteed.  For i such ' ...
            'that\n' ...
        '   sval_numbers_for_bounds(i) = stats.rank (that is for estimates\n'...
        '   of singular value stats.rank) if ' ...
            'stats.est_sval_upper_bounds(i)\n' ...
        '   is a large multiple of stats.est_sval_lower_bounds(i) then\n' ...
        '   solution vectors x produced by spqr_basic may be ' ...
            'inferior (i.e.\n'...
        '   be significanty larger) than solutions produced by ' ...
            'spqr_pinv or\n' ...
        '   spqr_cod.\n']) ;
    end

    if ( method == 2 || get_details >= 1 )
        fprintf ([ ...
        '\nstats.est_svals_of_R -- computed by spqr_ssi.\n' ...
        '   stats.est_svals_of_R contains estimates of the smallest ' ...
            'singular\n' ...
        '   of R.\n']) ;
    end

    if ( method == 3|| get_details >= 1 )
        fprintf ([ ...
        '\nstats.est_svals -- computed by spqr_ssp.\n' ...
        '   stats.est_svals(i) is an estimate of the ith largest ' ...
            'singular of\n' ...
        '   A or of A*N.  Also for i = 1:nsval, stats.est_svals(i) is a ' ...
            'lower\n' ...
        '   bound on the ith largest singular value of A (or A*N).\n']) ;
    end

    if ( method == 2 || method == 3 || get_details >= 1 )
        fprintf ([ ...
        '\nstats.est_error_bounds -- computed by spqr_ssi and spqr_ssp.\n' ...
        '   stats.est_error_bounds(i) is an estimated bound on the ' ...
            'absolute\n'...
        '   error in singular value number ' ...
            'stats.sval_numbers_for_bounds(i).\n' ...
        '   of R (for spqr_ssi) or of A or A*N (for spqr_ssp). It is ' ...
            'also a\n' ...
        '   rigorous bound on abs (s(i) - some true singular value of ' ...
            '(B+E)),\n'...
        '   where E is O(norm(B)*eps) and B = R (for spqr_ssi) and B =\n' ...
        '   A or A*N (for spqr_ssp).\n']) ;
     end

    fprintf ([ ...
    '\nstats.sval_numbers_for_bounds -- component i in the error bounds is ' ...
        'an estimated\n'...
    '   error bound for singular value number sval_numbers_for_bounds(i).\n']) ;

    if ( method == 1 || get_details >= 1 )
        fprintf ([ ...
        '\nstats.est_norm_A_transpose_times_NT -- an estimate of ' ...
            'norm(A''*NT).\n']) ;

        fprintf ( ...
        '\nstats.est_norm_A_times_N -- an estimate of norm(A*N).\n') ;
    end

    if (get_details >= 1)

        fprintf (['\n***** Additional statistics when opts.get_details ' ...
            'is 2: *****\n']) ;

        fprintf ([ ...
        '\nstats.rank_spqr -- the rough estimate of the numerical rank\n'...
        '   computed by spqr.  This is typically correct if the numerical\n'...
        '   rank is well-defined.\n']) ;

        fprintf ('\nstats.stats_ssi -- statistics returned by spqr_ssi.\n') ;

        fprintf ([ ...
        '\nstats_ssi.ssi_max_block_used -- the maximum block size ' ...
               'used by spqr_ssi.\n']) ;

        fprintf ([ ...
        '\nstats_ssi.ssi_min_block_used -- the initial block size ' ...
               'used by spqr_ssi.\n']) ;
    end

    if  (get_details == 1)

        fprintf (['\n***** Additions statistics when opts.get_details is ' ...
            '1: *****\n']) ;

        fprintf ([ ...
        '\nstats.normest_A  -- an estimate of the Euclidean norm of A. '...
            'Calculated using\n' ...
        '   normest(A,0.01).\n']) ;

        fprintf ([ ...
        '\nstats.normest_R  -- an estimate of the Euclidean norm of R. '...
            'Calculated for spqr_ssi\n' ...
        '   using normest(R,0.01).\n']) ;

        fprintf ([ ...
        '\nstats.est_err_bound_norm_A_times_N  -- an estimate of an\n'...
        '   error bound on stats.est_norm_A_times_N.  It is also a\n'...
        '   rigorous bound on abs (stats.est_norm_A_times_N - s)\n'...
        '   where s is some singular value of (A+E)*N and where E is\n' ...
        '   O(norm(A)*eps). Usually the singular value is the first '...
           'singular\n'...
        '   value but this is not guaranteed.\n']) ;

        fprintf ([ ...
        '\nstats.est_err_bound_norm_A_transpose_times_NT  -- an estimate '...
            'of an\n'...
        '   error bound on stats.est_norm_A_transpose_times_NT.  It is '...
            'also a\n'...
        '   rigorous bound on abs (stats.est_norm_A_transpose_times_NT '...
            '- s)\n'...
        '   where s is some singular value of (A+E)''*NT and where E is\n' ...
        '   O(norm(A)*eps). Usually the singular value is the first '...
            'singular\n'...
        '   value but this is not guaranteed.\n']) ;

        fprintf ([ ...
        '\nstats_ssi.norm_R_times_N -- Euclidean norm of (R*N), from ' ...
              'spqr_ssi.\n']) ;

        fprintf ([ ...
        '\nstats_ssi.norm_R_transpose_times_NT -- Eucliean norm of ' ...
              '(R''*NT), from spqr_ssi.\n']) ;

        fprintf ([ ...
        '\nstats_ssi.iters or stats_ssp_N.iters or stats_ssp_NT.iters -- ' ...
            'number of\n' ...
        '    iterations for subspace iteration in spqr_ssi or spqr_ssp.\n']) ;

        fprintf ([ ...
        '\nstats_ssi.nsvals_large_found -- the number of ''large'' (larger ' ...
        'than tol) singular\n' ...
        '   values found, from spqr_ssi.\n']) ;

        fprintf ([ ...
        '\nstats_ssi.final_blocksize -- final block size for subspace ' ...
        'iteration in \n   spqr_ssi. \n']) ;

        fprintf ([ ...
        '\nstats.stats_spqr_basic -- statistics returned when spqr_basic ' ...
        'is called by spqr_pinv.\n']) ;

        fprintf ([ ...
        '\nstats.stats_spqr_null -- statistics returned when spqr_null ' ...
        'is called by spqr_pinv.\n']) ;

        fprintf ([ ...
        '\nstats.info_spqr1 -- statistics from spqr for the first QR ' ...
           'factorization.\n' ...
        '   See ''help spqr'' for details.\n']) ;

        fprintf ([ ...
        '\nstats.info_spqr2 -- statistics from spqr for the second QR ' ...
           'factorization, if\n' ...
        '   required.  See ''help spqr'' for details.\n']) ;

        fprintf ([ ...
        '\nstats.stats_ssp_N -- statistics from spqr_ssp when calculating ' ...
           'the basis\n' ...
        '   N for the null space of A.\n']) ;

        fprintf ([ ...
        '\nstats.stats_ssp_NT -- statistics from spqr_ssp when calculating ' ...
           'the basis\n' ...
        '   NT for the null space of A transpose.\n']) ;

        fprintf ([ ...
        '\nstats.opts_used, stats_ssi.opts_used, or stats_ssp.opts_used -- ' ...
            'values of\n' ...
        '   options used.  These can be different from values in opts ' ...
            'since, for example,\n' ...
        '   the size of A can restrict some values in opts.\n']) ;

        fprintf ([ ...
        '\nstats.time, stats_ssi.time, etc.  -- the total time of the ' ...
           'routine including\n' ...
        '   the times described below.\n']) ;

        fprintf ([ ...
        '\nstats.time_initialize, stats_ssi.time_initialize, etc. -- the '...
            'time to\n' ...
        '   set default values of opts, including calculating ' ...
            'normest(A,0.01),\n' ...
        '   or normest(R,0.01) if needed.\n']) ;

        fprintf([ ...
        '\nstats.time_basis -- the time to compute the basis for the ' ...
            'numerical null space\n' ...
        '   following any calls to spqr and spqr_ssi. This will be small\n'...
        '   if the null space basis is returned in implicit form but can, '...
           'in some cases,\n' ...
        '   be significant if the null space basis is returned as an ' ...
           'explicit matrix.\n']) ;

        fprintf ([ ...
        '\nstats_ssi.time_iters, stats_ssp_N.time_iters, etc.  -- the time ' ...
           'for the\n' ...
        '   subspace iterations in spqr_ssi or spqr_ssp. Excludes time '...
           'for initialization,\n' ...
        '   error flag calculation, etc..\n']) ;

        fprintf ([ ...
        '\nstats_ssi.time_est_error_bounds, ' ...
           'stats_ssp_N.time_est_error_bounds, etc.  -- the time\n' ...
           '   for estimating the singular value error bounds in ' ...
               'spqr_ssi or spqr_ssp.\n']) ;

        fprintf ([ ...
        '\nstats.time_svd, stats_ssi.time_svd, etc.  -- the total time ' ...
           'for calls to MATLAB''s SVD\n' ...
        '   in the current routine and its subroutines.\n']) ;
    end
end

%-------------------------------------------------------------------------------
% print_vector
%-------------------------------------------------------------------------------

function print_vector (x)
n = length (x) ;
for k = 1:n
fprintf (' %g', x (k)) ;
end
fprintf ('\n') ;

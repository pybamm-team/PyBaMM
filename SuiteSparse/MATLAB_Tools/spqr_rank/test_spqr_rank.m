function [ number_fail, SJid_fail ] = test_spqr_rank (ids, figures)
%TEST_SPQR_RANK extensive functionality test of spqr_rank functions
% Returns the number of failures and, optionally, a list of matrices where
% failure occurred.  The first argument can be a negative scalar
% -k, in which case the k smallest matrices in the SJ Collection
% are tested.  Otherwise, the first argument gives a list of matrix IDs
% (as defined by SJget) that are used for the tests. If the optional second
% parameter is zero no plots are produced, if the second parameter is 1 then
% runs which produce no figures and which produce one figure are carried out
% and if the second parameter is 2 then runs producing zero, one and four 
% figures (illustrating numerical ranks, null space accuracy, basic solutions,
% and psuedoinverse solutions) are carried out. The last option is the default
% option. It can be slower than runs with figures = 0 or 1.
%
% Example
%
%   test_spqr_rank ;            % test with 100 smallest sample matrices
%   test_spqr_rank (-200) ;     % test with 200 smallest sample matrices
%   test_spqr_rank (-5, 0) ;    % test with 5 matrices, and do not plot anything
%
%   test_spqr_rank (list) ;     % run tests with a set of matrices.  Each
%                               % matrix is defined by an ID by SJget, and
%                               % 'list' is a vector of matrix IDs to test with.
%
% See also demo_spqr_rank, test_spqr_coverage

% Copyright 2012, Leslie Foster and Timothy A Davis

% Potential run times:
%        test_spqr_rank can require half an hour
%        test_spqr_rank(-200) can require 1.5 hours
%        test_spqr_rank(-200,1) can require an hour
%        test_spqr_rank(-400,0) can require ten hours

if (nargin < 1)
    ids = -100 ;
end
if (isscalar (ids) && ids < 0)
    nsamples_run = -ids ;
else
    nsamples_run = length (ids) ;
end

if (nargin < 2)
    figures = 2 ;
end

nfail = 0 ;

%-------------------------------------------------------------------------------
% extensive tests
%-------------------------------------------------------------------------------

ncases = 0 ;
cnt_fail = 0 ;
SJid_fail = [ ] ;
for figures_to_plot = 0:figures
   demo_opts.figures = figures_to_plot ;
   if figures_to_plot == 2
       null_spaces_limits = 1:2 ; %for figures_to_plot=2, null_spaces>0 required
   else
       null_spaces_limits = 0:2 ;
   end
   for repeatable = 0:1
       demo_opts.repeatable = repeatable;
       for null_spaces = null_spaces_limits
          demo_opts.null_spaces = null_spaces ;
          for start_with_A_transpose = 0:1
             demo_opts.start_with_A_transpose = start_with_A_transpose ;
             for implicit_null_space_basis = 0:1
               demo_opts.implicit_null_space_basis = implicit_null_space_basis ;
               for nsvals = [1 3]
                  demo_opts.nsvals = nsvals ;
                  if (nsamples_run == 1)
                      demo_opts.doprint = -1 ;
                  else
                      demo_opts.doprint = 0 ;
                      fprintf (['\nTest %4d matrices, figures: %d null ' ...
                      'spaces: %d A_trans: %d implicit: %d repeatable: %d ' ...
                      'nsvals: %d\n'], nsamples_run, figures_to_plot, ...
                      null_spaces, start_with_A_transpose, ...
                      implicit_null_space_basis, repeatable, nsvals) ;
                  end
                  demo_opts.tol_norm_type = 0 ; % fixed at 0 to reduce cases
                  [nfail_run, SJid_fail_run] = demo_spqr_rank (ids, demo_opts) ;
                  nfail = nfail + nfail_run ;
                  ncases = ncases + nsamples_run * 4 ;  % 4 for four methods
                  if nfail_run > 0
                     SJid_fail = union(SJid_fail, SJid_fail_run) ;
                     cnt_fail = cnt_fail + 1;
                     % To save statistics files in (rare) case of failure
                     % in files with names demo_spqr_rank_failure_(#).mat
                     % uncomment the following four lines:
                     %dest = ['''demo_spqr_rank_failure_',int2str(cnt_fail),...
                     %    '.mat'')'] ;
                     %com = ['copyfile(''save_samples_demo_spqr_rank.mat'',',...
                     %    dest] ;
                     %eval( com ) ;
                  end
               end
            end
          end
      end
   end
end

if (nsamples_run > 1 || nfail > 0)
    fprintf (['\nTests complete.  Total number of failures: %d for %d ' ...
    'matrix / option choices.\n'], nfail, ncases) ;
    if ( nfail > 0 )
       disp (['Failures for matrices with SJid = ', int2str(SJid_fail),'.']) ;
    end
end

if nargout > 0
    number_fail = nfail ;
end


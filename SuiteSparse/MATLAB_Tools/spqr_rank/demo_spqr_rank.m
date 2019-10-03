function [nfailures SJid_failures] = demo_spqr_rank (ids,args)
%DEMO_SPQR_RANK lengthy demo for spqr_rank functions (requires SJget)
% Usage: demo_spqr_rank(ids,args)
%
% This is a demonstration program for the routines spqr_basic, spqr_null,
% spqr_pinv, spqr_cod discussed in the paper "Algorithm xxx: Reliable
% Calculation of Numerical Rank, Null Space Bases, Pseudoinverse Solutions and
% Basic Solutions using SuiteSparseQR" by Leslie Foster and Timothy Davis,
% submitted ACM Transactions on Mathematical Software, 2011.  Plots similar to
% those in Figures 2 - 5 in the paper are reproduced.
%
% If the first argument ids is a negative scalar, than the smallest (-ids)
% matrices from the SJ Collection are used for the tests.  Otherwise, ids is
% a list of matrix IDs to use for the tests.
%
% If demo_spqr_rank has a second input parameter the second parameter
% controls options in the demonstration program.  For example if the second
% parameter is one then the routine produces plots similar to only those in
% Figure 2 of the paper and if the second parameter is zero then no plots
% are produced. These last two cases run more quickly than the default case.
% The second parameter can also be a structure (for details see the comments
% in the body of the code prior to the first executable statement).
%
% Examples:
%    demo_spqr_rank    % test code for 100 matrices, create 4 figures
% or
%    demo_spqr_rank(-200); % test code for 200 matrices, create 4 figures
% or
%    demo_spqr_rank(-300,1); % test code for 300 matrices, create one figure
%
% See also spqr_basic, spqr_null, spqr_pinv, spqr_cod, SJget.

% Copyright 2012, Leslie Foster and Timothy A Davis.

% Potential run times:
%        demo_spqr_rank can require 20 seconds
%        demo_spqr_rank(-300,1) can require a couple of minutes
%        demo_spqr_rank(-300,2) can require 15 minutes
%        demo_spqr_rank(-640,1) can require 5 hours

% The function creates a file save_samples_demo_spqr_rank containing many
% of the local variable used in demo_spqr_rank.  demo_spqr_rank(0) uses
% the file to redraw plots of the last run of demo_spqr_rank.   Also
%
%   load save_samples_demo_spqr_rank
%
% can be used to load and examine these local variables in the main MATLAB
% workspace.

% The second parameter can be a structure with any of the following fields 
% demo_opts.figures (default 2)
%    0 -- produce no plots
%    1 -- produce plots similar to those in figure 2
%    2 -- produce plots similar to those in figures 2 through 5
% demo_opts.null_spaces (default is 1)
%    0 -- return no null space bases (when allowed)
%    1 -- return one null space basis (of A or of A')
%    2 -- return null spaces bases of A and A' (when allowed)
% demo_opts.doprint (default is 0)
%   -1 -- do not print out anything
%    0 -- only print out a summary of the success of the programs
%    1 -- in demo_spqr_rank print out a summary of the
%         purpose of the demo_spqr_rank, a discussion of the plots
%         produced and information about the success of the programs
% demo_opts.start_with_A_transpose (default is 0)
%    0 -- in spqr_cod start with a QR  factorization of A
%    1 -- in spqr_cod start with a QR factorization of A transpose
% demo_opts.implicit_null_space_basis (default is 1)
%    1 -- return the null space basis in implicit form as Householder
%         transformations
%    0 -- return the null space basis as columns of an explicit matrix
% demo_opts.repeatable (default is 1)
%    1 -- reproduce exactly the same random numbers
%         for each run so that results are repeatable
%    0 -- the random number stream will be different on repeated runs
% demo_opts.nsvals (default is 1)                       
%    the number of small and large singular values to estimate
% demo_opts.get_details (default is 0, but can be modified by demo_opts.figures)
%    0 -- return basic statistics in spqr_basic, spqr_null, etc.
%    1 -- return detailed statistics in spqr_basic, spqr_null, etc.
%    2 -- return basic statistics and a few additional statistics needed
%         for communication between routines
% demo_opts.tol_norm_type (default is 0)
%    0 -- let tol = max( size(A) ) * eps( norm(A) ) where norm(A) is determined
%         using the precomputed singular values in the SJSingular Data Base
%         or (for figures = 2) by MATLAB's dense matrix SVD
%    1 -- compute the default tolerance tol = max( size(A) ) * eps( norm (A,1) )
%    2 -- compute the default tolerance tol = max( size(A) ) *
%                                             eps( normest (A,0.01) )

if (nargin < 1)
    ids = -100;
end

demo_opts = struct('figures', 2, ...
                   'null_spaces', 1, ...
                   'doprint', 1, ...
                   'start_with_A_transpose', 0, ...
                   'implicit_null_space_basis', 1, ...
                   'repeatable', 1 , ...
                   'nsvals', 1 , ...
                   'get_details', 0, ...
                   'tol_norm_type', 0) ;

if (nargin == 2)
    %override default values using second argument
   if isreal(args)
       demo_opts.figures = args ;
   else
       % assumes that args is a structure
        names_args = fieldnames(args) ;
        for i = 1 : length(names_args)
            demo_opts.(names_args{i}) = args.(names_args{i}) ;
        end
   end
end

% values of get_details and null_spaces restricted by demo_opts.figures:
if demo_opts.figures == 1
    demo_opts.get_details = max(1, demo_opts.get_details) ;
elseif demo_opts.figures == 2
    demo_opts.get_details = 1 ;
    demo_opts.null_spaces = max(1,demo_opts.null_spaces) ;
end

% SPQR_BASIC, SPQR_NULL, SPQR_PINV and SPQR_COD return null space bases stored
% in an implicit form by default.  To have the routines return null space bases
% as explicit matrices, use args(4) = 0.  To have the routines return a null
% space bases in the form that requires less memory, use args(4) = 2.

opts.get_details = demo_opts.get_details ;
opts.repeatable = demo_opts.repeatable ;
opts.implicit_null_space_basis = demo_opts.implicit_null_space_basis ;

if (demo_opts.doprint > 0)
    disp(' ')
    disp(['This program demonstrates use of the routines SPQR_BASIC, '...
          'SPQR_NULL, SPQR_PINV,'])
    disp(['and SPQR_COD discussed in the paper "Algorithm xxx: ',...
          'Reliable Calculation of'])
    disp(['Numerical Rank, Null Space Bases, Pseudoinverse Solutions and ',...)
          ' Basic Solutions'])
    disp(['using SuiteSparseQR" by Leslie Foster and Timothy Davis, ',...
         'submitted ACM'])
    disp(['Transactions on Mathematical Software, 2011.  Plots similar ',...
           'to those in Figures'])
    disp(['2 - 5 or, optionally, just Figure 2 in the paper are ',...
          'reproduced, except the'])
    disp(['sample set is restricted to small matrices so that the demo ',...
          'runs quickly. The'])
    disp(['matrices come from the San Jose State University Singular ',...
          'Matrix Database.'])
    disp(' ')
    disp('The routines are designed to work with rank deficient matrices.')
    disp('The primary use of each routine is:')
    disp('    SPQR_BASIC -- determine a basic solution to min ||b - A x||')
    disp(['    SPQR_NULL  -- determine an orthonormal basis for the ',...
          'numerical nullspace'])
    disp('                  of A')
    disp(['    SPQR_PINV  -- determine a pseudoinverse or mininimum ',...
          'norm solutio to'])
    disp('                  min || b - A x||')
    disp(['    SPQR_COD   -- determine a pseudoinverse or mininimum norm ',...
          'solution to'])
    disp(['                  min || b - A x|| using a complete orthogonal ',...
          'decomposition.'])
    disp('The demonstration program creates plots that illustrate the accuracy')
    disp('of rank determination, the accuracy of the null space bases, the')
    disp(['accuracy of the basic solutions and the accuracy of the ',...
          'pseudoinverse'])
    disp('solutions.  The above routines are compared with the calculations')
    disp('using MATLAB''s SVD, MATLAB''s dense matrix QR factorization and')
    disp(['with SPQR_SOLVE, part of SuiteSparseQR. In the demonstration ',...
          'program'])
    disp('the tolerance defining the numerical rank is min(m,n)*eps(||A||)')
    disp('where the matrix A is m by n.')
    %disp(' ')
    % if (demo_opts.dopause)
    %     disp('Press enter to begin demonstration')
    %     pause
    % end
    disp(' ')
end

mfilename ('fullpath') ;
install_SJget ;

% intitialze
index = SJget;

if (isscalar (ids) && ids < 0)
    % test with matrices 1 to (-ids)
    dim = max (index.nrows, index.ncols) ;
    % R2009b introduced '~' to denote unused output arguments, but we avoid that
    % feature so that this code can run on R2008a (and perhaps earlier).
    [ignore,indexs] = sort (dim) ;                                          %#ok
    clear ignore
    indexs = indexs (1:(-ids)) ;

elseif (isscalar (ids) && (ids == 0))

    % the file save_samples_demo_spqr_rank is created
    %    when demo_spqr_rank is run
    if exist('save_samples_demo_spqr_rank.mat','file')
        load save_samples_demo_spqr_rank
    else
        error (['prior to running demo_spqr_rank(0) run ' ...
                'demo_spqr_rank(ids) with ids < 0 or ids a list of IDs']) ;
    end
    % demo_spqr_rank(0) can be used to redraw plots of the last
    % run of demo_spqr_rank.   Also the command
    % load save_samples_demo_spqr_rank
    % can be used to load many of the local variables used in
    % demo_spqr_rank into the main MATLAB workspace.

else
    % list of matrix ID's has been passed in directly
    indexs = ids ;
end

cnt = 0;
time_start = clock;

%-------------------------------------------------------------------------------
% allocate space for vectors containing statistics calculated
%-------------------------------------------------------------------------------

nothing = -ones (1,length (indexs)) ;
rank_svd_v = nothing ;
gap_v = nothing ;
m_v = nothing ;
n_v = nothing ;
flag_spqr_basic_v = nothing ;
rank_svd_basic_v = nothing ;
rank_spqr_basic_v = nothing ;
rank_spqr_from_spqr_basic_v = nothing ;
flag_spqr_null_v = nothing ;
rank_svd_null_v = nothing ;
rank_spqr_null_v = nothing ;
rank_spqr_from_spqr_null_v = nothing ;
flag_spqr_pinv_v = nothing ;
rank_svd_pinv_v = nothing ;
rank_spqr_pinv_v = nothing ;
rank_spqr_from_spqr_pinv_v = nothing ;
flag_spqr_cod_v = nothing ;
rank_svd_cod_v = nothing ;
rank_spqr_cod_v = nothing ;
rank_spqr_from_spqr_cod_v = nothing ;
tol_v = nothing ;
norm_A_v = nothing ;
norm_A_N_svd_v = nothing ;
norm_A_NT_svd_v = nothing ;
norm_A_NT_spqr_basic_v = nothing ;
norm_A_N_spqr_null_v = nothing ;
norm_A_N_spqr_pinv_v = nothing ;
norm_A_N_spqr_cod_v = nothing ;
norm_x_pinv_v = nothing ;
norm_x_QR_dense_v = nothing ;
norm_r_QR_dense_v = nothing ;
norm_x_spqr_basic_v = nothing ;
norm_r_spqr_basic_v = nothing ;
norm_x_spqr_solve_v = nothing ;
norm_r_spqr_solve_v = nothing ;
norm_x_spqr_pinv_minus_x_pinv_v = nothing ;
norm_x_spqr_cod_minus_x_pinv_v = nothing ;
cond1_pinv_v = nothing ;
cond1_cod_v = nothing ;
norm_w_cod_v = nothing ;
norm_w_pinv_v = nothing ;

% ignore warnings from inverse power method in ssi
user_warning_state = warning ('off', 'spqr_rank:overflow') ;

% begin the calculations

if (demo_opts.doprint > 0)
    disp('Begin calculations')
    disp(' ')
end

for i = indexs

    cnt = cnt + 1;

    if (demo_opts.doprint > 0)
        fprintf ('.') ;
        if (mod (cnt,50) == 0)
            fprintf ('\n') ;
        end
    end

    %---------------------------------------------------------------------------
    % generate the problem
    %---------------------------------------------------------------------------

    % select matrix from SJSU singular matrices
    Problem = SJget(i,index) ;
    A=Problem.A ;
    [m,n]=size(A);
    private_stream = spqr_repeatable (opts.repeatable) ;
    if (~isempty (private_stream))
        b1 = randn (private_stream, m, 1) ;
        x2 = randn (private_stream, n, 1) ;
    else
        b1 = randn (m,1) ;
        x2 = randn (n,1) ;
    end
    b2 = A*x2;         % consistent right hand side
    b = [ b1, b2 ];

    %---------------------------------------------------------------------------
    % for demo_opts.figures == 2 find numerical rank, null space basis, 
    %     pseudoinverse soln using MATLAB svd
    % for demo_opts.figures <= 1 find numerical rank using precomputed sing.
    %     values in SJsingular database
    %---------------------------------------------------------------------------

    if demo_opts.figures <= 1
        % use the precomputed singular values from the SJsingular database
        s = Problem.svals;
        normA = max(s);
        tol = max(m,n) * eps( normA );
        rank_svd = SJrank(Problem,tol);
    else
        [U,S,V] = svd(full(A));
        if m > 1, s = diag(S);
        elseif m == 1
            s = S(1);
        else
            s = 0;
        end
        normA= max(s);
        tol = max(m,n) * eps(normA);
        rank_svd = sum(s > tol);
    end

    if (rank_svd == 0)
        gap = NaN;
    else
       if rank_svd < min(m,n)
           gap = s(rank_svd) / abs(s(rank_svd + 1));
       else
           gap = Inf;
       end
    end

    if demo_opts.figures == 2
        if (rank_svd == 0)
           pseudoinverse_svd = zeros(size(A'));
           N_svd = eye(n,n);
           NT_svd = eye(m,m);
        else
           S_pinv = diag(ones(rank_svd,1)./s(1:rank_svd));
           pseudoinverse_svd = V(:,1:rank_svd)*S_pinv*U(:,1:rank_svd)';
           N_svd = V(:,rank_svd+1:end);
           NT_svd = U(:,rank_svd+1:end);
        end
        x_pinv = pseudoinverse_svd * b(:,1);  % the plots only use x_pinv
                                              % for b(:,1), the random rhs
        norm_x_pinv = norm(x_pinv);
        norm_A_N_svd = norm(A*N_svd);
        norm_A_transpose_NT_svd = norm(A'*NT_svd);

        % find basic solution using MATLAB's dense QR routine
        warning_state = warning ('off','MATLAB:rankDeficientMatrix') ;
        if m ~= n
            x = full(A) \ b;
        else
            x = full([A,zeros(m,1)]) \ b;
            x = x(1:n,:);
        end
        warning (warning_state) ;           % restore the warnings

        norm_x_QR_dense = norm(x(:,1)) ;  %the plots only use ||x||
                                          % for the random rhs b(:,1)
        r = b(:,2)- A*x(:,2);             % the plots only use r for
                                          % the consistent rhs b(:,2)
        norm_r_QR_dense = norm(r) / norm(b(:,2));
    end

    %---------------------------------------------------------------------------
    % run spqr_basic, spqr_null, spqr_pinv and spqr_cod:
    %---------------------------------------------------------------------------

    if demo_opts.tol_norm_type == 0
       opts.tol = tol;
    else
       opts.tol_norm_type = demo_opts.tol_norm_type;
       % calculate tol inside spqr_basic, spqr_null, spqr_pinv and spqr_cod
    end

    % SPQR_COD uses a complete orthogonal decomposition (COD) of A.  By
    % default the COD is constructed by first factoring A. To select the
    % option for SPQR_COD which initially factors A', which can have
    % different numerical properties, use args(1) = 1.

    opts.start_with_A_transpose = demo_opts.start_with_A_transpose ;
    opts.nsvals_small = demo_opts.nsvals;
    opts.nsvals_large = demo_opts.nsvals;

    if demo_opts.null_spaces == 2
        [x_spqr_basic, stats_basic, NT_spqr_basic] = spqr_basic (A,b,opts) ;
        [N_spqr_null, stats_null] = spqr_null (A,opts) ;
        [x_spqr_pinv, stats_pinv, N_spqr_pinv, NT] = spqr_pinv (A,b,opts) ; %#ok
        [x_spqr_cod, stats_cod, N_spqr_cod, NT] = spqr_cod (A,b,opts) ;     %#ok
    elseif demo_opts.null_spaces == 1
        [x_spqr_basic, stats_basic, NT_spqr_basic] = spqr_basic (A,b,opts) ;
        [N_spqr_null, stats_null] = spqr_null (A,opts) ;
        [x_spqr_pinv, stats_pinv, N_spqr_pinv] = spqr_pinv (A,b,opts) ;
        [x_spqr_cod, stats_cod, N_spqr_cod] = spqr_cod (A,b,opts) ;
    else
        [x_spqr_basic, stats_basic] = spqr_basic (A,b,opts) ;
        [N_spqr_null, stats_null] = spqr_null (A,opts) ;
        [x_spqr_pinv, stats_pinv] = spqr_pinv (A,b,opts) ;
        [x_spqr_cod, stats_cod] = spqr_cod (A,b,opts) ;
    end

    %---------------------------------------------------------------------------
    % calculate and save results for figures displaying ranks:
    %---------------------------------------------------------------------------

    rank_svd_v(cnt) = rank_svd;
    gap_v(cnt) = gap;
    m_v(cnt) = m;
    n_v(cnt) = n;

    % spqr_basic results:
    flag_spqr_basic_v(cnt) = stats_basic.flag;
    if stats_basic.flag == 1
        % use tol_alt returned by spqr_basic to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_basic_v(cnt)=  sum(s > stats_basic.tol_alt);
        else
            rank_svd_basic_v(cnt)=  SJrank(Problem,stats_basic.tol_alt);
        end
    else
        % rank_svd_basic_v(cnt)=  rank_svd_v(cnt);
        % use tol returned by spqr_basic to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_basic_v(cnt)=  sum(s > stats_basic.tol);
        else
            rank_svd_basic_v(cnt)=  SJrank(Problem,stats_basic.tol);
        end
    end
    if stats_basic.flag <= 3
        rank_spqr_basic_v(cnt) = stats_basic.rank ;
    end
    if demo_opts.figures >= 1
        rank_spqr_from_spqr_basic_v(cnt) = stats_basic.rank_spqr ;
    else
        rank_spqr_from_spqr_null_v(cnt) = -1;   % not calculated
    end

    % spqr_null results:
    flag_spqr_null_v(cnt) = stats_null.flag;
    if stats_null.flag == 1
        % use tol_alt returned by spqr_null to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_null_v(cnt) =  sum(s > stats_null.tol_alt);
        else
            rank_svd_null_v(cnt) =  SJrank(Problem,stats_null.tol_alt);
        end
    else
        % rank_svd_null_v(cnt)=  rank_svd_v(cnt);  
        % use tol returned by spqr_null to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_null_v(cnt)=  sum(s > stats_null.tol);
        else
            rank_svd_null_v(cnt)=  SJrank(Problem,stats_null.tol);
        end
    end
    if stats_null.flag <= 3
        rank_spqr_null_v(cnt) = stats_null.rank ;
    end
    if demo_opts.figures >= 1
        rank_spqr_from_spqr_null_v(cnt) = stats_null.rank_spqr ;
    else
        rank_spqr_from_spqr_null_v(cnt) = -1 ;    % not calculated
    end

    % spqr_pinv results:
    flag_spqr_pinv_v(cnt) = stats_pinv.flag;
    if stats_pinv.flag == 1
        % use tol_alt returned by spqr_pinv to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_pinv_v(cnt) =  sum(s > stats_pinv.tol_alt);
        else
           rank_svd_pinv_v(cnt) =  SJrank(Problem,stats_pinv.tol_alt);
        end
    else
        % rank_svd_pinv_v(cnt)=  rank_svd_v(cnt);
        % use tol returned by spqr_pinv to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_pinv_v(cnt)=  sum(s > stats_pinv.tol);
        else
            rank_svd_pinv_v(cnt)=  SJrank(Problem,stats_pinv.tol);
        end
        
    end
    if stats_pinv.flag <= 3
        rank_spqr_pinv_v(cnt) = stats_pinv.rank ;
    end
    if demo_opts.figures >= 1
        rank_spqr_from_spqr_pinv_v(cnt) = stats_pinv.rank_spqr ;
    else
        rank_spqr_from_spqr_pinv_v(cnt) = -1 ;     % not calculated
    end

    % spqr_cod results:
    flag_spqr_cod_v(cnt) = stats_cod.flag;
    if stats_cod.flag == 1
        % use tol_alt returned by spqr_cod to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_cod_v(cnt)=  sum(s > stats_cod.tol_alt);
        else
            rank_svd_cod_v(cnt) =  SJrank(Problem,stats_cod.tol_alt);
        end
    else
        % rank_svd_cod_v(cnt)=  rank_svd_v(cnt);
        % use tol returned by spqr_cod to determine true num. rank
        if demo_opts.figures == 2
            rank_svd_cod_v(cnt)=  sum(s > stats_cod.tol);
        else
            rank_svd_cod_v(cnt)=  SJrank(Problem,stats_cod.tol);
        end
        
    end
    if stats_cod.flag <= 3
        rank_spqr_cod_v(cnt) = stats_cod.rank ;
    end
    if demo_opts.figures >= 1
        rank_spqr_from_spqr_cod_v(cnt) = stats_cod.rank_spqr ;
    else
        rank_spqr_from_spqr_cod_v(cnt) = -1 ;     % not calculated
    end

    tol_v(cnt) = tol;
    norm_A_v(cnt) = normA;

    if demo_opts.figures == 2

        %----------------------------------------------------------------------
        % calculate and save results for figures displaying null space accuracy
        %----------------------------------------------------------------------

        norm_A_N_svd_v(cnt) = norm_A_N_svd;
        norm_A_NT_svd_v(cnt) = norm_A_transpose_NT_svd;

        if stats_basic.flag <= 3
           % use spqr_null_mult to form NT_spqr_basic' * A
           A_transpose_times_NT = spqr_null_mult(NT_spqr_basic,A,0);
           norm_A_NT_spqr_basic_v(cnt) = norm(full( A_transpose_times_NT ));
           % For large matrices, rather than forming A_transpose_times_NT,
           % it is more efficient to use the estimate of
           % ||A' * NT_spqr_basic|| calculated by SPQR_BASIC (using spqr_ssp)
           % when SPQR_BASIC returns a null space bases:
           % norm_A_NT_spqr_basic_v(cnt) = ...
           %            stats_basic.est_norm_A_transpose_times_NT;
        end

        if stats_null.flag <= 3
           % use spqr_null_mult to form  A * N_spqr_null
           A_times_N = spqr_null_mult(N_spqr_null,A,3);
           norm_A_N_spqr_null_v(cnt) = norm(full( A_times_N ));
           % For large matrices, rather than forming A_times_N,
           % it is more efficient to use the estimate of
           % ||A * N|| calculated by SPQR_NULL (using spqr_ssp):
           % norm_A_N_spqr_null_v(cnt) = stats_null.est_norm_A_times_N;
        end

        if stats_pinv.flag <= 3
           % use spqr_null_mult to form  A * N_spqr_pinv
           A_times_N = spqr_null_mult(N_spqr_pinv,A,3);
           norm_A_N_spqr_pinv_v(cnt) = norm(full( A_times_N ));
           % For large matrices, rather than forming A_times_N,
           % it is more efficient to use the estimate of
           % ||A * N|| calculated by SPQR_PINV (using spqr_ssp):
           % norm_A_N_spqr_pinv_v(cnt) = stats_pinv.est_norm_A_times_N;
        end

        if stats_cod.flag <= 3
           % use spqr_null_mult to form  A * N_spqr_cod
           A_times_N = spqr_null_mult(N_spqr_cod,A,3);
           norm_A_N_spqr_cod_v(cnt) = norm(full( A_times_N ));
           % For large matrices, rather than forming A_times_N,
           % it is more efficient to use the estimate of
           % ||A * N|| calculated by SPQR_COD (using spqr_ssp):
           % norm_A_N_spqr_cod_v(cnt) = stats_cod.est_norm_A_times_N;
        end

        %----------------------------------------------------------------------
        % calculate and save results for basic solutions plots
        %----------------------------------------------------------------------

        % norm_x_pinv, norm_x_QR_dense and norm_r_QR_dense already computed
        norm_x_pinv_v(cnt) = norm_x_pinv;
        norm_x_QR_dense_v(cnt) = norm_x_QR_dense;
        norm_r_QR_dense_v(cnt) = norm_r_QR_dense;

        norm_x_spqr_basic_v(cnt) = norm(x_spqr_basic(:,1)) ;
        norm_r_spqr_basic_v(cnt) = norm(b(:,2) - A * x_spqr_basic(:,2)) / ...
             norm(b(:,2));
        opts_spqr_solve.tol = stats_basic.tol;
        warning_state = warning ('off','MATLAB:rankDeficientMatrix') ;
        x_spqr_solve = spqr_solve (sparse(A), b, opts_spqr_solve) ;
        warning (warning_state) ;
        norm_x_spqr_solve_v(cnt) = norm(x_spqr_solve(:,1));
        norm_r_spqr_solve_v(cnt) = norm(b(:,2) - A * x_spqr_solve(:,2)) / ...
            norm(b(:,2));

        %----------------------------------------------------------------------
        % calculate and save results for pinv solutions plots
        %----------------------------------------------------------------------

        if stats_pinv.flag <= 3
            norm_x_spqr_pinv_minus_x_pinv_v(cnt) = ...
               norm(x_spqr_pinv(:,1)-x_pinv) / norm(x_pinv);
        end

        if stats_cod.flag <= 3
            norm_x_spqr_cod_minus_x_pinv_v(cnt) = ...
               norm(x_spqr_cod(:,1)-x_pinv) / norm(x_pinv);
        end

        cond1_cod_v(cnt) = s(1) / s( rank_svd_cod_v(cnt) );
        cond1_pinv_v(cnt) = s(1) / s( rank_svd_pinv_v(cnt) );
        norm_w_cod_v(cnt) = stats_cod.info_spqr1.norm_E_fro ;    
        norm_w_pinv_v(cnt) = max( ...
           stats_pinv.stats_spqr_basic.info_spqr1.norm_E_fro, ...
           stats_pinv.stats_spqr_null.info_spqr1.norm_E_fro) ;
    end

    %---------------------------------------------------------------------------
    % save results to a *.mat file for future reference
    %---------------------------------------------------------------------------

    if ( mod(cnt,10) == 0 || cnt == length(indexs)  && cnt > 1)
        % save the information needed to draw the plots;
        % if later the code is run with ids = 0
        %      save_samples_demo_spqr_rank will be loaded
        time_required = etime(clock,time_start);
        save save_samples_demo_spqr_rank ...
             rank_svd_basic_v rank_spqr_basic_v flag_spqr_basic_v ...
             rank_svd_null_v  rank_spqr_null_v  flag_spqr_null_v ...
             rank_svd_pinv_v  rank_spqr_pinv_v  flag_spqr_pinv_v ...
             rank_svd_cod_v   rank_spqr_cod_v   flag_spqr_cod_v  ...
             rank_spqr_from_spqr_basic_v gap_v tol_v ...
             rank_spqr_from_spqr_null_v ...
             rank_spqr_from_spqr_pinv_v ...
             rank_spqr_from_spqr_cod_v ...
             norm_A_NT_svd_v norm_A_NT_spqr_basic_v ...
             norm_A_N_svd_v norm_A_N_spqr_null_v ...
             norm_A_N_spqr_pinv_v ...
             norm_A_N_spqr_cod_v ...
             norm_x_pinv_v norm_x_QR_dense_v  norm_r_QR_dense_v ...
             norm_x_spqr_basic_v  norm_r_spqr_basic_v ...
             norm_x_spqr_solve_v  norm_r_spqr_solve_v ...
             norm_x_spqr_pinv_minus_x_pinv_v ...
             norm_x_spqr_cod_minus_x_pinv_v cond1_pinv_v cond1_cod_v ...
             norm_w_cod_v  norm_w_pinv_v norm_A_v ...
             cnt time_required indexs m_v n_v rank_svd_v ...
             demo_opts
    end

end

% restore user's warning state
warning (user_warning_state) ;

% if (demo_opts.dopause)
%     disp('Press enter to see the first plot')
%     pause
% end

if demo_opts.figures >= 1
    %---------------------------------------------------------------------------
    % plot information for calculated ranks
    %---------------------------------------------------------------------------

    figure(1)
    subplot(2,2,1)
    plot_ranks(rank_svd_basic_v,rank_spqr_basic_v,...
        rank_spqr_from_spqr_basic_v,flag_spqr_basic_v,gap_v,'SPQR\_BASIC')
    subplot(2,2,2)
    plot_ranks(rank_svd_null_v,rank_spqr_null_v,rank_spqr_from_spqr_null_v,...
        flag_spqr_null_v,gap_v,'SPQR\_NULL')
    subplot(2,2,3)
    plot_ranks(rank_svd_pinv_v,rank_spqr_pinv_v,rank_spqr_from_spqr_pinv_v,...
        flag_spqr_pinv_v,gap_v,'SPQR\_PINV')
    subplot(2,2,4)
    plot_ranks(rank_svd_cod_v,rank_spqr_cod_v,rank_spqr_from_spqr_cod_v,...
        flag_spqr_cod_v,gap_v,'SPQR\_COD')

    if (demo_opts.doprint > 0)
        disp(' ')
        if demo_opts.figures == 1
           disp('In the figure for each of SPQR_BASIC, SPQR_NULL, ')
        else
           disp('In the first figure for each of SPQR_BASIC, SPQR_NULL, ')
        end
        disp('SPQR_PINV, SPQR_COD and for SPQR the plots summarize the percent')
        disp('of matrices where the calculated numerical rank is correct and')
        disp('the percent of the matrices where the warning flag indicates')
        disp('that the calculated numerical rank is correct with a warning')
        disp('flag either 0 or 1 versus the singular value gap, the ratio of')
        disp('singular number r over singular value number r+1, where r is the')
        disp('calculated numerical rank.')
        disp(' ')
        disp('Note that the percent of the matrices where the new routines ')
        disp('calculate the correct numerical rank approaches 100 percent')
        disp('as the singular value gap increases.')
        disp(' ')
        disp('The plot is best seen as a full screen plot.')
        disp(' ')
        % if (demo_opts.dopause)
        %     disp('Press enter to view the second figure')
        %     pause
        % end
    end
end

drawnow

if demo_opts.figures == 2

    %---------------------------------------------------------------------------
    % plot information for null spaces
    %---------------------------------------------------------------------------

    figure(2)
    subplot(2,2,1)
    plot_null_spaces(norm_A_NT_svd_v,tol_v, ...
           norm_A_NT_spqr_basic_v,flag_spqr_basic_v,'SPQR\_BASIC')
    subplot(2,2,2)
    plot_null_spaces(norm_A_N_svd_v,tol_v, ...
           norm_A_N_spqr_null_v,flag_spqr_null_v,'SPQR\_NULL')
    subplot(2,2,3)
    plot_null_spaces(norm_A_N_svd_v,tol_v, ...
           norm_A_N_spqr_pinv_v,flag_spqr_pinv_v,'SPQR\_PINV')
    subplot(2,2,4)
    plot_null_spaces(norm_A_N_svd_v,tol_v, ...
           norm_A_N_spqr_cod_v,flag_spqr_cod_v,'SPQR\_COD')

    if (demo_opts.doprint > 0)
        disp(' ')
        disp('In the second figure ||AN||, where N is a calculated')
        disp('orthonormal basis for the numerical null space, or, in the case')
        disp('of SPQR_BASIC, ||(transpose of A) N||, normalized by the ')
        disp('tolerance defining the numerical rank, is plotted for null')
        disp('space bases calculated by MATLAB''s SVD and by SPQR_BASIC,')
        disp('SPQR_NULL, SPQR_PINV and SPQR_COD.')
        disp(' ')
        disp('Note that the null space bases calculated by the new routines')
        disp('are generally as good as the bases calculated by MATLAB''s SVD.')
        disp('The tolerance used for normalization in the plots is ')
        disp('O(relative machine precision times ||A||).')
        disp(' ')
        disp('The plot is best seen as a full screen plot.')
        disp(' ')
        % if (demo_opts.dopause)
        %     disp('Press enter to view the third figure')
        %     pause
        % end
    end

    drawnow

    %---------------------------------------------------------------------------
    % plot information for basic solutions
    %---------------------------------------------------------------------------

    figure(3)
    plot_basic(norm_x_pinv_v,norm_x_QR_dense_v, norm_r_QR_dense_v, ...
        norm_x_spqr_basic_v, norm_r_spqr_basic_v, ...
        norm_x_spqr_solve_v, norm_r_spqr_solve_v,flag_spqr_basic_v)

    if (demo_opts.doprint)
        disp(' ')
        disp('In the third figure the left plot pictures ||x|| / ||x_PINV|| ')
        disp('where x is a basic solution to min || b - A x ||')
        disp('calculated by MATLAB''s dense matrix QR algorithm,')
        disp('by SPQR_SOLVE or by SPQR_BASIC. SPQR_SOLVE is part of')
        disp('SuiteSparseQR and can be used to construct basic solutions to')
        disp('min ||b - A x||. x_PINV is computed using MATLAB''s PINV. In the')
        disp('left hand plot the vectors b in min || b - A x || are random')
        disp('vectors. The right plot pictures  ||r|| = || b - A x || for x')
        disp('vectors calculated using MATLAB''s dense matrix QR algorithm,')
        disp('by SPQR_SOLVE or by SPQR_BASIC.  In the right hand plot the')
        disp('vectors b in min || b - A x || are of the form b = Ax where x')
        disp('is a random vector.')
        disp(' ')
        disp('In the left hand plot note that for most, but not all, of the')
        disp('matrices the norm of the basic solution calculated by SPQR_BASIC')
        disp('or by SPQR_SOLVE is the same order of magnitude as the norm of')
        disp('the pseudoinverse solution. Also note that SPQR_SOLVE calculates')
        disp('a large norm solution more frequently than does SPQR_BASIC.')
        disp(' ')
        disp('In the right hand plot note that for most, but not all, matrices')
        disp('the residuals for solutions calculated by SPQR_BASIC or by ')
        disp('SPQR_SOLVE are similar in size to the residual for solutions')
        disp('calcluated by MATLAB''s dense QR factorization. ')
        disp(' ')
        disp('The plot is best seen as a full screen plot.')
        disp(' ')
        % if (demo_opts.dopause)
        %     disp('Press enter to view the fourth figure.')
        %     pause
        % end
    end

    %---------------------------------------------------------------------------
    % plot information for pseudoinverse solutions
    %---------------------------------------------------------------------------

    figure(4)
    plot_pinv(norm_x_spqr_pinv_minus_x_pinv_v, ...
        norm_x_spqr_cod_minus_x_pinv_v, cond1_pinv_v, cond1_cod_v, ...
        norm_w_pinv_v, norm_w_cod_v, flag_spqr_pinv_v,  ... % tol_v, ...
        flag_spqr_cod_v, norm_A_v)

    if (demo_opts.doprint > 0)
        disp(' ')
        disp('In the fourth figure the left graph  plots || x - x_PINV ||')
        disp(' / ||x_PINV|| for x produced by SPQR_PINV for the matrices')
        disp('where SPQR_PINV returns a flag of 0. x_PINV is')
        disp('calculated using MATLAB''s PINV routine. Also part of a')
        disp('perturbation theory result from "Matrix Perturbation Theory" by')
        disp('Stewart and Sun, page 157, is plotted. The right hand graph')
        disp('is the same plot for x produced by SPQR_COD for the matrices')
        disp('where SPQR_COD returns a flag of 0. For the plots the ')
        disp('vectors b in min||b-Ax|| are random vectors')
        disp(' ')
        disp('Note that the accuracies of the pseudoinverse solutions')
        disp('calculated by SPQR_COD and, in most cases, by SPQR_PINV are as')
        disp('good as or nearly as good as predicted by the perturbation')
        disp('theory.')
        disp(' ')
        disp('The plot is best seen as a full screen plot.')
    end

    drawnow

end

if (demo_opts.doprint >= 0)
    fprintf ('\n') ;
end
failures = 0 ;

%-------------------------------------------------------------------------------
% check that numerical rank calculations are accurate
%-------------------------------------------------------------------------------

ifail_spqr_basic = find( rank_svd_basic_v ~= rank_spqr_basic_v & ...
    (flag_spqr_basic_v == 0 | flag_spqr_basic_v == 1) );
ifail_spqr_null = find( rank_svd_null_v ~= rank_spqr_null_v & ...
    (flag_spqr_null_v == 0 | flag_spqr_null_v == 1) );
ifail_spqr_pinv = find( rank_svd_pinv_v ~= rank_spqr_pinv_v & ...
    (flag_spqr_pinv_v == 0 | flag_spqr_pinv_v == 1) );
ifail_spqr_cod = find( rank_svd_cod_v ~= rank_spqr_cod_v & ...
    (flag_spqr_cod_v == 0 | flag_spqr_cod_v == 1) );

nfail_spqr_basic = length(ifail_spqr_basic);
nfail_spqr_null = length(ifail_spqr_null);
nfail_spqr_pinv = length(ifail_spqr_pinv);
nfail_spqr_cod = length(ifail_spqr_cod);

SJid_fail_spqr_basic = sort( indexs( ifail_spqr_basic ) );
SJid_fail_spqr_null = sort( indexs( ifail_spqr_null ) );
SJid_fail_spqr_pinv = sort( indexs( ifail_spqr_pinv ) );
SJid_fail_spqr_cod = sort( indexs( ifail_spqr_cod ) );

SJid_fail = union(SJid_fail_spqr_basic, SJid_fail_spqr_null) ; 
SJid_fail = union(SJid_fail, SJid_fail_spqr_pinv) ;
SJid_fail = union(SJid_fail, SJid_fail_spqr_cod) ;

if (demo_opts.doprint > 0)
    disp(' ')
    disp('Check that the routines reliably calculate the numerical rank.')
end

if (demo_opts.doprint >= 0)
    fprintf ('SPQR_BASIC: ') ;
end
iflagis0_or_1 = find(flag_spqr_basic_v == 0 | flag_spqr_basic_v == 1);
nflagis0_or_1 = length(iflagis0_or_1);

failures = failures + nfail_spqr_basic ;

if (demo_opts.doprint > 0)
    disp(['   ',int2str(nflagis0_or_1 - nfail_spqr_basic),...
        ' matrices have the correct numerical rank from the set of'])
    if ( nfail_spqr_basic == 0 )
        disp(['   ',int2str(nflagis0_or_1),...
            ' matrices with a warning flag of 0 or 1.'])
    elseif ( nfail_spqr_basic == 1 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrix from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_basic),'.'])
    elseif ( nfail_spqr_basic >= 2 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrices from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_basic),'.'])
    end
elseif (demo_opts.doprint >= 0)
    if ( nfail_spqr_basic == 0 )
        fprintf ('%3d matrices, failed: %d\n', nflagis0_or_1, ...
            nfail_spqr_basic) ;
    else
        fprintf ('%3d matrices, failed: %d with SJid =', ...
           nflagis0_or_1, nfail_spqr_basic) ;
       fprintf(' %d', SJid_fail_spqr_basic ) ;
       fprintf('\n') ;
    end
end

if (demo_opts.doprint >= 0)
    fprintf ('SPQR_NULL:  ') ;
end
iflagis0_or_1 = find(flag_spqr_null_v == 0 | flag_spqr_null_v == 1);
nflagis0_or_1 = length(iflagis0_or_1);

failures = failures + nfail_spqr_null ;

if (demo_opts.doprint > 0)
    disp(['   ',int2str(nflagis0_or_1 - nfail_spqr_null),...
        ' matrices have the correct numerical rank from the set of'])
    if ( nfail_spqr_null == 0 )
        disp(['   ',int2str(nflagis0_or_1),...
            ' matrices with a warning flag of 0 or 1.'])
    elseif ( nfail_spqr_null == 1 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrix from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_null),'.'])
    elseif ( nfail_spqr_null >= 2 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrices from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_null),'.'])
    end
elseif (demo_opts.doprint >= 0)
    if ( nfail_spqr_null == 0 )
        fprintf ('%3d matrices, failed: %d\n', nflagis0_or_1, ...
            nfail_spqr_null) ;
    else
        fprintf ('%3d matrices, failed: %d with SJid =', ...
           nflagis0_or_1, nfail_spqr_null) ;
       fprintf(' %d', SJid_fail_spqr_null ) ;
       fprintf('\n') ;
    end
end

if (demo_opts.doprint >= 0)
    fprintf ('SPQR_PINV:  ') ;
end
iflagis0_or_1 = find(flag_spqr_pinv_v == 0 | flag_spqr_pinv_v == 1);
nflagis0_or_1 = length(iflagis0_or_1);

failures = failures + nfail_spqr_pinv ;

if (demo_opts.doprint > 0)
    disp(['   ',int2str(nflagis0_or_1 - nfail_spqr_pinv),...
        ' matrices have the correct numerical rank from the set of'])
    if ( nfail_spqr_pinv == 0 )
        disp(['   ',int2str(nflagis0_or_1),...
            ' matrices with a warning flag of 0 or 1.'])
    elseif ( nfail_spqr_pinv == 1 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrix from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_pinv),'.'])
    elseif ( nfail_spqr_pinv >= 2 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrices from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_pinv),'.'])
    end
elseif (demo_opts.doprint >= 0)
    if ( nfail_spqr_pinv == 0 )
        fprintf ('%3d matrices, failed: %d\n', nflagis0_or_1, ...
            nfail_spqr_pinv) ;
    else
        fprintf ('%3d matrices, failed: %d with SJid =', ...
           nflagis0_or_1, nfail_spqr_pinv) ;
       fprintf(' %d', SJid_fail_spqr_pinv ) ;
       fprintf('\n') ;
    end
end

if (demo_opts.doprint >= 0)
    fprintf ('SPQR_COD:   ') ;
end
iflagis0_or_1 = find(flag_spqr_cod_v == 0 | flag_spqr_cod_v == 1);
nflagis0_or_1 = length(iflagis0_or_1);

failures = failures + nfail_spqr_cod ;

if (demo_opts.doprint > 0)
    disp(['   ',int2str(nflagis0_or_1 - nfail_spqr_cod),...
        ' matrices have the correct numerical rank from the set of'])
    if ( nfail_spqr_cod == 0 )
        disp(['   ',int2str(nflagis0_or_1),...
            ' matrices with a warning flag of 0 or 1.'])
    elseif ( nfail_spqr_cod == 1 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrix from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_cod),'.'])
    elseif ( nfail_spqr_cod >= 2 )
        disp(['   ',int2str(nflagis0_or_1),' matrices with a warning ',...
            'flag of 0 or 1.  Failure is for matrices from'])
        disp(['   ','the SJSU Singular Matrix Database with SJid = ',...
            int2str(SJid_fail_spqr_cod),'.'])
    end
elseif (demo_opts.doprint >= 0)
    if ( nfail_spqr_cod == 0 )
        fprintf ('%3d matrices, failed: %d\n', nflagis0_or_1, ...
            nfail_spqr_cod) ;
    else
        fprintf ('%3d matrices, failed: %d with SJid =', ...
           nflagis0_or_1, nfail_spqr_cod) ;
       fprintf(' %d', SJid_fail_spqr_cod ) ;
       fprintf('\n') ;
    end
end

%-------------------------------------------------------------------------------
% return results
%-------------------------------------------------------------------------------

if (nargout > 0)
    nfailures = failures ;
end

if nargout > 1
    SJid_failures = SJid_fail;
end

if (failures > 0)
    fprintf ('demo_spqr_rank: %d failures\n', failures) ;
end

%-------------------------------------------------------------------------------
% subfunctions
%-------------------------------------------------------------------------------


%****************************************************************
%****    plot_ranks
%****************************************************************


function plot_ranks(rank_svd_v,rank_spqr_cod_v,rank_spqr_v, ...
    flag_spqr_cod_v, gap_v, method)
% plot the percent of matrices with a gap larger than a specified gap
%     that have the correct numerical rank, that have a
%     warning flag = 0 or 1' and that have a correct rank for spqr
%     versus gap in singular values
% the input is computed by demo_reliable_spqr

% ncor = sum (rank_svd_v == rank_spqr_cod_v);

gap_tol = 10 .^ ((0:32)/2);
z = zeros (1, length (gap_tol)) ;
per_cor = z ;
per_cor_spqr = z ;
per_flag_0_1 = z ;

i = 0;
for gap = gap_tol
    i = i+1;
    igt = find( gap_v >= gap & rank_svd_v > 0 );
    % for igt use rank_svd_v > 0 to exclude rare cases where SJrank returns -1
    ncorrect = sum( rank_svd_v(igt) == rank_spqr_cod_v(igt) );
    per_cor(i) = 100*ncorrect / length(igt);
    ncorrect = sum( rank_svd_v(igt) == rank_spqr_v(igt) );
    per_cor_spqr(i) = 100*ncorrect / length(igt);
    n_flag_0_1 = sum( flag_spqr_cod_v(igt) <= 1 );
    per_flag_0_1(i) = 100 * n_flag_0_1 / length(igt);
end

semilogx(gap_tol,per_cor_spqr,'ks--',gap_tol,per_flag_0_1, ...
    'bo--',gap_tol,per_cor,'rx--');
axisv=axis;
axisv3 = axisv(3);
%axisv3=60;
axisv2=[axisv(1) gap_tol(end) axisv3 axisv(4)+.01];
axis(axisv2);
fs = 12;
ylabel('Percent','fontsize',fs)
xlabel('Gap in the singular value spectrum bigger than','fontsize',fs)
title(['Percent numerical rank correct and warning flag = 0 or 1',char(10),...
    'versus gap in singular values'],'fontsize',fs)
legend('% SPQR rank correct',['% flag = 0 or 1 in ',method],...
    ['% ',method,' rank correct'],'location','se')
grid
set(gca,'fontsize',fs)


%****************************************************************
%****    plot_ basic
%****************************************************************

function plot_basic(norm_x_pinv_v,norm_x_QR_dense_v, norm_r_QR_dense_v, ...
    norm_x_spqr_basic_v, norm_r_spqr_basic_v, ...
    norm_x_spqr_solve_v, norm_r_spqr_solve_v,flag_v)
% plot the quality of the basic solutions
% the input is computed by demo_spqr_rank

iflagis0 = find( flag_v == 0 );
nflagis0 = length(iflagis0);

norm_x_pinv0 = norm_x_pinv_v(iflagis0);
norm_x_spqr_basic0 = norm_x_spqr_basic_v(iflagis0);
norm_x_QR_dense0 = norm_x_QR_dense_v(iflagis0);
norm_x_spqr_solve0 = norm_x_spqr_solve_v(iflagis0);
norm_ratio_spqr_basic = norm_x_spqr_basic0 ./ norm_x_pinv0;
norm_ratio_QR_dense = norm_x_QR_dense0 ./ norm_x_pinv0;
norm_ratio_spqr_solve = norm_x_spqr_solve0 ./ norm_x_pinv0;
[norm_ratio_spqr_basic,isort]=sort(norm_ratio_spqr_basic);
norm_ratio_QR_dense = norm_ratio_QR_dense(isort);
norm_ratio_spqr_solve = norm_ratio_spqr_solve(isort);


subplot(1,2,1)

semilogy(1:nflagis0,norm_ratio_QR_dense, 'bo',...
    1:nflagis0,norm_ratio_spqr_solve, 'ks',...
    1:nflagis0,norm_ratio_spqr_basic, 'rx')
fs = 12;
ylabel(' || x || / ||x_{  PINV} ||','fontsize',fs)
xlabel('matrix: ordered by ||x_{SPQR\_BASIC} || / ||x_{PINV} ||','fontsize',fs)
title([' Comparison of the norms of basic ',...
    'solutions divided by ||x_{PINV}||',char(10),'for ', ...
    int2str(nflagis0) , ' matrices with flag = 0', ...
    ' in SPQR\_BASIC'],'fontsize',fs)
legend('dense QR','SPQR\_SOLVE',...
    ' SPQR\_BASIC ','location','best')
grid
set(gca,'fontsize',fs)

subplot(1,2,2)
norm_r_spqr_basic0 = norm_r_spqr_basic_v(iflagis0);
norm_r_QR_dense0 = norm_r_QR_dense_v(iflagis0);
norm_r_spqr_solve0 = norm_r_spqr_solve_v(iflagis0);
[norm_r_spqr_basic,isort]=sort(norm_r_spqr_basic0);
norm_r_QR_dense = norm_r_QR_dense0(isort);
norm_r_spqr_solve = norm_r_spqr_solve0(isort);

semilogy(1:nflagis0,norm_r_QR_dense,'bo',...
    1:nflagis0,norm_r_spqr_solve,'ks',...
    1:nflagis0,norm_r_spqr_basic,'rx')

fs = 12;
ylabel(' || r || / || b ||','fontsize',fs)
xlabel('matrix: ordered by ||r_{SPQR\_BASIC} || / ||b||','fontsize',fs)
title([' Comparison of the norms of residuals, ',...
    'r = b-A*x, divided by ||b||',char(10),' for ', ...
    int2str(length(iflagis0)) , ' matrices with flag = 0', ...
    ' in SPQR\_BASIC'],'fontsize',fs)
legend('dense QR','SPQR\_SOLVE',...
    ' SPQR\_BASIC ','location','best')
set(gca,'fontsize',fs)
grid

%****************************************************************
%****    plot_null_spaces
%****************************************************************

function plot_null_spaces(norm_A_N_svd_v,tol_v, ...
    norm_A_N_v,flag_v,method)
% plot the quality of the null spaces
% the input is computed by demo_spqr_rank

iflagis0 = find( flag_v == 0 );
nflagis0 = length(iflagis0);
n_method_better = sum( norm_A_N_v(iflagis0) <= norm_A_N_svd_v(iflagis0) );
percent_method_better = 100*n_method_better / length(iflagis0);
quality_method = norm_A_N_v(iflagis0) ./ tol_v(iflagis0);
quality_svd = norm_A_N_svd_v(iflagis0) ./ tol_v(iflagis0);
[ignore, isort]=sort(quality_method);                                       %#ok
clear ignore
fs = 12;
semilogy(1:nflagis0,quality_svd(isort),'bo',1:nflagis0, ...
    quality_method(isort),'rx');
%axis1 = axis;
%axis1(3)=1.e-8;
%axis(axis1)
if strcmp(method,'SPQR\_BASIC')
    ylabel('|| A^T N || / tolerance','fontsize',fs)
    xlabel(['matrix: ordered by ||A^TN|| / tolerance for ',method],...
        'fontsize',fs)
    title(['Null space quality when flag in ',method,' is 0.',char(10),...
        method,' has ||A^T N|| smaller in ',...
        int2str(percent_method_better),'% of cases.'],'fontsize',fs)
else
    ylabel('|| A N || / tolerance','fontsize',fs)
    xlabel(['matrix: ordered by ||AN|| / tolerance for ',method],...
        'fontsize',fs)
    title(['Null space quality when flag in ',method,' is 0.',char(10),...
        method,' has ||AN|| smaller in ',...
        int2str(percent_method_better),'% of cases.'],'fontsize',fs)
end
legend('SVD null space',[method,' null space'],'location','SE')
grid
set(gca,'fontsize',fs)

%****************************************************************
%****    plot_pinv
%****************************************************************

function  plot_pinv(norm_x_spqr_pinv_minus_x_pinv_v, ...
        norm_x_spqr_cod_minus_x_pinv_v, cond1_pinv_v, cond1_cod_v, ...
        norm_w_pinv_v, norm_w_cod_v, flag_spqr_pinv_v,  ... % tol_v, ...
        flag_spqr_cod_v, norm_A_v)

% plot the quality of the pseudoinverse solutions
% the input is computed by demo_spqr_rank

% draw plot for results from spqr_pinv
subplot(1,2,1)
iflagis0 = find(flag_spqr_pinv_v == 0);
nflagis0 = length(iflagis0);
%perturbation_theory_v = cond1_v .* (max(tol_v,norm_w_pinv_v) ./ norm_A_v );
perturbation_theory_v = cond1_pinv_v .* max(10*eps,norm_w_pinv_v ./ norm_A_v );

perturbation_theory0 = perturbation_theory_v(iflagis0);
[perturbation_theory,isort] = sort(perturbation_theory0);
norm_x_spqr_pinv_minus_x_pinv0 = norm_x_spqr_pinv_minus_x_pinv_v(iflagis0);
norm_x_spqr_pinv_minus_x_pinv = norm_x_spqr_pinv_minus_x_pinv0(isort);
fs = 12;
loglog(perturbation_theory, norm_x_spqr_pinv_minus_x_pinv,'o',...
    [perturbation_theory(1),max(1,perturbation_theory(end))], ...
    [perturbation_theory(1),max(1,perturbation_theory(end))],...
    'r--','linewidth',2 )
ylabel(' || x_{SPQR\_PINV } - x_{PINV} || / ||x_{PINV} ||','fontsize',fs)
xlabel(' ( \sigma_1(A) / \sigma_r(A) ) max(10 \epsilon, ||w|| / ||A|| ) ',...
    'fontsize',fs)
title([' Comparison of the pseudoinverse solutions ',...
    'returned by SPQR\_PINV ',char(10),' and MATLAB''s PINV for ', ...
    int2str(nflagis0) , ' matrices with flag = 0', ...
    ' in SPQR\_PINV'],'fontsize',fs)
legend('|| x_{SPQR\_PINV } - x_{PINV} || / ||x_{PINV} ||',...
    '( \sigma_1(A)/\sigma_r(A) )  max(10\epsilon, ||w|| / ||A|| ) ',...
    'location','best')
set(gca,'fontsize',fs)
grid
shg

% draw plot for results from spqr_cod
subplot(1,2,2)
iflagis0 = find(flag_spqr_cod_v == 0);
nflagis0 = length(iflagis0);
%perturbation_theory_v = cond1_v .* (max(tol_v,norm_w_cod_v) ./ norm_A_v);
perturbation_theory_v = cond1_cod_v .* max(10*eps,norm_w_cod_v ./ norm_A_v );
perturbation_theory0 = perturbation_theory_v(iflagis0);
[perturbation_theory,isort] = sort(perturbation_theory0);
norm_x_spqr_cod_minus_x_pinv0 = norm_x_spqr_cod_minus_x_pinv_v(iflagis0);
norm_x_spqr_cod_minus_x_pinv = norm_x_spqr_cod_minus_x_pinv0(isort);
fs = 12;
loglog(perturbation_theory, norm_x_spqr_cod_minus_x_pinv,'o',...
    [perturbation_theory(1),max(1,perturbation_theory(end))], ...
    [perturbation_theory(1),max(1,perturbation_theory(end))],...
    'r--','linewidth',2 )
ylabel('|| x_{SPQR\_COD } - x_{PINV} || / ||x_{PINV} ||','fontsize',fs)
xlabel('( \sigma_1(A)/\sigma_r(A) ) max(10\epsilon, ||w|| / ||A||) ',...
    'fontsize',fs)
title([' Comparison of the pseudoinverse solutions ',...
    'returned by SPQR\_COD ',char(10),' and MATLAB''s PINV for ', ...
    int2str(nflagis0) , ' matrices with flag = 0', ...
    ' in SPQR\_COD'],'fontsize',fs)
legend('|| x_{SPQR\_COD } - x_{PINV} || / ||x_{PINV} ||',...
    '( \sigma_1(A) / \sigma_r(A) )  max(10 \epsilon, ||w|| / ||A||) ',...
    'location','best')
set(gca,'fontsize',fs)
grid
shg

function [N, stats, stats_ssp, est_sval_upper_bounds] = ...
    spqr_rank_form_basis(call_from, A, U, V ,Q1, rank_spqr, numerical_rank, ...
    stats, opts, est_sval_upper_bounds, nsvals_small, nsvals_large,  p1, Q2, p2)
%SPQR_RANK_FORM_BASIS forms the basis for the null space of a matrix.
%
% Called from spqr_basic and spqr_cod after these routines call spqr and
% spqr_rank_ssi.  The input parameters are as used in spqr_basic and spqr_cod.
% The parameter call_from indicates the type of call:
%      call_from = 1 -- a call from spqr_basic, form null space basis for A'
%      call_from = 2 -- a call from spqr_cod, form null space basis for A
%      call_from = 3 -- a call from spqr_cod, form null space basis for A'
% Output:
%   N -- basis for the null space in implicit or explicit form as required
%      by opts
%   stats -- an update of the stats vector in spqr_basic or spqr_cod
%   stats_ssp -- information about the call to spqr_ssp
%   est_sval_upper_bounds -- an update to the estimated singular value upper
%      bounds
% Not user-callable.

% Copyright 2012, Leslie Foster and Timothy A Davis.

get_details = opts.get_details ;
implicit_null_space_basis = opts.implicit_null_space_basis ;
start_with_A_transpose = opts.start_with_A_transpose ;
if (get_details == 1)
    t = tic ;
end
[m , n] = size(A);
nullity_R11 = rank_spqr - numerical_rank;

%-------------------------------------------------------------------------------
% form X which contains basis for null space of trapezoidal matrix
%-------------------------------------------------------------------------------

if call_from == 1    % call from spqr_basic (for null space basis for A')
    if nullity_R11 == 0
        X = [sparse(rank_spqr, m-rank_spqr) ; ...
             speye(m-rank_spqr, m-rank_spqr)];
    else
        X = [ [sparse(U(:, end-nullity_R11+1:end)) ; ...
             sparse(m-rank_spqr, nullity_R11) ], ...
             [ sparse(rank_spqr, m-rank_spqr) ; ...
             speye(m-rank_spqr, m-rank_spqr)]];
    end
elseif call_from == 2    % call from spqr_cod (for null space basis of A)
    if nullity_R11 == 0
        X = [sparse(rank_spqr, n-rank_spqr) ; ...
             speye(n-rank_spqr, n-rank_spqr)];
    else
        if (start_with_A_transpose)
            X = [ [sparse(V(:, end-nullity_R11+1:end)) ; ...
                 sparse(n-rank_spqr, nullity_R11) ], ...
                 [ sparse(rank_spqr, n-rank_spqr) ; ...
                 speye(n-rank_spqr, n-rank_spqr)]];
        else
            X = [ [sparse(U(:, end-nullity_R11+1:end)) ; ...
                 sparse(n-rank_spqr, nullity_R11) ], ...
                 [ sparse(rank_spqr, n-rank_spqr) ; ...
                 speye(n-rank_spqr, n-rank_spqr)]];
        end
    end
elseif call_from == 3  % call from spqr_cod (for null space basis for A')
    if nullity_R11 == 0
        X = [sparse(rank_spqr, m-rank_spqr) ; ...
          speye(m-rank_spqr, m-rank_spqr)];
    else
        if (start_with_A_transpose)
            X = [ [sparse(U(:, end-nullity_R11+1:end)) ; ...
              sparse(m-rank_spqr, nullity_R11) ], ...
              [ sparse(rank_spqr, m-rank_spqr) ; ...
              speye(m-rank_spqr, m-rank_spqr)]];
        else
            X = [ [sparse(V(:, end-nullity_R11+1:end)) ; ...
              sparse(m-rank_spqr, nullity_R11) ], ...
              [ sparse(rank_spqr, m-rank_spqr) ; ...
              speye(m-rank_spqr, m-rank_spqr)]];
        end
    end
end

%-------------------------------------------------------------------------------
% form null space basis for A or A' using X and Q from QR factorization
%-------------------------------------------------------------------------------

if call_from == 1    % call from spqr_basic (for null space basis for A')

    if implicit_null_space_basis
        % store null space in implicit form (store Q and X in N = Q*X)
        N.Q = Q1;
        N.X = X;
        N.kind = 'Q*X' ;
    else
        % store null space in explicit form
        N = spqr_qmult(Q1,X,1);
    end

 elseif call_from == 2    % call from spqr_cod (for null space basis of A)

    if implicit_null_space_basis
        % store null space basis in implicit form
        if (start_with_A_transpose)
            % store P, Q and X in N = Q*P*X
            p = 1:n;
            p(p2) = 1:length(p2);
            N.Q = Q1;
            N.P = p;    % a permutation vector
            N.X = X;
            N.kind = 'Q*P*X' ;
        else
            % store P, Q and X in N = P*Q*X
            p(p1) = 1:length(p1);
            N.P = p;    % a permutation vector
            N.Q = Q2;
            N.X = X;
            N.kind = 'P*Q*X' ;
        end
    else
        % store null space basis as an explicit matrix
        if (start_with_A_transpose)
            p = 1:n;
            p(p2) = 1:length(p2);
            N = spqr_qmult(Q1,X(p,:),1);
        else
            N = spqr_qmult(Q2,X,1);
            p(p1) = 1:length(p1);
            N = N(p,:);
        end
    end

elseif call_from == 3  % call from spqr_cod (for null space basis for A')

    p = 1:m;
    if (start_with_A_transpose)
        p(p1) = 1:length(p1);
    else
        p(p2) = 1:length(p2);
    end

    if implicit_null_space_basis
        % store null space basis in implicit form
        if (start_with_A_transpose)
            %          (store P, Q and X in N = P*Q*X)
            N.P = p;    % a permutation vector
            N.Q = Q2;
            N.X = X;
            N.kind = 'P*Q*X' ;
        else
            %       (store P, Q and X in N = Q*P*X)
            N.Q = Q1;
            N.P = p;    % a permutation vector
            N.X = X;
            N.kind = 'Q*P*X' ;
        end
    else
        % store null space explicitly forming Q*P*X;
        if (start_with_A_transpose)
            N = spqr_qmult(Q2,X,1);
            N = N(p,:) ;
        else
            N = spqr_qmult(Q1,X(p,:),1);
        end
    end
end

if (get_details == 1)
    stats.time_basis = toc (t) ;
end

%-------------------------------------------------------------------------------
% call spqr_ssp to enhance, potentially, est_sval_upper_bounds, the estimated
%    upper bounds on the singular values
% and / or
%    to estimate ||A * N || or || A' * N ||
%-------------------------------------------------------------------------------

% Note: nullity = m - numerical_rank;   % the number of columns in X and N

if call_from == 1    % call from spqr_basic (for null space basis for A')

    % Note: opts.k is not the same as k in the algorithm description above.

    % Note that, by the interleave theorem for singular values, for
    % i=1:nullity, singular value i of A'*N will be an upper bound on singular
    % value numerical_rank + i of A.  S(i,i) is an estimate of singular value i
    % of A'*N with an estimated accuracy of stats_ssp.est_error_bounds(i).
    % Therefore let

    [s_ssp,stats_ssp] = spqr_ssp (A', N, max (nsvals_small, opts.k), opts) ;

 elseif call_from == 2    % call from spqr_cod (for null space basis of A)

    % Note that, by the interleave theorem for singular
    % values, for i = 1, ..., nullity, singular value i of A*N will be
    % an upper bound on singular value numerical_rank + i of A.
    % S(i,i) is an estimate of singular value i of A*N with an estimated
    % accuracy of stats_ssp.est_error_bounds(i).  Therefore let

    [s_ssp,stats_ssp] = spqr_ssp (A, N, max (nsvals_small, opts.k), opts) ;

end

if call_from == 1 || call_from == 2 % call from spqr_basic (for null space
               %  basis for A') or from spqr_cod (for null space basis of A)

    % By the comments prior to the call to spqr_ssp we have
    % s_ssp + stats_ssp.est_error_bounds are estimated upper bounds
    % for singular values (numerical_rank+1):(numerical_rank+nsvals_small)
    % of A.  We have two estimates for upper bounds on these singular
    % values of A.  Choose the smaller of the two:
    if ( nsvals_small > 0 )
        est_sval_upper_bounds(nsvals_large+1:end) = ...
            min( est_sval_upper_bounds(nsvals_large+1:end) , ...
               s_ssp(1:nsvals_small)' + ...
               stats_ssp.est_error_bounds(1:nsvals_small) );
    end

elseif call_from == 3  % call from spqr_cod (for null space basis for A')

    % call ssp to estimate nsvals_small sing. values of A' * N
    %    (useful to estimate the norm of A' * N)
    [ignore,stats_ssp] = spqr_ssp (A', N, nsvals_small, opts) ;             %#ok
    clear ignore

end

if (get_details == 1)
    if call_from == 1 || call_from == 3
        stats.stats_ssp_NT = stats_ssp ;
    elseif call_from == 2
        stats.stats_ssp_N = stats_ssp ;
    end
    stats.time_svd = stats.time_svd + stats_ssp.time_svd ;
end
end

function [Q,R,C,p,info] = spqr_wrapper (A, B, tol, Q_option, get_details)
%SPQR_WRAPPER wrapper around spqr to get additional statistics
%   Not user-callable.  Usage:
%
%   [Q,R,C,p,info] = spqr_wrapper (A, B, tol, Q_option, get_details) ;

% Copyright 2012, Leslie Foster and Timothy A Davis.

if (get_details)
    % get detailed statistics for time and memory usage
    t = tic ;
end

% set the options
opts.econ = 0 ;                 % get the rank-sized factorization
opts.tol = tol ;                % columns with norm <= tol treated as zero
opts.permutation = 'vector' ;   % return permutation as a vector, not a matrix

if (~issparse (A))
    A = sparse (A) ;            % make sure A is sparse
end

m = size (A,1) ;

if (strcmp (Q_option, 'keep Q'))

    % compute Q*R = A(:,p) and keep Q in Householder form
    opts.Q = 'Householder' ;
    [Q,R,p,info] = spqr (A, opts) ;
    if (isempty (B))
        % C is empty
        C = zeros (m,0) ;
    else
        % also compute C = Q'*B if B is present
        C = spqr_qmult (Q, B, 0) ;
    end

else

    % compute Q*R = A(:,p), but discard Q
    opts.Q = 'discard' ;
    if (isempty (B))
        [Q,R,p,info] = spqr (A, opts) ;
        % C is empty
        C = zeros (m,0) ;
    else
        % also compute C = Q'*B if B is present
        [C,R,p,info] = spqr (A, B, opts) ;
        Q = [ ] ;
    end

end

if (get_details)
    info.time = toc (t) ;
end


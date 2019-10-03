function [ x ] = ...
    spqr_rank_deflation(call_from, R, U, V, C, m, n, rank_spqr, ...
    numerical_rank, nsvals_large, opts, p1, p2, N, Q1, Q2)
%SPQR_RANK_DEFLATION constructs pseudoinverse or basic solution using deflation.
%
% Called from spqr_basic and spqr_cod after these routines call spqr,
% spqr_rank_ssi and spqr_rank_form_basis.  The input parameters are as used
% in spqr_basic and spqr_cod.
% The parameter call_from indicates the type of call:
%      call_from = 1 indicates a call from spqr_basic
%      call_from = 2 indicates a call from spqr_cod
% Output:
%   x -- for call_from = 1 x is a basic solution to
%                 min || b - A * x ||.               (1)
%        The basic solution has n - (rank returned by spqr) zero components.
%        For call_from = 2 x is an approximate pseudoinverse solution to (1).
% Not user-callable.

% Algorithm:   R * wh = ch or R' * wh = ch is solved where ch is described
%              in the code and R comes from the QR factorizations in
%              spqr_basic or spqr_cod. R is triangular and potentially
%              numerically singular with left and right singular vectors for
%              small singular values stored in U and V.  When R is numerically
%              singular deflation (see SIAM SISC, 11:519-530, 1990) to
%              calculate an approximate truncated singular value solution to
%              the triangular system.  Orthogonal transformations
%              are applied to wh to obtain the solutions x to (1).

% Copyright 2012, Leslie Foster and Timothy A Davis.

% disable nearly-singular matrix warnings, and save the current state
warning_state = warning ('off', 'MATLAB:nearlySingularMatrix') ;

start_with_A_transpose = opts.start_with_A_transpose ;
implicit_null_space_basis = opts.implicit_null_space_basis ;

if (isempty (C))

    x = zeros (m,0) ;

elseif (start_with_A_transpose || call_from == 1)

    ch = C(1:rank_spqr,:);
    if numerical_rank == rank_spqr
        wh = R \ ch ;
    else
        % use deflation (see SIAM SISC, 11:519-530, 1990) to calculate an
        % approximate truncated singular value solution to R * wh = ch
        U = U(:,nsvals_large+1:end);
        wh = ch - U*(U'*ch);
        wh = R \ wh ;
        V = V(:,nsvals_large+1:end);
        wh = wh - V*(V'*wh);
    end
    if call_from == 2
        wh(p2,:)=wh;
    end
    wh = [wh ; zeros(n - rank_spqr,size(C,2)) ];
    if call_from == 1
        x(p1,:)=wh;
    else
        if implicit_null_space_basis
           x = spqr_qmult(N.Q,wh,1);
        else
           x = spqr_qmult(Q1,wh,1);
        end
    end

else

    ch = C(p2,:);
    if numerical_rank == rank_spqr
        wh = ( ch' / R )' ;    % wh = R' \ ch rewritten to save memory
    else
        % use deflation (see SIAM SISC, 11:519-530, 1990) to calculate an
        % approximate truncated singular value solution to R' * wh = ch
        V = V(:,nsvals_large+1:end);
        wh = ch - V*(V'*ch);
        wh = ( wh' / R )' ;    % wh = R' \ ch rewritten to save memory
        U = U(:,nsvals_large+1:end);
        wh = wh - U*(U'*wh);
    end
    wh = [wh ; zeros(n - rank_spqr,size(C,2)) ];
    if implicit_null_space_basis
       wh = spqr_qmult(N.Q,wh,1);
    else
       wh = spqr_qmult(Q2,wh,1);
    end
    x(p1,:)=wh;

end

% restore the warning back to what it was
warning (warning_state) ;


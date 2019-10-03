function qrdemo (matrix_name)
%QRDEMO a simple demo of SuiteSparseQR
% This is the MATLAB equivalent of the qrdemo.cpp program.  It reads in either
% a single matrix or a set of matrices, held as files in Matrix Market format
% in the ../Matrix directory.
%
% Example
%   qrdemo
%   qrdemo ('../Matrix/young1c.mtx')
%
% See also spqr, spqr_solve, mldivide

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

if (nargin == 0)

    % try all matrices
    qrdemo ('../Matrix/a2.mtx') ;
    qrdemo ('../Matrix/r2.mtx') ;
    qrdemo ('../Matrix/a04.mtx') ;
    qrdemo ('../Matrix/a2.mtx') ;
    qrdemo ('../Matrix/west0067.mtx') ;
    qrdemo ('../Matrix/c2.mtx') ;
    qrdemo ('../Matrix/a0.mtx') ;
    qrdemo ('../Matrix/lfat5b.mtx') ;
    qrdemo ('../Matrix/bfwa62.mtx') ;
    qrdemo ('../Matrix/LFAT5.mtx') ;
    qrdemo ('../Matrix/b1_ss.mtx') ;
    qrdemo ('../Matrix/bcspwr01.mtx') ;
    qrdemo ('../Matrix/lpi_galenet.mtx') ;
    qrdemo ('../Matrix/lpi_itest6.mtx') ;
    qrdemo ('../Matrix/ash219.mtx') ;
    qrdemo ('../Matrix/a4.mtx') ;
    qrdemo ('../Matrix/s32.mtx') ;
    qrdemo ('../Matrix/c32.mtx') ;
    qrdemo ('../Matrix/lp_share1b.mtx') ;
    qrdemo ('../Matrix/a1.mtx') ;
    qrdemo ('../Matrix/GD06_theory.mtx') ;
    qrdemo ('../Matrix/GD01_b.mtx') ;
    qrdemo ('../Matrix/Tina_AskCal_perm.mtx') ;
    qrdemo ('../Matrix/Tina_AskCal.mtx') ;
    qrdemo ('../Matrix/GD98_a.mtx') ;
    qrdemo ('../Matrix/Ragusa16.mtx') ;
    qrdemo ('../Matrix/young1c.mtx') ;
    fprintf ('All tests passed\n') ;

else

    %---------------------------------------------------------------------------
    % turn off warnings
    %---------------------------------------------------------------------------

    s = warning ('query', 'all') ;
    warning ('off', 'MATLAB:singularMatrix') ;
    warning ('off', 'MATLAB:rankDeficientMatrix') ;

    % --------------------------------------------------------------------------
    % the MATLAB equivalent of qrdemo.cpp
    % --------------------------------------------------------------------------

    fprintf ('qrdemo %s\n', matrix_name) ;
    A = mread (matrix_name) ;
    [m n] = size (A) ;
    anorm = norm (A,1) ;
    fprintf ('Matrix %6d-by-%-6d nnz: %6d ', m, n, nnz(A)) ;
    B = ones (m,1) ;
    [X, info] = spqr_solve (A,B) ;
    rnk = info.rank_A_estimate ;
    rnorm = norm (A*X-B,1) ;
    xnorm = norm (X,1) ;
    if (m <= n && anorm > 0 && xnorm > 0)
        rnorm = rnorm / (anorm * xnorm) ;
    end
    fprintf ('residual: %8.1e rank: %6d', rnorm, rnk) ;

    %---------------------------------------------------------------------------
    % compare with MATLAB
    % --------------------------------------------------------------------------

    try
        if (m ~= n || rnk == min (m,n))
            X = A\B ;
        else
            % The matrix is square and rank deficient.  Rather than getting a
            % divide-by-zero, force MATLAB to use a sparse QR by appending on a
            % blank row, in an attempt to find a consistent solution.
            X = [A ; sparse(1,n)] \ [B ; 0] ;
            X = X (1:m,:) ;
        end
        rnorm_matlab = norm (A*X-B,1) ;
        xnorm = norm (X,1) ;
        if (m <= n && anorm > 0 && xnorm > 0)
            rnorm_matlab = rnorm_matlab / (anorm * xnorm) ;
        end
    catch                                                                   %#ok
        % the above will fail for complex matrices that are rectangular and/or
        % rank deficient
        rnorm_matlab = nan ;
    end

    % find the true rank; this is costly
    rank_exact = rank (full (A)) ;

    if (rnk ~= rank_exact)
        % This can happen for some matrices; rnk is just an estimate, not
        % guaranteed to be equal to the true rank.  This does not occur
        % for any matrix in the ../Matrix test set.
        fprintf (' ?\n ... rank estimate is not exact; true rank is %d\n', ...
            rank_exact) ;
        warning ('rank estimate is incorrect') ;                            %#ok
    elseif (m <= n && rnk == m)
        % A*x=b is a full-rank square or under-determined system which spqr
        % should solve with low relative residual.
        if (rnorm > 1e-12)
            fprintf (' ?\n ... spqr solution is poor\n') ;
            error ('qrdemo failed') ;
        else
            fprintf (' OK\n') ;
        end
    elseif (m > n && rnk == n)
        % A*x=b is a full-rank least-squares problem.  Both spqr and MATLAB
        % should find a similar residual.
        err = abs (rnorm - rnorm_matlab) / max (rnorm_matlab, 1) ;
        if (err > 1e-12)
            fprintf (' ?\n ... spqr solution is poor\n') ;
            error ('qrdemo failed') ;
        else
            fprintf (' OK\n') ;
        end
    else
        % A*x=b is not full-rank; any solution will do ...
        fprintf (' %8.1e OK\n', rnorm_matlab) ;
    end

    %---------------------------------------------------------------------------
    % restore warning status
    %---------------------------------------------------------------------------

    warning (s) ;

end

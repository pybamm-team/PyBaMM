function err = test_svd (A)
%TEST_SVD  test factorize(A,'svd') and factorize(A,'cod') for a given matrix
%
% Example
%   err = test_svd (A) ;
%
% See also test_all

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('.') ;

if (nargin < 1)
    % has rank 3
    A = magic (4) ;
end

[m, n] = size (A) ;
err = 0 ;

for st = 0:1

    if (st == 0)
        F = factorize (A, 'svd') ;
        Acond = cond (F) ;
    else
        F = factorize (A, 'cod') ;
    end
    Apinv = pinv (full (A)) ;

    if (st == 0)
        assert (ismethod (F, 'norm')) ;
        assert (ismethod (F, 'pinv')) ;
        Anorm = norm (full (A)) ;
        Ainvnorm = norm (pinv (full (A))) ;
        Fnorm = norm (F) ;
        e = abs (Anorm - Fnorm) ;
        Anorm = max (Anorm, Fnorm) ;
        if (Anorm > 0)
            e = e / Anorm ;
        end
        err = check_err (err, e) ;
        Anorm = max (Anorm, 1) ;
    end

    % if B=pinv(A), then A*B*A=A and B*A*B=B
    B = inverse (F) ;
    Bnorm = norm (double (B), 1) ;
    err = check_err (err, ...
        norm (double(A*B*A) - A, 1) / (Anorm^2 * Bnorm)) ;
    err = check_err (err, ...
        norm (double(B*A*B) - double(B), 1) / (Anorm * Bnorm^2)) ;

    for bsparse = 0:1
        b = rand (m, 1) ;
        if (bsparse)
            b = sparse (b) ;
        end
        x = Apinv*b ;
        y = F\b ;
        if (st == 0)
            z = pinv(F)*b ;
        else
            z = inverse (F)*b ;
        end
        if (st == 0 || Acond < 1e13)
            % skip this for COD for very ill-conditioned problems
            x = double (x) ;
            y = double (y) ;
            z = double (z) ;
            err = check_err (err, norm (x - y) / (Anorm * norm (x) + norm (b)));
            err = check_err (err, norm (x - z) / (Anorm * norm (x) + norm (b)));
        end

        c = rand (1, n) ;
        if (bsparse)
            c = sparse (c) ;
        end
        x = c*Apinv ;
        y = c/F ;
        if (st == 0)
            z = c*pinv(F) ;
        else
            z = c*inverse(F) ;
        end
        if (st == 0 || Acond < 1e13)
            % skip this for COD for very ill-conditioned problems
            x = double (x) ;
            y = double (y) ;
            z = double (z) ;
            err = check_err (err, norm (x - y) / (Anorm * norm (x) + norm (b)));
            err = check_err (err, norm (x - z) / (Anorm * norm (x) + norm (b)));
        end
    end

    if (st == 0)
        assert (ismethod (F, 'rank')) ;
        arank = rank (full (A)) ;
        if (arank ~= rank (F))
            fprintf ('\nrank of A: %d, rank of F: %d\n', arank, rank (F)) ;
            fprintf ('singular values:\n') ;
            s1 = svd (A) ;
            s2 = F.Factors.S ;
            disp ([s1 s2])
            error ('rank mismatch!') ;
        end

        assert (ismethod (F, 'cond')) ;
        c1 = cond (full (A)) ;
        c2 = cond (F) ;
        if (rank (F) == min (m,n))
            e = abs (c1 - c2) ;
            if (max (c1, c2) > 0)
                e = e / max (c1, c2) ;
            end
            % fprintf ('cond err: %g\n', e) ;
            err = check_err (err, e) ;
        else
            % both c1 and c2 should be large
            tol = max (m,n) * eps (norm (F)) ;
            assert (c1 > norm (F) / tol) ;
            assert (c2 > norm (F) / tol) ;
        end

        Z = null (F) ;
        e = norm (A*Z) / Anorm  ;
        % fprintf ('null space err %g (%g)\n', e, err) ;
        err = check_err (err, e) ;

        [U1, S1, V1] = svd (F) ;
        [U2, S2, V2] = svd (full (A)) ;
        err = check_err (err, norm (U1-U2)) ;
        err = check_err (err, norm (S1-S2) / Anorm)  ;
        err = check_err (err, norm (V1-V2)) ;
        err = check_err (err, norm (U1*S1*V1'-U2*S2*V2') / Anorm) ;

        [U1, S1, V1] = svd (inverse (F)) ;
        [U2, S2, V2] = svd (pinv (full (A))) ;
        err = check_err (err, norm (S1-S2) / Ainvnorm)  ;
        err = check_err (err, norm (U1*S1*V1'-U2*S2*V2') / Ainvnorm) ;

        [U1, S1, V1] = svd (F, 0) ;
        [U2, S2, V2] = svd (full (A), 0) ;
        err = check_err (err, norm (U1-U2)) ;
        err = check_err (err, norm (S1-S2) / Anorm) ;
        err = check_err (err, norm (V1-V2)) ;
        err = check_err (err, norm (U1*S1*V1'-U2*S2*V2') / Anorm) ;

        [U1, S1, V1] = svd (F, 'econ') ;
        [U2, S2, V2] = svd (full (A), 'econ') ;
        err = check_err (err, norm (U1-U2)) ;
        err = check_err (err, norm (S1-S2) / Anorm) ;
        err = check_err (err, norm (V1-V2)) ;
        err = check_err (err, norm (U1*S1*V1'-U2*S2*V2') / Anorm) ;

        [U1, S1, V1] = svd (F, 'rank') ;
        err = check_err (err, norm (U1*S1*V1'-U2*S2*V2') / Anorm) ;

        % fprintf ('svd err %g (%g)\n', e, err) ;

        % test the p-norm
        for k = 0:3
            if (k == 0)
                p = 'inf' ;
            elseif (k == 3)
                p = 'fro' ;
            else
                p = k ;
            end

            n1 = norm (full (A), p) ;
            n2 = norm (F, p) ;
            e = abs (n1 - n2) ;
            if (n1 > 1)
                e = e / n1 ;
            end
            % fprintf ('norm (A,%d): %g\n', k, e) ;
            err = check_err (err, e) ;

            n1 = norm (full (A'), p) ;
            n2 = norm (F', p) ;
            e = abs (n1 - n2) ;
            if (n1 > 1)
                e = e / n1 ;
            end
            % fprintf ('norm (A'',%d): %g\n', k, e) ;
            err = check_err (err, e) ;

            n1 = norm (Apinv, p) ;
            n2 = norm (pinv (F), p) ;
            e = abs (n1 - n2) ;
            if (n1 > 1)
                e = e / n1 ;
            end
            % fprintf ('norm (pinv(A),%d): %g\n', k, e) ;
            err = check_err (err, e) ;

            n1 = norm (Apinv', p) ;
            n2 = norm (pinv (F)', p) ;
            e = abs (n1 - n2) ;
            if (n1 > 1)
                e = e / n1 ;
            end
            % fprintf ('norm (pinv(A)'',%d): %g\n', k, e) ;
            err = check_err (err, e) ;
        end
    end
end

function err = check_err (err, e)
err = max (err, e) ;
if (err > 1e-6)
    error ('%g error too high!\n', err) ;
end

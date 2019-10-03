function err = test_factorize (A, strategy)
%TEST_FACTORIZE test the accuracy of the factorization object
%
% Example
%   test_factorize (A) ;    % where A is square or rectangular, sparse or dense
%   test_factorize (A, strategy) ;  % forces a particular strategy;
%                           % works only if the matrix is compatible.
%
% See also test_all, factorize, inverse, mldivide

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

reset_rand ;
if (nargin < 1)
    A = rand (100) ;
end

if (nargin < 2)
    strategy = '' ;
end

% do not check the sparsity of the result when using the SVD
spcheck = ~(strcmp (strategy, 'svd')) ;

err = 0 ;
if (strcmp (strategy, 'ldl') && issparse (A) && ~isreal(A))
    % do not test ldl on sparse complex matrices
    return ;
end

[m, n] = size (A) ;
if (min (m,n) > 0)
    anorm = norm (A,1) ;
else
    anorm = 1 ;
end

is_symmetric = ((m == n) && (nnz (A-A') == 0)) ;

for nrhs = 1:3

    for bsparse = 0:1

        b = rand (m,nrhs) ;
        if (bsparse)
            b = sparse (b) ;
        end

        %-----------------------------------------------------------------------
        % test backslash and related methods
        %-----------------------------------------------------------------------

        for a = [1 pi (pi+1i)]

            % method 0:
            x = (a*A)\b ;
            err = check_resid (err, anorm, a*A, x, b, spcheck) ;

            % method 1:
            S = inverse (A)/a ;
            x = S*b ;
            err = check_resid (err, anorm, a*A, x, b, spcheck) ;

            % method 3:
            F = testfac (A, strategy) ;
            S = inverse (F)/a ;
            x = S*b ;
            err = check_resid (err, anorm, a*A, x, b, spcheck) ;

            % method 4:
            F = a*testfac (A, strategy) ;
            x = F\b ;
            err = check_resid (err, anorm, a*A, x, b, spcheck) ;

            % method 5:
            S = inverse (F) ;
            x = S*b ;
            err = check_resid (err, anorm, a*A, x, b, spcheck) ;

            % method 6:
            if (m == n)
                [L, U, p] = lu (A, 'vector') ;
                x = a * (U \ (L \ (b (p,:)))) ;
                err = check_resid (err, anorm, A/a, x, b, spcheck) ;

            % method 7:
                if (is_symmetric)
                    F = a*factorize (A, 'symmetric') ;
                    x = F\b ;
                    err = check_resid (err, anorm, a*A, x, b, spcheck) ;
                else
                    F = a*factorize (A, 'unsymmetric') ;
                    x = F\b ;
                    err = check_resid (err, anorm, a*A, x, b, spcheck) ;
                end
            end

        end

        clear S F

        %------------------------------------------------------------------
        % test transposed backslash and related methods
        %------------------------------------------------------------------

        b = rand (n,nrhs) ;
        if (bsparse)
            b = sparse (b) ;
        end

        for a = [1 pi (pi+1i)]

            % method 0:
            x = (a*A)'\b ;
            err = check_resid (err, anorm, (a*A)', x, b, spcheck) ;

            % method 1:
            S = inverse (A) / a ;
            x = S'*b ;
            err = check_resid (err, anorm, (a*A)', x, b, spcheck) ;

            % method 2:
            F = a*testfac (A, strategy) ;
            x = F'\b ;
            err = check_resid (err, anorm, (a*A)', x, b, spcheck) ;

            % method 3:
            S = inverse (F') ;
            x = S*b ;
            err = check_resid (err, anorm, (a*A)', x, b, spcheck) ;
        end

        clear S F

        %------------------------------------------------------------------
        % test mtimes, times, plus, minus, rdivide, and ldivide
        %------------------------------------------------------------------

        for a = [1 pi pi+2i]

            F = a*testfac (A, strategy) ;
            S = inverse (F) ;
            B = a*A ;   % F is the factorization of a*A

            % test mtimes
            d = rand (n,1)  ;
            x = F*d ;
            y = B*d ;
            z = S\d ;
            err = check_error (err, norm (mtx(x)-mtx(y),1)) ;
            err = check_error (err, norm (mtx(x)-mtx(z),1)) ;

            % test mtimes transpose
            d = rand (m,1)  ;
            x = F'*d ;
            y = B'*d ;
            z = S'\d ;
            err = check_error (err, norm (mtx(x)-mtx(y),1)) ;
            err = check_error (err, norm (mtx(x)-mtx(z),1)) ;

            % test for scalars
            for s = [1 42 3-2i]
                E = s\B  - double (s\F) ; err = check_error (err, norm (E,1)) ;
                E = B/s  - double (F/s) ; err = check_error (err, norm (E,1)) ;
%               E = B.*s - F.*s ;       err = check_error (err, norm (E,1)) ;
%               E = s.*B - s.*F ;       err = check_error (err, norm (E,1)) ;
%               E = s.\B - s.\F ;       err = check_error (err, norm (E,1)) ;
%               E = B./s - F./s ;       err = check_error (err, norm (E,1)) ;
                E = B*s  - double (F*s) ; err = check_error (err, norm (E,1)) ;
                E = s*B  - double (s*F) ; err = check_error (err, norm (E,1)) ;
            end

        end

        clear S F C

        %------------------------------------------------------------------
        % test slash and related methods
        %------------------------------------------------------------------

        b = rand (nrhs,n) ;
        if (bsparse)
            b = sparse (b) ;
        end

        % method 0:
        x = b/A ;
        err = check_resid (err, anorm, A', x', b', spcheck) ;

        % method 1:
        S = inverse (A) ;
        x = b*S ;
        err = check_resid (err, anorm, A', x', b', spcheck) ;

        % method 4:
        F = testfac (A, strategy) ;
        x = b/F ;
        err = check_resid (err, anorm, A', x', b', spcheck) ;

        % method 5:
        S = inverse (F) ;
        x = b*S ;
        err = check_resid (err, anorm, A', x', b', spcheck) ;

        % method 6:
        if (m == n)
            [L, U, p] = lu (A, 'vector') ;
            x = (b / U) / L ; x (:,p) = x ;
            err = check_resid (err, anorm, A', x', b', spcheck) ;
        end

        %------------------------------------------------------------------
        % test transpose slash and related methods
        %------------------------------------------------------------------

        b = rand (nrhs,m) ;
        if (bsparse)
            b = sparse (b) ;
        end

        % method 0:
        x = b/A' ;
        err = check_resid (err, anorm, A, x', b', spcheck) ;

        % method 1:
        S = inverse (A)' ;
        x = b*S ;
        err = check_resid (err, anorm, A, x', b', spcheck) ;

        % method 4:
        F = testfac (A, strategy)' ;
        x = b/F ;
        err = check_resid (err, anorm, A, x', b', spcheck) ;

        % method 5:
        S = inverse (F) ;
        x = b*S ;
        err = check_resid (err, anorm, A, x', b', spcheck) ;

        %------------------------------------------------------------------
        % test double
        %------------------------------------------------------------------

        Y = double (inverse (A)) ;
        if (m == n)
            Z = inv (A) ;
        else
            Z = pinv (full (A)) ;
        end
        e = norm (Y-Z,1) ;
        if (n > 0)
            e = e / norm (Z,1) ;
        end
        err = check_error (e, err) ;

        %------------------------------------------------------------------
        % test subsref
        %------------------------------------------------------------------

        F = testfac (A, strategy) ;
        Y = inverse (A) ;
        if (numel (A) > 1)
            if (F (end) ~= A (end))
                error ('factorization subsref error') ;
            end
            if (F.A (end) ~= A (end))
                error ('factorization subsref error') ;
            end
        end
        if (n > 0)
            if (F (1,1) ~= A (1,1))
                error ('factorization subsref error') ;
            end
            if (F.A (1,1) ~= A (1,1))
                error ('factorization subsref error') ;
            end
            e = abs (Y (1,1) - Z (1,1)) ;
            err = check_error (e,err) ;
            if (m > 1 && n > 1)
                e = norm (Y (1:2,1:2) - Z (1:2,1:2), 1) ;
                err = check_error (e,err) ;
            end
        end
        if (m > 3 && n > 1)
            if (any (F (2:end, 1:2) - A (2:end, 1:2)))
                error ('factorization subsref error') ;
            end
            if (any (F (2:4, :) - A (2:4, :)))
                error ('factorization subsref error') ;
            end
            if (any (F (:, 1:2) - A (:, 1:2)))
                error ('factorization subsref error') ;
            end
        end

        %------------------------------------------------------------------
        % test transposed subsref
        %------------------------------------------------------------------

        FT = F' ;
        YT = Y' ;
        AT = A' ;
        ZT = Z' ;
        if (numel (AT) > 1)
            if (FT (end) ~= AT (end))
                error ('factorization subsref error') ;
            end
            if (F.A (end) ~= A (end))
                error ('factorization subsref error') ;
            end
        end

        if (n > 0)
            if (FT (1,1) ~= AT (1,1))
                error ('factorization subsref error') ;
            end
            if (F.A (1,1) ~= A (1,1))
                error ('factorization subsref error') ;
            end
            e = abs (YT (1,1) - ZT (1,1)) ;
            err = check_error (e,err) ;
            if (m > 1 && n > 1)
                e = norm (YT (1:2,1:2) - ZT (1:2,1:2), 1) ;
                err = check_error (e,err) ;
            end
        end

        if (m > 1 && n > 3)
            if (any (FT (2:end, 1:2) - AT (2:end, 1:2)))
                error ('factorization subsref error') ;
            end
            if (any (FT (2:4, :) - AT (2:4, :)))
                error ('factorization subsref error') ;
            end
            if (any (FT (:, 1:2) - AT (:, 1:2)))
                error ('factorization subsref error') ;
            end
        end

        %------------------------------------------------------------------
        % test update/downdate
        %------------------------------------------------------------------

        if (isa (F, 'factorization_chol_dense'))
            w = rand (n,1) ;
            b = rand (n,1) ;
            % update
            G = cholupdate (F,w) ;
            x = G\b ;
            err = check_resid (err, anorm, A+w*w', x, b, spcheck) ;
            % downdate
            G = cholupdate (G,w,'-') ;
            x = G\b ;
            err = check_resid (err, anorm, A, x, b, spcheck) ;
            clear G
        end

        %------------------------------------------------------------------
        % test size
        %------------------------------------------------------------------

        [m1, n1] = size (F) ;
        [m, n] = size (A) ;
        if (m1 ~= m || n1 ~= n)
            error ('size error') ;
        end
        [m1, n1] = size (Y) ;
        if (m1 ~= n || n1 ~= m)
            error ('pinv size error') ;
        end
        if (size (Y,1) ~= n || size (Y,2) ~= m)
            error ('pinv size error') ;
        end
        if (size (F,1) ~= m || size (F,2) ~= n)
            error ('size error') ;
        end

        %------------------------------------------------------------------
        % test mtimes
        %------------------------------------------------------------------

        clear S F Y

        for a = [1 pi pi+2i]

            F = a * testfac (A, strategy) ;
            S = inverse (F) ;
            d = rand (1,m) ;
            x = d*F ;
            y = d*(A*a) ;
            z = d/S ;

            err = check_error (err, norm (mtx(x)-mtx(y),1)) ;
            err = check_error (err, norm (mtx(x)-mtx(z),1)) ;

            d = rand (1,n) ;
            x = d*F' ;
            y = d*(A*a)' ;
            z = d/S' ;

            err = check_error (err, norm (mtx(x)-mtx(y),1)) ;
            err = check_error (err, norm (mtx(x)-mtx(z),1)) ;

        end

        %------------------------------------------------------------------
        % test inverse
        %------------------------------------------------------------------

        Y = double (inverse (inverse (A))) ;
        e = norm (A-Y,1) ;
        if (e > 0)
            error ('inverse error') ;
        end

        %------------------------------------------------------------------
        % test mldivide and mrdivide with a matrix b, transpose, etc
        %------------------------------------------------------------------

        if (max (m,n) < 100)
            F = testfac (A, strategy) ;
            B = rand (m,n) ;
            err = check_error (err, norm (B\A - mtx(B\F), 1) / anorm) ;
            err = check_error (err, norm (A/B - mtx(F/B), 1) / anorm) ;
            err = check_error (err, norm (B*pinv(full(A))-mtx(B/F), 1) / anorm);
        end
    end
end

fprintf ('.') ;

%--------------------------------------------------------------------------

function err = check_resid (err, anorm, A, x, b, spcheck)
[m, n] = size (A) ;

x = mtx (x) ;

if (m <= n)
    e = norm (A*x-b,1) / (anorm + norm (x,1)) ;
else
    e = norm (A'*(A*x)-A'*b,1) / (anorm + norm (x,1)) ;
end

if (min (m,n) > 1 && spcheck)
    if (issparse (A) && issparse (b))
        if (~issparse (x))
            error ('x must be sparse') ;
        end
    else
        if (issparse (x))
            error ('x must be full') ;
        end
    end
end

err = check_error (e, err) ;

%--------------------------------------------------------------------------

function x = mtx (x)
% make sure that x is a matrix.  It might be a factorization.
if (isobject (x))
    x = double (x) ;
end

%--------------------------------------------------------------------------

function err = check_error (err1, err2)
err = max (err1, err2) ;
if (err > 1e-8)
    fprintf ('error: %8.3e\n', full (err)) ;
    error ('error too high') ;
end
err = full (err) ;

%--------------------------------------------------------------------------

function F = testfac (A, strategy)
if (isempty (strategy))
    F = factorize (A) ;
else
    F = factorize (A, strategy) ;
end

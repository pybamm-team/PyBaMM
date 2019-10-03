classdef factorization_cod_sparse < factorization
%FACTORIZATION_COD_SPARSE complete orthogonal factorization: A = U*R*V' where A is sparse.
% A fairly accurate estimate of rank is found.  double(inverse(F)) is a fairly
% accurate estimate of pinv(A).

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_cod_sparse (A)
            %FACTORIZATION_SPARSE_COD A = U*R*V'
            [f.U, f.R, f.V, f.r] = cod_sparse (A) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = f.r ;
            F.kind = 'sparse COD factorization: A = U*R*V''' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            U = cod_qmult (f.U, speye (size (f.U.H,1)), 1) ;
            V = cod_qmult (f.V, speye (size (f.V.H,1)), 1) ;
            e = norm (F.A - U*f.R*V', 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBLCASS x = A\b using a sparse COD factorization
            % If the estimated rank is correct, this is x = pinv(A)*b
            f = F.Factors ;
            r = f.r ;
            c = cod_qmult (f.U, b, 0) ;                 % c = U' * b
            c = f.R (1:r,1:r) \ c (1:r,:) ;             % c = R \ c
            n = size (f.R, 2) ;
            if (r < n)
                c = [c ; sparse(n-r,size(c,2))] ;       % make sure c has n rows
                if (~issparse (b))
                    c = full (c) ;
                end
            end
            x = cod_qmult (f.V, c, 1) ;                 % x = V * c
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using sparse COD factorization
            % If the estimated rank is correct, this is x = b*pinv(A)
            f = F.Factors ;
            r = f.r ;
            c = cod_qmult (f.V, b, 3) ;                 % c = b * V
            c = c (:,1:r) / f.R (1:r,1:r) ;             % c = c / R
            m = size (f.R, 1) ;
            if (r < m)
                c = [c , sparse(size(c,1),m-r)] ;       % make sure c has m cols
                if (~issparse (b))
                    c = full (c) ;
                end
            end
            x = cod_qmult (f.U, c, 2) ;                 % x = c * U'
        end
    end
end

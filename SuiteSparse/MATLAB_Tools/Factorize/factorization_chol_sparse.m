classdef factorization_chol_sparse < factorization
%FACTORIZATION_CHOL_SPARSE P'*A*P = L*L' where A is sparse and sym. pos. def.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_chol_sparse (A)
            %FACTORIZATION_CHOL_SPARSE : P'*A*P = L*L'
            [f.L, g, f.P] = chol (A, 'lower') ;
            if (g ~= 0)
                error ('MATLAB:posdef', 'Matrix must be positive definite.') ;
            end
            F.A = A ;
            F.Factors = f ;
            F.A_rank = size (A,1) ;
            F.kind = 'sparse Cholesky factorization: P''*A*P = L*L''' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (f.P'*F.A*f.P - f.L*f.L', 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using sparse Cholesky
            % x = P * (L' \ (L \ (P' * b)))
            f = F.Factors ;
            x = f.P * (f.L' \ (f.L \ (f.P' * b))) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using sparse Cholesky
            % x = (P * (L' \ (L \ (P' * b'))))'
            x = (mldivide_subclass (F, b'))' ;
        end
    end
end

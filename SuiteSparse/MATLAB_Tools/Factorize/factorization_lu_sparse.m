classdef factorization_lu_sparse < factorization
%FACTORIZATION_LU_SPARSE P*A*Q = L*U where A is square and sparse.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_lu_sparse (A, fail_if_singular)
            %FACTORIZATION_LU_SPARSE : P*(R\A)*Q = L*U
            [m, n] = size (A) ;
            if (m ~= n)
                error ('FACTORIZE:wrongdim', ...
                    'LU for rectangular matrices not supported.  Use QR.') ;
            end
            if (n == 0)
                nil = sparse ([ ]) ;
                f.L = nil ;
                f.U = nil ;
                f.P = nil ;
                f.Q = nil ;
                f.R = nil ;
                F.A_condest = 1 ;
            else
                [f.L, f.U, f.P, f.Q, f.R] = lu (A) ;
                F.A_condest = cheap_condest (get_diag (f.U), fail_if_singular) ;
            end
            F.A = A ;
            F.Factors = f ;
            F.A_rank = n ;
            F.kind = 'sparse LU factorization: P*(R\A)*Q = L*U' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (f.P*(f.R\F.A)*f.Q - f.L*f.U, 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using sparse LU
            % x = Q * (U \ (L \ (P * (R \ b))))
            f = F.Factors ;
            x = f.Q * (f.U \ (f.L \ (f.P * (f.R \ b)))) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using sparse LU
            % x = ((((b * Q) / U) / L) * P) / R ; 
            f = F.Factors ;
            x = ((((b * f.Q) / f.U) / f.L) * f.P) / f.R ; 
        end
    end
end

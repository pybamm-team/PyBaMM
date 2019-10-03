classdef factorization_ldl_sparse < factorization
%FACTORIZATION_LDL_SPARSE P'*A*P = L*D*L' where A is sparse and symmetric

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_ldl_sparse (A)
            %FACTORIZATION_LDL_SPARSE : P'*A*P = L*D*L'
            [f.L, f.D, f.P] = ldl (A) ;
            c = full (condest (f.D)) ;
            if (c > 1/(2*eps))
                error ('MATLAB:singularMatrix', ...
                    'Matrix is singular to working precision.') ;
            end
            F.A = A ;
            F.Factors = f ;
            F.A_rank = size (A,1) ;
            F.A_condest = c ;
            F.kind = 'sparse LDL factorization: P''*A*P = L*D*L''' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (f.P'*F.A*f.P - f.L*f.D*f.L', 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using sparse LDL
            % x = P * (L' \ (L \ (P' * b)))
            f = F.Factors ;
            x = f.P * (f.L' \ (f.D \ (f.L \ (f.P' * b)))) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using sparse LDL
            % x = (P * (L' \ (L \ (P' * b'))))'
            x = (mldivide_subclass (F,b'))' ;
        end
    end
end

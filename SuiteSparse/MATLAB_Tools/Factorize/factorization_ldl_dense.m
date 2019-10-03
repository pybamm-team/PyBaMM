classdef factorization_ldl_dense < factorization
%FACTORIZATION_LDL_DENSE P'*A*P = L*D*L' where A is full and symmetric

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_ldl_dense (A)
            %FACTORIZATION_LDL_DENSE : A(p,p) = L*D*L'
            if (isempty (A))
                f.L = [ ] ;
                f.D = [ ] ;
                f.p = [ ] ;
            else
                [f.L, f.D, f.p] = ldl (A, 'vector') ;
            end
            % D is block diagonal with 2-by-2 blocks, and is best stored as
            % a *sparse* matrix, not full.  This saves storage and speeds up
            % D\b in the solve phases below.
            f.D = sparse (f.D) ;
            c = full (condest (f.D)) ;
            if (c > 1/(2*eps))
                error ('MATLAB:singularMatrix', ...
                    'Matrix is singular to working precision.') ;
            end
            n = size (A,1) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = n ;
            F.A_condest = c ;
            F.kind = 'dense LDL factorization: A(p,p) = L*D*L''' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (F.A (f.p, f.p) - f.L*f.D*f.L', 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using dense LDL
            % x = P * (L' \ (D \ (L \ (P' * b))))
            f = F.Factors ;
            opL.LT = true ;
            opLT.LT = true ;
            opLT.TRANSA = true ;
            y = b (f.p, :) ;
            if (issparse (y))
                y = full (y) ;
            end
            x = linsolve (f.L, f.D \ linsolve (f.L, y, opL), opLT) ;
            x (f.p, :) = x ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense LDL
            % x = (P * (L' \ (D \ (L \ (P' * b')))))'
            x = (mldivide_subclass (F, b'))' ;
        end
    end
end

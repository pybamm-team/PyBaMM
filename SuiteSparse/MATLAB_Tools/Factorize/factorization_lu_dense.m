classdef factorization_lu_dense < factorization
%FACTORIZATION_LU_DENSE A(p,:) = L*U where A is square and full.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_lu_dense (A, fail_if_singular)
            %FACTORIZATION_LU_DENSE : A(p,:) = L*U
            [m, n] = size (A) ;
            if (m ~= n)
                error ('FACTORIZE:wrongdim', ...
                    'LU for rectangular matrices not supported.  Use QR.') ;
            end
            [f.L, f.U, f.p] = lu (A, 'vector') ;
            F.A_condest = cheap_condest (get_diag (f.U), fail_if_singular) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = n ;
            F.kind = 'dense LU factorization: A(p,:) = L*U' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (F.A (f.p,:) - f.L*f.U, 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x=A\b using dense LU
            % x = U \ (L \ (b (p,:))) ;
            f = F.Factors ;
            opL.LT = true ;
            opU.UT = true ;
            y = b (f.p, :) ;
            if (issparse (y))
                y = full (y) ;
            end
            x = linsolve (f.U, linsolve (f.L, y, opL), opU) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense LU
            % x (:,p) = (L' \ (U' \ b'))'
            f = F.Factors ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            opLT.LT = true ;
            opLT.TRANSA = true ;
            y = b' ;
            if (issparse (y))
                y = full (y) ;
            end
            x = (linsolve (f.L, linsolve (f.U, y, opUT), opLT))';
            x (:, f.p) = x ;
        end
    end
end

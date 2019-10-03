classdef factorization_cod_dense < factorization
%FACTORIZATION_COD_DENSE complete orthogonal factorization: A = U*R*V' where A is full.
% A fairly accurate estimate of rank is found.  double(inverse(F)) is a fairly
% accurate estimate of pinv(A).

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_cod_dense (A)
            %FACTORIZATION_COD_DENSE A = U*R*V'
            [f.U, f.R, f.V, F.A_rank] = cod (A) ;
            F.A = A ;
            F.Factors = f ;
            F.kind = 'dense COD factorization: A = U*R*V''' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (F.A - f.U*f.R*f.V', 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBLCASS x = A\b using a dense COD factorization
            % x = V * (R \ (U' * b))
            f = F.Factors ;
            op.UT = true ;
            y = f.U' * b ;
            if (issparse (y))
                y = full (y) ;
            end
            x = f.V * linsolve (f.R, y, op) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense COD factorization
            % x = ((b * V) / R) * U' = (U * (R' \ (b*V)'))'
            f = F.Factors ;
            op.UT = true ;
            op.TRANSA = true ;
            y = (b * f.V)' ;
            if (issparse (y))
                y = full (y) ;
            end
            x = (f.U * linsolve (f.R, y, op))' ;
        end
    end
end

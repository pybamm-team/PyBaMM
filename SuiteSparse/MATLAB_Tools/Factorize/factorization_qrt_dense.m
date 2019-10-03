classdef factorization_qrt_dense < factorization
%FACTORIZATION_QRT_DENSE A' = Q*R where A is full.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_qrt_dense (A, fail_if_singular)
            %FACTORIZATION_QRT_DENSE : A' = Q*R
            [m, n] = size (A) ;
            if (m > n)
                error ('FACTORIZE:wrongdim', 'QR(A'') method requires m<=n.') ;
            end
            [f.Q, f.R] = qr (A',0) ;
            F.A_condest = cheap_condest (get_diag (f.R), fail_if_singular) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = rank_est (f.R, m, n) ;
            F.kind = 'dense economy QR factorization: A'' = Q*R' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (F.A' - f.Q*f.R, 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using a dense QR factorization of A'
            % minimum 2-norm solution of an underdetermined system
            % x = Q * (R' \ b)
            f = F.Factors ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            y = b ;
            if (issparse (y))
                y = full (y) ;
            end
            x = f.Q * linsolve (f.R, y, opUT) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense QR of A'
            % least-squares solution of a overdetermined problem
            % x = (R \ (Q' * b'))'
            f = F.Factors ;
            opU.UT = true ;
            y = f.Q' * b' ;
            if (issparse (y))
                y = full (y) ;
            end
            x = linsolve (f.R, y, opU)' ;
        end
    end
end

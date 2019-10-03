classdef factorization_chol_dense < factorization
%FACTORIZATION_CHOL_DENSE A = R'*R where A is full and symmetric pos. def.
% Adds an extra method, cholupdate, which acts just like the builtin cholupdate.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_chol_dense (A)
            %FACTORIZATION_CHOL_DENSE : A = R'*R
            [f.R, g] = chol (A) ;
            if (g ~= 0)
                error ('MATLAB:posdef', 'Matrix must be positive definite.') ;
            end
            F.A = A ;
            F.Factors = f ;
            F.A_rank = size (A,1) ;
            F.kind = 'dense Cholesky factorization: A = R''*R' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            f = F.Factors ;
            e = norm (F.A - f.R'*f.R, 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x=A\b using a dense Cholesky factorization
            % x = R \ (R' \ b)
            f = F.Factors ;
            opU.UT = true ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            y = b ;
            if (issparse (y))
                y = full (y) ;
            end
            x = linsolve (f.R, linsolve (f.R, y, opUT), opU) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense Cholesky
            % x = (R \ (R' \ b')'
            x = (mldivide_subclass (F, b'))' ;
        end

        function F = cholupdate (F,w,updown)
            %CHOLUPDATE update/downdate a dense Cholesky factorization
            %
            % Example
            %   % F becomes the Cholesky factorization of A+w*w'
            %   F = factorize (A) ;
            %   w = rand (size (A,1),1) ;
            %   G = cholupdate (F,w) ;
            %   x = G\b ;               % computes x = (A+w*w')\b
            %   G = cholupdate (F,w,'-') ;
            %   x = G\b ;               % computes x = (A-w*w')\b
            %
            % See also factorize, cholupdate.
            if (nargin < 3)
                updown = '+' ;
            end
            switch updown
                case '+'
                    F.Factors.R = cholupdate (F.Factors.R, w, '+') ;
                    F.A = F.A + w*w' ;
                case '-'
                    F.Factors.R = cholupdate (F.Factors.R, w, '-') ;
                    F.A = F.A - w*w' ;
                otherwise
                    error ('MATLAB:cholupdate:incorrectThirdInputArgument', ...
                        'Third argument must be ''+'' or ''-''.') ;
            end
        end
    end
end

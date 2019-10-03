classdef factorization_svd < factorization
%FACTORIZATION_SVD A = U*S*V'
% Adds the following extra methods that act just like the builtin functions.
% Most take little or no time to compute, since they rely on the precomputed
% SVD.  The exceptions are cond(F,p) and norm(F,p) when p is not 2.
%
%   c = cond (F,p)      the p-norm condition number.  p=2 is the default.
%                       cond(F,2) takes no time to compute, since it was
%                       computed when the SVD factorization was found.
%   a = norm (F,p)      the p-norm.  see the cond(F,p) discussion above.
%   r = rank (F)        returns the rank of A, precomputed by the SVD.
%   Z = null (F)        orthonormal basis for the null space of A
%   Q = orth (F)        orthonormal basis for the range of A
%   C = pinv (F)        the pseudo-inverse, V'*(S\V).
%   [U,S,V] = svd (F)   SVD of A or pinv(A), regular, economy, or rank-sized

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_svd (A)
            %FACTORIZATION_SVD singular value decomponsition, A = U*S*V'
            [f.U, f.S, f.V] = svd (full (A)) ;
            [m, n] = size (A) ;
            % convert S into a vector of singular values
            if (isempty (A))
                f.S = 0 ;
            elseif (isvector (A))
                f.S = f.S (1) ;
            else
                f.S = diag (f.S) ;
            end
            % compute rank(A), and save it in F
            f.r = sum (f.S > max (m,n) * eps (f.S (1))) ;
            F.A_rank = f.r ;
            % compute cond(A), and save it in F
            if (isempty (A))
                F.A_cond = 0 ;          % cond ([]) is zero, according to MATLAB
            elseif (f.r < min (m,n))
                F.A_cond = inf ;        % matrix is singular
            else
                F.A_cond = f.S (1) / f.S (end) ;
            end
            F.A = A ;
            F.Factors = f ;
            F.kind = 'singular value decomposition: A = U*S*V''' ;
        end

        function e = error_check (F)
            %ERROR_CHECK : return relative 1-norm of error in factorization
            % meant for testing only
            [U, S, V] = svd (F) ; % extracts the pre-computed SVD of A from F
            e = norm (F.A - U*S*V', 1) / norm (F.A, 1) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x=A\b using a singular value decomposition
            % Only svd(A,'econ') is needed.
            f = F.Factors ;
            r = f.r ;
            x = f.V (:,1:r) * (diag (f.S (1:r)) \ (f.U (:,1:r)' * b)) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x=b/A using a singular value decomposition
            % Only svd(A,'econ') is needed.
            f = F.Factors ;
            r = f.r ;
            x = ((b * f.V (:,1:r)) / diag (f.S (1:r))) * f.U (:,1:r)' ;
        end

        function c = cond (F, p)
            %COND the 2-norm condition number
            % cond(F,2) takes O(1) time to compute once the SVD is known.
            % Otherwise, pinv(A) (or pinv(A') if F has been transposed) is
            % explicitly computed using the pre-computed [U,S,V]=svd(A).
            if (nargin == 1 || isequal (p,2) || isempty (F))
                % The 2-norm condition number has been pre-computed.
                c = F.A_cond ;
            else
                % Compute the p-norm of a non-empty matrix, where p is not 2.
                [m, n] = size (F) ;
                if (m ~= n)
                    error ('MATLAB:cond:normMismatchSizeA', ...
                        'A is rectangular.  Use the 2 norm.') ;
                end
                r = F.A_rank ;
                if (r < min (m,n))
                    % matrix is rank-deficient so cond (A,p) is always inf
                    c = inf ;
                else
                    % The matrix is square, non-empty, and has full rank.
                    % One of these requires the explicit computation of pinv(A),
                    % where U,S,V are already pre-computed.  The same result is
                    % computed whether F represents A or its (pseudo) inverse.
                    c = norm (double (F), p) * norm (double (inverse (F)), p) ;
                end
            end
        end

        function nrm = norm (F, p)
            %NORM see the description of cond, above.
            if (nargin == 1 || isequal (p,2) || isempty (F))
                f = F.Factors ;
                r = f.r ;
                if (isempty (F))
                    nrm = 0 ;
                elseif (r == 0)
                    if (F.is_inverse)
                        nrm = inf ;
                    else
                        nrm = 0 ;
                    end
                else
                    if (F.is_inverse)
                        nrm = 1 / (f.S (r)) ;       % norm (pinv (A))
                    else
                        nrm = f.S (1) ;             % norm (A)
                    end
                end
            else
                % If F represents the inverse, then double (F) is pinv (A),
                % which is computed with V*(S\U') via mldivide.  U,S,V are
                % already computed, so double(F) is not too hard to compute.
                nrm = norm (double (F), p) ;
            end
        end

        function r = rank (F)
            % The rank of A has been pre-computed.  Just return it.
            r = F.A_rank ;
        end

        function Z = null (F)
            %NULL orthonormal basis for the null space of A
            f = F.Factors ;
            r = f.r ;
            Z = f.V (:, r+1:end) ;
        end

        function Q = orth (F)
            %ORTH orthonormal basis for the range of A
            % This function makes theta=subspace(A,B) easy to compute,
            % 
            f = F.Factors ;
            r = f.r ;
            Q = f.U (:, 1:r) ;
        end

        function C = pinv (F)
            % PINV is just another name for inverse(factorize(A,'svd'))
            C = inverse (F) ;
        end

        function [U,S,V] = svd (F, kind)
            % SVD return the svd of A, A', pinv(A), or pinv(A'). [U,S,V]=svd(A)
            % has already been computed.  Truncate / transpose / reshape it
            % as needed, also considering svd(",'econ') and svd(",0).
            f = F.Factors ;
            U = f.U ;
            S = f.S ;
            V = f.V ;
            [m, n] = size (F.A) ;
            if (nargin > 1)
                switch kind
                    case 'econ'
                        % return svd(A,'econ')
                        k = min (m,n) ;
                        U = U (:, 1:k) ;
                        S = S (1:k) ;
                        V = V (:, 1:k) ;
                    case 'rank'
                        % return the rank-sized SVD
                        k = f.r ;
                        U = U (:, 1:k) ;
                        S = S (1:k) ;
                        V = V (:, 1:k) ;
                    case 0
                        % return svd(A,0)
                        if (m > n)
                            k = n ;
                            U = U (:, 1:k) ;
                            S = S (1:k) ;
                        end
                    otherwise
                        error ('unrecognized kind') ;
                end
            end
            if (F.is_inverse ~= F.is_ctrans)
                % returning svd(A') or svd(pinv(A)).  swap U and V
                T = U ;
                U = V ;
                V = T ;
            end
            if (F.is_inverse)
                % returning svd(pinv(A)) or svd(pinv(A')).  Invert and reverse
                % S(1:r) and the corresponding singular vectors.
                r = f.r ;
                S = [(1 ./ S (r:-1:1)) ; (zeros (length (S) - r, 1))] ;
                U (:, 1:r) = U (:, r:-1:1) ;
                V (:, 1:r) = V (:, r:-1:1) ;
            end
            % The user expects S as a matrix of the proper size, so expand it.
            k = length (S) ;
            if (isempty (F))
                k = 0 ;
            end
            S = full (sparse (1:k, 1:k, S, size (U,2), size (V,2))) ;
        end
    end
end

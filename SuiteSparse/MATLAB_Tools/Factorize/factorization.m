classdef factorization
%FACTORIZATION a generic matrix factorization object
% Normally, this object is created via the F=factorize(A) function.  Users
% do not need to use this method directly.
%
% This is an abstract class that is specialized into 13 different kinds of
% matrix factorizations:
%
%   factorization_chol_dense    dense Cholesky      A = R'*R
%   factorization_lu_dense      dense LU            A(p,:) = L*U
%   factorization_qr_dense      dense QR of A       A = Q*R
%   factorization_qrt_dense     dense QR of A'      A' = Q*R
%   factorization_ldl_dense     dense LDL           A(p,p) = L*D*L'
%   factorization_cod_dense     dense COD           A = U*R*V'
%
%   factorization_chol_sparse   sparse Cholesky     P'*A*P = L*L'
%   factorization_lu_sparse     sparse LU           P*(R\A)*Q = L*U
%   factorization_qr_sparse     sparse QR of A      (A*P)'*(A*P) = R'*R
%   factorization_qrt_sparse    sparse QR of A'     (P*A)*(P*A)' = R'*R
%   factorization_ldl_sparse    sparse LDL          P'*A*P = L*D*L'
%   factorization_cod_sparse    sparse COD          A = U*R*V'
%
%   factorization_svd           SVD                 A = U*S*V'
%
% The abstract class provides the following functions.  In the descriptions,
% F is a factorization.  The arguments b, y, and z may be factorizations or
% matrices.  The output x is normally matrix unless it can be represented as a
% scaled factorization.  For example, G=F\2 and G=inverse(F)*2 both return a
% factorization G.  Below, s is always a scalar, and C is always a matrix.
%
%   These methods return a matrix x, unless one argument is a scalar (in which
%   case they return a scaled factorization object):
%   x = mldivide (F, b)     x = A \ b
%   x = mrdivide (b, F)     x = b / A
%   x = mtimes (y, z)       y * z
%
%   These methods always return a factorization:
%   F = uplus (F)           +F
%   F = uminus (F)          -F
%   F = inverse (F)         representation of inv(A), without computing it
%   F = ctranspose (F)      F'
%
%   These built-in methods return a scalar:
%   s = isreal (F)
%   s = isempty (F)
%   s = isscalar (F)
%   s = issingle (F)
%   s = isnumeric (F)
%   s = isfloat (F)
%   s = isvector (F)
%   s = issparse (F)
%   s = isfield (F,f)
%   s = isa (F, s)
%   s = condest (F)
%
%   This method returns the estimated rank from the factorization.
%   s = rankest (F)
%
%   These methods support access to the contents of a factorization object
%   e = end (F, k, n)
%   [m,n] = size (F, k)
%   S = double (F)
%   C = subsref (F, ij)
%   S = struct (F)
%   disp (F)
%
% The factorization_chol_dense object also provides cholupdate, which acts
% just like the builtin cholupdate.
%
% The factorization_svd object provides:
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
%
% See also mldivide, lu, chol, ldl, qr, svd.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    properties (SetAccess = protected)
        % The abstract class holds a QR, LU, Cholesky factorization:
        A = [ ] ;           % a copy of the input matrix
        Factors = [ ] ;
        is_inverse = false ;% F represents the factorization of A or inv(A)
        is_ctrans = false ; % F represents the factorization of A or A'
        kind = '' ;         % a string stating what kind of factorization F is
        alpha = 1 ;         % F represents the factorization of A or alpha*A.
        A_rank = [ ] ;      % rank of A, from SVD, or estimate from COD
        A_cond = [ ] ;      % 2-norm condition number of A, from SVD
        A_condest = [ ] ;   % quick and dirty estimate of the condition number
        % If F is inverted, alpha doesn't change.  For example:
        %   F = alpha*factorize(A) ; % F = alpha*A, in factorized form.
        %   G = inverse(F) ;         % G = inv(alpha*A)
        %   H = beta*G               % H = beta*inv(alpha*A) 
        %                                = inv((alpha/beta)*A)
        % So to update alpha via scaling, beta*F, the new scale factor beta
        % is multiplied with F.alpha if F.is_inverse is false.  Otherwise,
        % F.alpha is divided by beta to get the new scale factor.
    end

    methods (Abstract)
        x = mldivide_subclass (F, b) ;
        x = mrdivide_subclass (b, F) ;
        e = error_check (F) ;
    end

    methods

        %-----------------------------------------------------------------------
        % mldivide and mrdivide: return a scaled factorization or a matrix
        %-----------------------------------------------------------------------

        % Let b be a double scalar, F a non-scalar factorization, and g a scalar
        % factorization.  Then these operations return scaled factorization
        % objects (unless flatten is true, in which case a matrix is returned):
        %
        %   F\b = inverse (F) * b           |   F/b = F / b
        %   F\g = inverse (F) * double (g)  |   F/g = F / double (g)
        %   b\F = F / b                     |   b/F = b * inverse (F)
        %   g\F = F / double (g)            |   g/F = double (g) * inverse (F)
        %
        % Otherwise mldivide & mrdivide always return a matrix as their result.

        function x = mldivide (y, z, flatten)
            %MLDIVIDE x = y\z where either y or z or both are factorizations.
            flatten = (nargin > 2 && flatten) ;
            if (isobject (y) && isscalar (z) && ~flatten)
                % x = y\z where y is an object and z is scalar (perhaps object).
                % result is a scaled factorization object x.
                x = scale_factor (inverse (y), ~(y.is_inverse), double (z)) ;
            elseif (isscalar (y) && isobject (z) && ~flatten)
                % x = y\z where y is scalar (perhaps object) and z is an object.
                % result is a scaled factorization object x.
                x = scale_factor (z, ~(z.is_inverse), double (y)) ;
            else
                % result x will be a matrix.  b is coerced to be a matrix.
                [F, b, first_arg_is_F] = getargs (y, z) ;
                if (~first_arg_is_F)
                    % x = b\F where F is a factorization.
                    error_if_inverse (F, 1) ;
                    x = b \ F.A ;            % use builtin backslash
                elseif (F.is_ctrans)
                    % x = F\b where F represents (alpha*A)' or inv(alpha*A)'
                    if (F.is_inverse)
                        % x = inv(alpha*A)'\b = (A'*b)*alpha'
                        x = (F.A'*b) * F.alpha' ;
                    else
                        % x = (alpha*A)'\b = (b'/A)' / alpha'
                        x = mrdivide_subclass (b', F)' / F.alpha' ;
                    end
                else
                    % x = F\b where F represents (alpha*A) or inv(alpha*A)
                    if (F.is_inverse)
                        % x = inv(alpha*A)\b = (A*b)*alpha
                        x = (F.A*b) * F.alpha ;
                    else
                        % x = (alpha*A)\b = (A\b) / alpha
                        x = mldivide_subclass (F, b) / F.alpha ;
                    end
                end
            end
        end

        function x = mrdivide (y, z, flatten)
            %MRDIVIDE x = y/z where either y or z or both are factorizations.
            flatten = (nargin > 2 && flatten) ;
            if (isobject (y) && isscalar (z) && ~flatten)
                % x = y/z where y is an object and z is scalar (perhaps object).
                % result is a scaled factorization object x.
                x = scale_factor (y, ~(y.is_inverse), double (z)) ;
            elseif (isscalar (y) && isobject (z) && ~flatten)
                % x = y/z where y is scalar (perhaps object) and z is an object.
                % result is a scaled factorization object x.
                x = scale_factor (inverse (z), ~(z.is_inverse), double (y)) ;
            else
                % result x will be a matrix.  b is coerced to be a matrix.
                [F, b, first_arg_is_F] = getargs (y, z) ;
                if (first_arg_is_F)
                    % x = F/b where F is a factorization object
                    error_if_inverse (F, 2)
                    x = F.A / b ;            % use builtin slash
                elseif (F.is_ctrans)
                    % x = b/F where F represents (alpha*A)' or inv(alpha*A)'
                    if (F.is_inverse)
                        % x = b/inv(alpha*A)' = (b*A')*alpha'
                        x = (b*F.A') * F.alpha' ;
                    else
                        % x = b/(alpha*A)' = (A\b')' / alpha'
                        x = mldivide_subclass (F, b')' / F.alpha' ;
                    end
                else
                    % x = b/F where F represents (alpha*A) or inv(alpha*A)
                    if (F.is_inverse)
                        % x = b/inv(alpha*A) = (b*A)*alpha
                        x = (b*F.A) * F.alpha ;
                    else
                        % x = b/(alpha*A) = (b/A) / alpha
                        x = mrdivide_subclass (b, F) / F.alpha ;
                    end
                end
            end
        end

        %-----------------------------------------------------------------------
        % mtimes: a simple and clean wrapper for mldivide and mrdivide
        %-----------------------------------------------------------------------

        function x = mtimes (y, z)
            %MTIMES x=y*z where y or z is a factorization object (or both).
            % Since inverse(F) is so cheap, and does the right thing inside
            % mldivide and mrdivide, this is just a simple wrapper.
            if (isobject (y))
                % A*b               becomes inverse(A)\b
                % inverse(A)*b      becomes A\b 
                % A'*b              becomes inverse(A)'\b
                % inverse(A)'*b     becomes A'\b
                x = mldivide (inverse (y), z) ;
            else
                % b*A               becomes b/inverse(A)
                % b*inverse(A)      becomes b/A
                % b*A'              becomes b/inverse(A)'
                % b*inverse(A)'     becomes b/A'
                % y is a scalar or matrix, z must be an object
                x = mrdivide (y, inverse (z)) ;
            end
        end

        %-----------------------------------------------------------------------
        % uplus, uminus, ctranspose, inverse: always return a factorization
        %-----------------------------------------------------------------------

        function F = uplus (F)
            %UPLUS +F
        end

        function F = uminus (F)
            %UMINUS -F
            F.alpha = -(F.alpha) ;
        end

        function F = inverse (F)
            %INVERSE "inverts" F by flagging it as factorization of inv(A)
            F.is_inverse = ~(F.is_inverse) ;
        end

        function F = ctranspose (F)
            %CTRANSPOSE "transposes" F by flagging it as factorization of A'
            F.is_ctrans = ~(F.is_ctrans) ;
        end

        %-----------------------------------------------------------------------
        % is* methods that return a scalar
        %-----------------------------------------------------------------------

        function s = isreal (F)
            %ISREAL for F=factorize(A): same as isreal(A)
            s = isreal (F.A) ;
        end

        function s = isempty (F)
            %ISEMPTY for F=factorize(A): same as isempty(A)
            s = any (size (F.A) == 0) ;
        end

        function s = isscalar (F)
            %ISSCALAR for F=factorize(A): same as isscalar(A)
            s = isscalar (F.A)  ;
        end

        function s = issingle (F)                                           %#ok
            %ISSINGLE for F=factorize(A) is always false
            s = false ;
        end

        function s = isnumeric (F)                                          %#ok
            %ISNUMERIC for F=factorize(A) is always true
            s = true ;
        end

        function s = isfloat (F)                                            %#ok
            %ISFLOAT for F=factorize(A) is always true
            s = true ;
        end

        function s = isvector (F)
            %ISVECTOR for F=factorize(A): same as isvector(A)
            s = isvector (F.A) ;
        end

        function s = issparse (F)
            %ISSPARSE for F=factorize(A): same as issparse(A)
            s = issparse (F.A) ;
        end

        function s = isfield (F, f)                                         %#ok
            %ISFIELD isfield(F,f) is true if F.f exists, false otherwise
            s = (ischar (f) && (strcmp (f, 'A') ...
                || strcmp (f, 'Factors') || strcmp (f, 'kind') ...
                || strcmp (f, 'is_inverse') || strcmp (f, 'is_ctrans') ...
                || strcmp (f, 'alpha') || strcmp (f, 'A_rank') ...
                || strcmp (f, 'A_cond') || strcmp (f, 'A_condest'))) ;
        end

        function s = isa (F, s)
            %ISA for F=factorize(A): 'double', 'numeric', 'float' are true.
            % For other types, the builtin isa does the right thing.
            s = strcmp (s, 'double') || strcmp (s, 'numeric') ||  ...
                strcmp (s, 'float') || builtin ('isa', F, s) ;
        end

        %-----------------------------------------------------------------------
        % condest, rankest
        %-----------------------------------------------------------------------

        function C = abs (F)
            %ABS abs(F) returns abs(A) or abs(inverse(A)), as appropriate.  The
            % ONLY reason abs is included here is to support the builtin
            % normest1 for small matrices (n <= 4).  Computing abs(inverse(A))
            % explicitly computes the inverse of A, so use with caution.
            C = abs (double (F)) ;
        end

        function s = condest (F)
            %CONDEST 1-norm condition number for square matrices.
            % Does not require another factorization of A, so it's very fast.
            % Does NOT explicitly compute the inverse of A.  Instead, if F
            % represents an inverse, F*x inside normest1 does the right thing,
            % and does A\b using the factorization F.
            A = F.A ;                                                       %#ok
            [m, n] = size (A) ;                                             %#ok
            if (m ~= n)
                error ('MATLAB:condest:NonSquareMatrix', ...
                       'Matrix must be square.') ;
            end
            if (n == 0)
                s = 0 ;
            elseif (F.is_inverse)
                % F already represents the factorization of the inverse of A
                s = F.alpha * norm (A,1) * normest1 (F) ;                   %#ok
            else
                % Note that the inverse is NOT explicitly computed.
                s = F.alpha * norm (A,1) * normest1 (inverse (F)) ;         %#ok
            end
        end

        function r = rankest (F)
            %RANKEST returns the estimated rank of A.
            % It is a very rough estimate if Cholesky, LU, QR, or LDL succeeded
            % (in which A is assumed to have full rank).  COD finds a more
            % accurate estimate, and SVD finds the exact rank.
            r = F.A_rank ;
        end

        %-----------------------------------------------------------------------
        % end, size
        %-----------------------------------------------------------------------

        function e = end (F, k, n)
            %END returns index of last item for use in subsref
            if (n == 1)
                e = numel (F.A) ;   % # of elements, for linear indexing
            else
                e = size (F, k) ;   % # of rows or columns in A or pinv(A)
            end
        end

        function [m, n] = size (F, k)
            %SIZE returns the size of the matrix F.A in the factorization F
            if (F.is_inverse ~= F.is_ctrans)
                % swap the dimensions to match pinv(A)
                if (nargout > 1)
                    [n, m] = size (F.A) ;
                else
                    m = size (F.A) ;
                    m = m ([2 1]) ;
                end
            else
                if (nargout > 1)
                    [m, n] = size (F.A) ;
                else
                    m = size (F.A) ;
                end
            end
            if (nargin > 1)
                m = m (k) ;
            end
        end

        %-----------------------------------------------------------------------
        % double: a wrapper for subsref
        %-----------------------------------------------------------------------

        function S = double (F)
            %DOUBLE returns the factorization as a matrix, A or inv(A)
            ij.type = '()' ;
            ij.subs = cell (1,0) ;
            S = subsref (F, ij) ;   % let factorize.subsref do all the work
        end

        %-----------------------------------------------------------------------
        % subsref: returns a matrix
        %-----------------------------------------------------------------------

        function C = subsref (F, ij)
            %SUBSREF A(i,j) or (i,j)th entry of inv(A) if F is inverted.
            % Otherwise, explicit entries in the inverse are computed.
            % This method also extracts the contents of F with F.whatever.
            switch (ij (1).type)
                case '.'
                    % F.A usage: extract one of the matrices from F
                    switch ij (1).subs
                        case 'A'
                            C = F.A ;
                        case 'Factors'
                            C = F.Factors ;
                        case 'is_inverse'
                            C = F.is_inverse ;
                        case 'is_ctrans'
                            C = F.is_ctrans ;
                        case 'kind'
                            C = F.kind ;
                        case 'alpha'
                            C = F.alpha ;
                        case 'A_cond'
                            C = F.A_cond ;
                        case 'A_condest'
                            C = F.A_condest ;
                        case 'A_rank'
                            C = F.A_rank ;
                        otherwise
                            error ('MATLAB:nonExistentField', ...
                            'Reference to non-existent field ''%s''.', ...
                            ij (1).subs) ;
                    end
                    % F.X(2,3) usage, return X(2,3), for component F.X
                    if (length (ij) > 1 && ~isempty (ij (2).subs))
                        C = subsref (C, ij (2)) ;
                    end
                case '()'
                    C = subsref_paren (F, ij) ;
                case '{}'
                    error ('MATLAB:cellRefFromNonCell', ...
                    'Cell contents reference from a non-cell array object.') ;
            end
        end

        %-----------------------------------------------------------------------
        % struct: extracts all contents of a factorization object
        %-----------------------------------------------------------------------

        function S = struct (F)
            %STRUCT convert factorization F into a struct.
            % S cannot be used for subsequent object methods here.
            S.A = F.A ;
            S.Factors = F.Factors ;
            S.is_inverse = F.is_inverse ;
            S.is_ctrans = F.is_ctrans ;
            S.alpha = F.alpha ;
            S.A_rank = F.A_rank ;
            S.A_cond = F.A_cond ;
            S.kind = F.kind ;
        end

        %-----------------------------------------------------------------------
        % disp: displays the contents of F
        %-----------------------------------------------------------------------

        function disp (F)
            %DISP displays a factorization object
            fprintf ('  class: %s\n', class (F)) ;
            fprintf ('  %s\n', F.kind) ;
            fprintf ('  A: [%dx%d double]\n', size (F.A)) ;
            fprintf ('  Factors:\n') ; disp (F.Factors) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
            fprintf ('  is_ctrans: %d\n', F.is_ctrans) ;
            fprintf ('  alpha: %g', F.alpha) ;
            if (~isreal (F.alpha))
                fprintf (' + (%g)i', imag (F.alpha)) ;
            end
            fprintf ('\n') ;
            if (~isempty (F.A_rank))
                fprintf ('  A_rank: %d\n', F.A_rank) ;
            end
            if (~isempty (F.A_condest))
                fprintf ('  A_condest: %d\n', F.A_condest) ;
            end
            if (~isempty (F.A_cond))
                fprintf ('  A_cond: %d\n', F.A_cond) ;
            end
        end
    end

    %---------------------------------------------------------------------------
    % methods that are not user-callable
    %---------------------------------------------------------------------------

    methods (Access = protected)

        function [F, b, first_arg_is_F] = getargs (y, z)
            first_arg_is_F = isobject (y) ;
            if (first_arg_is_F)
                F = y ;             % first argument is a factorization object
                b = double (z) ;    % 2nd one coerced to be a matrix
            else
                b = y ;             % first argument is not an object
                F = z ;             % second one must be an object
            end
        end

        function F = scale_factor (F, use_beta_inverse, beta)
            %SCALE_FACTOR scales a factorization
            if (use_beta_inverse)
                % F = inv(alpha*A), so F*beta = inv((alpha/beta)*A)
                if (F.is_ctrans)
                    F.alpha = F.alpha / beta' ;
                else
                    F.alpha = F.alpha / beta ;
                end
            else
                % F = alpha*A, so F*beta = (alpha*beta)*A
                if (F.is_ctrans)
                    F.alpha = F.alpha * beta' ;
                else
                    F.alpha = F.alpha * beta ;
                end
            end
        end
    end
end


%-------------------------------------------------------------------------------
% subsref_paren: support function for subsref
%-------------------------------------------------------------------------------

function C = subsref_paren (F, ij)
%SUBSREF_PAREN C = subsref_paren(F,ij) implements C=F(i,j) and C=F(i)

    % F(2,3) usage, return A(2,3) or the (2,3) of inv(A).
    assert (length (ij) == 1, 'Improper index matrix reference.') ;
    A = F.A ;
    is_ctrans = F.is_ctrans ;
    if (is_ctrans && length (ij.subs) > 1)   % swap i and j
        ij.subs = ij.subs ([2 1]) ;
    end

    if (F.is_inverse)

        % requesting explicit entries of the inverse

        if (length (ij.subs) == 1)
            % for linear indexing of the inverse (C=F(i)), first
            % convert to double and then use builtin subsref
            C = subsref (double (F), ij) ;
        else
            % standard indexing, C = F(i,j)
            if (is_ctrans)
                [n, m] = size (A) ;
            else
                [m, n] = size (A) ;
            end
            if (length (ij.subs) == 2)
                ilen = length (ij.subs {1}) ;
                if (strcmp (ij.subs {1}, ':'))
                    ilen = n ;
                end
                jlen = length (ij.subs {2}) ;
                if (strcmp (ij.subs {2}, ':'))
                    jlen = m ;
                end
                j = ij ;
                j.subs {1} = ':' ;
                i = ij ;
                i.subs {2} = ':' ;
                if (jlen <= ilen)
                    % compute X=S(:,j) of S=inv(A) and return X(i,:)
                    C = subsref (mldivide (...
                        inverse (F), ...
                        subsref (identity (A, m), j), 1), i) ;
                else
                    % compute X=S(i,:) of S=inv(A) and return X(:,j)
                    C = subsref (mrdivide (...
                        subsref (identity (A, n), i), ...
                        inverse (F), 1), j) ;
                end
            else
                % the entire inverse has been explicitly computed
                C = mldivide (inverse (F), identity (A, m), 1) ;
            end
        end

    else

        % F is not inverted, so just return A(i,j)
        if (isempty (ij (1).subs))
            C = A ;
        else
            C = subsref (A, ij) ;
        end
        C = C * F.alpha ;
        if (is_ctrans)
            C = C' ;
        end
    end
end


%-------------------------------------------------------------------------------
% identity: return a full or sparse identity matrix
%-------------------------------------------------------------------------------

function I = identity (A, n)
%IDENTITY return a full or sparse identity matrix.  Not user-callable
    if (issparse (A))
        I = speye (n) ;
    else
        I = eye (n) ;
    end
end

%-------------------------------------------------------------------------------
% throw an error if inv(A) is being inadvertently computed 
%-------------------------------------------------------------------------------

function error_if_inverse (F, kind)
    % x = b\F or F/b where F=inverse(A) and b is not a scalar is unsupported.
    % It could be done by coercing F into an explicit matrix representation of
    % inv(A), via x = b\double(F) or double(A)/b, but this is the same as
    % b\inv(A) or inv(A)/b respectively.  That is dangerous, and thus it is
    % not done here automatically.
    if (F.is_inverse)
        if (kind == 1)
            s1 = 'B\F' ;
            s2 = 'B\double(F)' ;
        else
            s1 = 'F/B' ;
            s2 = 'double(F)/B' ;
        end
        error ('FACTORIZE:unsupported', ...
        ['%s where F=inverse(A) requires the explicit computation of the ' ...
         'inverse.\nThis is ill-advised, so it is never done automatically.'...
         '\nTo force it, use %s instead of %s.\n'], s1, s2, s1) ;
    end
end

function err = test_function (A, strategy, burble)
%TEST_FUNCTION test various functions applied to a factorize object
%
% Example
%   test_functions (A) ;    % where A is square or rectangular, sparse or dense
%
% See also test_all, factorize, inverse, mldivide

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

reset_rand ;

state = warning ('off', 'MATLAB:illConditionedMatrix') ;

if (nargin < 1)
    A = rand (10) ;
end
if (nargin < 2)
    strategy = 'default' ;
end
if (nargin < 3)
    burble = 0 ;
end

err = 0 ;
[m, n] = size (A) ;
F = factorize (A, strategy) ;

%   C = rand (m,n) ;
%   if (~isempty (A)) 
%       C (1,1) = A (1,1) ;
%   end
%   if (issparse (A))
%       C = sparse (A) ;
%   end
%   H = factorize (C, strategy) ; %#ok

%-------------------------------------------------------------------------------
% implicitly-defined methods that return the correct result:
%-------------------------------------------------------------------------------

if (m == 10)
    burble = 1 ;
    % only do these pedantic tests on a few test matrices
    assert (~ismethod (F, 'ismethod')) ;
    assert (~ismethod (F, 'iscell')) ;      assert (~iscell (F)) ;
    assert (~ismethod (F, 'iscellstr')) ;   assert (~iscellstr (F)) ;
    assert (~ismethod (F, 'ischar')) ;      assert (~ischar (F)) ;
    assert (~ismethod (F, 'iscom')) ;       assert (~iscom (F)) ;
    assert (~ismethod (F, 'isinteger')) ;   assert (~isinteger (F)) ;
    assert (~ismethod (F, 'ishandle')) ;    assert (~ishandle (F)) ;
    assert (~ismethod (F, 'isobject')) ;    assert ( isobject (F)) ;
    assert (~ismethod (F, 'isinterface')) ; assert (~isinterface (F)) ;
    assert (~ismethod (F, 'isjava')) ;      assert (~isjava (F)) ;
    assert (~ismethod (F, 'iskeyword')) ;   assert (~iskeyword (F)) ;
    assert (~ismethod (F, 'isletter')) ;    assert (~isletter (F)) ;
    assert (~ismethod (F, 'isspace')) ;     assert (~isspace (F)) ;
    assert (~ismethod (F, 'islogical')) ;   assert (~islogical (F)) ;
    assert (~ismethod (F, 'isstruct')) ;    assert (~isstruct (F)) ;
    assert (~ismethod (F, 'isvarname')) ;   assert (~isvarname (F)) ;
    assert (~ismethod (F, 'isglobal')) ;    % not tested since it's deprecated
    assert (~ismethod (F, 'ndims')) ;       assert (ndims (F) == 2) ;
    assert (~ismethod (F, 'isequal')) ;
    assert (~ismethod (F, 'isequalwithequalnans')) ;
    % One could imagine that these return length(F.A) and numel(F.A), but if
    % they are defined that way, F.A fails because the size of F is wrong.
    assert (~ismethod (F, 'length')) ;      assert (length (F) == 1) ;
    assert (~ismethod (F, 'numel')) ;       assert (numel (F) == 1) ;
end

G = factorize (A, strategy) ;
any_nans = any (any (isnan (F.A))) || any (any (isnan (double (inverse (F))))) ;
if (~any_nans)
    assert ( isequal (F, G)) ;
    assert (~isequal (F, inverse (F))) ;
end
assert ( isequalwithequalnans (F, G)) ;
assert (~isequalwithequalnans (F, inverse (F))) ;
clear G

%-------------------------------------------------------------------------------
% explicit methods
%-------------------------------------------------------------------------------

if (min (m,n) < 2 || m == 10)

    % explicitly-defined methods, tested here:
    assert (ismethod (F, 'isfloat')) ;  assert ( isfloat (F)) ;
    assert (ismethod (F, 'isnumeric')); assert ( isnumeric (F)) ;
    assert (ismethod (F, 'issingle')) ; assert (~issingle (F)) ;
    assert (ismethod (F, 'isreal')) ;   assert (isreal (F) == isreal (A)) ;
    assert (ismethod (F, 'isempty')) ;  assert (isempty (F) == isempty (A)) ;
    assert (ismethod (F, 'isvector')) ; assert (isvector (F) == isvector (A)) ;
    assert (ismethod (F, 'isscalar')) ; assert (isscalar (F) == isscalar (A)) ;
    assert (ismethod (F, 'issparse')) ; assert (issparse (F) == issparse (A)) ;
    assert (ismethod (F, 'size')) ;     assert (isequal (size (A), size (F))) ;

    if (~any_nans)
        assert (ismethod (F, 'abs')) ;    assert (isequal (abs (F), abs (F.A)));
        assert (ismethod (F, 'double')) ; assert (isequal (A, double (F))) ;
    end

    assert (ismethod (F, 'isfield')) ;
    assert ( isfield (F, 'A')) ;
    assert ( isfield (F, 'Factors')) ;
    assert ( isfield (F, 'is_inverse')) ;
    assert ( isfield (F, 'is_ctrans')) ;
    assert ( isfield (F, 'alpha')) ;
    assert ( isfield (F, 'kind')) ;
    assert (~isfield (F, 'anthing_else')) ;

    assert (ismethod (F, 'isa')) ;
    assert ( isa (F, 'double')) ;
    assert (~isa (F, 'logical')) ;
    assert (~isa (F, 'char')) ;
    assert (~isa (F, 'single')) ;
    assert ( isa (F, 'float')) ;
    assert (~isa (F, 'int8')) ;
    assert (~isa (F, 'uint8')) ;
    assert (~isa (F, 'int16')) ;
    assert (~isa (F, 'uint16')) ;
    assert (~isa (F, 'int32')) ;
    assert (~isa (F, 'uint32')) ;
    assert (~isa (F, 'int64')) ;
    assert (~isa (F, 'uint64')) ;
    assert (~isa (F, 'integer')) ;
    assert ( isa (F, 'numeric')) ;
    assert (~isa (F, 'cell')) ;
    assert (~isa (F, 'struct')) ;
    assert (~isa (F, 'function_handle')) ;
    assert ( isa (F, 'factorization')) ;
    assert (~isa (F, 'anything_else')) ;

    % explicitly-defined methods, but tested elsewhere:
    assert (ismethod (F, 'mldivide')) ;
    assert (ismethod (F, 'mldivide_subclass')) ;
    assert (ismethod (F, 'mrdivide')) ;
    assert (ismethod (F, 'mrdivide_subclass')) ;
    assert (ismethod (F, 'inverse')) ;
    assert (ismethod (F, 'ctranspose')) ;
    assert (ismethod (F, 'end')) ;
    assert (ismethod (F, 'mtimes')) ;
    assert (ismethod (F, 'subsref')) ;
    assert (ismethod (F, 'disp')) ;
    assert (ismethod (F, 'condest')) ;
    assert (ismethod (F, 'struct')) ;
    if (isa (F, 'factorization_chol_dense'))
        assert (ismethod (F, 'cholupdate')) ;
        if (burble)
            methods (F)
        end
    end
    if (isa (F, 'factorization_svd'))
        assert (ismethod (F, 'cond')) ;
        assert (ismethod (F, 'norm')) ;
        assert (ismethod (F, 'rank')) ;
        assert (ismethod (F, 'null')) ;
        assert (ismethod (F, 'orth')) ;
        assert (ismethod (F, 'pinv')) ;
        assert (ismethod (F, 'svd')) ;
        if (burble)
            methods (F)
        end
    else
        assert (~ismethod (F, 'cond')) ;
        assert (~ismethod (F, 'norm')) ;
        assert (~ismethod (F, 'rank')) ;
        assert (~ismethod (F, 'null')) ;
        assert (~ismethod (F, 'orth')) ;
        assert (~ismethod (F, 'pinv')) ;
        assert (~ismethod (F, 'svd')) ;
    end
end

%-------------------------------------------------------------------------------
% 1-norm and condition estimates
%-------------------------------------------------------------------------------

if (m == n)

    % norm(A,1)
    reset_rand ;
    if (isempty (A))
        nest1 = 0 ;
    else
        nest1 = full (normest1 (A)) ;
    end
    if (isa (F, 'factorization_svd'))
        nexact = norm (F, 1) ;
    else
        nexact = norm (A, 1) ;
    end
    if (burble)
        fprintf ('\nnorm(A,1), exact:             %g\n', nexact) ;
        fprintf ('  MATLAB normest1(A)          %g\n', nest1) ;
    end

    % norm(inv(A),1)
    if (isempty (A))
        imine = 0 ;
    else
        imine = full (normest1 (inverse (A))) ;
    end
    if (isa (F, 'factorization_svd'))
        iexact = norm (inverse (F), 1) ;
    else
        iexact = norm (inv (A), 1) ;
    end
    try
        reset_rand ;
        iest = full (normest1 (inv (A))) ;
    catch %#ok
        iest = -1 ;
    end
    if (burble)
        fprintf ('\nnorm (inv(A),1) exact:        %g\n', iexact) ;
        fprintf ('  MATLAB normest1 (inv (A)):  %g\n', iest) ;
        fprintf ('  normest1 (inverse (F)):     %g\n', imine) ;
    end

    reset_rand ;
    kest = full (condest (A)) ;
    reset_rand ;
    kF = full (condest (F)) ;
    kS = full (condest (inverse (F))) ;
    err = max (err, abs (rankest (F) - F.A_rank)) ;

    kexact = -1 ;
    kexact2 = -1 ;
    kexact3 = -1 ;
    if (isa (F, 'factorization_svd'))
        try
            kexact = cond (F, 1) ;
            kexact1 = F.A_cond ;
            kexact2 = cond (F) ;
            err = max (err, abs (kexact1-kexact2)) ;
            kexact3 = cond (full (A)) ;
            err = max (err, abs (kexact2-kexact3) / max (1, abs (kexact3))) ;
        catch %#ok
        end
    end

    if (burble)
        fprintf ('\n  cond (A,1), exact:          %g\n', kexact) ;
        fprintf ('  MATLAB condest(A):          %g\n', kest) ;
        fprintf ('  condest(F):                 %g\n', kF) ;
        fprintf ('  condest(inverse(A)):        %g\n', kS) ;
        fprintf ('  cond (A,2), exact:          %g\n', kexact2) ;
        fprintf ('  cond (F,2), exact:          %g\n', kexact3) ;
        fprintf ('  rankest %d %d\n', rankest (F), F.A_rank) ;
        if (~isempty (F.A_condest))
            fprintf ('  cheap condest:              %g\n', F.A_condest) ;
        end
    end

    if (err > 1e-9)
        display (A) ;
        display (full (A)) ;
        display (strategy) ;
        display (err) ;
        error ('error too high!') ;
    end
end

%-------------------------------------------------------------------------------
% compute explicit entries of the inverse

if (~any_nans && ~isempty (A))
    P1 = pinv (full (A)) ;
    P2 = inverse (A) ;
    s1 = P1 (1) ;
    s2 = P2 (1) ;
    err = max (err, abs (s1-s2) / max (1, abs (s1))) ;
    s1 = P1 (:,1) ;
    s2 = P2 (:,1) ;
    err = max (err, norm (s1-s2,1) / max (1, norm (s1,1))) ;
    s1 = P1 (1,:) ;
    s2 = P2 (1,:) ;
    err = max (err, norm (s1-s2,1) / max (1, norm (s1,1))) ;
end

%-------------------------------------------------------------------------------
% test struct
%-------------------------------------------------------------------------------

K = struct (F) ;
if (burble)
    disp ('K =') ;
    disp (K) ;
end

if (~any_nans)
    assert (isequal (K.A, F.A)) ;
    assert (isequal (K.Factors, F.Factors)) ;
end
assert (isequal (K.kind, F.kind)) ;
assert (isequal (K.is_inverse, F.is_inverse)) ;
assert (isequal (K.is_ctrans, F.is_ctrans)) ;
assert (isequal (K.alpha, F.alpha)) ;

K = struct (inverse (F)') ;
if (burble)
    disp ('K =') ;
    disp (K) ;
end

if (~any_nans)
    assert (isequal (K.A, F.A)) ;
    assert (isequal (K.Factors, F.Factors)) ;
end
assert (isequal (K.is_inverse, ~F.is_inverse)) ;
assert (isequal (K.is_ctrans, ~F.is_ctrans)) ;
assert (isequal (K.alpha, F.alpha)) ;
assert (isequal (K.kind, F.kind)) ;

% restore user's warnings
warning (state) ;

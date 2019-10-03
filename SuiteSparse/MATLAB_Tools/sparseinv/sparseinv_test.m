function sparseinv_test (extensive)
%SPARSEINV_TEST tests the sparseinv function.
%
% Example
%   sparseinv_test ;        % basic test
%   sparseinv_test (1) ;    % extensive test (requires ssget)
%
% See also sparseinv, sparseinv_install, ssget.

% Copyright 2011, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    extensive = 0 ;
end

load west0479 ;
A = west0479 ;

for k = 1:2

    [Z, Zpattern, L, D, U, P, Q, stats] = sparseinv (A) ;

    n = size (A,1) ;
    I = speye (n) ;

    S = inv (A) ;
    Snorm = norm (S, 1) ;
    e1 = norm (Zpattern.*(Z-S), 1) / Snorm ;
    e2 = norm ((L+I)*D*(U+I) - P*A*Q, 1) / norm (A,1) ;
    c = condest (A) ;

    fprintf ('west0479: errors %g %g condest %g : ', e1, e2, c) ;
    if (e1 > 1e-10 || e2 > 1e-10)
        error ('west0479 test failed') ;
    end
    fprintf ('ok\n') ;
    disp (stats) ;

    % create a symmetric positive definite matrix to test with
    if (k == 1)
        A = A+A' + 1e6*I ;
    end
end

% check error-handling
fprintf ('testing error handling (errors below are expected)\n') ;
ok = 1 ;
A = ones (2) ;
try
    Z1 = sparseinv (A) ;                                                    %#ok
    ok = 0 ;
catch me
    fprintf ('    expected error: %s\n', me.message) ;
end
A = sparse (ones (3,2)) ;
try
    Z2 = sparseinv (A) ;                                                    %#ok
    ok = 0 ;
catch me
    fprintf ('    expected error: %s\n', me.message) ;
end
A = sparse (ones (3)) ;
try
    Z3 = sparseinv (A) ;                                                    %#ok
    ok = 0 ;
catch me
    fprintf ('    expected error: %s\n', me.message) ;
end
A = 1i * sparse (ones (3)) ;
try
    Z4 = sparseinv (A) ;                                                    %#ok
    ok = 0 ;
catch me
    fprintf ('    expected error: %s\n', me.message) ;
end

if (~ok)
    error ('error-handling tests failed') ;
end

% now try with lots of matrices from the SuiteSparse Matrix Collection
if (extensive && exist ('ssget', 'file') == 2)

    fprintf ('Now doing extensive tests with SuiteSparse Matrix Collection:\n') ;
    dofigures = (exist ('cspy', 'file') == 2) ;
    if (dofigures)
        clf ;
    end

    index = ssget ;
    f = find ((index.nrows == index.ncols) & (index.isReal == 1)) ;
    [ignore,i] = sort (index.nrows (f)) ;   %#ok
    f = f (i) ;
    nmat = length (f) ;

    s = warning ('off', 'MATLAB:nearlySingularMatrix') ;

    for k = 1:nmat
        id = f (k) ;
        Prob = ssget (id, index) ;
        A = Prob.A ;
        n = size (A,1) ;
        I = speye (n) ;
        fprintf ('id: %4d  n: %4d : %-30s', id, n, Prob.name) ;

        Z = [ ] ; 
        try
            [Z, Zpattern, L, D, U, P, Q] = sparseinv (A) ;
        catch me
            fprintf ('%s', me.message) ;
        end

        if (~isempty (Z))
            e = norm ((L+I) * D * (U+I) - P*A*Q, 1) / norm (A,1) ;
            fprintf ('errs:  %12.3e ', e) ;
            if (e > 1e-10)
                error ('error in factorization too high') ;
            end
            S = inv (A) ;           % normally S has MANY nonzero entries
            Snorm = norm (S,1) ;
            E = abs (Zpattern .* (Z-S)) / Snorm ;
            e = norm (E, 1) ;
            c = condest (A) ;
            fprintf (' %12.3e  condest: %12.2e', e, c) ;
            if (e/c  > 1e-8)
                error ('error in sparseinv too high') ;
            end
            fprintf (' ok') ;

            if (dofigures)
                subplot (2,2,1) ; cspy (A) ;
                title (Prob.name, 'Interpreter', 'none') ;
                subplot (2,2,2) ; cspy (P*A*Q) ;  title ('P*A*Q') ;
                subplot (2,2,3) ; cspy (Z) ;      title ('sparse inverse') ;
                subplot (2,2,4) ; cspy (S) ;      title ('inverse') ;
                drawnow
            end

        end
        fprintf ('\n') ;
        if (n >= 300)
            break ;
        end
    end

    warning (s) ;       % restore warning status
end

fprintf ('All sparseinv tests passed.\n') ;

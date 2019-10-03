function err = test_performance
%TEST_PERFORMANCE compare performance of factorization/solve methods.
% Returns the largest relative residual seen.
%
% Example
%   err = test_performance
%
% See also test_all, factorize, inverse, mldivide

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\nPerformance comparisons of 4 methods:\n') ;
fprintf ('    backslash:  A\\b, or L\\b (and related) for solve times.\n') ;
fprintf ('    linsolve:   a built-in MATLAB function\n') ;
fprintf ('    factorize:  the factorization object\n') ;
fprintf ('    inv:        x=inv(A)*b, the explicit inverse (ack!)\n') ;
fprintf ('Run times are in seconds.\n') ;
fprintf ('Time relative to best time is in parentheses (lower is better).\n') ;

% linsolve options:
A_is_posdef.POSDEF = true ;     % for x = A\b where A is sym. pos. definite
A_is_posdef.SYM = true ;
lsolve.LT = true ;              % for x = L\b where L is lower triangular
usolve.UT = true ;              % for x = U\b where U is upper triangular
ltsolve.LT = true ;             % for x = L'b where L is lower triangular
ltsolve.TRANSA = true ;

nn = [50 100 500 1000] ;        % matrix sizes to test
ns = length (nn) ;
tmax = 1 ;                      % minimum testing time
err = 0 ;                       % largest relative residual seen 

for posdef = 0:1

    if (posdef)
        fprintf ('\n------------------ For positive definite matrices:\n');
        strategy = 'chol' ;
    else
        fprintf ('\n------------------ For unsymmetric matrices:\n') ;
        strategy = 'default' ;
    end

    Tfactor = zeros (ns,4) ;
    Tsolve  = zeros (ns,4) ;

    %----------------------------------------------------------------------
    % compare factorizations times (plus a single solve)
    %----------------------------------------------------------------------

    fprintf ('\nCompare factorization times:\n') ;
    for i = 1:ns
        n = nn (i) ;

        fprintf ('n %4d ', n) ;

        A = rand (n) ;
        if (posdef)
            A = A'*A + eye (n) ;
        end
        anorm = norm (A,1) ;
        b = rand (n,1) ;

        % method 1: backslash
        t = 0 ;
        k = 0 ;
        tic
        while (t < tmax)
            x = A\b ;
            k = k + 1 ;
            t = toc ;
        end
        Tfactor (i,1) = t / k ;
        err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
        clear x

        if (posdef)

            % method 2: linsolve (pos. definite case)
            t = 0 ;
            k = 0 ;
            tic
            while (t < tmax)
                x = linsolve (A,b, A_is_posdef) ;
                k = k + 1 ;
                t = toc ;
            end
            Tfactor (i,2) = t / k ;
            err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
            clear x

        else

            % method 2: linsolve (unsymmetric case)
            t = 0 ;
            k = 0 ;
            tic
            while (t < tmax)
                x = linsolve (A,b) ;
                k = k + 1 ;
                t = toc ;
            end
            Tfactor (i,2) = t / k ;
            err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
            clear x

        end

        % method 3: factorize method
        t = 0 ;
        k = 0 ;
        tic
        while (t < tmax)
            F = factorize (A, strategy) ;
            x = F\b ;
            k = k + 1 ;
            t = toc ;
        end
        Tfactor (i,3) = t / k ;
        err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
        clear x F

        % method 4: inv method (ack!)
        t = 0 ;
        k = 0 ;
        tic
        while (t < tmax)
            S = inv (A) ;
            x = S*b ;                                                  %#ok
            k = k + 1 ;
            t = toc ;
        end
        Tfactor (i,4) = t / k ;
        err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
        clear x S

        tbest = min (Tfactor (i,:)) ;
        fprintf ('tbest %10.6f :\n', tbest) ;
        fprintf ('    backslash (%5.2f)\n', Tfactor (i,1) / tbest);
        fprintf ('    linsolve  (%5.2f)\n', Tfactor (i,2) / tbest) ;
        fprintf ('    factorize (%5.2f)\n', Tfactor (i,3) / tbest) ;
        fprintf ('    inv       (%5.2f)\n', Tfactor (i,4) / tbest) ;

    end

    %----------------------------------------------------------------------
    % compare solve times
    %----------------------------------------------------------------------

    fprintf ('\nCompare solve times:\n') ;
    for i = 1:ns
        n = nn (i) ;

        fprintf ('n %4d ', n) ;

        A = rand (n) ;
        if (posdef)
            A = A'*A + eye (n) ;
        end
        b = rand (n,1) ;
        anorm = norm (A,1) ;

        if (posdef)
            L = chol (A, 'lower') ;
        else
            [L, U, p] = lu (A,'vector') ;
        end

        S = inv (A) ;
        F = factorize  (A) ;

        if (posdef)

            % method 1: backslash (pos. definite case)
            t = 0 ;
            k = 0 ;
            tic
            while (t < tmax)
                x = L' \ (L\b) ;
                k = k + 1 ;
                t = toc ;
            end
            Tsolve (i,1) = t / k ;
            err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
            clear x

            % method 2: linsolve (pos. definite case)
            t = 0 ;
            k = 0 ;
            tic
            while (t < tmax)
                x = linsolve (L, linsolve (L, b, lsolve), ltsolve) ;
                k = k + 1 ;
                t = toc ;
            end
            Tsolve (i,2) = t / k ;
            err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
            clear x

        else

            % method 1: backslash (unsymmetric case)
            t = 0 ;
            k = 0 ;
            tic
            while (t < tmax)
                x = U \ (L \ (b (p))) ;
                k = k + 1 ;
                t = toc ;
            end
            Tsolve (i,1) = t / k ;
            err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
            clear x

            % method 2: linsolve (unsymmetric case)
            t = 0 ;
            k = 0 ;
            tic
            while (t < tmax)
                x = linsolve (U, linsolve (L, b (p), lsolve), usolve) ;
                k = k + 1 ;
                t = toc ;
            end
            Tsolve (i,2) = t / k ;
            err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
            clear x

        end

        % method 3: factorize object
        t = 0 ;
        k = 0 ;
        tic
        while (t < tmax)
            x = F\b ;
            k = k + 1 ;
            t = toc ;
        end
        Tsolve (i,3) = t / k ;
        err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
        clear x

        % method 4: "inv" method (ack!)
        t = 0 ;
        k = 0 ;
        tic
        while (t < tmax)
            x = S*b ;           %#ok
            k = k + 1 ;
            t = toc ;
        end
        Tsolve (i,4) = t / k ;
        err = max (err, norm (A*x-b,1) / (anorm + norm (x,1))) ;
        clear x

        tbest = min (Tsolve (i,:)) ;

        fprintf ('tbest %10.6f :\n', tbest) ;
        fprintf ('    backslash (%5.2f)\n', Tsolve (i,1) / tbest);
        fprintf ('    linsolve  (%5.2f)\n', Tsolve (i,2) / tbest) ;
        fprintf ('    factorize (%5.2f)\n', Tsolve (i,3) / tbest) ;
        fprintf ('    inv       (%5.2f)\n', Tsolve (i,4) / tbest) ;

        clear F S L U p

    end

    % determine break-even values for inv
    fprintf ('\nBreak-even values K for inv vs the other methods\n') ;
    fprintf ('(# of solves must exceed K for inv(A)*b to be faster):\n') ;
    for i = 1:ns
        n = nn (i) ;
        s = zeros (3,1) ;
        for t = 1:3
            if (Tfactor (i,4) > Tfactor (i,t) && ...
                Tsolve (i,4) > Tsolve (i,t))
                % inv is always slower, for any K
                s (t) = inf ;
            elseif (Tfactor (i,4) < Tfactor (i,t) && ...
                Tsolve (i,4) < Tsolve (i,t))
                % inv is always faster, for any K
                s (t) = 1 ;
            else
                s (t) = (Tfactor (i,4) - Tfactor (i,t)) / ...
                         (Tsolve (i,t) - Tsolve (i,4)) ;
            end
        end

        fprintf ('n %4d\n', n) ;
        fprintf ('    # solves vs backslash  %8.1f\n', max (1, s (1))) ;
        fprintf ('    # solves vs linsolve:  %8.1f\n', max (1, s (2))) ;
        fprintf ('    # solves vs factorize: %8.1f\n', max (1, s (3))) ;
    end

end

%--------------------------------------------------------------------------
% compute the Schur complement, S = A-B*inv(D)*C
%--------------------------------------------------------------------------

fprintf ('\nSchur complement, S=A-B*inv(D)*C or A-B(D\\C),\n') ;
fprintf ('where A, B, C, and D are square and unsymmetric.\n') ;
fprintf ('"inverse" means S=A-B*inverse(D)*C, which does not actually\n') ;
fprintf ('use the inverse, but uses the factorization object instead.\n') ;

Tschur = zeros (ns,5) ;

for i = 1:ns
    n = nn (i) ;
    fprintf ('n %4d ', n) ;

    A = rand (n) ;
    B = rand (n) ;
    C = rand (n) ;
    D = rand (n) ;

    % method 1: backslash
    t = 0 ;
    k = 0 ;
    tic
    while (t < tmax)
        S = A - B*(D\C) ;                                              %#ok
        k = k + 1 ;
        t = toc ;
    end
    Tschur (i,1) = t / k ;
    clear S

    % method 2: linsolve
    t = 0 ;
    k = 0 ;
    tic
    while (t < tmax)
        S = A - B * linsolve (D,C) ;                                   %#ok
        k = k + 1 ;
        t = toc ;
    end
    Tschur (i,2) = t / k ;
    clear S

    % method 3: factorize object
    t = 0 ;
    k = 0 ;
    tic
    while (t < tmax)
        S = A - B*(factorize(D)\C) ;                                   %#ok
        k = k + 1 ;
        t = toc ;
    end
    Tschur (i,3) = t / k ;
    clear F S

    % method 4: inverse, with the factorize object
    t = 0 ;
    k = 0 ;
    tic
    while (t < tmax)
        S = A - B*inverse(D)*C ;                                       %#ok
        k = k + 1 ;
        t = toc ;
    end
    Tschur (i,4) = t / k ;
    clear S

    % method 5: "inv" method, using the explicit inverse (ack!)
    t = 0 ;
    k = 0 ;
    tic
    while (t < tmax)
        S = A - B*inv(D)*C ;                                           %#ok
        k = k + 1 ;
        t = toc ;
    end
    Tschur (i,5) = t / k ;
    clear S

    tbest = min (Tschur (i,:)) ;

    fprintf ('tbest %10.6f :\n', tbest) ;
    fprintf ('    backslash (%5.2f)\n', Tschur (i,1) / tbest);
    fprintf ('    linsolve  (%5.2f)\n', Tschur (i,2) / tbest) ;
    fprintf ('    factorize (%5.2f)\n', Tschur (i,3) / tbest) ;
    fprintf ('    inv       (%5.2f)\n', Tschur (i,4) / tbest) ;

end


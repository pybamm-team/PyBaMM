function lxtest
%LXTEST test the lsubsolve mexFunction
% Example:
%   lxtest
% See also cholmod_test, ltest, ltest2

% Copyright 2013, Timothy A. Davis, http://www.suitesparse.com

rng ('default')
index = UFget ;

%{
f = find (index.posdef & index.amd_lnz > 0) ;
f = setdiff (f, 1425) ; % not really posdef
[ignore i] = sort (index.amd_lnz (f)) ;
f = f (i) ;
%}

f = [ 1440 1438 57 2203 72 2204 60 436 872 873 874 25 61 70 23 220 44 217 ...
    69 63 64 315 2 66 76 ] ;

nmat = length (f) ;

for k = 1:nmat
    id = f (k) ;
    Prob = UFget (id, index)
    A = Prob.A ;
    n = size (A,1) ;
    [LD gunk p] = ldlchol (A) ;
    C = A (p,p) ;
    [count h parent post Lpattern] = symbfact (C, 'sym', 'lower') ;
    if (~isequal (Lpattern, spones (LD)))
        error ('!') ;
    end

    P = sparse (1:n, p, 1) ;

    L = speye (n) + tril (LD,-1) ;
    D = triu (LD) ;
    err = norm (L*D*L' - C, 1) / norm (C, 1) ;
    fprintf ('err %g in LDL''-C\n', err) ;
    if (err > 1e-12)
        error ('!') ;
    end

    D2 = chol (D) ;
    L2 = L*D2 + 1e-50 * spones (L) ;
    if (~isequal (spones (L), spones (L2)))
        error ('oops') ;
    end
    err = norm (L2*L2' - C, 1) / norm (C, 1) ;
    fprintf ('err %g in LL''-C\n', err) ;
    if (err > 1e-12)
        error ('!') ;
    end

    % test lsubsolve
    for i = 1:n
        b = sparse (i, 1, rand(1), n, 1) ;
        [err x1 x2 xset] = ltest2 (LD, L, D, L2, P, p, b, err) ;
        if (err > 1e-12)
            error ('!') ;
        end
    end

    for trial = 1:100
        b = sprand (n, 1, trial/100) ;
        [err x1 x2 xset] = ltest2 (LD, L, D, L2, P, p, b, err) ;
        if (err > 1e-12)
            error ('!') ;
        end
    end

    b = sparse (rand (n,1)) ;
    [err x1 x2 xset] = ltest2 (LD, L, D, L2, P, p, b, err) ;
    fprintf ('err %g in solves\n', err) ;
    if (err > 1e-12)
        error ('!') ;
    end
end

fprintf ('lxtest: all tests passed\n') ;

function ltest
%LTEST test lxbpattern
% Example:
%   ltest
% See also cholmod_test, lxtest, ltest2

% Copyright 2013, Timothy A. Davis, http://www.suitesparse.com

rng ('default')
index = UFget ;

%{
f = find (index.nrows == index.ncols & index.amd_lnz > 0) ;
f = setdiff (f, 1425) ; % not really posdef
[ignore i] = sort (index.amd_lnz (f)) ;
f = f (i) ;
%}

f = [ 449 1440 185 56 238 1438 186 57 13 58 97 2203 14 59 72 1177 2204 60 ...
      436 103 133 232 274 132 109 ] ;

nmat = length (f) ;

for k = 1:nmat
    id = f (k) ;
    Prob = UFget (id, index)
    A = spones (Prob.A) ;
    n = size (A,1) ;
    A = A+A' ;
    p = amd (A) ;
    A = A (p,p) ;
    [count, h, parent, post, L] = symbfact (A, 'sym', 'lower') ;
    L = sparse (1:n, 1:n, count+1, n, n) - sprand (L) / n ;

    % test lxbpattern
    for i = 1:n
        b = sparse (i, 1, rand(1), n, 1) ;

        x1 = find (L\b) ;
        x2 = lxbpattern (L, b) ;
        s2 = sort (x2)' ;
        if (~isequal (x1, s2))
            error ('!') ;
        end

    end

    for trial = 1:100
        b = sprand (n, 1, trial/100) ;

        x1 = find (L\b) ;
        x2 = lxbpattern (L, b) ;
        s2 = sort (x2)' ;
        if (~isequal (x1, s2))
            error ('!') ;
        end

    end

    b = sparse (rand (n,1)) ;

    x1 = find (L\b) ;
    x2 = lxbpattern (L, b) ;
    s2 = sort (x2)' ;
    if (~isequal (x1, s2))
        error ('!') ;
    end
end

fprintf ('ltest: all tests passed\n') ;

function [err x1 x2 xset] = ltest2 (LD, L, D, L2, P, p, b, err)
%LTEST2 test the lsubsolve mexFunction
% Example:
%   [err x1 x2 xset] = ltest2 (LD, L, D, L2, P, p, b, err)
%
% See also cholmod_test, ltest, ltest2

% Copyright 2013, Timothy A. Davis, http://www.suitesparse.com

if (~isreal (L))
    b = b * (pi + 1i) ;
end

for sys = 0:8

    %---------------------------------------------------------------------------
    % solve with LDL' = A
    %---------------------------------------------------------------------------

    % solve for all of x
    switch sys

        case 0
            x1 = P' * (L' \ (D \ (L \ (P * b)))) ;        % solve Ax = b

        case 1
            x1 =      (L' \ (D \ (L \ (    b)))) ;        % solve LDL'x = b

        case 2
            x1 =      (     (D \ (L \ (    b)))) ;        % solve LDx = b

        case 3
            x1 =      (L' \ (D \ (    (    b)))) ;        % solve DL'x = b
            
        case 4
            x1 =      (     (    (L \ (    b)))) ;        % solve Lx = b

        case 5
            x1 =      (L' \ (    (    (    b)))) ;        % solve L'x = b

        case 6
            x1 =      (     (D \ (    (    b)))) ;        % solve Dx = b

        case 7
            x1 =      (     (    (    (P * b)))) ;        % x = Pb

        case 8
            x1 = P' * (     (    (    (    b)))) ;        % x = P'b
    end

    % solve only for entries in xset, using lsubsolve.
    % xset contains the pattern of b, and the reach of b in the graph of L
    kind = 1 ;  % LDL'
    [x2 xset] = lsubsolve (LD, kind, p, b, sys) ;
    xset = xset'' ;
    spok (xset) ;
    err = max (err, norm ((x1 - x2).*xset, 1) / norm (x1,1)) ;
    if (err > 1e-12)
        sys
        warning ('LDL''!') ;
        return
    end

    %---------------------------------------------------------------------------
    % solve with L2*L2' = A
    %---------------------------------------------------------------------------

    % solve for all of x
    switch sys

        case 0
            x1 = P' * (L2' \ (    (L2 \ (P * b)))) ;        % solve Ax = b

        case 1
            x1 =      (L2' \ (    (L2 \ (    b)))) ;        % solve L2L2'x = b

        case 2
            x1 =      (      (    (L2 \ (    b)))) ;        % solve L2x = b

        case 3
            x1 =      (L2' \ (    (     (    b)))) ;        % solve L2'x = b
            
        case 4
            x1 =      (      (    (L2 \ (    b)))) ;        % solve L2x = b

        case 5
            x1 =      (L2' \ (    (     (    b)))) ;        % solve L2'x = b

        case 6
            x1 =      (      (    (     (    b)))) ;        % solve Dx = b

        case 7
            x1 =      (      (    (     (P * b)))) ;        % x = Pb

        case 8
            x1 = P' * (      (    (     (    b)))) ;        % x = P'b
    end

    % solve only for entries in xset, using lsubsolve.
    % xset contains the pattern of b, and the reach of b in the graph of L2
    kind = 0 ;  % L2*L2'
    [x2 xset] = lsubsolve (L2, kind, p, b, sys) ;
    xset = xset'' ;
    spok (xset) ;
    err = max (err, norm ((x1 - x2).*xset, 1) / norm (x1,1)) ;
    if (err > 1e-12)
        sys
        warning ('LL''!') ;
        return
    end

end

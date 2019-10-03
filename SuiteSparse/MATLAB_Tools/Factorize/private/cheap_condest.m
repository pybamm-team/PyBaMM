function c = cheap_condest (d, fail_if_singular)
%CHEAP_CONDEST checks the diagonal of a triangular matrix

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (isempty (d))
    dmin = 1 ;
    dmax = 1 ;
else
    d = abs (d) ;
    dmin = min (d) ;
    dmax = max (d) ;
end
if (dmin == 0)
    if (fail_if_singular)
        error ('MATLAB:singularMatrix', ...
            'Matrix is singular to working precision.');
    else
        warning ('MATLAB:singularMatrix', ...
            'Matrix is singular to working precision.');
    end
elseif (dmin < 2 * eps * (dmax))
    % MATLAB treats this as a warning, but it is treated here as an error
    % so that F=factorize(A) will abandon this factorization and use a
    % better one, in its default strategy.
    if (fail_if_singular)
        error ('MATLAB:nearlySingularMatrix', ...
            ['Matrix is close to singular or badly scaled.\n' ...
             '         Results may be inaccurate. RCOND = %g'], dmin / dmax) ;
    else
        warning ('MATLAB:nearlySingularMatrix', ...
            ['Matrix is close to singular or badly scaled.\n' ...
             '         Results may be inaccurate. RCOND = %g'], dmin / dmax) ;
    end
end
if (dmin == 0) 
    c = inf ;
else
    c = dmax / dmin ;
end


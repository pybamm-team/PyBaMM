function test5
%TEST5 test cs_add
%
% Example:
%   test5
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0) ;

for trial = 1:200
    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    d = rand (1) ;
    A = sprandn (m,n,d) ;
    B = sprandn (m,n,d) ;

    if (~ispc)
        if (mod (trial, 4) == 0)
            A = A + 1i*sprand(A) ;
        end
        if (mod (trial, 2) == 0)
            B = B + 1i*sprand(B) ;
        end
    end

    C = A+B ;
    D = cs_add (A,B) ;
    err = nnz (spones (C) - spones (D)) ;
    if (err > 0)
        error ('nz!') ;
    end
    err = norm (C-D,1) ;
    fprintf ('m %3d n %3d nnz(A) %6d nnz(B) %6d nnz(C) %6d err %g\n', ...
        m, n, nnz(A), nnz(B), nnz(C), err) ;
    if (err > 1e-12)
        error ('!') ;
    end

    alpha = pi ;
    beta = 3 ;

    if (~ispc)
        if (rand () > .5)
            alpha = alpha + rand ( ) * 1i ;
        end
        if (rand () > .5)
            beta = beta + rand ( ) * 1i ;
        end
    end

    C = alpha*A+B ;
    D = cs_add (A,B,alpha) ;
    err = nnz (spones (C) - spones (D)) ;
    if (err > 0)
        error ('nz!') ;
    end
    err = norm (C-D,1) ;
    fprintf ('m %3d n %3d nnz(A) %6d nnz(B) %6d nnz(C) %6d err %g\n', ...
        m, n, nnz(A), nnz(B), nnz(C), err) ;
    if (err > 1e-12)
        error ('!') ;
    end

    C = alpha*A + beta*B ;
    D = cs_add (A,B,alpha,beta) ;
    err = nnz (spones (C) - spones (D)) ;
    if (err > 0)
        error ('nz!') ;
    end
    err = norm (C-D,1) ;
    fprintf ('m %3d n %3d nnz(A) %6d nnz(B) %6d nnz(C) %6d err %g\n', ...
        m, n, nnz(A), nnz(B), nnz(C), err) ;
    if (err > 1e-12)
        error ('!') ;
    end

end

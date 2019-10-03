function test4
%TEST4 test cs_multiply
%
% Example:
%   test4
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0) ;

for trial = 1:100
    m = fix (10 * rand (1)) ;
    n = fix (10 * rand (1)) ;
    k = fix (10 * rand (1)) ;
    d = rand (1) ;
    A = sprandn (m,n,d) ;
    B = sprandn (n,k,d) ;

    if (~ispc)
        if (mod (trial, 4) == 0)
            A = A + 1i * sprandn (A) ;
        end
        if (mod (trial, 2) == 0)
            B = B + 1i * sprandn (B) ;
        end
    end

    C = A*B ;
    D = cs_multiply (A,B) ;
    err = nnz (spones (C) - spones (D)) ;
    if (err > 0)
        error ('nz!') ;
    end
    err = norm (C-D,1) ;
    fprintf ('m %3d n %3d k %3d nnz(A) %6d nnz(B) %6d nnz(C) %6d err %g\n', ...
        m, n, k, nnz(A), nnz(B), nnz(C), err) ;
    if (err > 1e-12)
        error ('!') ;
    end
end

function err = test_cod (A, tol)
%TEST_COD test the COD, COD_SPARSE and RQ functions
%   This function does not test the factorize object itself, but some of the
%   factorization methods it relies on.
%
% Example
%   err = test_cod (A, tol)
%
% See also test_all_cod, test_all.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    A = magic (4) ;
end
err = 0 ;
[m, n] = size (A) ;

if (issparse (A))

    [U, R, V, r] = cod_sparse (A) ;                                         %#ok

    % 1-norm of A - U*R*V'
    err = max (err, norm (A - cod_qmult (U, cod_qmult (V, R, 2),1),1)) ;

    Umat = cod_qmult (U, speye (size (A,1)), 1) ;      % convert U into a matrix
    Vmat = cod_qmult (V, speye (size (A,2)), 1) ;      % convert V into a matrix
    err = max (err, norm (A - Umat*R*Vmat',1)) ;

    % test U'*x
    x = rand (size (Umat,1), 1) ;
    y1 = Umat'*x ;
    y2 = cod_qmult (U, x) ;                             % default method 0
    y3 = cod_qmult (Umat, x) ;
    err = max (err, norm (y1 - y2, 1)) ;
    err = max (err, norm (y1 - y3, 1)) ;

    % test U*x
    x = rand (size (Umat,2), 1) ;
    y1 = Umat*x ;
    y2 = cod_qmult (U, x, 1) ;
    y3 = cod_qmult (Umat, x, 1) ;
    err = max (err, norm (y1 - y2, 1)) ;
    err = max (err, norm (y1 - y3, 1)) ;

    % test x*U'
    x = rand (1, size (Umat,2)) ;
    y1 = x*Umat' ;
    y2 = cod_qmult (U, x, 2) ;
    y3 = cod_qmult (Umat, x, 2) ;
    err = max (err, norm (y1 - y2, 1)) ;
    err = max (err, norm (y1 - y3, 1)) ;

    % test x*U
    x = rand (1, size (Umat,1)) ;
    y1 = x*Umat ;
    y2 = cod_qmult (U, x, 3) ;
    y3 = cod_qmult (Umat, x, 3) ;
    err = max (err, norm (y1 - y2, 1)) ;
    err = max (err, norm (y1 - y3, 1)) ;

    % test cod_sparse with matrix form of Q
    opts.Q = 'matrix' ;
    if (nargin == 2)
        opts.tol = tol ;
    end
    [U, R, V] = cod_sparse (A,opts) ;
    err = max (err, norm (A - U*R*V',1)) ;

    if (nargin == 2)
        [U, R, V] = cod_sparse (A,tol) ;
        U = cod_qmult (U, speye (size (A,1)), 1) ; 
        V = cod_qmult (V, speye (size (A,2)), 1) ;
        err = max (err, norm (A - U*R*V',1)) ;
    end

    try
        % this should cause an error
        [R, Q] = rq (A) ;                                                   %#ok
        err = inf ;
    catch                                                                   %#ok
    end

    try
        % this should cause an error
        [U, R, V, r] = cod (A) ;                                            %#ok
        err = inf ;
    catch                                                                   %#ok
    end

else

    if (nargin < 2)
        [U, R, V, r] = cod (A) ;                                            %#ok
    else
        [U, R, V, r] = cod (A, tol) ;                                       %#ok
    end
    err = max (err, norm (A - U*R*V',1)) ;

    if (m <= n)
        [R, Q] = rq (A) ;
        err = max (err, norm (A - R*Q,1)) ;
    else
        [L, Q] = rq (A) ;
        err = max (err, norm (A - Q*L,1)) ;
    end

    try
        % this should cause an error
        [U, R, V, r] = cod_sparse (A) ;                                     %#ok
        err = inf ;
    catch                                                                   %#ok
    end

end

err = err / norm (A,1) ;

if (err > 1e-12)
    error ('error too high! %g\n', err) ;
end

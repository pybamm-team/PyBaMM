function err = test_accuracy
%TEST_ACCURACY test the accuracy of the factorize object
%
% Example
%   err = test_accuracy
%
% See also test_all, test_factorize.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\nTesting accuracy:\n') ;
reset_rand ;

A = [ 0.1482    0.3952    0.1783    1.1601
      0.3952    0.3784    0.2811    0.4893
      0.1783    0.2811    1.1978    1.3837
      1.1601    0.4893    1.3837    0.7520 ] ;

F = factorize (A, 'ldl', 1) ;                                               %#ok
err = test_factorize (sparse (A)) ;
err = max (err, test_factorize (A)) ;

rect   = {'', 'qr', 'cod', 'svd' } ;        % methods for rectangular matrices
square = [{'lu'} rect] ;                    % for square matrices
sym    = [{'ldl', 'symmetric'} square] ;    % for symmetric
spd    = [{'chol'} sym] ;                   % for symmetric positive definite
square = [{'unsymmetric'} square] ;

%   rect
%   square
%   sym
%   spd
%   pause

fprintf ('please wait\n') ;

% small matrices: full and sparse
for n = 0:6
    for im = 0:1
        fprintf ('test %2d of 14 ', 2*n+im+1) ;

        % unsymmetric
        A = rand (n) ;
        if (im == 1)
            A = A + 1i * rand (n) ;
        end
        err = tfac (A, err, square) ;

        % dense, symmetric but not always positive definite
        A = A+A' ;
        err = tfac (A, err, sym) ;

        % symmetric positive definite
        A = A'*A + eye (n) ;
        err = tfac (A, err, spd) ;

        % least-squares problem
        A = rand (2*n,n) ;
        err = tfac (A, err, rect) ;

        % under-determined problem
        A = A' ;
        err = tfac (A, err, rect) ;
        fprintf ('\n') ;
    end
end
% default dense 100-by-100 matrix
err = max (err, test_factorize) ;

fprintf ('\nerr so far: %g\nplease wait ', err) ;

for im = 0:1

    % sparse rectangular
    A = sprandn (5,10,0.6) + speye (5,10) ;
    if (im == 1)
        A = A + 1i * sprandn (5,10,0.2) ;
    end
    err = tfac (A, err, rect) ;
    err = tfac (A', err, rect) ;

    % sparse, unsymmetric
    load west0479
    A = west0479 ;
    if (im == 1)
        A = A + 1i * sprand (A) ;
    end
    err = tfac (A, err, square) ;

    % sparse, symmetric, but not positive definite
    A = abs (A+A') + eps * speye (size (A,1)) ;
    err = tfac (A, err, sym) ;

    % sparse symmetric positive definite
    A = delsq (numgrid ('L', 8)) ;
    if (im == 1)
        A = A + 1i * sprand (A) ;
        A = A'*A ;
    end
    err = tfac (A, err, spd) ;

end

if (err > 1e-6)
    error ('error to high!  %g\n', err) ;
end

fprintf ('\nmax error is OK: %g\n', err) ;

%-------------------------------------------------------------------------------

function err = tfac (A, err, list)
for k = 1 : length (list)
    method = list {k} ;
    err = max (err, test_factorize (A, method)) ;
    if (~issparse (A))
        err = max (err, test_factorize (sparse (A), method)) ;
    end
end


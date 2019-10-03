function test_errors
%TEST_ERRORS tests error handling for the factorize object methods
%
% Example
%   test_errors
%
% See also test_all, factorize.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\nTesting error handling (error messages are expected)\n\n') ;
reset_rand ;
ok = true ;

% the matrix A must be 2D for factorize
A = ones (2,2,2) ; 
try
    F = factorize (A, [ ], 1) ;                                             %#ok
    ok = false ;
    fprintf ('error not caught: A must be 2D for factorize\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% check if invalid strategy is caught
A = rand (4) ;
try
    F = factorize (A, 'gunk', 1) ;                                          %#ok
    ok = false ;
    fprintf ('error not caught: invalid strategy\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% cannot use COD on sparse matrices
try
    [U, R, V, r] = cod (sparse (A)) ;                                       %#ok
    ok = false ;
    fprintf ('error not caught: cannot use cod for sparse A\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% cannot use RQ on sparse matrices
try
    [R, Q] = rq (sparse (A)) ;                                              %#ok
    ok = false ;
    fprintf ('error not caught: cannot use rq for sparse A\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% cannot do B\inverse(A) or inverse(A)/B
A = rand (2) ;
B = rand (2) ;
try
    C = B \ inverse (A) ;                                                   %#ok
    ok = false ;
    fprintf ('error not caught: requires explicit inverseA\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end
try
    C = inverse (A) / B ;                                                   %#ok
    ok = false ;
    fprintf ('error not caught: requires explicit inverseA\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% cannot use COD_SPARSE on full matrices
try
    [U, R, V, r] = cod_sparse (A) ;                                         %#ok
    ok = false ;
    fprintf ('error not caught: cannot use cod_sparse for full A\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

if (~ok)
    error ('!') ;
end

% cannot factorize a logical array, sparse or full
for s = {'default', 'symmetric', 'qr', 'lu', 'ldl', 'chol', 'svd', 'cod'}
    for sp = 0:1
        A = logical (ones (3)) ;                                            %#ok
        if (sp)
            A = sparse (A) ;
        end
        try
            F = factorize (A, char (s), 1) ;                                %#ok
            ok = false ;
            fprintf ('error not caught:\n') ;
        catch me
            fprintf ('\nExpected error: [%s]\n', me.message) ;
        end
    end
end
try
    A = logical (sparse (rand (3,4))) ;
    F = factorize (A, 'qr', 1) ;                                            %#ok
    ok = false ;
    fprintf ('error not caught:\n') ;
catch me
    fprintf ('\nExpected error: [%s]\n', me.message) ;
end

if (~ok)
    error ('!') ;
end

% the matrix A must be full-rank for ldl and chol
for s = {'ldl', 'chol'}
    for sp = 0:1
        A = ones (3) ;
        if (sp)
            A = sparse (A) ;
        end
        try
            F = factorize (A, char (s), 1) ;                                %#ok
            ok = false ;
            fprintf ('error not caught\n') ;
        catch                                                               %#ok
        end
        A = zeros (3,2) ;
        if (sp)
            A = sparse (A) ;
        end
        try
            F = factorize (A, char (s), 1) ;                                %#ok
            ok = false ;
            fprintf ('error not caught\n') ;
        catch                                                               %#ok
        end
        A = zeros (2,3) ;
        if (sp)
            A = sparse (A) ;
        end
        try
            F = factorize (A, char (s), 1) ;                                %#ok
            ok = false ;
            fprintf ('error not caught\n') ;
        catch                                                               %#ok
        end
    end
end

% cannot do LU, CHOL, or LDL on rectangular matrices
for s = {'lu', 'ldl', 'chol'}
    A = rand (3,2) ;
    for sp = 0:1
        try
            F = factorize (A, char (s), 1) ;                                %#ok
            ok = false ;
            fprintf ('\nerror not caught: tall-and-thin case\n') ;
        catch me
            fprintf ('\nExpected error: [%s]\n', me.message) ;
        end
        A = sparse (A) ;
    end
end

if (~ok)
    error ('!') ;
end

% cannot do QR on short-and-fat matrices or QRT on tall-and-thin matrices
try
    F = factorization_qr_dense (rand (2,3), 0) ;                            %#ok
    ok = false ;
    fprintf ('\nerror not caught: short-and-fat case\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end
try
    F = factorization_qr_sparse (sparse (rand (2,3)), 0) ;                  %#ok
    ok = false ;
    fprintf ('\nerror not caught: short-and-fat case\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end
try
    F = factorization_qrt_dense (rand (3,2), 0) ;                           %#ok
    ok = false ;
    fprintf ('\nerror not caught: tall-and-thin case\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end
try
    F = factorization_qrt_sparse (sparse (rand (3,2)), 0) ;                 %#ok
    ok = false ;
    fprintf ('\nerror not caught: tall-and-thin case\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end

if (~ok)
    error ('!') ;
end

% cannot do CHOL, or LDL on singular matrices
for s = {'ldl', 'chol'}
    A = zeros (2) ;
    for sp = 0:1
        try
            F = factorize (A, char (s), 1) ;                                %#ok
            ok = false ;
            fprintf ('\nerror not caught: singular case\n') ;
        catch me
            fprintf ('\nExpected error: [%s]\n', me.message) ;
        end
        A = sparse (A) ;
    end
end

if (~ok)
    error ('!') ;
end

% cannot use cell indexing
A = rand (3,2) ;
F = factorize (A, [ ], 1) ;
try
    C = F {1} ;                                                        %#ok
    ok = false ;
    fprintf ('\nerror not caught: cannot use cell indexing\n') ;
catch me
    fprintf ('\nExpected error: [%s]\n', me.message) ;
end

% invalid indexing
try
    C = F (1,1).L ;                                                    %#ok
    ok = false ;
    fprintf ('error not caught: invalid indexing\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% invalid indexing
try
    C = F.L (1,1).stuff ;                                              %#ok
    ok = false ;
    fprintf ('error not caught: invalid indexing\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% non-existent field
try
    C = F.junk ;                                                       %#ok
    ok = false ;
    fprintf ('error not caught: invalid field\n') ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% can only update/downdate a dense Cholesky factorization
A = rand (2) ;
F = factorize (A, [ ], 1) ;
w = rand (2,1) ;
try
    F = cholupdate (F,w) ;
    fprintf ('error not caught: cannot update this type of matrix\n') ;
    disp (F) ;
    ok = false ;
catch me
    fprintf ('\nExpected error: [%s]\n', me.message) ;
end
try
    F = choldowndate (F,w,'-') ;
    fprintf ('error not caught: cannot downdate this type of matrix\n') ;
    disp (F) ;
    ok = false ;
catch me
    fprintf ('Expected error: [%s]\n', me.message) ;
end

% cannot do condest(F) or cond(F,1) for rectangular matrices
try
    F = factorize (rand (4,3)) ;
    c = condest (F) ;                                                       %#ok
    ok = false ;
    fprintf ('\nerror not caught: condest for rectangular case\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end
try
    F = factorize (rand (4,3), 'svd') ;
    c = cond (F,1) ;                                                        %#ok
    ok = false ;
    fprintf ('\nerror not caught: cond(A,1) for rectangular case\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end

% test for invalid kind of svd
try
    F = factorize (rand (4,3), 'svd') ;
    [U, S, V] = svd (F,'gunk') ;                                            %#ok
    ok = false ;
    fprintf ('\nerror not caught: invalid kind of svd\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end

% test for invalid cholupdate parameter
try
    A = rand (2) ;
    A = A*A' ;
    F = factorize (A) ;
    G = cholupdate (F, ones(2,1), 'gunk') ;                                 %#ok
    ok = false ;
    fprintf ('\nerror not caught: invalid kind of cholupdate\n') ;
catch me
   fprintf ('\nExpected error: [%s]\n', me.message) ;
end

%-------------------------------------------------------------------------------

if (~ok)
    error ('error-handling failed') ;
end
fprintf ('\nAll error-handing tests passed\n') ;

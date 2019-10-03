function ssmult_test
%SSMULT_TEST lengthy test of SSMULT and SSMULTSYM
%
%   Example
%       ssmult_test
%
% See also ssmult, ssmultsym

% Copyright 2007-2011, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\nTesting large sparse column vectors (1e7-by-1)\n') ;
x = sprandn (1e7,1,1e-4) ;
y = sprandn (1e7,1,1e-4) ;
x (1) = pi ;
y (1) = exp (1) ;
tic ; a = x'*y ; t1 = toc ;
tic ; b = ssmult (x, y, 1) ; t2 = toc ;
fprintf ('s=x''*y in MATLAB: %8.3f seconds\n', t1) ;
fprintf ('s=ssmult(x,y,1):  %8.3f seconds; error %g\n', t2, abs (full(a-b))) ;
fprintf ('SSMULT speedup: %8.3g\n\n', t1/t2) ;

load west0479
A = west0479 ;
B = sprand (A) ;
C = A*B ;
D = ssmult (A,B) ;
err = norm (C-D,1) / norm (C,1) ;
fprintf ('west0479 error: %g\n', err) ;

fprintf ('\ntesting large matrices (may fail if you are low on memory):\n') 
rand ('state', 0) ;

n = 10000 ;
A = sprand (n, n, 0.01) ;
B = sprand (n, n, 0.001) ;
test_large (A,B) ;

msg = { 'real', 'complex' } ;

% all of these calls to ssmult should fail:
fprintf ('\ntesting error handling (the errors below are expected):\n') ;
A = { 3, 'gunk', sparse(1), sparse(1), sparse(rand(3,2)) } ;
B = { 4,   0   , 5,         msg,       sparse(rand(3,4)) } ;
for k = 1:length(A)
    try
        % the following statement is supposed to fail 
        C = ssmult (A {k}, B {k}) ;                                         %#ok
        error ('test failed\n') ;
    catch me
        disp (me.message) ;
    end
end
fprintf ('error handling tests: ok.\n') ;

% err should be zero:
rand ('state', 0)
for Acomplex = 0:1
    for Bcomplex = 0:1
        err = 0 ;
        fprintf ('\ntesting C = A*B where A is %s, B is %s\n', ...
            msg {Acomplex+1}, msg {Bcomplex+1}) ;
        for m = [ 0:30 100 ]
            fprintf ('.') ;
            for n = [ 0:30 100 ]
                for k = [ 0:30 100 ]
                    A = sprand (m,k,0.1) ;
                    if (Acomplex)
                        A = A + 1i*sprand (A) ;
                    end
                    B = sprand (k,n,0.1) ;
                    if (Bcomplex)
                        B = B + 1i*sprand (B) ;
                    end
                    C = A*B ;
                    D = ssmult (A,B) ;
                    s = ssmultsym (A,B) ;
                    err = max (err, norm (C-D,1)) ;
                    err = max (err, nnz (C-D)) ;
                    err = max (err, isreal (D) ~= (norm (imag (D), 1) == 0)) ;
                    err = max (err, s.nz > nnz (C)) ;
                    [i j x] = find (D) ;                                    %#ok
                    if (~isempty (x))
                        err = max (err, any (x == 0)) ;
                    end
                end
            end
        end
        fprintf (' maximum error: %g\n', err) ;
    end
end

sstest ;
fprintf ('\nSSMULT tests complete.\n') ;


%-------------------------------------------------------------------------------
function test_large (A,B)
% test large matrices
n = size (A,1) ;
fprintf ('dimension %d   nnz(A): %d   nnz(B): %d\n', n, nnz (A), nnz (B)) ;
c = ssmultsym (A,B) ;
fprintf ('nnz(C): %d   flops: %g   memory: %g MB\n', ...
    c.nz, c.flops, c.memory/2^20) ;
try
    % warmup for accurate timings
    C = A*B ;                                                               %#ok
    D = ssmult (A,B) ;                                                      %#ok
    tic ;
    C = A*B ;
    t1 = toc ;
    tic ;
    D = ssmult (A,B) ;
    t2 = toc ;
    tic ;
    t3 = toc ;
    fprintf ('MATLAB time:          %g\n', t1) ;
    err = norm (C-D,1) ;
    fprintf ('SSMULT time:          %g err: %g\n', t2, err) ;
catch me
    disp (me.message)
    fprintf ('tests with large random matrices failed ...\n') ;
end
clear C D


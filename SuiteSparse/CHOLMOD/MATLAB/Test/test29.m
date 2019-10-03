function test29
%TEST29 test spsym
% Example:
%   spsym
% See also cholmod_test

% Copyright 2015, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0) ;

r = zeros (0,2) ;
for n = 0:5

    % real unsymmetric, diagonal all nonzero
    A = sparse (rand (n,n)) ;
    r = [r ; test_spsym(A)] ;

    % real symmetric, diagonal all nonzero
    A = A+A' ;
    r = [r ; test_spsym(A)] ;

    % real unsymmetric, diagonal all nonzero
    A (2,1) = 0 ;
    r = [r ; test_spsym(A)] ;

    % real symmetric, diagonal all zero
    A = sparse (n,n) ;
    r = [r ; test_spsym(A)] ;

    % real symmetric, diagonal all nonzero
    A = speye (n,n) ;
    r = [r ; test_spsym(A)] ;

    % real symmetric, diagonal mostly nonzero
    A (2,2) = 0 ;
    r = [r ; test_spsym(A)] ;

    % real rectangular, or square unsymmetric
    for m = 0:5
        A = sparse (rand (n,m)) ;
        r = [r ; test_spsym(A)] ;
    end

    % skew symmetric when n > 1
    A = tril (sparse (rand (n,n)), -1) ;
    A = A-A' ;
    c = test_spsym(A) ;
    r = [r ; c ] ;

    % complex Hermitian (when n > 1)
    A = sparse (rand (n,n)) + 1i * sparse (rand (n,n)) ;
    A = A+A' ;
    c = test_spsym(A) ;
    r = [r ; c ] ;

    % complex Hermitian but with non-positive diagonal (when n > 1)
    A (3,3) = -1 ;
    c = test_spsym(A) ;
    r = [r ; c] ;

end

r = unique (r, 'rows') ;
rtrue =  [
     1     1
     2     2
     3     2
     4     2
     5     2
     6     6
     7     7 ] ;
if (~isequal (r, rtrue))
    error ('failed.  Incomplete test cases') ;
end

% test with the UF sparse matrix collection
r = zeros (0,2) ;
index = UFget ;
for i = [168 27 2137 56 231 1621 -1621] ;
    Prob = UFget (abs (i),index)
    A = Prob.A ;
    if (i < 0)
        % UF collection does not contain any matrices for which spsym(A) = 4.
        % (complex Hermitian with zero nonpos. diagonal).  So make one.
        fprintf ('setting A (5,5) = 0\n') ;
        A (5,5) = 0 ;
    end
    c = test_spsym (A) ;
    c
    fprintf ('full  test:') ; print_result (c (1)) ;
    fprintf ('quick test:') ; print_result (c (2)) ;
    r = [r ; c] ;
end

r = unique (r, 'rows') ;
if (~isequal (r, rtrue))
    error ('failed.  Incomplete test cases') ;
end

%-------------------------------------------------------------------------------

function r = test_spsym (A)
s1 = spsym (A) ;
s2 = get_symmetry (A) ;
if (s1 ~= s2)
    error ('failed!')
end
s3 = spsym (A,0) ;
s4 = get_symmetry (A,0) ;
if (s3 ~= s1 || s4 ~= s1)
    error ('failed!')
end
s5 = spsym (A,1) ;
s6 = get_symmetry (A,1) ;
if (s5 ~= s6)
    error ('failed!')
end
r = [s1 s5] ;   % r(1) is the full test, r(2) is the quick test

%-------------------------------------------------------------------------------

function print_result (s)
switch (s)
case 1
    fprintf ('rectangular\n') ;
case 2
    fprintf ('unsymmetric (or not Cholesky candidate for quick test)\n') ;
case 3
    fprintf ('symmetric, but with one or more A(j,j) <= 0\n') ;
case 4
    fprintf ('Hermitian, but with one or more A(j,j) <= 0 or with nonzero imaginary part\n') ;
case 5
    fprintf ('skew symmetric\n') ;
case 6
    fprintf ('symmetric with real positive diagonal\n') ;
case 7
    fprintf ('Hermitian with real positive diagonal\n') ;
otherwise
    error ('unknown result') ;
end


function sstest3                                                            %#ok
%SSTEST3: an exhaustive test of ssmult
%
% For a list of all 64 functions computed by ssmult, look at the source code
% of this test.
%
% See also ssmult, mtimes.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\nsstest3: Please wait ') ;

for m = 0:30
    fprintf ('.') ;
    for n = 0:30
        for k = 0:30

            for Acomplex = 0:1
                for Bcomplex = 0:1

                    A = sprand (m, n, 0.3) ;
                    if (Acomplex)
                        A = 1i*sprand (m, n, 0.3) ;
                    end

                    %-----------------------------------------------------------
                    % y = A*B and variants
                    %-----------------------------------------------------------

                    B = sprand (n, k, 0.3) ;
                    if (Bcomplex)
                        B = 1i*sprand (n, k, 0.3) ;
                    end

                    y = A*B ;
                    z = ssmult (A, B, 0,0, 0,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A*B) ;
                    z = ssmult (A, B, 0,0, 0,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*B).' ;
                    z = ssmult (A, B, 0,0, 0,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*B)' ;
                    z = ssmult (A, B, 0,0, 0,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A*conj(B) and variants
                    %-----------------------------------------------------------

                    y = A*conj(B) ;
                    z = ssmult (A, B, 0,0, 0,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A*conj(B)) ;
                    z = ssmult (A, B, 0,0, 0,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*conj(B)).' ;
                    z = ssmult (A, B, 0,0, 0,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*conj(B))' ;
                    z = ssmult (A, B, 0,0, 0,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = conj(A)*B and variants
                    %-----------------------------------------------------------

                    y = conj(A)*B ;
                    z = ssmult (A, B, 0,1, 0,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(conj(A)*B) ;
                    z = ssmult (A, B, 0,1, 0,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*B).' ;
                    z = ssmult (A, B, 0,1, 0,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*B)' ;
                    z = ssmult (A, B, 0,1, 0,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = conj(A)*conj(B) and variants
                    %-----------------------------------------------------------

                    y = conj(A)*conj(B) ;
                    z = ssmult (A, B, 0,1, 0,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(conj(A)*conj(B)) ;
                    z = ssmult (A, B, 0,1, 0,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*conj(B)).' ;
                    z = ssmult (A, B, 0,1, 0,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*conj(B))' ;
                    z = ssmult (A, B, 0,1, 0,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A.'*B and variants
                    %-----------------------------------------------------------

                    B = sprand (m, k, 0.3) ;
                    if (Bcomplex)
                        B = 1i*sprand (m, k, 0.3) ;
                    end

                    y = A.'*B ;
                    z = ssmult (A, B, 1,0, 0,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A.'*B) ;
                    z = ssmult (A, B, 1,0, 0,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*B).' ;
                    z = ssmult (A, B, 1,0, 0,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*B)' ;
                    z = ssmult (A, B, 1,0, 0,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A.'*conj(B) and variants
                    %-----------------------------------------------------------

                    y = (A.'*conj(B))' ;
                    z = ssmult (A, B, 1,0, 0,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*conj(B)).' ;
                    z = ssmult (A, B, 1,0, 0,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*conj(B)) ;
                    z = ssmult (A, B, 1,0, 0,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A.'*conj(B)) ;
                    z = ssmult (A, B, 1,0, 0,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A'*B and variants
                    %-----------------------------------------------------------

                    y = A'*B ;
                    z = ssmult (A, B, 1,1, 0,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A'*B) ;
                    z = ssmult (A, B, 1,1, 0,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*B).' ;
                    z = ssmult (A, B, 1,1, 0,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*B)' ;
                    z = ssmult (A, B, 1,1, 0,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A'*conj(B) and variants
                    %-----------------------------------------------------------

                    y = A'*conj(B) ;
                    z = ssmult (A, B, 1,1, 0,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A'*conj(B)) ;
                    z = ssmult (A, B, 1,1, 0,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*conj(B))' ;
                    z = ssmult (A, B, 1,1, 0,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*conj(B)).' ;
                    z = ssmult (A, B, 1,1, 0,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A*B.' and variants
                    %-----------------------------------------------------------

                    B = sprand (k,n, 0.3) ;
                    if (Bcomplex)
                        B = 1i*sprand (k, n, 0.3) ;
                    end

                    y = A*B.' ;
                    z = ssmult (A, B, 0,0, 1,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A*B.') ;
                    z = ssmult (A, B, 0,0, 1,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*B.').' ;
                    z = ssmult (A, B, 0,0, 1,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*B.')' ;
                    z = ssmult (A, B, 0,0, 1,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A*B' and variants
                    %-----------------------------------------------------------

                    y = A*B' ;
                    z = ssmult (A, B, 0,0, 1,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj (A*B') ;
                    z = ssmult (A, B, 0,0, 1,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*B').' ;
                    z = ssmult (A, B, 0,0, 1,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A*B')' ;
                    z = ssmult (A, B, 0,0, 1,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = conj(A)*B.' and variants
                    %-----------------------------------------------------------

                    y = conj(A)*B.' ;
                    z = ssmult (A, B, 0,1, 1,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(conj(A)*B.') ;
                    z = ssmult (A, B, 0,1, 1,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*B.').' ;
                    z = ssmult (A, B, 0,1, 1,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*B.')' ;
                    z = ssmult (A, B, 0,1, 1,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = conj(A)*B' and variants
                    %-----------------------------------------------------------

                    y = conj(A)*B' ;
                    z = ssmult (A, B, 0,1, 1,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(conj(A)*B') ;
                    z = ssmult (A, B, 0,1, 1,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*B').' ;
                    z = ssmult (A, B, 0,1, 1,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (conj(A)*B')' ;
                    z = ssmult (A, B, 0,1, 1,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A.'*B.' and variants
                    %-----------------------------------------------------------

                    B = sprand (k,m, 0.3)  ;
                    if (Bcomplex)
                        B = 1i*sprand (k, m, 0.3) ;
                    end

                    y = A.'*B.' ;
                    z = ssmult (A, B, 1,0, 1,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A.'*B.') ;
                    z = ssmult (A, B, 1,0, 1,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*B.').' ;
                    z = ssmult (A, B, 1,0, 1,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*B.')' ;
                    z = ssmult (A, B, 1,0, 1,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A'*B.' and variants
                    %-----------------------------------------------------------

                    y = A'*B.' ;
                    z = ssmult (A, B, 1,1, 1,0, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A'*B.') ;
                    z = ssmult (A, B, 1,1, 1,0, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*B.').' ;
                    z = ssmult (A, B, 1,1, 1,0, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*B.')' ;
                    z = ssmult (A, B, 1,1, 1,0, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A.'*B' and variants
                    %-----------------------------------------------------------

                    y = A.'*B' ;
                    z = ssmult (A, B, 1,0, 1,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj (A.'*B') ;
                    z = ssmult (A, B, 1,0, 1,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*B').' ;
                    z = ssmult (A, B, 1,0, 1,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A.'*B')' ;
                    z = ssmult (A, B, 1,0, 1,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    %-----------------------------------------------------------
                    % y = A'*B' and variants
                    %-----------------------------------------------------------

                    y = A'*B' ;
                    z = ssmult (A, B, 1,1, 1,1, 0,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = conj(A'*B') ;
                    z = ssmult (A, B, 1,1, 1,1, 0,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*B').' ;
                    z = ssmult (A, B, 1,1, 1,1, 1,0) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                    y = (A'*B')' ;
                    z = ssmult (A, B, 1,1, 1,1, 1,1) ;
                    err = norm (y-z,1) ;
                    if (err > 0)
                        err
                        error ('!') ;
                    end

                end
            end
        end
    end
end
fprintf ('\nsstest3: all tests passed') ;

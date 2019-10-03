function gee_its_simple_test
%GEE_ITS_SIMPLE_TEST tests the "Gee! It's Simple!" package
% Exhaustive test of the "Gee! It's Simple!" package.  Returns the largest
% relative residual for any solution to A*x=b (using the inf norm).  This test
% exercises all statements in the package.  Note that the rand state is
% modified.
%
% Example:
%   gee_its_simple_test ;
%
% See also: gee_its_simple, gee_its_short, rand, mldivide, gee_its_simple_resid

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% error-handling tests
%-------------------------------------------------------------------------------

fprintf ('\nTesting error handling (expect error and warning messages):\n\n');

gunk = 0 ;
ok = 0 ;

lasterr ('') ;

try
    % too many inputs
    gee_its_simple_factorize (A,gunk) ;                                     %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too many outputs
    [LU,p,rcnd,gunk] = gee_its_simple_factorize (A) ;                       %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too few inputs
    gee_its_simple_factorize ;                                              %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too many inputs
    x = gee_its_simple (A,b,gunk) ;                                         %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too many outputs
    [x,rcnd,gunk] = gee_its_simple (A,b) ;                                  %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too few inputs
    x = gee_its_simple ;                                                    %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too many inputs
    x = gee_its_simple_forwardsolve (A,b,gunk) ;                            %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too many outputs
    [x,gunk] = gee_its_simple_forwardsolve (A,b) ;                          %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too few inputs
    x = gee_its_simple_forwardsolve ;                                       %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too many inputs
    x = gee_its_simple_backsolve (A,b,gunk) ;                               %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too many outputs
    [x,gunk] = gee_its_simple_backsolve (A,b) ;                             %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % too few inputs
    x = gee_its_simple_backsolve ;                                          %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % rectangular A
    x = gee_its_simple (eye (4,3), ones (4,1)) ;                            %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % A is 3D
    x = gee_its_simple (ones (9,3,3), ones (9,1)) ;                         %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % b is 3D
    x = gee_its_simple (eye (3,3), ones (3,3,3)) ;                          %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % dimensions of A and b do not matrix
    x = gee_its_simple (eye (3,3), ones (4,1)) ;                            %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % dimensions of L and b do not matrix
    x = gee_its_simple_forwardsolve (eye (3,3), ones (4,1)) ;               %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

try
    % dimensions of U and b do not matrix
    x = gee_its_simple_backsolve (eye (3,3), ones (4,1)) ;                  %#ok
catch
    ok = ok + 1 ;
    disp (lasterr) ;
end
fprintf ('\n') ;

% singular matrix
lastwarn ('') ;
x = gee_its_simple (0, 1) ;                                                 %#ok
[msg, id] = lastwarn ;
if (~isempty (msg) & ~isempty (id))                                         %#ok
    ok = ok + 1 ;
end
fprintf ('\n') ;

% ill-conditioned matrix
lastwarn ('') ;
x = gee_its_simple ([1e30 2e30 ; 1 1], [1 ; 1]) ;                           %#ok
[msg, id] = lastwarn ;
if (~isempty (msg) & ~isempty (id))                                         %#ok
    ok = ok + 1 ;
end

if (ok ~= 20)
    error ('test failed') ;
end

fprintf ('\n\nError-handing tests complete (all error messages and warnings\n');
fprintf ('shown above were expected).  Now testing for accuracy:\n\n') ;

%-------------------------------------------------------------------------------
% compare accuracy vs. backslash
%-------------------------------------------------------------------------------

maxerr1 = 0 ;   % largest residual for A\b (gee_its_sweet)
maxerr2 = 0 ;   % largest residual for gee_its_simple (A,b)
maxerr3 = 0 ;   % largest residual for gee_its_short (A,b)
maxerr4 = 0 ;   % largest residual for gee_its_too_short (A,b)
rmax = 0 ;      % largest relative difference in rcond

nmax = 50 ;     % largest dimension of A to test
cmax = 10 ;     % largest number of columns of b to test
ntrials = 2 ;   % number of trials for each x=A\b

rand ('state', 0) ;

for n = 0:nmax
    fprintf ('.') ;
    for c = 0:cmax
        for trial = 1:ntrials

            % set up the system
            A = rand (n) ;
            b = rand (n,c) ;

            % solve it four different ways
            x1 = gee_its_sweet (A,b) ;      % this is just a one-liner: x=A\b
            x2 = gee_its_simple (A,b) ;
            x3 = gee_its_short (A,b) ;
            x4 = gee_its_too_short (A,b) ;

            % get the relative residuals
            err1 = gee_its_simple_resid (A, x1, b) ;
            err2 = gee_its_simple_resid (A, x2, b) ;
            err3 = gee_its_simple_resid (A, x3, b) ;
            err4 = gee_its_simple_resid (A, x4, b) ;

            maxerr1 = max (maxerr1, err1) ;
            maxerr2 = max (maxerr2, err2) ;
            maxerr3 = max (maxerr3, err3) ;
            maxerr4 = max (maxerr4, err4) ;

            if (max ([err1 err2 err3]) > 1e-14)
                error ('test failed') ;
            end

            % test rcond
            if (n > 0)
                [L,U,p] = lu (A) ;                                          %#ok
                r1 = min (abs (diag (U))) / max (abs (diag (U))) ;
                [LU,p,r2] = gee_its_simple_factorize (A) ;
                if (r1 ~= 0)
                    r = abs (r1 - r2) / r1 ;
                    rmax = max (rmax, r) ;
                end
                if (r > 1e-10)
                    error ('test failed') ;
                end
            end
        end
    end
end
fprintf ('\n') ;

fprintf ('max residual for backslash:         %g\n', maxerr1) ;
fprintf ('max residual for gee_its_simple:    %g\n', maxerr2) ;
fprintf ('max residual for gee_its_short:     %g\n', maxerr3) ;
fprintf ('max residual for gee_its_too_short: %g (no pivoting!)\n', maxerr4) ;

fprintf ('\n\nAll tests passed OK\n') ;


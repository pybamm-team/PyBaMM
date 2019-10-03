function lintest (A,b)
%LINTEST test A*x=b, using linfactor, x=A\b, and (ack!) the explicit inv(A).
% The results printed include the breakeven point, which is the number of
% systems Ax=b that must be solved with the same A but different b for the
% inv(A) method to be faster than linfactor.  Using inv(A) is always
% numerically dubious, and typically slower.  However because of MATLAB's
% interpretive overhead, linfactor can be slightly slower in its
% forward/backsolves.  A true linfactor mexFunction would probably be just as
% fast as inv(A)*b when A is full.  You should never ever use inv(A) to solve
% a linear system.
%
% Example:
%   load west0479
%   b = rand (size (west0479,1), 1) ;
%   lintest (west0479, b) ;
%
% See also linfactor, lintests, mldivide

% Copyright 2007, Timothy A. Davis

%-------------------------------------------------------------------------------
% warmup, to make sure functions are loaded, for accurate timings
%-------------------------------------------------------------------------------

F = linfactor (1) ;
x = linfactor (F, 1) ;                                                      %#ok
F = linfactor (sparse (1)) ;
x = linfactor (F, 1) ;                                                      %#ok
F = linfactor (sparse (-1)) ;
x = linfactor (F, 1) ;                                                      %#ok
S = inv (1) ;
x = S*1 ;                                                                   %#ok
S = inv (sparse (1)) ;
x = S*1 ;                                                                   %#ok
S = inv (sparse (-1)) ;
x = S*1 ;                                                                   %#ok
x = rand (2) \ rand (2,1) ;                                                 %#ok
x = sparse (rand (2)) \ rand (2,1) ;                                        %#ok
clear x F S

%-------------------------------------------------------------------------------
% linfactor
%-------------------------------------------------------------------------------

% do this several times for accurate timings
t1 = 0 ;
trials = 0 ;
while (t1 < 1)
    [F, t] = linfactor (A) ;                % factorize A
    t1 = t1 + t ;
    trials = trials + 1 ;
end
t1 = t1 / trials ;

% do this several times for accurate timings
t2 = 0 ;
trials = 0 ;
while (t2 < 1)
    [x, t] = linfactor (F, b) ;             % use the factors to solve Ax=b
    t2 = t2 + t ;
    trials = trials + 1 ;
end
t2 = t2 / trials ;

resid = norm (A*x-b,1) / (norm (A,1) * norm (x,1) + norm (b,1)) ;

fprintf ('%-16s factor time: %10.6f solve time: %10.6f resid: %8.2e\n', ...
    F.kind (1:(find(F.kind == ':', 1, 'first'))), t1, t2, resid) ;

%-------------------------------------------------------------------------------
% inv
%-------------------------------------------------------------------------------

% Try again using inv(A)*b.  This is a really horrible way to solve Ax=b.  I'm 
% doing it here precisely to show that it is typically slower, except when there
% are a huge number of right-hand-sides to solve and A is small.  inv(A) also
% tends to be less accurate, but random matrices do not trigger that problem.
% It fails hopelessly when A is large, sparse, and where max(diff(r)) is large
% where [p,q,r,s] = dmperm(A).  Never, ever use inv(A) to solve a linear system.
% Oh, did I tell you never to use inv(A) to solve Ax=b?

try

    % do this several times for accurate timings
    t3 = 0 ;
    trials = 0 ;
    while (t3 < 1)
        tic ;
        S = inv (A) ;   %#ok
        t = toc ;
        t3 = t3 + t ;
        trials = trials + 1 ;
    end
    t3 = t3 / trials ;

    % do this several times for accurate timings
    t4 = 0 ;
    trials = 0 ;
    while (t4 < 1)
        tic ;
        x = S*b ;
        t = toc ;
        t4 = t4 + t ;
        trials = trials + 1 ;
    end
    t4 = t4 / trials ;

    resid = norm (A*x-b,1) / (norm (A,1) * norm (x,1) + norm (b,1)) ;

    fprintf ('%-16s factor time: %10.6f solve time: %10.6f resid: %8.2e\n', ...
        'inv(A)', t3, t4, resid) ;

    % determine the breakeven point where using inv(A) is faster
    nrhs = max (1, ceil ((t3 - t1) / (t2 - t4))) ;
    if (t1 < t3 & t2 < t4)                                      %#ok
        fprintf ('inv(A) breakeven: never\n') ;
    elseif (t3 < t1 & t4 < t2)                                 %#ok
        fprintf ('inv(A) breakeven: > 0\n') ;
    elseif (t1 < t3 & t4 < t2)                                 %#ok
        fprintf ('inv(A) breakeven: > %d\n', nrhs) ;
    else
        fprintf ('inv(A) breakeven: < %d\n', nrhs) ;
    end

catch

    % inv(A) probably ran out of memory
    fprintf ('inv(A) failed\n') ;
    fprintf ('inv(A) breakeven: never\n') ;
end

%-------------------------------------------------------------------------------
% backslash
%-------------------------------------------------------------------------------

% do this several times for accurate timings
t0 = 0 ;
trials = 0 ;
while (t0 < 1)
    tic ;
    x = A\b ;                               % solve Ax=b
    t = toc ;
    t0 = t0 + t ;
    trials = trials + 1 ;
end
t0 = t0 / trials ;

resid = norm (A*x-b,1) / (norm (A,1) * norm (x,1) + norm (b,1)) ;

fprintf ('%-16s  total time: %10.6f                        resid: %8.2e\n', ...
    'x = A\b', t0, resid) ;
fprintf ('\n') ;


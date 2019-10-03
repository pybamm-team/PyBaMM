function sstest2
%SSTEST2 exhaustive performance test for SSMULT.  Requires ssget.
% ssget is available at http://www.suitesparse.com 
%
% Example
%   sstest2
%
% See also ssmult, ssmultsym, ssmult_install, sstest, ssget, mtimes.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

help sstest2

try
    index = ssget ;
catch
    fprintf ('\nsstest2 requires ssget.\n') ;
    fprintf ('see http://www.suitesparse.com\n') ;
    return ;
end
[ignore, f] = sort (index.nnz) ;                                            %#ok

nmat = length (f) ;
TM = zeros (nmat, 4) ;
T1 = zeros (nmat, 4) ;
tlim = 0.01 ;

tmin = 1 ;
tmax = 0 ;

check = 1 ;
rand ('state', 0)

for k = 1:nmat

    Prob = ssget (f (k), index) ;
    A = Prob.A ;
    clear Prob

    for kind = 1:4

        try
            if (~isreal (A))
                A = spones (A) ;
            end
            B = A' ;

            if (kind == 2)
                A = sprand (A) + 1i*sprand(A) ;
                B = sprand (B) ;
            elseif (kind == 3)
                A = sprand (A) ;
                B = sprand (B) + 1i*sprand(B) ;
            elseif (kind == 4)
                A = sprand (A) + 1i*sprand(A) ;
                B = sprand (B) + 1i*sprand(B) ;
            end

            C = A*B     ; % warmup

            if (check)

                D = ssmult (A,B) ;
                err = norm (C-D,1) ;
                if (err > 0)
                    fprintf ('err: %g\n', err) ;
                    error ('!')
                end
                clear D

            else

                % warmup, for accurate timings
                C = ssmult (A,B) ;                                          %#ok
                clear C

            end

            tr = 0 ;
            tm = 0 ;
            tic
            while (tm < tlim)
                C = A*B ;                                                   %#ok
                clear C
                tr = tr + 1 ;
                tm = toc ;
            end
            tm = tm / tr ;

            tr = 0 ;
            t1 = 0 ;
            tic
            while (t1 < tlim)
                C = ssmult (A,B) ;                                          %#ok
                clear C
                tr = tr + 1 ;
                t1 = toc ;
            end
            t1 = t1 / tr ;

            fprintf ('%4d: %4d ', k, f(k)) ;
            fprintf (...
            'MATLAB %12.6f SSMULT %12.6f  speedup %12.3f', ....
                tm, t1, tm / t1) ;
            
            if (tm < t1)
                fprintf (' ****') ;
            end
            fprintf ('\n') ;

            TM (k,kind) = tm ;
            T1 (k,kind) = t1 ;

            tmin = min (tmin, tm) ;
            tmax = max (tmax, tm) ;

        catch me
            disp (me.message)
            TM (k,kind) = 1 ;
            T1 (k,kind) = 1 ;
        end

    end

    for kind = 1:4

        subplot (2,4,kind) ;
        r = TM (1:k,kind) ./ T1 (1:k,kind) ;
        rmin = min (r) ;
        rmax = max (r) ;
        loglog (TM (1:k,kind), r, 'o', ...
            [tmin tmax], [1 1], 'r-', ...
            [tmin tmax], [1.1 1.1], 'r-', ...
            [tmin tmax], [1/1.1 1/1.1], 'r-', ...
            [tmin tmax], [2 2], 'g-', ...
            [tmin tmax], [1.5 1.5], 'g-', ...
            [tmin tmax], [1/1.5 1/1.5], 'g-', ...
            [tmin tmax], [.5 .5], 'g-' );
        if (k > 2)
            axis ([tmin tmax rmin rmax]) ;
        end
        xlabel ('MATLAB time') ; 
        ylabel ('MATLAB/SM time') ; 
        if (kind == 1)
            title ('real*real') ;
        elseif (kind == 2)
            title ('complex*real') ;
        elseif (kind == 3)
            title ('real*complex') ;
        elseif (kind == 4)
            title ('complex*complex') ;
        end

    end

    drawnow

    clear A B C
    save sstest2_results.mat TM T1 f
    diary off
    diary on

end


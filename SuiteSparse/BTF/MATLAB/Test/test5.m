function test5 (nmat)
%TEST5 test for BTF
% Requires UFget
% Example:
%   test5
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

index = UFget ;

[ignore f] = sort (index.nnz) ;

% time intensive
skip_costly = [1514 1297 1876 1301] ;
f = setdiff (f, skip_costly) ;

if (nargin < 1)
    nmat = 1000 ;
end
nmat = min (nmat, length (f)) ;
f = f (1:nmat) ;

h = waitbar (0, 'BTF test 5 of 6') ;

try
    for k = 1:nmat

        i = f(k) ;
        Prob = UFget (i, index) ;
        A = Prob.A ;

        waitbar (k/nmat, h) ;

        for tr = [1 -1]

            if (tr == -1)
                AT = A' ;
                [m n] = size (A) ;
                if (m == n)
                    if (nnz (spones (AT) - spones (A)) == 0)
                        fprintf ('skip test with transpose\n') ;
                        continue ;
                    end
                end
                A = AT ;
            end

            tic
            q1 = maxtrans (A) ;
            t1 = toc ;
            r1 = sum (q1 > 0) ;

            tic
            q2 = maxtrans (A, 10) ;
            t2 = toc ;
            r2 = sum (q2 > 0) ;

            fprintf (...
                '%4d %4d : %10.4f %8d  : %10.4f %8d', k, f(k), t1, r1, t2, r2) ;
            fprintf (' rel sprank %8.4f', r2 / (max (1, r1))) ;
            if (t2 ~= 0)
                fprintf (': rel time %8.4f', t1 / t2) ;
            end
            fprintf ('\n') ;

            if (r1 ~= r2)
                disp (Prob) ;
                fprintf ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n') ;
            end

        end
    end

catch
    % out-of-memory is OK, other errors are not
    disp (lasterr) ;
    if (isempty (strfind (lasterr, 'Out of memory')))
        error (lasterr) ;                                                   %#ok
    else
        fprintf ('test terminated early, but otherwise OK\n') ;
    end
end

close (h) ;

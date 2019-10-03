function test4 (nmat)
%TEST4 test for BTF
% Requires UFget
% Example:
%   test4
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

index = UFget ;
f = find (index.nrows == index.ncols) ;
[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;

% time intensive
skip_costly = [1514 1297 1876 1301] ;
f = setdiff (f, skip_costly) ;

if (nargin < 1)
    nmat = 1000 ;
end
nmat = min (nmat, length (f)) ;
f = f (1:nmat) ;

h = waitbar (0, 'BTF test 4 of 6') ;

try
    for k = 1:nmat

        Prob = UFget (f (k), index) ;
        A = Prob.A ;

        waitbar (k/nmat, h) ;

        for tr = [1 -1]

            if (tr == -1)
                AT = A' ;
                [m n] = size (A) ;
                if (m == n)
                    if (nnz (spones (AT) - spones (A)) == 0)
                        fprintf ('skip transpose\n') ;
                        continue ;
                    end
                end
                A = AT ;
            end

            tic
            [p1,q1,r1,work1] = btf (A) ;
            t1 = toc ;
            n1 = length (r1) - 1 ;

            tic
            [p2,q2,r2,work2] = btf (A, 10) ;
            t2 = toc ;
            n2 = length (r2) - 1 ;

            fprintf (...
                '%4d %4d : %10.4f %8d  %8g : %10.4f %8d  %8g :', ...
                k, f(k), t1, n1, work1, t2, n2, work2) ;
            if (t2 ~= 0)
                fprintf (' rel %8.4f %8.4f' , t1 / t2, n2 / (max (1, n1))) ;
            end
            fprintf ('\n') ;

            if (n1 ~= n2 | work1 ~= work2)                                  %#ok
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

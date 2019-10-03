function test6
%TEST6 test for BTF
% Requires UFget
% Example:
%   test6
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

quick2 = [ ...
 1522 -272  1463  1521   460 1507  -838 1533 -1533 -1456 -1512   734   211 ...
 -385 -735   394  -397  1109 -744  ...
 -734 -375 -1200 -1536  -837  519  -519  520  -520   189  -189   454   385 ...
  387 -387   384  -384   386 -386   388 -388   525  -525   526  -526   735 ...
 1508  209   210  1243 -1243 1534  -840 1234 -1234   390  -390   392  -392 ...
 -394 1472  1242 -1242   389 -389   391 -391   393  -393  1215 -1215  1216 ...
-1216  736  -736   737  -737  455  -455 -224  -839  1426 -1426 -1473   396 ...
 -396  398  -398   400  -400  402  -402  404  -404 -1531   395  -395   397 ...
  399 -399   401  -401   403 -403   405 -405  -738  -739  1459 -1459  1111 ...
 1110  376  -376   284  -284 -740  -742 -741  -743  1293 -1293   452   920 ...
 -745 -446  1462 -1461   448 -448   283 -283  1502 -1502  1292 -1292  1503 ...
-1503 1291 -1291   445  -445 -746  -747 1300 -1300   435  -435 -1343 -1345 ...
-1344 1305 -1305   921 -1513 1307 -1307 1369 -1369  1374 -1374  1377 ...
-1377  748  -748  -749  1510  922  -922 ] ;

index = UFget ;
nmat = length (quick2) ;
dopause = 0 ;

h = waitbar (0, 'BTF test 6 of 6') ;

try

    for k = 1:nmat

        waitbar (k/nmat, h) ;

        i = quick2 (k) ;
        Prob = UFget (abs (i), index) ;
        disp (Prob) ;
        if (i < 0)
            fprintf ('transposed\n') ;
            A = Prob.A' ;
            [m n] = size (A) ;
            if (m == n)
                if (nnz (spones (A) - spones (Prob.A)) == 0)
                    fprintf ('skip...\n') ;
                    continue ;
                end
            end
        else
            A = Prob.A ;
        end

        tic
        [p1,q1,r1,work1] = btf (A) ;
        t1 = toc ;
        n1 = length (r1) - 1 ;
        m1 = nnz (diag (A (p1, abs (q1)))) ;

        limit = work1/nnz(A) ;

        fprintf ('full search: %g * nnz(A)\n', limit) ;

        works = linspace(0,limit,9) ;
        works (1) = eps ;
        nw = length (works) ;

        T2 = zeros (nw, 1) ;
        N2 = zeros (nw, 1) ;
        M2 = zeros (nw, 1) ;

        T2 (end) = t1 ;
        N2 (end) = n1 ;
        M2 (end) = m1 ;

        fprintf ('full time %10.4f   blocks %8d  nnz(diag) %8d\n\n', t1, n1, m1) ;

        subplot (3,4,4) ;
        drawbtf (A, p1, abs (q1), r1) ;
        title (Prob.name, 'Interpreter', 'none') ;

        for j = 1:nw-1

            maxwork = works (j) ;

            tic
            [p2,q2,r2,work2] = btf (A, maxwork) ;
            t2 = toc ;
            n2 = length (r2) - 1 ;
            m2 = nnz (diag (A (p2, abs (q2)))) ;
            T2 (j) = t2 ;
            N2 (j) = n2 ;
            M2 (j) = m2 ;

            fprintf ('%9.1f %10.4f   blocks %8d  nnz(diag) %8d\n', ...
                maxwork, t2, n2, m2) ;

            subplot (3,4,4+j) ;
            drawbtf (A, p2, abs (q2), r2) ;
            title (sprintf ('%g', maxwork)) ;

            ss = [1:j nw] ;

            subplot (3,4,1) ;
            plot (works(ss), T2(ss), 'o-') ;  title ('time vs work') ;
            axis ([0 limit 0 max(0.1,max(T2))]) ;

            subplot (3,4,2) ;
            plot (works(ss), N2(ss), 'o-') ; title ('blocks vs work') ;
            axis ([0 limit 0 n1]) ;

            subplot (3,4,3) ;
            plot (works(ss), M2(ss), 'o-') ; title ('nnz(diag) vs work') ;
            axis ([0 limit 0 m1]) ;
            drawnow

        end
        fprintf ('full time %10.4f   blocks %8d  nnz(diag) %8d\n', t1, n1, m1) ;

        if (dopause)
            input ('hit enter: ') ;
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

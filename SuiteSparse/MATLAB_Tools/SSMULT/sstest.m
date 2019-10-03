function sstest
%SSTEST exhaustive performance test for SSMULT.
%
% Example
%   sstest
%
% See also ssmult, ssmultsym, ssmult_install, sstest2, mtimes.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

N = [500:50:1000 1100:100:3000 3200:200:5000 ] ;

% warmup for more accurate timings
A = sparse (1) ;
B = sparse (1) ;
C = A*B ;
D = ssmult(A,B) ;
err = norm (C-D,1) ;
if (err > 0)
    error ('test failure') ;
end
clear C D

titles = { ...
    'C=A*B blue, C=B*A red, both real', ...
    'A real, B complex', 'A complex, B real', 'both complex' } ;

xlabels = { '(A random, B diagonal)', '(A random, B permutation)', ...
    '(A random, B tridiagonal)' } ;

fprintf ('\nIn the next plots, speedup is the time for MATLAB C=A*B divided\n');
fprintf ('by the time for C=ssmult(A,B).  The X-axis is n, the dimension\n') ;
fprintf ('of the square matrices A and B.  A is a sparse random matrix with\n');
fprintf ('1%% nonzero values.  B is diagonal in the first row of plots,\n') ;
fprintf ('a permutation in the 2nd row, and tridiagonal in the third.\n') ;
fprintf ('C=A*B is in blue, C=B*A is in red.  A and B are both real in the\n') ;
fprintf ('first column of plots, B is complex in the 2nd, A in the 3rd, and\n');
fprintf ('both are complex in the 4th column of plots.  You will want to\n') ;
fprintf ('maximize the figure; otherwise the text is too hard to read.\n') ; 
fprintf ('\nBe aware that in MATLAB 7.6 and later, C=A*B in MATLAB uses\n') ;
fprintf('SSMULT (but with some additional MathWorks-specific optimizations)\n');
fprintf ('so you are comparing nearly identical codes.\n');
% input ('Hit enter to continue: ', 's') ;

tlim = 0.1 ;
clf ;

for fig = 1:3

    fprintf ('Testing C=A*B and C=B*A %s\n', xlabels {fig}) ;

    T = zeros (length(N),4,4) ;

    for k = 1:length(N)

        n = N (k) ;
        try

            A = sprand (n,n,0.01) ;
            if (fig == 1)
                % B diagonal
                B = spdiags (rand (n,1), 0, n, n) ;
            elseif (fig == 2)
                % B permutation
                B = spdiags (rand (n,1), 0, n, n) ;
                B = B (:,randperm(n)) ;
            else
                % B tridiagonal
                B = spdiags (rand (n,3), -1:1, n, n) ;
            end

            for kind = 1:4

                if (kind == 2)
                    % A complex, B real
                    A = A + 1i*sprand (A) ;
                elseif (kind == 3)
                    % A real, B complex
                    A = real (A) ;
                    B = B + 1i*sprand (B) ;
                elseif (kind == 4)
                    % both complex
                    A = A + 1i*sprand (A) ;
                    B = B + 1i*sprand (B) ;
                end

                %---------------------------------------------------------------
                % C = A*B
                %---------------------------------------------------------------

                t1 = 0 ;
                trials = 0 ;
                tic
                while (t1 < tlim)
                    C = A*B ;
                    trials = trials + 1 ;
                    t1 = toc ;
                end
                t1 = t1 / trials ;

                t2 = 0 ;
                trials = 0 ;
                tic
                while (t2 < tlim)
                    D = ssmult (A,B) ;
                    trials = trials + 1 ;
                    t2 = toc ;
                end
                t2 = t2 / trials ;

                err = norm (C-D,1) ;
                if (err > 0)
                    error ('test failure') ;
                end
                clear C
                clear D

                %---------------------------------------------------------------
                % C = B*A
                %---------------------------------------------------------------

                t3 = 0 ;
                trials = 0 ;
                tic
                while (t3 < tlim)
                    C = B*A ;
                    trials = trials + 1 ;
                    t3 = toc ;
                end
                t3 = t3 / trials ;

                t4 = 0 ;
                trials = 0 ;
                tic
                while (t4 < tlim)
                    D = ssmult (B,A) ;
                    trials = trials + 1 ;
                    t4 = toc ;
                end
                t4 = t4 / trials ;

                err = norm (C-D,1) ;
                if (err > 0)
                    error ('test failure') ;
                end
                clear C
                clear D

                %---------------------------------------------------------------

                T (k,kind,1) = t1 ;
                T (k,kind,2) = t2 ;
                T (k,kind,3) = t3 ;
                T (k,kind,4) = t4 ;
                subplot (3,4,kind + 4*(fig-1)) ;
                plot (N(1:k), T (1:k,kind,1) ./ T (1:k,kind,2), 'o', ...
                      N(1:k), T (1:k,kind,3) ./ T (1:k,kind,4), 'rx', ...
                      [N(1) n], [1 1], 'k') ;
                xlabel (['n ' xlabels{fig}]) ;
                ylabel ('speedup') ;
                axis tight
                title (titles {kind}) ;
                drawnow

            end

        catch me
            % probably because we ran out of memory ...
            disp (me.message) ;
            break ;
        end
    end
end


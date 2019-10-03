function test2 (nmat)
%test2: KLU test
% Example:
%   test2
% See also klu

% Copyright 2004-2012, University of Florida

clear functions
% rand ('state', 0) ;
% warning ('off', 'MATLAB:singularMatrix') ;
% warning ('off', 'MATLAB:nearlySingularMatrix') ;
% warning ('off', 'MATLAB:divideByZero') ;

index = UFget ;
f = find (index.nrows == index.ncols) ;
[ignore i] = sort (index.nnz (f)) ;                                         %#ok
f = f (i) ;

if (nargin < 1)
    nmat = 500 ;
end
nmat = min (nmat, length (f)) ;
f = f (1:nmat) ;

if (~isempty (strfind (computer, '64')))
    is64 = 1 ;
else
    is64 = 0 ;
end

Tklu = 1e-6 * ones (2*nmat,1) ;
Tmatlab = zeros (2*nmat,1) ;
Tcsparse = zeros (2*nmat,1) ;
LUnz = zeros (2*nmat, 1) ;
k = 0 ;

h = waitbar (0, 'KLU test 2 of 5') ;

clf

% try

    for kk = 1:nmat

        Prob = UFget (f (kk), index) ;

        waitbar (kk/nmat, h) ;

        disp (Prob) ;
        if (isfield (Prob, 'kind'))
            if (~isempty (strfind (Prob.kind, 'subsequent')))
                fprintf ('skip ...\n') ;
                continue
            end
        end
        A = Prob.A ;

        for do_complex = 0:1
            
            k = k + 1 ;
            if (do_complex)
                A = sprand (A) + 1i * sprand (A) ;
            end

            try
                [L,U,p,q] = lu (A, 'vector') ;
            catch                                                           %#ok
                % older version of MATLAB, which doesn't have 'vector' option
                [L,U,P,Q] = lu (A) ;
                [p ignore1 ignore2] = find (P') ;                           %#ok
                [q ignore1 ignore2] = find (Q) ;                            %#ok
                clear ignore1 ignore2 P Q
            end

            LU.L = L ;
            LU.U = U ;
            if (is64)
                LU.p = int64 (p) ;
                LU.q = int64 (q) ;
            else
                LU.p = int32 (p) ;
                LU.q = int32 (q) ;
            end
            LUnz (k) = nnz (L) + nnz (U) ;

            n = size (A,1) ;

            do_klu = (nnz (diag (U)) == n) ;
            if (do_klu)

                fprintf ('klu...\n') ;
                err = 0 ;
                erc = 0 ;
                er2 = 0 ;
                for nrhs = 10:-1:1

                    b = rand (n,nrhs) ;

                    tic ;
                    x = klu (LU,'\',b) ;
                    Tklu (k) = max (1e-6, toc) ;

                    tic ;
                    y = U \ (L \ b (p,:)) ;
                    y (q,:) = y ;
                    Tmatlab (k) = max (1e-6, toc) ;

                    if (nrhs == 1 & isreal (U) & isreal (L) & isreal (b))   %#ok
                        tic ;
                        z = cs_usolve (U, cs_lsolve (L, b (p))) ;
                        z (q) = z ;
                        Tcsparse (k) = max (1e-6, toc) ;
                        erc = norm (A*z-b,1) / norm (A,1) ;
                    end

                    err = max (err, norm (A*x-b,1) / norm (A,1)) ;
                    er2 = max (er2, norm (A*y-b,1) / norm (A,1)) ;
                    if (err > 100*er2)
                        fprintf ('error %g %g\n', err, er2) ;
                        error ('?') ;
                    end
                end

                fprintf ('klu... with randomized scaling for L\n') ;
                er3 = 0 ;
                er4 = 0 ;
                D = spdiags (rand (n,1), 0, n, n) ;
                LU.L = D * L ;
                A2 = D * A (p,q) ;
                if (is64)
                    LU.p = int64 (1:n) ;
                    LU.q = int64 (1:n) ;
                else
                    LU.p = int32 (1:n) ;
                    LU.q = int32 (1:n) ;
                end
                for nrhs = 1:10
                    b = rand (n,nrhs) ;
                    x = klu (LU,'\',b) ;
                    y = U \ (LU.L \ b) ;
                    er3 = max (er3, norm (A2*x-b,1) / norm (A,1)) ;
                    er4 = max (er4, norm (A2*y-b,1) / norm (A,1)) ;
                    if (er3 > 1e3*er4)
                        fprintf ('error %g %g\n', er3, er4) ;
                        error ('?') ;
                    end
                end


            else
                err = Inf ;
                er2 = Inf ;
                erc = Inf ;
            end

            lumax = max (LUnz (1:k)) ;
            loglog (...
                LUnz (1:k), Tmatlab (1:k) ./ Tklu (1:k), 'o', ...
                LUnz (1:k), Tcsparse (1:k) ./ Tklu (1:k), 'x', ...
                [20 lumax], [1 1], 'r-') ;
            axis ([20 lumax .1 20]) ;
            drawnow

            fprintf ('err %g %g %g\n', err, er2, erc) ;
        end
    end

% catch me
%     disp (me.message) ;
% end

close (h) ;

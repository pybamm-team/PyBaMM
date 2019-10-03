function test4 (nmat)
%test4: KLU test
% Example:
%   test4
% See also klu

% Copyright 2004-2012, University of Florida

% rand ('state', 0) ;
% warning ('off', 'MATLAB:singularMatrix') ;
% warning ('off', 'MATLAB:nearlySingularMatrix') ;
% warning ('off', 'KLU:rcond') ;
% warning ('off', 'MATLAB:Axes:NegativeDataInLogAxis') ;

index = UFget ;
f = find (index.nrows == index.ncols & index.isReal & index.amd_lnz > 0) ;
[ignore i] = sort (index.amd_lnz (f)) ;                                     %#ok
f = f (i) ;
% f = f (1:100) ;

if (nargin < 1)
    nmat = 500 ;
end
nmat = min (nmat, length (f)) ;
f = f (1:nmat) ;

if (1)
    Tlu = -ones (nmat,1) ;
    Tklu = -ones (nmat,1) ;
    Tklu2 = -ones (nmat,1) ;
    LUnz = -ones (nmat,1) ;
    k = 0 ;
end

if (~isempty (strfind (computer, '64')))
    is64 = 1 ;
else
    is64 = 0 ;
end

% 589: Sieber

h = waitbar (0, 'KLU test 4 of 5') ;

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
            if (~isempty (strfind (Prob.kind, 'random')))
                fprintf ('skip ...\n') ;
                continue
            end
        end
        k = k + 1 ;
        A = Prob.A ;
        n = size (A,1) ;
        err1 = 0 ;
        err2 = 0 ;
        err4 = 0 ;
        terr1 = 0 ;
        terr2 = 0 ;
        terr4 = 0 ;


        for do_imag = 0:1
            if (do_imag)
                A = sprand (A) + 1i * sprand (A) ;
            end

            % compare with UMFPACK
            try
                tic
                [L,U,p,q,R1] = lu (A, 'vector') ;
                t1 = max (1e-6, toc) ;
            catch                                                           %#ok
                % older version of MATLAB, which doesn't have 'vector' option
                tic
                [L,U,P,Q] = lu (A) ;
                t1 = max (1e-6, toc) ;
                [p ignore1 ignore2] = find (P') ;                           %#ok
                [q ignore1 ignore2] = find (Q) ;                            %#ok
                clear ignore1 ignore2 P Q
                R1 = speye (n) ;
            end

            if (Tlu (k) == -1)
                Tlu (k) = t1 ;
                LUnz (k) = nnz (L) + nnz (U) ;
            end

            % note that the scaling R1 and R2 are different with KLU and UMFPACK
            % UMFPACK:  L*U-P*(R1\A)*Q
            % KLU:      L*U-R2\(P*A*Q)
            %
            % R1 and R2 are related, via P, where R2 = P*R*P', or equivalently
            % R2 = R1 (p,p).

            rcond = min (abs (diag (U))) / max (abs (diag (U))) ;
            if (rcond < 1e-15)
                fprintf ('skip...\n') ;
                break ;
            end

            F.L = L ;
            F.U = U ;
            if (is64)
                F.p = int64(p) ;
                F.q = int64(q) ;
            else
                F.p = int32(p) ;
                F.q = int32(q) ;
            end

            F.R = R1(p,p) ;
            b = rand (n,1) ;
            x = klu (F, '\', b) ;
            y = klu (b', '/', F) ;

            fprintf ('solve with klu %g\n', ...
                norm (A*x-b,1)/norm(A,1)) ;
            fprintf ('solve with klu %g transpose\n', ...
                norm (y*A-b',1)/norm(A,1)) ;


            for nrhs = 1:10
                for do_b_imag = 0:1
                    b = rand (n, nrhs) ;
                    if (do_b_imag)
                        b = b + 1i * rand (n, nrhs) ;
                    end

                    % KLU backslash
                    tic ;
                    x = klu (A,'\',b) ;
                    t2 = max (1e-6, toc) ;

                    % KLU slash
                    xt = klu (b','/',A) ;

                    % KLU backslash with precomputed LU
                    tic
                    LU = klu (A) ;
                    z = klu (LU,'\',b) ;
                    t4 = max (1e-6, toc) ;

                    % KLU slash with precomputed LU
                    zt = klu (b','/',LU) ;

                    % UMFPACK
                    tic
                    rb = R1 \ b ;
                    y = U \ (L \ rb (p,:)) ;
                    y (q,:) = y ;
                    t3 = max (1e-6, toc) ;

                    yt = (L' \ (U' \ b (q,:))) ;
                    yt (p,:) = yt ;
                    yt = R1 \ yt ;
                    yt = yt' ;

                    if (Tklu (k) == -1)
                        Tlu (k) = Tlu (k) + t3 ;
                        Tklu (k) = t2 ;
                        Tklu2 (k) = t4 ;
                    end

                    err1 = max (err1, norm (A*x-b,1) / norm (A,1)) ;
                    err2 = max (err2, norm (A*y-b,1) / norm (A,1)) ;
                    err4 = max (err4, norm (A*z-b,1) / norm (A,1)) ;

                    terr1 = max (terr1, norm (xt*A-b',1) / norm (A,1)) ;
                    terr2 = max (terr2, norm (yt*A-b',1) / norm (A,1)) ;
                    terr4 = max (terr4, norm (zt*A-b',1) / norm (A,1)) ;
                end
            end
        end

        fprintf ('err %g %g %g\n', err1, err2, err4) ;
        if (err1 > 1e4*err2 | err4 > 1e4*err2)                              %#ok
            fprintf ('warning: KLU inaccurate!\n')
        end

        fprintf ('terr %g %g %g\n', terr1, terr2, terr4) ;
        if (terr1 > 1e4*terr2 | terr4 > 1e4*terr2)                          %#ok
            fprintf ('warning: KLU T inaccurate!\n')
        end

        lunzmax = max (LUnz (1:k)) ;
        loglog ( ...
            LUnz (1:k), Tklu (1:k) ./ Tlu (1:k), 'o', ...
            LUnz (1:k), Tklu2 (1:k) ./ Tlu (1:k), 'x', ...
            [10 lunzmax], [1 1], 'r-') ;
        drawnow

    end

% catch me
%     disp (me.message) ;
% end

close (h) ;

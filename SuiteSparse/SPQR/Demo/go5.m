% compares SPQR with SPQR+GPU on lots of sparse matrices

clear
index = UFget ;
f = find ((index.isReal == 1) & (index.isBinary == 0) & (index.isGraph == 0)) ;
nmat = length (f) ;
howbig = max (index.amd_rnz (f), index.amd_lnz (f)) ;
[ignore i] = sort (howbig) ;
howbig = howbig (i) ;
f = f (i) ;
nmat = length (f) ;

keep = ones (nmat,1) ;
kinds = UFkinds ;
for k = 1:nmat
    id = f (k) ;
    if (~isempty (strfind (kinds {id}, 'subsequent')))
        keep (k) = 0 ;
    end
    if (~isempty (strfind (kinds {id}, 'duplicate')))
        keep (k) = 0 ;
    end
end
f = f (find (keep)) ;
nmat = length (f) ;

rankdeficient = [317] ;
skip_list = [1657] ;

revisit = [253 317 1556]

for k = 1:nmat
    id = f (k) ;
    kind = kinds {id} ;
    fprintf ('%4d %4d %s/%s : %s\n', ...
        k, id, index.Group {id}, index.Name {id}, kind);
end

fprintf ('\n# of matrices %d\n', nmat) ;

fprintf ('Hit enter to continue: ') ; pause

have_gpu = 1 ;

opts.Q = 'discard' ;

try
    load Results7
catch
    % 1: spqr on CPU with colamd
    % 2: spqr on CPU with metis
    % 3: spqr on GPU with colamd
    % 4: spqr on GPU with metis
    Info        = cell (nmat,4) ;
    Flops       = zeros (nmat, 4) ;
    Mem         = zeros (nmat, 4) ;
    Rnz         = zeros (nmat, 4) ;
    T_analyze   = zeros (nmat, 4) ;
    T_factorize = zeros (nmat, 4) ;

    % matrix stats
    Rank_est    = zeros (nmat, 1) ;
    Anz         = zeros (nmat, 1) ;
    Augment     = zeros (nmat, 1) ; % 1 if A was augmented, 0 otherwise
end

for k = 1:nmat

    %---------------------------------------------------------------------------
    % skip the problem if it is already done
    %---------------------------------------------------------------------------

    if (all (T_factorize (k,:) > 0))
        continue
    end

    %---------------------------------------------------------------------------
    % get the problem
    %---------------------------------------------------------------------------

    id = f (k) ;
    kind = kinds {id} ;
    fprintf ('\n%4d %4d %s/%s : %s\n', ...
        k, id, index.Group {id}, index.Name {id}, kind) ;

    if (any (id == skip_list))
        fprintf ('skip\n') ;
        continue
    end

    Prob = UFget (id, index) ;
    A = Prob.A ;
    anorm = norm (A,1) ;
    [m n] = size (A) ;
    if (m < n)
        A = A' ;
        [m n] = size (A) ;
    end
    b = ones (m,1) ;

    %---------------------------------------------------------------------------
    % run SPQR without the GPU
    %---------------------------------------------------------------------------

    for ordering = 1:2

        if (ordering == 1)
            opts.ordering = 'colamd' ;
        else
            opts.ordering = 'metis' ;
        end

        [C R E info] = spqr (A, b, opts) ;
        rnz = nnz (R) ;

        if (ordering == 1)

            Rank_est (k) = info.rank_A_estimate ;
            fprintf ('rankest %d ', Rank_est (k)) ;

            if (info.rank_A_estimate < min (m,n))
                % oops, A is rank deficient.  Try it again
                fprintf ('(rank deficient)\n') ;
                Augment (k) = 1 ;
                A = [A ; anorm*speye(n)] ;
                [m n] = size (A) ;
                b = ones (m,1) ;
                clear Q R E
                [C R E info] = spqr (A, b, opts) ;
                rnz = nnz (R) ;
            else
                fprintf ('(ok)\n') ;
            end

            Anz (k) = nnz (A) ;
        end
    
        x = E*(R\C) ;

        Info {k,ordering} = info ;
        info
        Flops (k,ordering) = info.flops ;
        Mem (k,ordering) = info.memory_usage_in_bytes ;
        Rnz (k,ordering) = rnz ;
        T_analyze (k,ordering) = info.analyze_time ;
        T_factorize (k,ordering) = info.factorize_time ;

        atrnorm (ordering) = norm (A'*(A*x-b)) / norm (A'*A,1) ;;
        fprintf ('relative atrnorm for CPU: %g\n', atrnorm (ordering)) ;

        clear Q R E

    end

    if (any (id == rankdeficient))
        fprintf ('skipping rank deficient case for the GPU\n') ;
        continue ;
    end

    %---------------------------------------------------------------------------
    % run SPQR with the GPU
    %---------------------------------------------------------------------------

    mwrite ('A.mtx', A) ;

    for ordering = 1:2

%       try
            info = spqr_gpu (ordering, A) ;
            info
            Info {k,2 + ordering} = info ;
            Flops (k,2 + ordering) = info.flops ;
            Mem (k,2 + ordering) = info.memory_usage_in_bytes ;
            Rnz (k,2 + ordering) = info.nnzR ;
            T_analyze (k,2 + ordering) = info.analyze_time ;
            T_factorize (k,2 + ordering) = info.factorize_time ;

            % Protect the data against caught OOM errors
            if (T_factorize(k,2+ordering) == 0)
                T_factorize(k,2+ordering) = nan ;
                % pause
                error ('oom') ;
            end

            [C,R,E,B,X,err] = spqr_gpu2 (ordering, A) ;
            clear C R E B X

%       catch
%           % GPU version failed
%           T_factorize (k,2 + ordering) = nan ;
%           pause
%       end

    end

%    if (Flops (k,1) ~= Flops (k,3))
%        Flops (k,[1 3])
%        error ('colamd flops are not the same ... why?') ;
%    end
%    if (Flops (k,2) ~= Flops (k,4))
%        Flops (k,[2 4])
%        error ('metis flops are not the same ... why?') ;
%    end

    %---------------------------------------------------------------------------
    % save the results and plot them
    %---------------------------------------------------------------------------

    save Results7 ...
        Info Flops Mem Rnz T_analyze T_factorize Rank_est Anz Augment f

    intensity1 = Flops (1:k,1) ./ Mem (1:k,1) ;
    intensity2 = Flops (1:k,2) ./ Mem (1:k,2) ;

    gflops1 = 1e-9 * Flops (1:k,1) ./ T_factorize (1:k,1) ;
    gflops2 = 1e-9 * Flops (1:k,2) ./ T_factorize (1:k,2) ;
    gflops3 = 1e-9 * Flops (1:k,1) ./ T_factorize (1:k,3) ;
    gflops4 = 1e-9 * Flops (1:k,2) ./ T_factorize (1:k,4) ;

    fprintf ('CPU colamd factime %8.2f  gflops : %8.2f\n', ...
        T_factorize (k,1), gflops1 (k)) ;

    fprintf ('CPU metis  factime %8.2f  gflops : %8.2f\n', ...
        T_factorize (k,2), gflops2 (k)) ;

    fprintf ('GPU colamd factime %8.2f  gflops : %8.2f\n', ...
        T_factorize (k,3), gflops3 (k)) ;

    fprintf ('GPU metis  factime %8.2f  gflops : %8.2f\n', ...
        T_factorize (k,4), gflops4 (k)) ;

%{
    subplot (2,3,1) ; plot (intensity1, gflops1, 'o') ; title ('CPU:colamd') ;
    ylabel ('Gflops') ; xlabel ('flops/byte') ;
    subplot (2,3,2) ; plot (intensity1, gflops3, 'o') ; title ('GPU:colamd') ;
    ylabel ('Gflops') ; xlabel ('flops/byte') ;
    subplot (2,3,3) ; plot (intensity1, gflops3./gflops1, 'o') ;
    title ('GPU speedup (colamd)') ;

    subplot (2,3,4) ; plot (intensity2, gflops2, 'o') ; title ('CPU:metis') ;
    ylabel ('Gflops') ; xlabel ('flops/byte') ;
    subplot (2,3,5) ; plot (intensity2, gflops4, 'o') ; title ('GPU:metis') ;
    ylabel ('Gflops') ; xlabel ('flops/byte') ;
    subplot (2,3,6) ; plot (intensity2, gflops4./gflops2, 'o') ;
    title ('GPU speedup (metis)') ;
    drawnow
%}

    diary off
    diary on

    % fprintf ('Hit enter: ') ; pause

end

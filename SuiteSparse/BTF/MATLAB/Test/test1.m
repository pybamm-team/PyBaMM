function test1 (nmat)
%TEST1 test for BTF
% Requires CSparse and UFget
% Example:
%   test1
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

index = UFget ;
% f = find (index.sprank < min (index.nrows, index.ncols)) ;
f = 1:length (index.nrows) ;

% too much time:
skip = [1514 1297 1876 1301] ;

f = setdiff (f, skip) ;

[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;

if (nargin < 1)
    nmat = 1000 ;
end
nmat = min (nmat, length (f)) ;
f = f (1:nmat) ;

T0 = zeros (nmat,1) ;
T1 = zeros (nmat,1) ;
Anz = zeros (nmat,1) ;
MN = zeros (nmat, 2) ;
Nzdiag = zeros (nmat,1) ;
clf

% warmup
p = maxtrans (sparse (1)) ;         %#ok
p = cs_dmperm (sparse (1)) ;        %#ok
a = cs_transpose (sparse (1)) ;     %#ok

h = waitbar (0, 'BTF test 1 of 6') ;

try

    for k = 1:nmat

        Prob = UFget (f (k), index) ;
        A = Prob.A ;
        clear Prob
        t = 0 ;

        waitbar (k/nmat, h) ;

        r = full (sum (spones (A), 2)) ;
        c = full (sum (spones (A))) ;
        m2 = length (find (r > 0)) ;
        n2 = length (find (c > 0)) ;

        if (m2 < n2)
            tic
            A = cs_transpose (A) ;
            t = toc ;
        end

        Nzdiag (k) = nnz (diag (A)) ;

        [m n] = size (A) ;
        Anz (k) = nnz (A) ;
        MN (k,:) = [m n] ;

        tic
        q = maxtrans (A) ;
        t0 = toc ;
        s0 = sum (q > 0) ;
        T0 (k) = max (1e-9, t0) ;

        tic
        p = cs_dmperm (A) ;
        t1 = toc ;
        s1 = sum (p > 0) ;
        T1 (k) = max (1e-9, t1) ;

        fprintf (...
            '%4d maxtrans %10.6f %10.6f  cs_dmperm %10.6f m/n %8.2f', ...
            f(k), t, t0, t1, m/n) ;
        if (t1 ~= 0)
            fprintf (' rel: %8.4f', t0 / t1) ;
        end
        fprintf ('\n') ;
        if (s0 ~= s1)
            error ('!') ;
        end

        if (s0 == n & m == n)                                               %#ok
            B = A (:, q) ;
            subplot (2,2,1) ;
            cspy (B) ;
            if (nnz (diag (B)) ~= n)
                error ('?')
            end
            clear B
        else
            cspy (0) ;
        end

        maxnz = nnz (A) ;

        zfree  = find (MN (1:k,1) == MN (1:k,2) & Nzdiag (1:k) == MN(1:k,1)) ;
        square = find (MN (1:k,1) == MN (1:k,2) & Nzdiag (1:k) ~= MN(1:k,1)) ;
        tall   = find (MN (1:k,1) >  MN (1:k,2)) ;
        squat  = find (MN (1:k,1) <  MN (1:k,2)) ;

        subplot (2,2,2) ;
        loglog (Anz (square), T0 (square) ./ T1 (square), ...
            'o', [1 maxnz], [1 1], 'r-') ;
        title ('square') ;
        subplot (2,2,3) ;
        loglog (Anz (tall), T0 (tall) ./ T1 (tall), ...
            'o', [1 maxnz], [1 1], 'r-') ;
        title ('tall') ;
        subplot (2,2,4) ;
        title ('square, intially zero-free') ;
        loglog (Anz (zfree), T0 (zfree) ./ T1 (zfree), ...
            'o', [1 maxnz], [1 1], 'r-') ;
        title ('square, zero-free diag') ;

        drawnow

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

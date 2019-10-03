function test2 (nmat)
%TEST2 test for BTF
% Requires CSparse and UFget
% Example:
%   test2
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5.

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

index = UFget ;
f = find (index.nrows == index.ncols) ;

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
p = maxtrans (sparse (1)) ;             %#ok
p = btf (sparse (1)) ;                  %#ok
p = cs_dmperm (sparse (1)) ;            %#ok
a = cs_transpose (sparse (1)) ;         %#ok

h = waitbar (0, 'BTF test 2 of 6') ;

try
    for k = 1:nmat

        Prob = UFget (f (k), index) ;
        A = Prob.A ;

        waitbar (k/nmat, h) ;

        Nzdiag (k) = nnz (diag (A)) ;

        [m n] = size (A) ;
        Anz (k) = nnz (A) ;
        MN (k,:) = [m n] ;

        tic
        [p,q,r] = btf (A) ;
        t0 = toc ;
        s0 = sum (q > 0) ;
        T0 (k) = max (1e-9, t0) ;

        tic
        [p2,q2,r2] = cs_dmperm (A) ;
        t1 = toc ;
        s1 = sum (dmperm (A) > 0) ;
        T1 (k) = max (1e-9, t1) ;

        fprintf ('%4d btf %10.6f cs_dmperm %10.6f', f(k), t0, t1) ;
        if (t1 ~= 0)
            fprintf (' rel: %8.4f', t0 / t1) ;
        end
        fprintf ('\n') ;

        if (s0 ~= s1)
            error ('!') ;
        end

        C = A (p, abs (q)) ;
        subplot (1,2,1) ;
        cspy (C) ;
        z = find (q < 0) ;
        zd = nnz (diag (C (z,z))) ;
        if (zd > 0)
            error ('?') ;
        end

        minnz = Anz (1) ;
        maxnz = nnz (A) ;

        subplot (1,2,2) ;
        loglog (Anz (1:k), T0 (1:k) ./ T1 (1:k), ...
            'o', [minnz maxnz], [1 1], 'r-') ;
        drawnow

        clear C A Prob
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

function meshnd_example
%MESHND_EXAMPLE example usage of meshnd and meshsparse.
%
% Example:
%   meshnd_example
%
% See also meshnd.

% Copyright 2009, Timothy A. Davis, http://www.suitesparse.com

help meshnd

% 2D mesh, compare with Cleve Moler's demos

m = 7 ;
n = 7 ;

[G p pinv Gnew] = meshnd (m,n) ;
fprintf ('Original mesh:\n') ;
disp (G) ;
fprintf ('Permuted node numbers using meshnd.m (nested dissection):\n') ;
disp (Gnew) ;

Moler = nested (n+2) ;
Moler = Moler (2:n+1,2:n+1) ;
fprintf ('Cleve Moler''s nested dissection ordering, using nested.m\n') ;
disp (Moler) ;
fprintf ('Difference between nested.m and meshnd.m:\n') ;
disp (Gnew-Moler) ;

% 2D and 3D meshes

stencils = [5 9 7 27] ;
mm = [7 7 7 7] ;
nn = [7 7 7 7] ;
kk = [1 1 7 7] ;

for s = 1:4

    m = mm (s) ;
    n = nn (s) ;
    k = kk (s) ;
    [G p] = meshnd (mm (s), nn (s), kk (s)) ;
    A = meshsparse (G, stencils (s)) ;
    C = A (p,p) ;
    parent = etree (C) ;
    try
        L = chol (C, 'lower') ;
    catch
        % old version of MATLAB
        L = chol (C)' ;
    end
    subplot (4,5,(s-1)*5 + 1) ;
    do_spy (A) ;
    if (k > 1)
	title (sprintf ('%d-by-%d-by-%d mesh, %d-point stencil', ...
	    m, n, k, stencils (s))) ;
    else
	title (sprintf ('%d-by-%d mesh, %d-point stencil', ...
	    m, n, stencils (s))) ;
    end
    subplot (4,5,(s-1)*5 + 2) ;
    do_spy (C) ;
    title ('nested dissection') ;
    subplot (4,5,(s-1)*5 + 3) ;
    treeplot (parent) ;
    title ('etree') ;
    xlabel ('') ;
    subplot (4,5,(s-1)*5 + 4) ;
    do_spy (L) ;
    title (sprintf ('Cholesky with nd, nnz %d', nnz (L))) ;
    try
        % use the built-in AMD
        p = amd (A) ;
    catch
        try
            % use AMD from SuiteSparse
            p = amd2 (A) ;
        catch
            % use the older built-in SYMAMD
            p = symamd (A) ;
        end
    end
    try
        L = chol (A (p,p), 'lower') ;
    catch
        % old version of MATLAB
        L = chol (A (p,p))' ;
    end
    subplot (4,5,(s-1)*5 + 5) ;
    do_spy (L) ;
    title (sprintf ('Cholesky with amd, nnz %d', nnz (L))) ;

end

%-------------------------------------------------------------------------------

function do_spy (A)
%DO_SPY use cspy(A) to plot a matrix, or spy(A) if cspy not installed.
try
    % This function is in CSparse.  It generates better looking plots than spy.
    cspy (A) ;
catch
    spy (A) ;
end


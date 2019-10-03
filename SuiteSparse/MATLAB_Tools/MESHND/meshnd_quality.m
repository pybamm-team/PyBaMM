function meshnd_quality (do_metis)
%MESHND_QUALITY test the ordering quality computed by meshnd.
% The fill-in and flop count for sparse Cholesky factorization using the meshnd
% nested dissection ordering is computed with AMD.  If SuiteSparse is installed
% with METIS, and if requested, then the metis nested dissection ordering is
% also compared.
%
% Example:
%   meshnd_quality          % compare MESHND and AMD
%   meshnd_quality (1)      % also compare with METIS
%
% See also meshnd, meshsparse, nested, amd, metis.

% Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com

stencils = [5 9 7 27] ;

if (nargin < 1)
    do_metis = 0 ;
end
if (do_metis)
    if (exist ('metis') ~= 3)                                               %#ok
        % METIS not installed
        do_metis = 0 ;
    end
end

clf

for sk = 1:4

    stencil = stencils (sk) ;

    is3D = (stencil == 7 | stencil == 27) ;     %#ok
    if (is3D)
	s = 2.^(3:.1:7) ;		% mesh size up to 127-by-127-by-127
    else
	s = 2.^(3:.1:10) - 1 ;		% mesh size up to 1023-by-1023
    end
    t = length (s) ;
    lnz = nan * zeros (3,t) ;
    fl  = nan * zeros (3,t) ;

    try

        for t = 1:length (s)

            n = floor (s (t)) ;

            % create the mesh and the matrix, and get nested dissection ordering
            if (is3D)
                fprintf ('3D mesh: %d-by-%d-by-%d, %d-point stencil\n', ...
                    n, n, n, stencil) ;
                [G p] = meshnd (n, n, n) ;
            else
                fprintf ('2D mesh: %d-by-%d, %d-point stencil\n', n, n,stencil);
                [G p] = meshnd (n, n) ;
            end
            A = meshsparse (G, stencil) ;

            % ND results
            c = symbfact (A (p,p)) ;
            lnz (1,t) = sum (c) ;
            fl  (1,t) = sum (c.^2) ;
            fprintf ('    MESHND:            nnz(L) %8.3e  flops %8.3e\n', ...
                lnz (1,t), fl (1,t)) ;
            clear G

            % AMD results
            try
                p = amd (A) ;
            catch
                % assume SuiteSparse is installed
                p = amd2 (A) ;
            end
            c = symbfact (A (p,p)) ;
            lnz (2,t) = sum (c) ;
            fl  (2,t) = sum (c.^2) ;
            fprintf ('    AMD:               nnz(L) %8.3e  flops %8.3e\n', ...
                lnz (2,t), fl (2,t)) ;

            % METIS results (requires SuiteSparse and METIS)
            if (do_metis)
                p = metis (A) ;
                c = symbfact (A (p,p)) ;
                lnz (3,t) = sum (c) ;
                fl  (3,t) = sum (c.^2) ;
                fprintf (...
                    '    METIS:             nnz(L) %8.3e  flops %8.3e\n', ...
                    lnz (3,t), fl (3,t)) ;
            end

            % plot the relative nnz(L) results
            subplot (2, 4, 2*sk - 1) ;
            loglog (s (1:t), lnz (2,1:t) ./ lnz (1,1:t), 'b-') ;
            hold on
            if (do_metis)
                loglog (s (1:t), lnz (3,1:t) ./ lnz (1,1:t), 'r-') ;
            end
            loglog (s (1:t), ones (1,t), 'k-') ;
            if (do_metis)
                ylabel ('nnz(L) for AMD or METIS / nnz(L) for meshnd') ;
                legend ('AMD', 'METIS') ;
            else
                ylabel ('nnz(L) for AMD / nnz(L) for meshnd') ;
            end
            xlabel ('mesh size') ;
            axis ([min(s) max(s) .1 10]) ;
            set (gca, 'YTick', [.1 .25 .5 .8 1 1.25 2 4 10]) ;
            if (is3D)
                set (gca, 'XTick', [1 10 100]) ;
                title (sprintf ('3D mesh, %d-point stencil', stencil)) ;
            else
                set (gca, 'XTick', [1 10 100 1000]) ;
                title (sprintf ('2D mesh, %d-point stencil', stencil)) ;
            end

            % plot the relative flop results
            subplot (2, 4, 2*sk) ;
            loglog (s (1:t), fl (2,1:t) ./ fl (1,1:t), 'b-') ;
            hold on
            if (do_metis)
                loglog (s (1:t), fl (3,1:t) ./ fl (1,1:t), 'r-') ;
            end
            loglog (s (1:t), ones (1,t), 'k-') ;
            ylabel ('flops for AMD or METIS / flops for meshnd') ;
            if (do_metis)
                ylabel ('flops for AMD or METIS / flops for meshnd') ;
                legend ('AMD', 'METIS') ;
            else
                ylabel ('nnz(L) for AMD / nnz(L) for meshnd') ;
            end
            xlabel ('mesh size') ;
            axis ([min(s) max(s) .1 10]) ;
            set (gca, 'YTick', [.1 .25 .5 .8 1 1.25 2 4 10]) ;
            if (is3D)
                set (gca, 'XTick', [1 10 100]) ;
                title (sprintf ('3D mesh, %d-point stencil', stencil)) ;
            else
                set (gca, 'XTick', [1 10 100 1000]) ;
                title (sprintf ('2D mesh, %d-point stencil', stencil)) ;
            end

            drawnow

        end

    catch
        % out-of-memory is OK, other errors are not
        disp (lasterr) ;
        if (isempty (strfind (lasterr, 'Out of memory')))
            error (lasterr) ;                                               %#ok
        else
            fprintf ('test terminated early, but otherwise OK\n') ;
        end
    end

end

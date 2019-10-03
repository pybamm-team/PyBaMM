%NN Compare nesdis with metis, in both quality and run time
%
% Example:
%   nn
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

index = UFget ;
f = find (index.amd_lnz > 0) ;
[ignore i] = sort (index.amd_lnz (f)) ;
f = f (i) ;
nmat = length (f) ;

T1 = zeros (1,nmat) ;
T2 = zeros (1,nmat) ;
TM = zeros (1,nmat) ;
Lnz1 = zeros (1,nmat) ;
Lnz2 = zeros (1,nmat) ;
LnzM = zeros (1,nmat) ;
Fl1 = zeros (1,nmat) ;
Fl2 = zeros (1,nmat) ;
FlM = zeros (1,nmat) ;

for k = 1:nmat

    i = f (k) ;
    Prob = UFget (i,index) ;
    A = Prob.A ;
    [m n] = size (A) ;

    if (m ~= n)
	continue ;
    end

    fprintf ('%35s: ', Prob.name) ;

    if (m == n)
	mode = 'sym' ;
	A = A + A' ;
	len = n ;
    elseif (m < n)
	mode = 'row' ;
	len = m ;
    else
	mode = 'col' ;
	len = n ;
    end
    
    fprintf (' %s ', mode) ;

    % try nesdis using camd, and splitting connected components
    tic
    [p2 cparent2 cmember2] = nesdis (A, mode, [200 1]) ;
    t2 = toc ;

    % subplot (3,3,7) ; treeplot (cparent2) ;

    % try nesdis using camd
    tic
    [p1 cparent1 cmember1] = nesdis (A, mode) ;
    t1 = toc ;

    % subplot (3,3,8) ; treeplot (cparent1) ;

    % try metis
    tic
    pm = metis (A, mode) ;
    tm = toc ;

    if (any (sort (p1) ~= 1:len))
	error ('p1!') ;
    end

    if (any (sort (p2) ~= 1:len))
	error ('p2!') ;
    end

    % compare ordering quality

    if (m == n)
	c2 = symbfact2 (A (p2,p2), mode) ; fl2 = sum (c2.^2) ; c2 = sum (c2) ;
	c1 = symbfact2 (A (p1,p1), mode) ; fl1 = sum (c1.^2) ; c1 = sum (c1) ;
	cm = symbfact2 (A (pm,pm), mode) ; flm = sum (cm.^2) ; cm = sum (cm) ;
    elseif (m < n)
	c2 = symbfact2 (A (p2, :), mode) ; fl2 = sum (c2.^2) ; c2 = sum (c2) ;
	c1 = symbfact2 (A (p1, :), mode) ; fl1 = sum (c1.^2) ; c1 = sum (c1) ;
	cm = symbfact2 (A (pm, :), mode) ; flm = sum (cm.^2) ; cm = sum (cm) ;
    else
	c2 = symbfact2 (A ( :,p2), mode) ; fl2 = sum (c2.^2) ; c2 = sum (c2) ;
	c1 = symbfact2 (A ( :,p1), mode) ; fl1 = sum (c1.^2) ; c1 = sum (c1) ;
	cm = symbfact2 (A ( :,pm), mode) ; flm = sum (cm.^2) ; cm = sum (cm) ;
    end

    T1 (k) = t1 ;
    T2 (k) = t2 ;
    TM (k) = tm ;
    Lnz1 (k) = c1 ;
    Lnz2 (k) = c2 ;
    LnzM (k) = cm ;
    Fl1 (k) = fl1 ;
    Fl2 (k) = fl2 ;
    FlM (k) = flm ;
    tmax = max ([max(T1 (1:k)) max(T2 (1:k)) max(TM (1:k))]) ;
    tmin = min ([min(T1 (1:k)) min(T2 (1:k)) min(TM (1:k))]) ;
    cmax = max ([max(Lnz1 (1:k)) max(Lnz2 (1:k)) max(LnzM (1:k))]) ;
    flmax =max ([max(Fl1 (1:k)) max(Fl2 (1:k)) max(FlM (1:k))]) ;

    fprintf (...
	'time %8.2f %8.2f %8.2f speedup %8.2f  flop %8.2e %8.2e %8.2e ratio %8.2f\n', ...
	t2, t1, tm, tm/t1, fl2, fl1, flm, flm/fl1) ;

    if (mod (k, 20) ~= 0)
	continue 
    end

    subplot (3,3,1) ;
    x = T1 (1:k) ./ T2 (1:k) ;
    semilogy (1:k, x, 'o', [1 k], [1 1], 'r-') ;
    axis tight
    title (sprintf ('(nesdis default)/(with split) time, median: %g', ...
	median (x))) ;

    subplot (3,3,2) ;
    x = (Lnz1 (1:k) ./ Lnz2 (1:k)) ;
    semilogy (1:k, x, 'o', [1 k], [1 1], 'r-') ;
    axis tight
    title (sprintf ('(nesdis default)/(with split) lnz, median: %g', ...
	median (x))) ;

    subplot (3,3,3) ;
    x = (Fl1 (1:k) ./ Fl2 (1:k)) ;
    semilogy (1:k, x, 'o', [1 k], [1 1], 'r-') ;
    axis tight
    title (sprintf ('(nesdis default)/(with split) flops, median: %g', ...
	median (x))) ;

    subplot (3,3,4) ;
    x = T1 (1:k) ./ TM (1:k) ;
    semilogy (1:k, x, 'o', [1 k], [1 1], 'r-') ;
    axis tight
    title (sprintf ('(nesdis default)/metis time, median %g', median (x))) ;

    subplot (3,3,5) ;
    x = (Lnz1 (1:k) ./ LnzM (1:k)) ;
    semilogy (1:k, x, 'o', [1 k], [1 1], 'r-') ;
    axis tight
    title (sprintf ('(nesdis default)/metis lnz, median: %g', median (x))) ;

    subplot (3,3,6) ;
    x = (Fl1 (1:k) ./ FlM (1:k)) ;
    semilogy (1:k, x, 'o', [1 k], [1 1], 'r-') ;
    axis tight
    title (sprintf ('(nesdis default)/metis flops, median: %g', median (x))) ;

    drawnow

    % pause
end

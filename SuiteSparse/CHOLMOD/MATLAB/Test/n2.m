%N2 script to test CHOLMOD septree function
% Example:
%   n2
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

index = UFget ;
f = find ((index.amd_lnz > 0) & (index.nrows > 200)) ;
[ignore i] = sort (index.amd_lnz (f)) ;
f = f (i) ;
nmat = length (f) ;

for i = f
    
    Prob = UFget (i, index) ;
    disp (Prob) ;
    A = spones (Prob.A) ;
    [m n] = size (A) ;
    name = Prob.name ;
    clear Prob

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

    [p cp cmem] = nesdis (A, mode) ;

    subplot (2,4,1) ;
    treeplot (cp) ;

    [cp2 cmem2] = septree (cp, cmem, 0.5, 200) ;	    %#ok
    subplot (2,4,2) ;
    treeplot (cp2) ;

    [cp3 cmem3] = septree (cp, cmem, 0.2, 300) ;	    %#ok
    subplot (2,4,3) ;
    treeplot (cp3) ;

    [cp4 cmem4] = septree (cp, cmem, 0.12, 500) ;	    %#ok
    subplot (2,4,4) ;
    treeplot (cp4) ;


    [p cp cmem] = nesdis (A, mode, [200 1]) ;

    subplot (2,4,5) ;
    treeplot (cp) ;

    [cp2 cmem2] = septree (cp, cmem, 0.5, 200) ;	    %#ok
    subplot (2,4,6) ;
    treeplot (cp2) ;

    [cp3 cmem3] = septree (cp, cmem, 0.2, 300) ;	    %#ok
    subplot (2,4,7) ;
    treeplot (cp3) ;

    [cp4 cmem4] = septree (cp, cmem, 0.12, 500) ;	    %#ok
    subplot (2,4,8) ;
    treeplot (cp4) ;

    drawnow
    % pause

end


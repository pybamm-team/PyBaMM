%BTF_DEMO demo for BTF
%
% Example:
%   btf_demo
%
% See also btf, dmperm, strongcomp, maxtrans

% Copyright 2004-2007, University of Florida

load west0479 ;
A = west0479 ;

clf

subplot (2,3,1) ;
spy (A)
title ('west0479') ;

subplot (2,3,2) ;
[p, q, r] = btf (A) ;
% spy (A (p, abs(q))) ;
drawbtf (A, p, q, r) ;
title ('btf') ;

fprintf ('\nbtf_demo: n %d nnz(A) %d  # of blocks %d\n', ...
    size (A,1), nnz (A), length (r) - 1) ;

subplot (2,3,3) ;
[p, q, r, s] = dmperm (A) ;
drawbtf (A, p, q, r) ;
title ('dmperm btf') 

subplot (2,3,4) ;
[p, r] = strongcomp (A) ;
% spy (A (p, abs(q))) ;
drawbtf (A, p, p, r) ;
title ('strongly conn. comp.') ;

subplot (2,3,5) ;
q = maxtrans (A) ;
spy (A (:,q))
title ('max transversal') ;

subplot (2,3,6) ;
p = dmperm (A) ;
spy (A (p,:))
title ('dmperm maxtrans') ;

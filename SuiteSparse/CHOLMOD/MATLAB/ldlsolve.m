function x = ldlsolve (LD,b)						    %#ok
%LDLSOLVE solve LDL'x=b using a sparse LDL' factorization
%
%   Example:
%   x = ldlsolve (LD,b)
%
%   solves the system L*D*L'*x=b for x.  This is equivalent to
%
%   [L,D] = ldlsplit (LD) ;
%   x = L' \ (D \ (L \ b)) ;
%
%   LD is from ldlchol, or as updated by ldlupdate or ldlrowmod.  You must not
%   modify LD as obtained from ldlchol, ldlupdate, or ldlrowmod prior to passing
%   it to this function.  See ldlupdate for more details.
%
%   See also LDLCHOL, LDLUPDATE, LDLSPLIT, LDLROWMOD

%   Copyright 2006-2017, Timothy A. Davis, http://www.suitesparse.com

error ('ldlsolve mexFunction not found') ;

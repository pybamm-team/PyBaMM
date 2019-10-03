function [v,beta,s] = house (x)
%HOUSE find a Householder reflection.
% real or complex case.
% Example:
%   [v,beta,s] = house (x)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

v = x ;
s = norm (x) ;
if (s == 0)
    beta = 0 ;
    v (1) = 1 ;
else
    if (x (1) ~= 0)
        s = sign (x (1)) * s ;
    end
    v (1) = v (1) + s ;
    beta = 1 / real (conj (s) * v (1)) ;
end
s = - s ;


function p = metis (A, mode)						    %#ok
%METIS nested dissection ordering via METIS_NodeND.
%
%   Example:
%   p = metis(A)        returns p such chol(A(p,p)) is typically sparser than
%                       chol(A).  Uses tril(A) and assumes A is symmetric.
%   p = metis(A,'sym')  the same as p=metis(A).
%   p = metis(A,'col')  returns p so that chol(A(:,p)'*A(:,p)) is typically
%                       sparser than chol(A'*A).
%   p = metis(A,'row')  returns p so that chol(A(p,:)*A(p,:)') is typically
%                       sparser than chol(A'*A).
%
%   A must be square for p=metis(A) or metis(A,'sym')
%
%   Requires METIS, authored by George Karypis, Univ. of Minnesota.  This
%   MATLAB interface, via CHOLMOD, is by Tim Davis.
%
%   See also NESDIS, BISECT

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('metis mexFunction not found') ;

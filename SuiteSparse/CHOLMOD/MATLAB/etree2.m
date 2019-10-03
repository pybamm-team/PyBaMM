function [parent, post] = etree2 (A, mode)				    %#ok
%ETREE2  sparse elimination tree.
%   Finds the elimination tree of A, A'*A, or A*A', and optionaly postorders
%   the tree.  parent(j) is the parent of node j in the tree, or 0 if j is a
%   root.  The symmetric case uses only the upper or lower triangular part of
%   A (etree2(A) uses the upper part, and etree2(A,'lo') uses the lower part).
%
%   Example:
%   parent = etree2 (A)         finds the elimination tree of A, using triu(A)
%   parent = etree2 (A,'sym')   same as etree2(A)
%   parent = etree2 (A,'col')   finds the elimination tree of A'*A
%   parent = etree2 (A,'row')   finds the elimination tree of A*A'
%   parent = etree2 (A,'lo')    finds the elimination tree of A, using tril(A)
%
%   [parent,post] = etree2 (...) also returns a post-ordering of the tree.
%
%   If you have a fill-reducing permutation p, you can combine it with an
%   elimination tree post-ordering using the following code.  Post-ordering has
%   no effect on fill-in (except for lu), but it does improve the performance
%   of the subsequent factorization.
%
%   For the symmetric case, suitable for chol(A(p,p)):
%
%       [parent post] = etree2 (A (p,p)) ;
%       p = p (post) ;
%
%   For the column case, suitable for qr(A(:,p)) or lu(A(:,p)):
%
%       [parent post] = etree2 (A (:,p), 'col') ;
%       p = p (post) ;
%
%   For the row case, suitable for qr(A(p,:)') or chol(A(p,:)*A(p,:)'):
%
%       [parent post] = etree2 (A (p,:), 'row') ;
%       p = p (post) ;
%
%   See also TREELAYOUT, TREEPLOT, ETREEPLOT, ETREE

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('etree2 mexFunction not found') ;

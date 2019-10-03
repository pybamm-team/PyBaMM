function [cp_new, cmember_new] = septree (cp, cmember, nd_oksep, nd_small)  %#ok
%SEPTREE prune a separator tree.
%
%   Example:
%   [cp_new, cmember_new] = septree (cp, cmember, nd_oksep, nd_small) ;
%
%   cp and cmember are outputs of nesdis.  cmember(i)=c means that node i is in
%   component c, where c is in the range of 1 to the number of components.
%   length(cp) is the number of components found.  cp is the separator tree;
%   cp(c) is the parent of component c, or 0 if c is a root.  There can be
%   anywhere from 1 to n components, where n is the number of rows of A, A*A',
%   or A'*A.
%
%   On output, cp_new and cmember_new are the new tree and graph-to-tree
%   mapping.  A subtree is collapsed into a single node if the number of nodes
%   in the separator is > nd_oksep times the total size of the subtree, or if
%   the subtree has fewer than nd_small nodes.
%
%   Requires the CHOLMOD Partition Module.
%
%   See also NESDIS.

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('septree mexFunction not found') ;

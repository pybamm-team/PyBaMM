function [p, cparent, cmember] = nesdis (A, mode, opts)			    %#ok
%NESDIS nested dissection ordering via CHOLMOD's nested dissection.
%
%   Example:
%   p = nesdis(A)         returns p such chol(A(p,p)) is typically sparser than
%                         chol(A).  Uses tril(A) and assumes A is symmetric.
%   p = nesdis(A,'sym')   the same as p=nesdis(A).
%   p = nesdis(A,'col')   returns p so that chol(A(:,p)'*A(:,p)) is typically
%                         sparser than chol(A'*A).
%   p = nesdis(A,'row')   returns p so that chol(A(p,:)*A(p,:)') is typically
%                         sparser than chol(A'*A).
%
%   A must be square for p=nesdis(A) or nesdis(A,'sym').
%
%   With three output arguments, [p cp cmember] = nesdis(...), the separator
%   tree and node-to-component mapping is returned.  cmember(i)=c means that
%   node i is in component c, where c is in the range of 1 to the number of
%   components.  length(cp) is the number of components found.  cp is the
%   separator tree; cp(c) is the parent of component c, or 0 if c is a root.
%   There can be anywhere from 1 to n components, where n is dimension of A,
%   A*A', or A'*A.  cmember is a vector of length n.
%
%   An optional 3rd input argument, nesdis (A,mode,opts), modifies the default
%   parameters.  opts(1) specifies the smallest subgraph that should not be
%   partitioned (default is 200).  opts(2) is 0 by default; if nonzero,
%   connected components (formed after the node separator is removed) are
%   partitioned independently.  The default value tends to lead to a more
%   balanced separator tree, cp.  opts(3) defines when a separator is kept; it
%   is kept if the separator size is < opts(3) times the number of nodes in the
%   graph being cut (valid range is 0 to 1, default is 1).
%
%   opts(4) specifies graph is to be ordered after it is dissected.  For the
%   'sym' case: 0: natural ordering, 1: CAMD, 2: CSYMAMD.  For other cases:
%   0: natural ordering, nonzero: CCOLAMD.  The default is 1, to use CAMD for
%   the symmetric case and CCOLAMD for the other cases.
%
%   If opts is shorter than length 4, defaults are used for entries
%   that are not present.
%
%   NESDIS uses METIS' node separator algorithm to recursively partition the
%   graph.  This gives a set of constraints (cmember) that is then passed to
%   CCOLAMD, CSYMAMD, or CAMD, constrained minimum degree ordering algorithms.
%   NESDIS typically takes slightly more time than METIS (METIS_NodeND), but
%   tends to produce better orderings.
%
%   Requires METIS, authored by George Karypis, Univ. of Minnesota.  This
%   MATLAB interface, via CHOLMOD, is by Tim Davis.
%
%   See also METIS, BISECT, AMD

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('nesdis mexFunction not found') ;

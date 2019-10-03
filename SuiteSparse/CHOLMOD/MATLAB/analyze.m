function [p, count] = analyze (A, mode, k)				    %#ok
%ANALYZE order and analyze a matrix using CHOLMOD's best-effort ordering.
%
%   Example:
%   [p count] = analyze (A)         orders A, using just tril(A)
%   [p count] = analyze (A,'sym')   orders A, using just tril(A)
%   [p count] = analyze (A,'row')   orders A*A'
%   [p count] = analyze (A,'col')   orders A'*A
%
%   an optional 3rd parameter modifies the ordering strategy:
%
%   [p count] = analyze (A,'sym',k) orders A, using just tril(A)
%   [p count] = analyze (A,'row',k) orders A*A'
%   [p count] = analyze (A,'col',k) orders A'*A
%
%   Returns a permutation and the count of the number of nonzeros in each
%   column of L for the permuted matrix A.  That is, count is returned as:
%
%      count = symbfact2 (A (p,p))       if ordering A
%      count = symbfact2 (A (p,:),'row') if ordering A*A'
%      count = symbfact2 (A (:,p),'col') if ordering A'*A
%
%   CHOLMOD uses the following ordering strategy:
%
%       k = 0:  Try AMD.  If that ordering gives a flop count >= 500 * nnz(L)
%          and a fill-in of nnz(L) >= 5*nnz(C), then try METIS_NodeND (where
%          C = A, A*A', or A'*A is the matrix being ordered.  Selects the best
%          ordering tried.  This is the default.
%
%       if k > 0, then multiple orderings are attempted.
%
%       k = 1 or 2: just try AMD
%       k = 3: also try METIS_NodeND
%       k = 4: also try NESDIS, CHOLMOD's nested dissection (NESDIS), with
%            default parameters.  Uses METIS's node bisector and CCOLAMD.
%       k = 5: also try the natural ordering (p = 1:n)
%       k = 6: also try NESDIS with large leaves of the separator tree
%       k = 7: also try NESDIS with tiny leaves and no CCOLAMD ordering
%       k = 8: also try NESDIS with no dense-node removal
%       k = 9: also try COLAMD if ordering A'*A or A*A', (AMD if ordering A).
%       k > 9 is treated as k = 9
%
%       k = -1: just use AMD
%       k = -2: just use METIS
%       k = -3: just use NESDIS
%
%       The method returning the smallest nnz(L) is used for p and count.
%       k = 4 takes much longer than (say) k = 0, but it can reduce nnz(L) by
%       a typical 5% to 10%.  k = 5 to 9 is getting extreme, but if you have
%       lots of time and want to find the best ordering possible, set k = 9.
%
%   If METIS is not installed for use in CHOLMOD, then the strategy is
%   different:
%
%       k = 1 to 4: just try AMD
%       k = 5 to 8: also try the natural ordering (p = 1:n)
%       k = 9: also try COLAMD if ordering A'*A or A*A', (AMD if ordering A).
%       k > 9 is treated as k = 9
%
%   See also METIS, NESDIS, BISECT, SYMBFACT, AMD

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('analyze mexFunction not found') ;

function [R,p,q] = chol2 (A)						    %#ok
%CHOL2 sparse Cholesky factorization, A=R'R.
%   Note that A=L*L' (LCHOL) and A=L*D*L' (LDLCHOL) factorizations are faster
%   than R'*R (CHOL2 and CHOL) and use less memory.  The LL' and LDL'
%   factorization methods use tril(A).  This method uses triu(A), just like
%   the built-in CHOL.
%
%   Example:
%   R = chol2 (A)                 same as R = chol (A), just faster
%   [R,p] = chol2 (A)             same as [R,p] = chol(A), just faster
%   [R,p,q] = chol2 (A)           factorizes A(q,q) into R'*R, where q is
%                                 a fill-reducing ordering
%
%   A must be sparse.
%
%   See also LCHOL, LDLCHOL, CHOL, LDLUPDATE.

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('chol2 mexFunction not found') ;

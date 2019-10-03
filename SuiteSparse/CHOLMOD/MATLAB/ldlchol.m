function [LD,p,q] = ldlchol (A,beta)					    %#ok
%LDLCHOL sparse A=LDL' factorization
%   Note that L*L' (LCHOL) and L*D*L' (LDLCHOL) factorizations are faster than
%   R'*R (CHOL2 and CHOL) and use less memory.  The LL' and LDL' factorization
%   methods use tril(A).  A must be sparse.
%
%   Example:
%   LD = ldlchol (A)            return the LDL' factorization of A
%   [LD,p] = ldlchol (A)        similar [R,p] = chol(A), but for L*D*L'
%   [LD,p,q] = ldlchol (A)      factorizes A(q,q) into L*D*L', where q is a
%                               fill-reducing ordering
%   LD = ldlchol (A,beta)       return the LDL' factorization of A*A'+beta*I
%   [LD,p] = ldlchol (A,beta)   like [R,p] = chol(A*A'+beta+I)
%   [LD,p,q] = ldlchol (A,beta) factorizes A(q,:)*A(q,:)'+beta*I into L*D*L'
%
%   The output matrix LD contains both L and D.  D is on the diagonal of LD, and
%   L is contained in the strictly lower triangular part of LD.  The unit-
%   diagonal of L is not stored.  You can obtain the L and D matrices with
%   [L,D] = ldlsplit (LD).  LD is in the form needed by ldlupdate.
%
%   Explicit zeros may appear in the LD matrix.  The pattern of LD matches the
%   pattern of L as computed by symbfact2, even if some entries in LD are
%   explicitly zero.  This is to ensure that ldlupdate and ldlsolve work
%   properly.  You must NOT modify LD in MATLAB itself and then use ldlupdate
%   or ldlsolve if LD contains explicit zero entries; ldlupdate and ldlsolve
%   will fail catastrophically in this case.
%
%   You MAY modify LD in MATLAB if you do not pass it back to ldlupdate or
%   ldlsolve.  Just be aware that LD contains explicit zero entries, contrary
%   to the standard practice in MATLAB of removing those entries from all
%   sparse matrices.  LD = sparse2 (LD) will remove any zero entries in LD.
%
%   See also LDLUPDATE, LDLSOLVE, LDLSPLIT, CHOL2, LCHOL, CHOL, SPARSE2

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('ldlchol mexFunction not found') ;

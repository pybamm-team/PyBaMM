function [LU_or_x,info,c] = klu (A,operation,b,opts)                        %#ok
%KLU sparse left-looking LU factorization, using a block triangular form.
%
%   Example:
%   LU = klu (A)            factorizes R\A(p,q) into L*U+F, returning a struct
%   x = klu (A,'\',b)       x = A\b, using KLU
%   x = klu (b,'/',A)       x = b/A, using KLU
%   x = klu (LU,'\',b)      x = A\b, where LU = klu(A)
%   x = klu (b,'/',LU)      x = b/A, where LU = klu(A)
%
%   KLU(A) factorizes a square sparse matrix, L*U+F = R\A(p,q), where L and U
%   are the factors of the diagonal blocks of the block, F are the entries
%   above the diagonal blocks.  r corresponds to the 3rd output of dmperm; it
%   specifies where the block boundaries are.  The kth block consists of
%   rows/columns r(k) to r(k+1)-1 of A(p,q).
%
%   Note that the use of the scale factor R differs between KLU and UMFPACK
%   (and the LU function, which is based on UMFPACK).  In LU, the factorization
%   is L*U = P*(R1\A)*Q; in KLU it is L*U+F = R2\(P*A*Q).  R1 and R2 are related
%   via R2 = P*R1*P', or equivalently R2 = R1(p,p).
%
%   The LU output is a struct containing members L, U, p, q, R, F, and r.
%
%   opts is an optional input struct which appears as the last input argument.
%   Entries not present are set to their defaults:
%
%                       default
%       opts.tol        0.001   partial pivoting tolerance; valid range 0 to 1.
%       opts.btf        1       use block triangular form (BTF) if nonzero
%       opts.ordering   0       how each block is ordered:
%                               0: AMD, 1: COLAMD, 2: natural,
%                               3: CHOLMOD's ordering of (A'*A),
%                               4: CHOLMOD's ordering of (A+A')
%       opts.scale      2       1: R = diag(sum(abs(A)')), row-sum
%                               2: R = diag(max(abs(A)')), max in each row
%                               otherwise: none (R=I)
%       opts.maxwork    0       if > 0, limit work in BTF ordering to
%                               opts.maxwork*nnz(A); no limit if <= 0.
%
%       The CHOLMOD ordering is to try AMD (for A+A') or COLAMD (for A'*A)
%       first.  If the fill-in with AMD or COLAMD is high, METIS is tried (on
%       A+A' or A'*A), and the best ordering found is selected.  CHOLMOD, METIS,
%       CAMD, and CCOLAMD are required.  If not available, only ordering options
%       0, 1, and 2 may be used (AMD and COLAMD are always required by KLU).
%
%   Two optional outputs, [LU,info,c] = klu (A) or [x,info,c] = klu (A,'\',b)
%   provide statistics about the factorization:
%
%       info.noffdiag   number of off-diagonal pivots chosen (after preordering)
%       info.nrealloc   number of memory reallocations of L and U
%       info.rcond      a very cheap estimate of 1/(condition number)
%       info.rgrowth    reciprocal pivot growth
%       info.flops      flop count
%       info.nblocks    # of blocks in BTF form (1 if not computed)
%       info.ordering   AMD, COLAMD, natural, cholmod(AA'), cholmod(A+A')
%       info.scale      scaling (<=0: none, 1: sum, 2: max)
%       info.lnz        nnz(L), including diagonal
%       info.unz        nnz(U), including diagonal
%       info.offnz      nnz(F)
%       info.tol        pivot tolerance used
%       info.memory     peak memory usage in bytes
%       c               the same as MATLAB's condest
%
%   info and c are relevant only if the matrix is factorized (LU = klu (A),
%   x = klu (A,'/',b), or x = klu (b,'/',A) usages).
%
%   See also BTF, LU, DMPERM, CONDEST, CHOLMOD, AMD, COLAMD, CAMD, CCOLAMD.

% Copyright 2004-2009, Univ. of Florida
% http://www.suitesparse.com

error ('klu mexFunction not found') ;

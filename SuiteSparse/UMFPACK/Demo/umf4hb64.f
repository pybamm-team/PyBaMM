c=======================================================================
c== umf4hb =============================================================
c=======================================================================

c-----------------------------------------------------------------------
c UMFPACK Copyright (c) 2005-2012 by Timothy A. Davis,
c http://www.suitesparse.com.  All Rights Reserved.
c See ../Doc/License.txt for License.
c-----------------------------------------------------------------------

c umf4hb64:
c       read a sparse matrix in the Harwell/Boeing format, factorizes
c       it, and solves Ax=b.  Also saves and loads the factors to/from a
c       file.  Saving to a file is not required, it's just here to
c       demonstrate how to use this feature of UMFPACK.  This program
c       only works on square RUA-type matrices.
c
c       This is HIGHLY non-portable.  It may not work with your C and
c       FORTRAN compilers.  See umf4_f77wrapper.c for more details.
c
c usage (for example):
c
c       in a Unix shell:
c       umf4hb64 < HB/arc130.rua

        integer*8
     $          nzmax, nmax
        parameter (nzmax = 5000000, nmax = 160000)
        integer*8
     $          Ap (nmax), Ai (nzmax), n, nz, totcrd, ptrcrd, i, j, p,
     $          indcrd, valcrd, rhscrd, ncol, nrow, nrhs, nzrhs, nel,
     $          numeric, symbolic, status, sys, filenum

        character title*72, key*30, type*3, ptrfmt*16,
     $          indfmt*16, valfmt*20, rhsfmt*20
        double precision Ax (nzmax), x (nmax), b (nmax), aij, xj,
     $          r (nmax), control (20), info (90)
        character rhstyp*3

c       ----------------------------------------------------------------
c       read the Harwell/Boeing matrix
c       ----------------------------------------------------------------

        read (5, 10, err = 998)
     $          title, key,
     $          totcrd, ptrcrd, indcrd, valcrd, rhscrd,
     $          type, nrow, ncol, nz, nel,
     $          ptrfmt, indfmt, valfmt, rhsfmt
        if (rhscrd .gt. 0) then
c          new Harwell/Boeing format:
           read (5, 20, err = 998) rhstyp, nrhs, nzrhs
           endif
10      format (a72, a8 / 5i14 / a3, 11x, 4i14 / 2a16, 2a20)
20      format (a3, 11x, 2i14)

        print *, 'Matrix key: ', key

        n = nrow
        if (type .ne. 'RUA' .or. nrow .ne. ncol) then
           print *, 'Error: can only handle square RUA matrices'
           stop
        endif
        if (n .ge. nmax .or. nz .gt. nzmax) then
           print *, ' Matrix too big!'
           stop
        endif

c       read the matrix (1-based)
        read (5, ptrfmt, err = 998) (Ap (p), p = 1, ncol+1)
        read (5, indfmt, err = 998) (Ai (p), p = 1, nz)
        read (5, valfmt, err = 998) (Ax (p), p = 1, nz)

c       ----------------------------------------------------------------
c       create the right-hand-side, assume x (i) = 1 + i/n
c       ----------------------------------------------------------------

        do 30 i = 1,n
            b (i) = 0
30      continue
c       b = A*x
        do 50 j = 1,n
            xj = j
            xj = 1 + xj / n
            do 40 p = Ap (j), Ap (j+1)-1
                i = Ai (p)
                aij = Ax (p)
                b (i) = b (i) + aij * xj
40          continue
50      continue

c       ----------------------------------------------------------------
c       convert from 1-based to 0-based
c       ----------------------------------------------------------------

        do 60 j = 1, n+1
            Ap (j) = Ap (j) - 1
60      continue
        do 70 p = 1, nz
            Ai (p) = Ai (p) - 1
70      continue

c       ----------------------------------------------------------------
c       factor the matrix and save to a file
c       ----------------------------------------------------------------

c       set default parameters
        call umf4def (control)

c       print control parameters.  set control (1) to 1 to print
c       error messages only
        control (1) = 2
        call umf4pcon (control)

c       pre-order and symbolic analysis
        call umf4sym (n, n, Ap, Ai, Ax, symbolic, control, info)

c       print statistics computed so far
c       call umf4pinf (control, info) could also be done.
        print 80, info (1), info (16),
     $      (info (21) * info (4)) / 2**20,
     $      (info (22) * info (4)) / 2**20,
     $      info (23), info (24), info (25)
80      format ('symbolic analysis:',/,
     $      '   status:  ', f5.0, /,
     $      '   time:    ', e10.2, ' (sec)'/,
     $      '   estimates (upper bound) for numeric LU:', /,
     $      '   size of LU:    ', f10.2, ' (MB)', /,
     $      '   memory needed: ', f10.2, ' (MB)', /,
     $      '   flop count:    ', e10.2, /
     $      '   nnz (L):       ', f10.0, /
     $      '   nnz (U):       ', f10.0)

c       check umf4sym error condition
        if (info (1) .lt. 0) then
            print *, 'Error occurred in umf4sym: ', info (1)
            stop
        endif

c       numeric factorization
        call umf4num (Ap, Ai, Ax, symbolic, numeric, control, info)

c       print statistics for the numeric factorization
c       call umf4pinf (control, info) could also be done.
        print 90, info (1), info (66),
     $      (info (41) * info (4)) / 2**20,
     $      (info (42) * info (4)) / 2**20,
     $      info (43), info (44), info (45)
90      format ('numeric factorization:',/,
     $      '   status:  ', f5.0, /,
     $      '   time:    ', e10.2, /,
     $      '   actual numeric LU statistics:', /,
     $      '   size of LU:    ', f10.2, ' (MB)', /,
     $      '   memory needed: ', f10.2, ' (MB)', /,
     $      '   flop count:    ', e10.2, /
     $      '   nnz (L):       ', f10.0, /
     $      '   nnz (U):       ', f10.0)

c       check umf4num error condition
        if (info (1) .lt. 0) then
            print *, 'Error occurred in umf4num: ', info (1)
            stop
        endif

c       save the symbolic analysis to the file s0.umf
c       note that this is not needed until another matrix is
c       factorized, below.
	filenum = 0
        call umf4ssym (symbolic, filenum, status)
        if (status .lt. 0) then
            print *, 'Error occurred in umf4ssym: ', status
            stop
        endif

c       save the LU factors to the file n0.umf
        call umf4snum (numeric, filenum, status)
        if (status .lt. 0) then
            print *, 'Error occurred in umf4snum: ', status
            stop
        endif

c       free the symbolic analysis
        call umf4fsym (symbolic)

c       free the numeric factorization
        call umf4fnum (numeric)

c       No LU factors (symbolic or numeric) are in memory at this point.

c       ----------------------------------------------------------------
c       load the LU factors back in, and solve the system
c       ----------------------------------------------------------------

c       At this point the program could terminate and load the LU
C       factors (numeric) from the n0.umf file, and solve the
c       system (see below).  Note that the symbolic object is not
c       required.

c       load the numeric factorization back in (filename: n0.umf)
        call umf4lnum (numeric, filenum, status)
        if (status .lt. 0) then
            print *, 'Error occurred in umf4lnum: ', status
            stop
        endif

c       solve Ax=b, without iterative refinement
        sys = 0
        call umf4sol (sys, x, b, numeric, control, info)
        if (info (1) .lt. 0) then
            print *, 'Error occurred in umf4sol: ', info (1)
            stop
        endif

c       free the numeric factorization
        call umf4fnum (numeric)

c       No LU factors (symbolic or numeric) are in memory at this point.

c       print final statistics
        call umf4pinf (control, info)

c       print the residual.  x (i) should be 1 + i/n
        call resid (n, nz, Ap, Ai, Ax, x, b, r)

c       ----------------------------------------------------------------
c       load the symbolic analysis back in, and factorize a new matrix
c       ----------------------------------------------------------------

c       Again, the program could terminate here, recreate the matrix,
c       and refactorize.  Note that umf4sym is not called.

c       load the symbolic factorization back in (filename: s0.umf)
        call umf4lsym (symbolic, filenum, status)
        if (status .lt. 0) then
            print *, 'Error occurred in umf4lsym: ', status
            stop
        endif

c       arbitrarily change the values of the matrix but not the pattern
        do 100 p = 1, nz
            Ax (p) = Ax (p) + 3.14159 / 100.0
100     continue

c       numeric factorization of the modified matrix
        call umf4num (Ap, Ai, Ax, symbolic, numeric, control, info)
        if (info (1) .lt. 0) then
            print *, 'Error occurred in umf4num: ', info (1)
            stop
        endif

c       free the symbolic analysis
        call umf4fsym (symbolic)

c       create a new right-hand-side, assume x (i) = 7 - i/n
        do 110 i = 1,n
            b (i) = 0
110     continue
c       b = A*x, with the modified matrix A (note that A is now 0-based)
        do 130 j = 1,n
            xj = j
            xj = 7 - xj / n
            do 120 p = Ap (j) + 1, Ap (j+1)
                i = Ai (p) + 1
                aij = Ax (p)
                b (i) = b (i) + aij * xj
120         continue
130     continue

c       ----------------------------------------------------------------
c       solve Ax=b, with iterative refinement
c       ----------------------------------------------------------------

        sys = 0
        call umf4solr (sys, Ap, Ai, Ax, x, b, numeric, control, info)
        if (info (1) .lt. 0) then
            print *, 'Error occurred in umf4solr: ', info (1)
            stop
        endif

c       print the residual.  x (i) should be 7 - i/n
        call resid (n, nz, Ap, Ai, Ax, x, b, r)

c       ----------------------------------------------------------------
c       solve Ax=b, without iterative refinement, broken into steps
c       ----------------------------------------------------------------

c       the factorization is PAQ=LU, PRAQ=LU, or P(R\A)Q=LU.

c       x = R*b (or x=R\b, or x=b, as appropriate)
        call umf4scal (x, b, numeric, status)
        if (status .lt. 0) then
            print *, 'Error occurred in umf4scal: ', status
            stop
        endif

c       solve P'Lr=x for r (using r as workspace)
        sys = 3
        call umf4sol (sys, r, x, numeric, control, info)
        if (info (1) .lt. 0) then
            print *, 'Error occurred in umf4sol: ', info (1)
            stop
        endif

c       solve UQ'x=r for x
        sys = 9
        call umf4sol (sys, x, r, numeric, control, info)
        if (info (1) .lt. 0) then
            print *, 'Error occurred in umf4sol: ', info (1)
            stop
        endif

c       free the numeric factorization
        call umf4fnum (numeric)

c       print the residual.  x (i) should be 7 - i/n
        call resid (n, nz, Ap, Ai, Ax, x, b, r)

        stop
998     print *, 'Read error: Harwell/Boeing matrix'
        stop
        end

c=======================================================================
c== resid ==============================================================
c=======================================================================

c Compute the residual, r = Ax-b, its max-norm, and print the max-norm
C Note that A is zero-based.

        subroutine resid (n, nz, Ap, Ai, Ax, x, b, r)
        integer*8
     $      n, nz, Ap (n+1), Ai (n), j, i, p
        double precision Ax (nz), x (n), b (n), r (n), rmax, aij

        do 10 i = 1, n
            r (i) = -b (i)
10      continue

        do 30 j = 1,n
            do 20 p = Ap (j) + 1, Ap (j+1)
                i = Ai (p) + 1
                aij = Ax (p)
                r (i) = r (i) + aij * x (j)
20          continue
30      continue

        rmax = 0
        do 40 i = 1, n
            rmax = max (rmax, r (i))
40      continue

        print *, 'norm (A*x-b): ', rmax
        return
        end

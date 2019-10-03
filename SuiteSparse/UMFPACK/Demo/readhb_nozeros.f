c=======================================================================
c== readhb_nozeros =====================================================
c=======================================================================

c-----------------------------------------------------------------------
c UMFPACK Copyright (c) 2005-2012 by Timothy A. Davis,
c http://www.suitesparse.com.  All Rights Reserved.
c See ../Doc/License.txt for License.
c-----------------------------------------------------------------------

c readhb_nozeros:
c       read a sparse matrix in the Harwell/Boeing format and
c       output a matrix in triplet format.
c       Identical to readhb, except that this version removes explicit
c       zero entries from the matrix.
c
c usage (for example):
c
c       in a Unix shell:
c       readhb_nozeros < HB/arc130.rua > tmp/A
c
c       Then, in MATLAB, you can do the following:
c       >> load tmp/A
c       >> A = spconvert (A) ;
c       >> spy (A)

        integer nzmax, nmax
        parameter (nzmax = 10000000, nmax = 250000)
        integer Ptr (nmax), Index (nzmax), n, nz, totcrd, ptrcrd,
     $          indcrd, valcrd, rhscrd, ncol, nrow, nrhs, row, col, p
        character title*72, key*30, type*3, ptrfmt*16,
     $          indfmt*16, valfmt*20, rhsfmt*20
        logical sym
        double precision Value (nzmax), skew
        character rhstyp*3
        integer nzrhs, nel

        integer ne, nnz

c-----------------------------------------------------------------------

c       read header information from Harwell/Boeing matrix

        read (5, 10, err = 998)
     $          title, key,
     $          totcrd, ptrcrd, indcrd, valcrd, rhscrd,
     $          type, nrow, ncol, nz, nel,
     $          ptrfmt, indfmt, valfmt, rhsfmt
        if (rhscrd .gt. 0) then
c          new Harwell/Boeing format:
           read (5, 20, err = 998) rhstyp,nrhs,nzrhs
           endif
10      format (a72, a8 / 5i14 / a3, 11x, 4i14 / 2a16, 2a20)
20      format (a3, 11x, 2i14)

        skew = 0.0
        if (type (2:2) .eq. 'Z' .or. type (2:2) .eq. 'z') skew = -1.0
        if (type (2:2) .eq. 'S' .or. type (2:2) .eq. 's') skew =  1.0
        sym = skew .ne. 0.0

        write (0, 31) key
31      format ('Matrix key: ', a8)

        n = max (nrow, ncol)

        if (n .ge. nmax .or. nz .gt. nzmax) then
           write (0, *) 'Matrix too big!'
           write (0, *) '(recompile readhb_nozeros.f with larger',
     $		' nzmax, nmax)'
           stop
        endif

        read (5, ptrfmt, err = 998) (Ptr (p), p = 1, ncol+1)
        read (5, indfmt, err = 998) (Index (p), p = 1, nz)

        do 55 col = ncol+2, n+1
           Ptr (col) = Ptr (ncol+1)
55      continue

c       read the values
        if (valcrd .gt. 0) then
           read (5, valfmt, err = 998) (Value (p), p = 1, nz)
        else
           do 50 p = 1, nz
              Value (p) = 1
50         continue
        endif

c  create the triplet form of the input matrix

        ne = 0
        nnz = 0
        do 100 col = 1, n
           do 90 p = Ptr (col), Ptr (col+1) - 1
              row = Index (p)

c             remove zeros, to compare fairly with LU in MATLAB
c             (MATLAB always removes explicit zeros)
              ne = ne + 1
              if (Value (p) .ne. 0) then
                  nnz = nnz + 1
                  write (6, 200) row, col, Value (p)
              endif

              if (sym .and. row .ne. col) then
                 ne = ne + 1
                 if (Value (p) .ne. 0) then
                    nnz = nnz + 1
                    write (6, 200) col, row, skew * Value (p)
                 endif
              endif

90            continue
100        continue
200     format (2i7, e30.18e3)

c       write (0,*) 'Number of entries: ',ne,' True nonzeros: ', nnz
        stop

998     write (0,*) 'Read error: Harwell/Boeing matrix'
        stop
        end

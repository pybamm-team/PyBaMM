c=======================================================================
c== readhb_size ========================================================
c=======================================================================

c-----------------------------------------------------------------------
c UMFPACK Copyright (c) 2005-2012 by Timothy A. Davis,
c http://www.suitesparse.com.  All Rights Reserved.
c See ../Doc/License.txt for License.
c-----------------------------------------------------------------------

c readhb_size:
c       read a sparse matrix in the Harwell/Boeing format and output the
c       size of the matrix (# rows, # columns, and # of entries)
c
c usage (for example):
c
c       readhb_size < HB/arc130.rua > tmp/Asize

        integer nz, totcrd, ptrcrd,
     $          indcrd, valcrd, rhscrd, ncol, nrow, nrhs
        character title*72, key*30, type*3, ptrfmt*16,
     $          indfmt*16, valfmt*20, rhsfmt*20
        character rhstyp*3
        integer nzrhs, nel

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

        write (6, *) nrow, ncol, nz
        stop
998     write (0, *) 'Read error'
        stop
        end


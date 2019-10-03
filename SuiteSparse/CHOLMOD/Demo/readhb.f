c-----------------------------------------------------------------------
c Read a sparse matrix in the Harwell/Boeing format and output a
c matrix in triplet format.  Only the lower triangular part of a
c symmetric matrix is provided.  Does not handle skew-symmetric
c matrices.
c-----------------------------------------------------------------------

        integer nzmax, nmax
        parameter (nzmax = 100000000, nmax = 1000000)
        integer Ptr (nmax), Index (nzmax), n, nz, totcrd, ptrcrd,
     $		indcrd, valcrd, rhscrd, ncol, nrow, nrhs, row, col, p
        character title*72, key*30, type*3, ptrfmt*16,
     $          indfmt*16, valfmt*20, rhsfmt*20
        logical sym
        double precision Value (nzmax), skew
        character rhstyp*3
        integer nzrhs, nel, stype

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

c       write (0, 30) title, key, type, nrow, ncol, nz
        if (rhscrd .gt. 0) then
c          new Harwell/Boeing format:
c          write (0, 40) rhstyp,nrhs,nzrhs
        endif
30      format (
     $          ' title: ', a72 /
     $          ' key: ', a8 /
     $          ' type: ', a3, ' nrow: ', i14, ' ncol: ', i14 /
     $          ' nz: ', i14)
40      format (' rhstyp: ', a3, ' nrhs: ', i14, ' nzrhs: ', i14)
*       write (0, *) ' sym: ', sym, ' skew: ', skew

	if (skew .eq. -1) then
	   write (0, *) 'Cannot handle skew-symmetric matrices'
	   stop
	endif

        n = max (nrow, ncol)

        if (ncol .ge. nmax .or. nz .gt. nzmax) then
           write (0, *) ' Matrix too big!'
           stop
        endif

        read (5, ptrfmt, err = 998) (Ptr (p), p = 1, ncol+1)
        read (5, indfmt, err = 998) (Index (p), p = 1, nz)

c       read the values
        if (valcrd .gt. 0) then
           read (5, valfmt, err = 998) (Value (p), p = 1, nz)
        endif

c  create the triplet form of the input matrix
c  stype = 0: unsymmetric
c  stype = -1: symmetric, lower triangular part present

	stype = -skew

	write (6, 101) title
	write (6, 102) key
101	format ('% title:', a72)
102	format ('% key:  ', a8)
	write (6, 110) nrow, ncol, nz, stype
110	format (2i8, i12, i3)

        do 100 col = 1, ncol
           do 90 p = Ptr (col), Ptr (col+1) - 1
              row = Index (p)
	      if (valcrd .gt. 0) then
                 write (6, 200) row, col, Value (p)
c                if (sym .and. row .ne. col) then
c		    write (6, 200) col, row, skew * Value (p)
c		 endif
	      else
                 write (6, 201) row, col
	      endif
90         continue
100     continue
200	format (2i8, e30.18e3)
201	format (2i8)
        stop

998     write (0,*) 'Read error: Harwell/Boeing matrix'
        stop
        end

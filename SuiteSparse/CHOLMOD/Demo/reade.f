c-----------------------------------------------------------------------
c Read a sparse matrix in the Harwell/Boeing format and output a
c matrix in triplet format.
c
c not for:   **A
c use for:   RSE and PSE
c-----------------------------------------------------------------------

        integer nzmax, nmax
        parameter (nzmax = 50000000, nmax = 1000000)
        integer Ptr (nmax), Index (nzmax), n, nz, totcrd, ptrcrd,
     $		indcrd, valcrd, rhscrd, ncol, nrow, nrhs, row, col, p,
     $		nguess, nexact, nrhsix, nrhsvl
        character title*72, key*30, type*3, ptrfmt*16,
     $          indfmt*16, valfmt*20, rhsfmt*20
        logical sym, unsorted
        double precision Value (nzmax)
	double precision skew
        character rhstyp*3
        integer nel
	integer lastrow

	integer prow, el, pval, hasval
	integer stype

c-----------------------------------------------------------------------
c read header information from Harwell/Boeing matrix
c-----------------------------------------------------------------------

	nrhs = 0
	nrhsix = 0

        read (5, 10, err = 998)
     $          title, key,
     $          totcrd, ptrcrd, indcrd, valcrd, rhscrd,
     $          type, nrow, ncol, nz, nel,
     $          ptrfmt, indfmt, valfmt, rhsfmt
        if (rhscrd .gt. 0) then
c          new Harwell/Boeing format:
           read (5, 20, err = 998) rhstyp,nrhs,nrhsix
           endif
10      format (a72, a8 / 5i14 / a3, 11x, 4i14 / 2a16, 2a20)
20      format (a3, 11x, 2i14)

        skew = 0.0
        if (type (2:2) .eq. 'Z' .or. type (2:2) .eq. 'z') then
            write (0, *) '*ZE not supported!'
	    stop
	endif
        if (type (2:2) .eq. 'S' .or. type (2:2) .eq. 's') skew =  1.0
        if (type (2:2) .eq. 'H' .or. type (2:2) .eq. 'h') then
            write (0, *) '*HE not supported!'
	    stop
	endif
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

        n = max (nrow, ncol)

        if (n .ge. nmax .or. nz .gt. nzmax) then
            write (0, *) 'Matrix too big!'
            stop
        endif

	if (type (3:3) .ne. 'E' .and. type (3:3) .ne. 'a') then
            write (0, *) 'Can only handle **E types!'
            stop
	endif

	if (.not. (
     $		type (1:1) .eq. 'P' .or.
     $		type (1:1) .eq. 'p' .or.
     $		type (1:1) .eq. 'R' .or.
     $		type (1:1) .eq. 'r')) then
            write (0, *) 'Can only handle R*E or P*E types!'
            stop
	endif

c-----------------------------------------------------------------------
c read the pattern
c-----------------------------------------------------------------------

        read (5, ptrfmt, err = 998) (Ptr (p), p = 1, ncol+1)
        read (5, indfmt, err = 998) (Index (p), p = 1, nz)

c-----------------------------------------------------------------------
c  check if the columns are sorted
c-----------------------------------------------------------------------

	unsorted = .false.
        do 101 col = 1, ncol
	   lastrow = 0
           do 91 p = Ptr (col), Ptr (col+1) - 1
              row = Index (p)
	      if (row .lt. lastrow) then
	          unsorted = .true.
	          write (0,*) ' ********* Columns are unsorted'
		  goto 102
	      endif
	      lastrow = row
91            continue
101        continue
102	continue

c-----------------------------------------------------------------------
c read the values
c-----------------------------------------------------------------------

        if (valcrd .gt. 0) then
           read (5, valfmt, err = 998) (Value (p), p = 1, nel)
	endif

c-----------------------------------------------------------------------
c  write the triplet form of the input matrix
c-----------------------------------------------------------------------

c  stype = 0: unsymmetric
c  stype = -1: symmetric, lower triangular part present

	stype = -1

	nz = 0
        do 300 el = 1, ncol
	      do 390 p = Ptr (el), Ptr (el+1) - 1
		 do 392 prow = p, Ptr (el+1) - 1
		    nz = nz + 1 
392		 continue
390	     continue
300	continue

	write (6, 701) title
	write (6, 702) key
701	format ('% title:', a72)
702	format ('% key:  ', a8)
	write (6, 710) nrow, nrow, nz, stype
710	format (2i8, i12, i3)

	pval = 0
        do 100 el = 1, ncol
	      do 90 p = Ptr (el), Ptr (el+1) - 1
	         col = Index (p)
		 do 92 prow = p, Ptr (el+1) - 1
	             row = Index (prow)
		     if (row .lt. col) then
			write (0, *) 'bad format!'
			stop
		     endif
		     if (valcrd .gt. 0) then
		        pval = pval + 1
			if (pval .gt. nel) then
			    write (0, *) 'bad format!'
			    stop
			endif
			write (6, 200) row, col, Value (pval)
		     else
			write (6, 201) row, col
		     endif
92		 continue
90	     continue
100        continue

200	format (2i8, e30.18e3)
201	format (2i8)

c-----------------------------------------------------------------------
        stop

998     write (0,*) 'Read error: Harwell/Boeing matrix'
        stop
        end


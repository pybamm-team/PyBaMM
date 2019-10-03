C ======================================================================
C === AMD_cross ========================================================
C ======================================================================

C ----------------------------------------------------------------------
C AMD, Copyright (c) by Timothy A. Davis, Patrick R.
C Amestoy, and Iain S. Duff.  See ../README.txt for License.
C email: DrTimothyAldenDavis@gmail.com
C ----------------------------------------------------------------------

C This program provides an example of how to call the C version of AMD
C from a Fortran program.  It is HIGHLY non-portable.

C The amd_order routine returns PERM (1) < 0 if an error occurs.
C (-1: out of memory, -2: invalid matrix)

C Note that the input matrix is 0-based.  From Fortran, column j of the
C matrix is in AI (AP (I)+1 ... AP (I+1)).  The row indices in this
C set are in the range 0 to N-1.  To demonstrate this translation,
C the input matrix is printed in 1-based form.  This program uses
C the same 5-by-5 test matrix as amd_simple.c.

        INTEGER N, NZ, K, P
        PARAMETER (N = 5, NZ = 14)
        INTEGER AP (N+1), AI (NZ), PERM (N)
        DATA AP / 0,   2,       6,       10,  12, 14 /
        DATA AI / 0,1, 0,1,2,4, 1,2,3,4, 2,3, 1,4    /
        DOUBLE PRECISION CONTROL (5), INFO (20)

C       print the input matrix
        PRINT 10, N, N, NZ
10      FORMAT ('Input matrix:', I2, '-by-', I2, ' with',I3,' entries')
        DO 40 J = 1, N
            PRINT 20, J, AP (J+1) - AP (J), AP (J)+1, AP (J+1)
20          FORMAT ( /, 'Column: ', I2, ' number of entries: ', I2,
     $          ' with row indices in AI (', I3, ' ... ', I3, ')')
            PRINT 30, ((AI (P) + 1), P = AP (J) + 1, AP (J+1))
30          FORMAT ('    row indices: ', 24I3)
40      CONTINUE

        CALL AMDDEFAULTS (CONTROL)
        CALL AMDORDER (N, AP, AI, PERM, CONTROL, INFO)
        CALL AMDINFO (INFO)

        DO 60 K = 1, N
            PRINT 50, K, PERM (K) + 1
50          FORMAT ('PERM (',I2,') = ', I2)
60      CONTINUE
        END


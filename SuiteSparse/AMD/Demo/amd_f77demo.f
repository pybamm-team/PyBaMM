C ======================================================================
C === Fortran AMD demo main program ====================================
C ======================================================================

C ----------------------------------------------------------------------
C AMD, Copyright (c) by Timothy A. Davis, Patrick R.
C Amestoy, and Iain S. Duff.  See ../README.txt for License.
C email: DrTimothyAldenDavis@gmail.com
C ----------------------------------------------------------------------

C A simple Fortran 77 main program that illustrates the use of the
C Fortran version of AMD (both the AMD and AMDBAR routines).  Note
C that aggressive absorption has no effect on this particular matrix.

C       AP and AI contain the symmetric can_24 Harwell/Boeing matrix,
C       including upper and lower triangular parts, but excluding the
C       diagonal entries.  Note that this matrix is 1-based, with row
C       and column indices in the range 1 to N.

        INTEGER N, NZ, IWLEN, PFREE, I, J, K, JNEW, P, INEW,
     $      METHOD, NCMPA
        PARAMETER (N = 24, NZ = 136, IWLEN = 200)
        INTEGER PE (N), DEGREE (N), NV (N), NEXT (N), PERM (N), W (N),
     $      HEAD (N), PINV (N), LEN (N), AP (N+1), AI (NZ), IW (IWLEN)
        CHARACTER A (24,24)

        DATA AP
     $      / 1, 9, 14, 19, 24, 29, 34, 42, 50, 53, 61, 66, 71,
     $       76, 81, 86, 91, 94, 102, 110, 118, 123, 131, 134, 137 /
        DATA AI /
     $      6, 7, 13, 14, 18, 19, 20, 22,
     $      9, 10, 14, 15, 18,
     $      7, 12, 21, 22, 23,
     $      8, 11, 16, 19, 20,
     $      8, 10, 15, 16, 17,
     $      1, 7, 13, 14, 18,
     $      1, 3, 6, 12, 13, 20, 22, 24,
     $      4, 5, 10, 15, 16, 17, 18, 19,
     $      2, 10, 15,
     $      2, 5, 8, 9, 14, 15, 18, 19,
     $      4, 19, 20, 21, 22,
     $      3, 7, 13, 22, 24,
     $      1, 6, 7, 12, 24,
     $      1, 2, 6, 10, 18,
     $      2, 5, 8, 9, 10,
     $      4, 5, 8, 17, 19,
     $      5, 8, 16,
     $      1, 2, 6, 8, 10, 14, 19, 20,
     $      1, 4, 8, 10, 11, 16, 18, 20,
     $      1, 4, 7, 11, 18, 19, 21, 22,
     $      3, 11, 20, 22, 23,
     $      1, 3, 7, 11, 12, 20, 21, 23,
     $      3, 21, 22,
     $      7, 12, 13 /

C       print the input matrix
        PRINT 11, N, N, NZ
11      FORMAT ('AMD Fortran 77 demo, with the 24-by-24',
     $      ' Harwell/Boeing matrix, can_24:'
     $      /, 'Input matrix: ', I2, '-by-', I2,' with ',I3,' entries',
     $      /, 'Note that the Fortran version of AMD requires that'
     $      /, 'no diagonal entries be present.')
        DO 20 J = 1, N
            PRINT 21, J, AP (J+1) - AP (J), AP (J), AP (J+1)-1
21          FORMAT ( /, 'Column: ', I2, ' number of entries: ', I2,
     $          ' with row indices in AI (', I3, ' ... ', I3, ')')
            PRINT 10, ((AI (P)), P = AP (J), AP (J+1) - 1)
10          FORMAT ('    row indices: ', 24I3)
20      CONTINUE

C       print a character plot of the input matrix.  This is only
C       reasonable because the matrix is small.
        PRINT 31
31      FORMAT ('Plot of input matrix pattern:')
        DO 50 J = 1,N
            DO 30 I = 1,N
                A (I, J) = '.'
30          CONTINUE
C           add the diagonal entry to the plot
            A (J, J) = 'X'
            DO 40 P = AP (J), AP (J+1) - 1
                I = AI (P)
                A (I, J) = 'X'
40          CONTINUE
50      CONTINUE
        PRINT 60, ((MOD (J, 10)), J = 1,N)
60      FORMAT ('     ', 24I2)
        DO 80 I = 1,N
            PRINT 70, I, (A (I, J), J = 1,N)
70          FORMAT (' ', I2, ': ', 24A2)
80      CONTINUE

        DO 190 METHOD = 1,2

C           load the matrix into AMD's workspace
            DO 90 J = 1,N
                PE (J) = AP (J)
                LEN (J) = AP (J+1) - AP (J)
90          CONTINUE
            DO 100 P = 1,NZ
                IW (P) = AI (P)
100         CONTINUE
            PFREE = NZ + 1

C           order the matrix using AMD or AMDBAR
            IF (METHOD .EQ. 1) THEN
                PRINT 101 
101             FORMAT (/, '------------------------------------------',
     $                  /, 'ordering the matrix with AMD',
     $                  /, '------------------------------------------')
                CALL AMD (N, PE, IW, LEN, IWLEN, PFREE, NV, NEXT,
     $              PERM, HEAD, PINV, DEGREE, NCMPA, W)
            ELSE
                PRINT 102 
102             FORMAT (/, '------------------------------------------',
     $                  /, 'ordering the matrix with AMDBAR',
     $                  /, '------------------------------------------')
                CALL AMDBAR (N, PE, IW, LEN, IWLEN, PFREE, NV, NEXT,
     $              PERM, HEAD, PINV, DEGREE, NCMPA, W)
            ENDIF

C           print the permutation vector, PERM, and its inverse, PINV.
C           row/column J = PERM (K) is the Kth row/column in the
C           permuted matrix.
            PRINT 110, (PERM (K), K = 1,N)
110         FORMAT (/, 'Permutation vector: ', /, 24I3)
            PRINT 120, (PINV (J), J = 1,N)
120         FORMAT (/, 'Inverse permutation vector: ', /, 24I3)

C           print a character plot of the permuted matrix.
            PRINT 121
121         FORMAT ('Plot of permuted matrix pattern:')
            DO 150 JNEW = 1,N
                J = PERM (JNEW)
                DO 130 INEW = 1,N
                    A (INEW, JNEW) = '.'
130             CONTINUE
C               add the diagonal entry to the plot
                A (JNEW, JNEW) = 'X'
                DO 140 P = AP (J), AP (J+1) - 1
                    INEW = PINV (AI (P))
                    A (INEW, JNEW) = 'X'
140             CONTINUE
150         CONTINUE
            PRINT 60, ((MOD (J, 10)), J = 1,N)
            DO 160 I = 1,N
                PRINT 70, I, (A (I, J), J = 1,N)
160         CONTINUE

C           print the permuted matrix, PERM*A*PERM'
            DO 180 JNEW = 1,N
                J = PERM (JNEW)
                PRINT 171, JNEW, J, AP (J+1) - AP (J)
171             FORMAT (/, 'New column: ', I2, ' old column: ', I2,
     $              ' number of entries: ', I2)
                PRINT 170, (PINV (AI (P)), P = AP (J), AP (J+1) - 1)
170             FORMAT ('    new row indices: ', 24I3)
180         CONTINUE
190     CONTINUE
        END

C ----------------------------------------------------------------------
C AMD, Copyright (c) by Timothy A. Davis, Patrick R.
C Amestoy, and Iain S. Duff.  See ../README.txt for License.
C email: DrTimothyAldenDavis@gmail.com
C ----------------------------------------------------------------------

C This program provides an example of how to call the Fortran version
C of AMD.  It uses the same matrix as the amd_simple.c demo (in C).
C Note that the diagonal entries are not present, and the matrix is
C symmetric.

        INTEGER N, NZ, J, K, P, IWLEN, PFREE, NCMPA
        PARAMETER (N = 5, NZ = 10, IWLEN = 17)
        INTEGER AP (N+1), AI (NZ), LAST (N), PE (N), LEN (N), ELEN (N),
     $      IW (IWLEN), DEGREE (N), NV (N), NEXT (N), HEAD (N), W (N)
        DATA AP / 1, 2,     5,     8,  9,  11/
        DATA AI / 2, 1,3,5, 2,4,5, 3,  2,3   /

C       load the matrix into the AMD workspace
        DO 10 J = 1,N
            PE (J) = AP (J)
            LEN (J) = AP (J+1) - AP (J)
10      CONTINUE
        DO 20 P = 1,NZ
            IW (P) = AI (P)
20      CONTINUE
        PFREE = NZ + 1

C       order the matrix (destroys the copy of A in IW, PE, and LEN)
        CALL AMD (N, PE, IW, LEN, IWLEN, PFREE, NV, NEXT, LAST, HEAD,
     $      ELEN, DEGREE, NCMPA, W)

        DO 60 K = 1, N
            PRINT 50, K, LAST (K)
50          FORMAT ('P (',I2,') = ', I2)
60      CONTINUE
        END

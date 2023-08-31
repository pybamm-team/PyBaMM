
#if SUNDIALS_VERSION_MAJOR < 6

  #define SUN_PREC_NONE PREC_NONE
  #define SUN_PREC_LEFT PREC_LEFT

  // Compatibility layer - wrap older sundials functions in new-style calls
  void SUNContext_Create(void *comm, SUNContext *ctx)
  {
    // Function not available
    return;
  }

  int SUNContext_Free(SUNContext *ctx)
  {
    // Function not available
    return;
  }

  void* IDACreate(SUNContext sunctx)
  {
    return IDACreate();
  }

  N_Vector N_VNew_Serial(sunindextype vec_length, SUNContext sunctx)
  {
    return N_VNew_Serial(vec_length);
  }

  N_Vector N_VNew_OpenMP(sunindextype vec_length, SUNContext sunctx)
  {
    return N_VNew_OpenMP(vec_length);
  }

  N_Vector N_VNew_Cuda(sunindextype vec_length, SUNContext sunctx)
  {
    return N_VNew_Cuda(vec_length);
  }

  SUNMatrix SUNSparseMatrix(sunindextype M, sunindextype N, sunindextype NNZ, int sparsetype, SUNContext sunctx)
  {
    return SUNMatrix SUNSparseMatrix(M, N, NNZ, sparsetype);
  }

  SUNMatrix SUNMatrix_cuSparse_NewCSR(int M, int N, int NNZ, cusparseHandle_t cusp, SUNContext sunctx)
  {
    return SUNMatrix_cuSparse_NewCSR(M, N, NNZ, cusp);
  }

  SUNMatrix SUNBandMatrix(sunindextype N, sunindextype mu, sunindextype ml, SUNContext sunctx)
  {
    return SUNMatrix SUNBandMatrix(N, mu, ml);
  }

  SUNMatrix SUNDenseMatrix(sunindextype M, sunindextype N, SUNContext sunctx)
  {
    return SUNDenseMatrix(M, N, sunctx);
  }

  SUNLinearSolver SUNLinSol_Dense(N_Vector y, SUNMatrix A, SUNContext sunctx)
  {
    return SUNLinSol_Dense(y, A, sunctx);
  }

  SUNLinearSolver SUNLinSol_KLU(N_Vector y, SUNMatrix A, SUNContext sunctx)
  {
    return SUNLinSol_KLU(y, A, sunctx);
  }

  SUNLinearSolver SUNLinSol_Band(N_Vector y, SUNMatrix A, SUNContext sunctx)
  {
    return SUNLinSol_Band(y, A, sunctx);
  }

  SUNLinearSolver SUNLinSol_SPBCGS(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPBCGS(y, pretype, maxl);
  }

  SUNLinearSolver SUNLinSol_SPFGMR(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPFGMR(y, pretype, maxl);
  }

  SUNLinearSolver SUNLinSol_SPGMR(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPGMR(y, pretype, maxl);
  }

  SUNLinearSolver SUNLinSol_SPTFQMR(N_Vector y, int pretype, int maxl, SUNContext sunctx)
  {
    return SUNLinSol_SPTFQMR(y, pretype, maxl);
  }
#endif

#include "CasadiSolver_Cuda.hpp"

/*
 * CasadiSolver implementations specialised for CUDA jobs
 */

void CasadiSolver_cuSolverSp_batchQR::SetLinearSolver() {
  cusolverStatus_t cusol_status = cusolverSpCreate(&cusol);
  if (cusol_status != CUSOLVER_STATUS_SUCCESS) {
    DEBUG("ERROR: could not create cuSPARSE cusol handle\n");
    std::bad_alloc exception;
    throw exception;
  }
  
  LS = SUNLinSol_cuSolverSp_batchQR(yy, J, cusol, sunctx);
}

void CasadiSolver_cuSolverSp_batchQR::AllocateVectors() {
  yy = N_VNew_Cuda(number_of_states, sunctx);
  yp = N_VNew_Cuda(number_of_states, sunctx);
  avtol = N_VNew_Cuda(number_of_states, sunctx);
  id = N_VNew_Cuda(number_of_states, sunctx);
}

void CasadiSolver_cuSolverSp_batchQR::SetMatrix() {
  cusparseStatus_t cusp_status = cusparseCreate(&cusp);
  if (cusp_status != CUSPARSE_STATUS_SUCCESS) {
    DEBUG("ERROR: could not create cuSPARSE cusp handle\n");
    std::bad_alloc exception;
    throw exception;
  }
  
  J = SUNMatrix_cuSparse_NewCSR(
    number_of_states,
    number_of_states,
    jac_times_cjmass_nnz,
    cusp,
    sunctx
  );
}

void CasadiSolver_cuSolverSp_batchQR::ChildDestructors() {
  cusparseDestroy(cusp);
  cusolverSpDestroy(cusol);
}

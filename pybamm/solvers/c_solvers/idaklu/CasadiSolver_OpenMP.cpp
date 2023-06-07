#include "CasadiSolver_OpenMP.hpp"

/*
 * CasadiSolver implementations compatible with the OPENMP vector class
 */

void CasadiSolver_Dense::SetLinearSolver() {
  LS = SUNLinSol_Dense(yy, J, sunctx);
}

void CasadiSolver_KLU::SetLinearSolver() {
  LS = SUNLinSol_KLU(yy, J, sunctx);
}

void CasadiSolver_Band::SetLinearSolver() {
  LS = SUNLinSol_Band(yy, J, sunctx);
}

void CasadiSolver_SPBCGS::SetLinearSolver() {
  LS = SUNLinSol_SPBCGS(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

void CasadiSolver_SPFGMR::SetLinearSolver() {
  LS = SUNLinSol_SPFGMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

void CasadiSolver_SPGMR::SetLinearSolver() {
  LS = SUNLinSol_SPGMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

void CasadiSolver_SPTFQMR::SetLinearSolver() {
  LS = SUNLinSol_SPTFQMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

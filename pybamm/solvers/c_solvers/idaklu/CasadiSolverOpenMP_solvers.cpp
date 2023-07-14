#include "CasadiSolverOpenMP_solvers.hpp"

/*
 * CasadiSolver implementations compatible with the OPENMP vector class
 */

void CasadiSolverOpenMP_Dense::SetLinearSolver() {
  LS = SUNLinSol_Dense(yy, J, sunctx);
}

void CasadiSolverOpenMP_KLU::SetLinearSolver() {
  LS = SUNLinSol_KLU(yy, J, sunctx);
}

void CasadiSolverOpenMP_Band::SetLinearSolver() {
  LS = SUNLinSol_Band(yy, J, sunctx);
}

void CasadiSolverOpenMP_SPBCGS::SetLinearSolver() {
  LS = SUNLinSol_SPBCGS(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

void CasadiSolverOpenMP_SPFGMR::SetLinearSolver() {
  LS = SUNLinSol_SPFGMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

void CasadiSolverOpenMP_SPGMR::SetLinearSolver() {
  LS = SUNLinSol_SPGMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

void CasadiSolverOpenMP_SPTFQMR::SetLinearSolver() {
  LS = SUNLinSol_SPTFQMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
}

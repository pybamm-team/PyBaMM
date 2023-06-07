#include "CasadiSolverCuda_solvers.hpp"

/*
 * CasadiSolver implementations specialised for CUDA jobs
 */

void CasadiSolverCuda_cuSolverSp_batchQR::Initialize() {}

Solution CasadiSolverCuda_cuSolverSp_batchQR::solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs) {

}

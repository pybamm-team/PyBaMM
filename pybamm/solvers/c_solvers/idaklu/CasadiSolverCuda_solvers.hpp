#ifndef PYBAMM_IDAKLU_CASADISOLVERCUDA_HPP
#define PYBAMM_IDAKLU_CASADISOLVERCUDA_HPP

#include "CasadiSolver.hpp"
#include <casadi/casadi.hpp>
using Function = casadi::Function;

#include "casadi_functions.hpp"
#include "common.hpp"
#include "options.hpp"
#include "solution.hpp"
#include "sundials_legacy_wrapper.hpp"

class CasadiSolverCuda_cuSolverSp_batchQR : public CasadiSolver
{
public:
  void *ida_mem;            // pointer to memory
  np_array atol_np;
  double rel_tol;
  np_array rhs_alg_id;
  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  int precon_type;
  N_Vector yy, yp, avtol;   // y, y', and absolute tolerance
  N_Vector *yyS, *ypS;      // y, y' for sensitivities
  N_Vector id;              // rhs_alg_id
  realtype rtol;
  const int jac_times_cjmass_nnz;
  int jac_bandwidth_lower;
  int jac_bandwidth_upper;
  SUNMatrix J;
  SUNLinearSolver LS;
  std::unique_ptr<CasadiFunctions> functions;
  Options options;

  SUNMatrix Ja;
  int block_nnz;
  int nblocks;
  cusparseStatus_t cusp_status;
  cusolverStatus_t cusol_status;
  cusparseHandle_t cusp_handle;
  cusolverSpHandle_t cusol_handle;

  int rtn;

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext sunctx;
#endif

public:
  CasadiSolverCuda_cuSolverSp_batchQR(
    np_array atol_np,
    double rel_tol,
    np_array rhs_alg_id,
    int number_of_parameters,
    int number_of_events,
    int jac_times_cjmass_nnz,
    int jac_bandwidth_lower,
    int jac_bandwidth_upper,
    std::unique_ptr<CasadiFunctions> functions,
    const Options& options);
  ~CasadiSolverCuda_cuSolverSp_batchQR();
  Solution solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs) override;
  void sync_device();
  void Initialize() override;
  void AllocateVectors();
  void SetMatrix();
  void SetLinearSolver();
};

#endif // PYBAMM_IDAKLU_CASADISOLVERCUDA_HPP

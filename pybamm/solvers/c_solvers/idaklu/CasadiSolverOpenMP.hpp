#ifndef PYBAMM_IDAKLU_CASADISOLVEROPENMP_HPP
#define PYBAMM_IDAKLU_CASADISOLVEROPENMP_HPP

#include "CasadiSolver.hpp"
#include <casadi/casadi.hpp>
using Function = casadi::Function;

#include "casadi_functions.hpp"
#include "common.hpp"
#include "options.hpp"
#include "solution.hpp"
#include "sundials_legacy_wrapper.hpp"

class CasadiSolverOpenMP : public CasadiSolver
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
  realtype *res;
  realtype *res_dvar_dy;
  realtype *res_dvar_dp;
  Options options;

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext sunctx;
#endif

public:
  CasadiSolverOpenMP(
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
  ~CasadiSolverOpenMP();
  void CalcVars(
    realtype *y_return,
    size_t length_of_return_vector,
    size_t t_i,
    realtype *tret,
    realtype *yval,
    std::vector<realtype*> ySval,
    realtype *yS_return,
    size_t *ySk);
  void CalcVarsSensitivities(
    realtype *tret,
    realtype *yval,
    std::vector<realtype*> ySval,
    realtype *yS_return,
    size_t *ySk);
  Solution solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs) override;
  void Initialize() override;
  void AllocateVectors();
  void SetMatrix();
  virtual void SetLinearSolver() = 0;
};

#endif // PYBAMM_IDAKLU_CASADISOLVEROPENMP_HPP

#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_HPP

#include <casadi/casadi.hpp>
using Function = casadi::Function;

#include "casadi_functions.hpp"
#include "common.hpp"
#include "options.hpp"
#include "solution.hpp"
#include "sundials_legacy_wrapper.hpp"

class CasadiSolver
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

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext sunctx;
#endif

public:
  CasadiSolver(
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
  ~CasadiSolver();
  Solution solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs);
  void Initialize();
  void AllocateVectors();
  void SetMatrix();
  virtual void SetLinearSolver() = 0;
  void ChildDestructors() {};
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP

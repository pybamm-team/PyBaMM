#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_HPP

#include <casadi/casadi.hpp>
using Function = casadi::Function;

#include "solution.hpp"
#include "casadi_functions.hpp"
#include "common.hpp"

class CasadiSolver
{
public:
  CasadiSolver(np_array atol_np, int number_of_parameters, bool use_jacobian,
               int jac_times_cjmass_nnz, CasadiFunctions &functions);

  void *ida_mem; // pointer to memory

  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  N_Vector yy, yp, avtol; // y, y', and absolute tolerance
  N_Vector *yyS, *ypS;    // y, y' for sensitivities
  N_Vector id;            // rhs_alg_id
  realtype rtol, *yval, *ypval, *atval, *ySval;
  const int jac_times_cjmass_nnz;

  SUNMatrix J;
  SUNLinearSolver LS;

  CasadiFunctions functions;

  Solution solve(np_array t_np, np_array y0_np, np_array yp0_np,
                 np_array_dense inputs);
};


CasadiSolver create_casadi_solver(int number_of_states, int number_of_parameters,
                      const Function &rhs_alg, const Function &jac_times_cjmass,
                      const np_array_int &jac_times_cjmass_colptrs,
                      const np_array_int &jac_times_cjmass_rowvals,
                      const int jac_times_cjmass_nnz,
                      const Function &jac_action, const Function &mass_action,
                      const Function &sens, const Function &event,
                      const int number_of_events, int use_jacobian,
                      np_array rhs_alg_id, np_array atol_np, double rel_tol,
                      np_array_dense inputs);


#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP

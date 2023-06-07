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
  CasadiSolver();
  ~CasadiSolver();
  virtual Solution solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs) = 0;
  virtual void Initialize() = 0;
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP

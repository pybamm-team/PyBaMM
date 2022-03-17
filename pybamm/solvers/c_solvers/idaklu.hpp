#ifndef PYBAMM_IDAKLU_HPP
#define PYBAMM_IDAKLU_HPP

#include "solution.hpp"

Solution solve_python(np_array t_np, np_array y0_np, np_array yp0_np,
               residual_type res, jacobian_type jac, 
               sensitivities_type sens,
               jac_get_type gjd, jac_get_type gjrv, jac_get_type gjcp, 
               int nnz, event_type event,
               int number_of_events, int use_jacobian, np_array rhs_alg_id,
               np_array atol_np, double rel_tol, int number_of_parameters);

#endif // PYBAMM_IDAKLU_HPP

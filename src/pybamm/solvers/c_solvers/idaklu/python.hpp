#ifndef PYBAMM_IDAKLU_HPP
#define PYBAMM_IDAKLU_HPP

#include "common.hpp"
#include "Solution.hpp"
#include <functional>

using residual_type = std::function<
    np_array(realtype, np_array, np_array, np_array)
  >;
using sensitivities_type = std::function<void(
    std::vector<np_array>&, realtype, const np_array&,
    const np_array&,
    const np_array&, const std::vector<np_array>&,
    const std::vector<np_array>&
  )>;
using jacobian_type = std::function<np_array(realtype, np_array, np_array, realtype)>;

using event_type =
    std::function<np_array(realtype, np_array, np_array)>;

using jac_get_type = std::function<np_array()>;


/**
 * @brief Interface to the python solver
 */
Solution solve_python(np_array t_np, np_array y0_np, np_array yp0_np,
               residual_type res, jacobian_type jac,
               sensitivities_type sens,
               jac_get_type gjd, jac_get_type gjrv, jac_get_type gjcp,
               int nnz, event_type event,
               int number_of_events, int use_jacobian, np_array rhs_alg_id,
               np_array atol_np, double rel_tol, np_array inputs,
               int number_of_parameters);

#endif // PYBAMM_IDAKLU_HPP


#ifndef PYBAMM_IDAKLU_CASADI_HPP
#define PYBAMM_IDAKLU_CASADI_HPP

#include <casadi/casadi.hpp>

using Function = casadi::Function;

#include "solution.hpp"

Solution solve_casadi(np_array t_np, np_array y0_np, np_array yp0_np,
               const Function &rhs_alg, 
               const Function &jac_times_cjmass, 
               const np_array_int &jac_times_cjmass_colptrs, 
               const np_array_int &jac_times_cjmass_rowvals, 
               const int jac_times_cjmass_nnz,
               const Function &jac_action, 
               const Function &mass_action, 
               const Function &sens, 
               const Function &event, 
               const int number_of_events, 
               int use_jacobian, 
               np_array rhs_alg_id, 
               np_array atol_np,  
               double rel_tol, 
               np_array_dense inputs,
               int number_of_parameters);




#endif // PYBAMM_IDAKLU_CASADI_HPP

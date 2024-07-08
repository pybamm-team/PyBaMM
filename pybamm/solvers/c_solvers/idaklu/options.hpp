#ifndef PYBAMM_OPTIONS_HPP
#define PYBAMM_OPTIONS_HPP

#include "common.hpp"

/**
 * @brief Options passed to the idaklu solver by pybamm
 */
struct Options {
  bool print_stats;
  bool using_sparse_matrix;
  bool using_banded_matrix;
  bool using_iterative_solver;
  std::string jacobian;
  std::string preconditioner; // spbcg
  int precon_half_bandwidth;
  int precon_half_bandwidth_keep;
  int num_threads;
  // IDA main solver
  int max_order_bdf;
  int max_num_steps;
  double dt_init;
  double dt_max;
  int max_error_test_failures;
  int max_nonlinear_iterations;
  int max_convergence_failures;
  double nonlinear_convergence_coefficient;
  // IDA initial conditions calculation
  sunbooleantype suppress_algebraic_error;
  double nonlinear_convergence_coefficient_ic;
  int max_num_steps_ic;
  int max_number_jacobians_ic;
  int max_number_iterations_ic;
  int max_linesearch_backtracks_ic;
  sunbooleantype linesearch_off_ic;
  bool calc_ic;
  bool init_all_y_ic;
  // IDALS linear solver interface
  std::string linear_solver; // klu, lapack, spbcg
  int linsol_max_iterations;
  sunbooleantype linear_solution_scaling;
  double epsilon_linear_tolerance;
  double increment_factor;
  explicit Options(py::dict options);

};

#endif // PYBAMM_OPTIONS_HPP

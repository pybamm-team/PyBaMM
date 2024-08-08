#ifndef PYBAMM_OPTIONS_HPP
#define PYBAMM_OPTIONS_HPP

#include "common.hpp"

/**
 * @brief SetupOptions passed to the idaklu setup by pybamm
 */
struct SetupOptions {
  bool using_sparse_matrix;
  bool using_banded_matrix;
  bool using_iterative_solver;
  std::string jacobian;
  std::string preconditioner; // spbcg
  int precon_half_bandwidth;
  int precon_half_bandwidth_keep;
  int num_threads;
  // IDALS linear solver interface
  std::string linear_solver; // klu, lapack, spbcg
  int linsol_max_iterations;
  explicit SetupOptions(py::dict &py_opts);
};

/**
 * @brief SolverOptions passed to the idaklu solver by pybamm
 */
struct SolverOptions {
  bool print_stats;
  // IDA main solver
  int max_order_bdf;
  int max_num_steps;
  double dt_init;
  double dt_max;
  int max_error_test_failures;
  int max_nonlinear_iterations;
  int max_convergence_failures;
  double nonlinear_convergence_coefficient;
  double nonlinear_convergence_coefficient_ic;
  sunbooleantype suppress_algebraic_error;
  // IDA initial conditions calculation
  bool calc_ic;
  bool init_all_y_ic;
  int max_num_steps_ic;
  int max_num_jacobians_ic;
  int max_num_iterations_ic;
  int max_linesearch_backtracks_ic;
  sunbooleantype linesearch_off_ic;
  // IDALS linear solver interface
  sunbooleantype linear_solution_scaling;
  double epsilon_linear_tolerance;
  double increment_factor;
  explicit SolverOptions(py::dict &py_opts);
};

#endif // PYBAMM_OPTIONS_HPP

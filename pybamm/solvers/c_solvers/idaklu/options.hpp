#ifndef PYBAMM_OPTIONS_HPP
#define PYBAMM_OPTIONS_HPP

#include "common.hpp"

struct Options {
  bool print_stats;
  bool using_sparse_matrix;
  bool using_iterative_solver;
  std::string jacobian;
  int linsol_max_iterations;
  int precon_half_bandwidth;
  int precon_half_bandwidth_keep;
  std::string linear_solver; // klu, lapack, spbcg 
  Options(py::dict options);

};

#endif // PYBAMM_OPTIONS_HPP

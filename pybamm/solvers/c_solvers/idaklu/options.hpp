#ifndef PYBAMM_OPTIONS_HPP
#define PYBAMM_OPTIONS_HPP

#include "common.hpp"

struct Options {
  bool print_stats;
  bool use_jacobian;
  bool dense_jacobian;
  std::string linear_solver; // klu, lapack, spbcg 
  Options(py::dict options);

};

#endif // PYBAMM_OPTIONS_HPP

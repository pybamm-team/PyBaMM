#ifndef PYBAMM_OPTIONS_HPP
#define PYBAMM_OPTIONS_HPP

#include "common.hpp"

struct Options {
  bool print_stats;
  bool use_jacobian;

  Options(py::dict options);

};

#endif // PYBAMM_OPTIONS_HPP

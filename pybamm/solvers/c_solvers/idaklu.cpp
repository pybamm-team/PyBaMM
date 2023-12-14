#include "idaklu/casadi_solver.hpp"
#include "idaklu/common.hpp"
#include "idaklu/python.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <iostream>

#include "pybind11_kernel_helpers.h"

Function generate_function(const std::string &data)
{
  return Function::deserialize(data);
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<np_array>);

template <typename T>
void cpu_idaklu(void *out_tuple, const void **in) {
  // Parse the inputs
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t vars = *reinterpret_cast<const std::int64_t *>(in[1]);
  const T *t = reinterpret_cast<const T *>(in[2]);
  const T *in1 = reinterpret_cast<const T *>(in[3]);
  const T *in2 = reinterpret_cast<const T *>(in[4]);
  std::cout << "size: " << size << std::endl;
  std::cout << "vars: " << vars << std::endl;
  for (std::int64_t n = 0; n < size; ++n) {
    std::cout << "t: " << t[n] << std::endl;
    std::cout << "in1: " << in1[n] << std::endl;
    std::cout << "in2: " << in2[n] << std::endl;
  }

  // We have a single output, which is a multi-dimensional array; recurse over the dimensions
  void *out = reinterpret_cast<T *>(out_tuple);
  std::int64_t i = 0;
  for (std::int64_t n = 0; n < vars; ++n) {
    for (std::int64_t m = 0; m < size; ++m) {
      reinterpret_cast<T *>(out)[i] = (T) (t[m] + n);
      std::cout << " out" << reinterpret_cast<T *>(out)[i];
      ++i;
    }
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_idaklu_f32"] = EncapsulateFunction(cpu_idaklu<float>);
  dict["cpu_idaklu_f64"] = EncapsulateFunction(cpu_idaklu<double>);
  return dict;
}

PYBIND11_MODULE(idaklu, m)
{
  m.doc() = "sundials solvers"; // optional module docstring

  py::bind_vector<std::vector<np_array>>(m, "VectorNdArray");

  m.def("solve_python", &solve_python,
    "The solve function for python evaluators",
    py::arg("t"),
    py::arg("y0"),
    py::arg("yp0"),
    py::arg("res"),
    py::arg("jac"),
    py::arg("sens"),
    py::arg("get_jac_data"),
    py::arg("get_jac_row_vals"),
    py::arg("get_jac_col_ptr"),
    py::arg("nnz"),
    py::arg("events"),
    py::arg("number_of_events"),
    py::arg("use_jacobian"),
    py::arg("rhs_alg_id"),
    py::arg("atol"),
    py::arg("rtol"),
    py::arg("inputs"),
    py::arg("number_of_sensitivity_parameters"),
    py::return_value_policy::take_ownership);

  py::class_<CasadiSolver>(m, "CasadiSolver")
  .def("solve", &CasadiSolver::solve,
    "perform a solve",
    py::arg("t"),
    py::arg("y0"),
    py::arg("yp0"),
    py::arg("inputs"),
    py::return_value_policy::take_ownership);

  //py::bind_vector<std::vector<Function>>(m, "VectorFunction");
  //py::implicitly_convertible<py::iterable, std::vector<Function>>();

  m.def("create_casadi_solver", &create_casadi_solver,
    "Create a casadi idaklu solver object",
    py::arg("number_of_states"),
    py::arg("number_of_parameters"),
    py::arg("rhs_alg"),
    py::arg("jac_times_cjmass"),
    py::arg("jac_times_cjmass_colptrs"),
    py::arg("jac_times_cjmass_rowvals"),
    py::arg("jac_times_cjmass_nnz"),
    py::arg("jac_bandwidth_lower"),
    py::arg("jac_bandwidth_upper"),
    py::arg("jac_action"),
    py::arg("mass_action"),
    py::arg("sens"),
    py::arg("events"),
    py::arg("number_of_events"),
    py::arg("rhs_alg_id"),
    py::arg("atol"),
    py::arg("rtol"),
    py::arg("inputs"),
    py::arg("var_casadi_fcns"),
    py::arg("dvar_dy_fcns"),
    py::arg("dvar_dp_fcns"),
    py::arg("options"),
    py::return_value_policy::take_ownership);

  m.def("generate_function", &generate_function,
    "Generate a casadi function",
    py::arg("string"),
    py::return_value_policy::take_ownership);
  
  m.def("registrations", &Registrations);

  py::class_<Function>(m, "Function");

  py::class_<Solution>(m, "solution")
  .def_readwrite("t", &Solution::t)
  .def_readwrite("y", &Solution::y)
  .def_readwrite("yS", &Solution::yS)
  .def_readwrite("flag", &Solution::flag);
}

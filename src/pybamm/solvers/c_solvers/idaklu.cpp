#include <vector>
#include <iostream>
#include <functional>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "idaklu/idaklu_solver.hpp"
#include "idaklu/IdakluJax.hpp"
#include "idaklu/common.hpp"
#include "idaklu/python.hpp"
#include "idaklu/Expressions/Casadi/CasadiFunctions.hpp"

#ifdef IREE_ENABLE
#include "idaklu/Expressions/IREE/IREEFunctions.hpp"
#endif


casadi::Function generate_casadi_function(const std::string &data)
{
  return casadi::Function::deserialize(data);
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<np_array>);

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

  py::class_<IDAKLUSolver>(m, "IDAKLUSolver")
  .def("solve", &IDAKLUSolver::solve,
    "perform a solve",
    py::arg("t"),
    py::arg("y0"),
    py::arg("yp0"),
    py::arg("inputs"),
    py::return_value_policy::take_ownership);

  m.def("create_casadi_solver", &create_idaklu_solver<CasadiFunctions>,
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
    py::arg("var_fcns"),
    py::arg("dvar_dy_fcns"),
    py::arg("dvar_dp_fcns"),
    py::arg("options"),
    py::return_value_policy::take_ownership);

#ifdef IREE_ENABLE
  m.def("create_iree_solver", &create_idaklu_solver<IREEFunctions>,
    "Create a iree idaklu solver object",
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
    py::arg("var_fcns"),
    py::arg("dvar_dy_fcns"),
    py::arg("dvar_dp_fcns"),
    py::arg("options"),
    py::return_value_policy::take_ownership);
#endif

  m.def("generate_function", &generate_casadi_function,
    "Generate a casadi function",
    py::arg("string"),
    py::return_value_policy::take_ownership);

  // IdakluJax interface routines
  py::class_<IdakluJax>(m, "IdakluJax")
    .def(
      "register_callback_eval",
      &IdakluJax::register_callback_eval,
      "Register a callback for function evaluation",
      py::arg("callback")
    )
    .def(
      "register_callback_jvp",
      &IdakluJax::register_callback_jvp,
      "Register a callback for JVP evaluation",
      py::arg("callback")
    )
    .def(
      "register_callback_vjp",
      &IdakluJax::register_callback_vjp,
      "Register a callback for the VJP evaluation",
      py::arg("callback")
    )
    .def(
      "register_callbacks",
      &IdakluJax::register_callbacks,
      "Register callbacks for function evaluation, JVP evaluation, and VJP evaluation",
      py::arg("callback_eval"),
      py::arg("callback_jvp"),
      py::arg("callback_vjp")
    )
    .def(
      "get_index",
      &IdakluJax::get_index,
      "Get the index of the JAXified instance"
    );
  m.def(
    "create_idaklu_jax",
    &create_idaklu_jax,
    "Create an idaklu jax object"
  );
  m.def(
    "registrations",
    &Registrations
  );

  py::class_<casadi::Function>(m, "Function");

#ifdef IREE_ENABLE
  py::class_<IREEBaseFunctionType>(m, "IREEBaseFunctionType")
    .def(py::init<>())
    .def_readwrite("mlir", &IREEBaseFunctionType::mlir)
    .def_readwrite("kept_var_idx", &IREEBaseFunctionType::kept_var_idx)
    .def_readwrite("nnz", &IREEBaseFunctionType::nnz)
    .def_readwrite("numel", &IREEBaseFunctionType::numel)
    .def_readwrite("col", &IREEBaseFunctionType::col)
    .def_readwrite("row", &IREEBaseFunctionType::row)
    .def_readwrite("pytree_shape", &IREEBaseFunctionType::pytree_shape)
    .def_readwrite("pytree_sizes", &IREEBaseFunctionType::pytree_sizes)
    .def_readwrite("n_args", &IREEBaseFunctionType::n_args);
#endif

  py::class_<Solution>(m, "solution")
    .def_readwrite("t", &Solution::t)
    .def_readwrite("y", &Solution::y)
    .def_readwrite("yS", &Solution::yS)
    .def_readwrite("y_term", &Solution::y_term)
    .def_readwrite("flag", &Solution::flag);
}

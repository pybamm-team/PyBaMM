#include <vector>
#include <iostream>
#include <functional>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "idaklu_source/idaklu_solver.hpp"
#include "idaklu_source/observe.hpp"
#include "idaklu_source/IDAKLUSolverGroup.hpp"
#include "idaklu_source/IdakluJax.hpp"
#include "idaklu_source/common.hpp"
#include "idaklu_source/Expressions/Casadi/CasadiFunctions.hpp"

#ifdef IREE_ENABLE
#include "idaklu_source/Expressions/IREE/IREEFunctions.hpp"
#endif


casadi::Function generate_casadi_function(const std::string &data)
{
  return casadi::Function::deserialize(data);
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<np_array>);
PYBIND11_MAKE_OPAQUE(std::vector<np_array_realtype>);
PYBIND11_MAKE_OPAQUE(std::vector<Solution>);

PYBIND11_MODULE(idaklu, m)
{
  m.doc() = "sundials solvers"; // optional module docstring

  py::bind_vector<std::vector<np_array>>(m, "VectorNdArray");
  py::bind_vector<std::vector<np_array_realtype>>(m, "VectorRealtypeNdArray");
  py::bind_vector<std::vector<Solution>>(m, "VectorSolution");

  py::class_<IDAKLUSolverGroup>(m, "IDAKLUSolverGroup")
  .def("solve", &IDAKLUSolverGroup::solve,
    "perform a solve",
    py::arg("t_eval"),
    py::arg("t_interp"),
    py::arg("y0"),
    py::arg("yp0"),
    py::arg("inputs"),
    py::return_value_policy::take_ownership);

  m.def("create_casadi_solver_group", &create_idaklu_solver_group<CasadiFunctions>,
    "Create a group of casadi idaklu solver objects",
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

  m.def("observe", &observe,
    "Observe variables",
    py::arg("ts"),
    py::arg("ys"),
    py::arg("inputs"),
    py::arg("funcs"),
    py::arg("is_f_contiguous"),
    py::arg("shape"),
    py::return_value_policy::take_ownership);

  m.def("observe_hermite_interp", &observe_hermite_interp,
    "Observe and Hermite interpolate variables",
    py::arg("t_interp"),
    py::arg("ts"),
    py::arg("ys"),
    py::arg("yps"),
    py::arg("inputs"),
    py::arg("funcs"),
    py::arg("shape"),
    py::return_value_policy::take_ownership);

#ifdef IREE_ENABLE
  m.def("create_iree_solver_group", &create_idaklu_solver_group<IREEFunctions>,
    "Create a group of iree idaklu solver objects",
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
    .def_readwrite("yp", &Solution::yp)
    .def_readwrite("yS", &Solution::yS)
    .def_readwrite("ypS", &Solution::ypS)
    .def_readwrite("y_term", &Solution::y_term)
    .def_readwrite("flag", &Solution::flag);
}

#include <vector>
#include <iostream>
#include <functional>
#include <optional>

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
#include "idaklu_source/sundials_error_handler.hpp"
#include "idaklu_source/reduce.hpp"
#include "idaklu_source/StandaloneNewtonSolver.hpp"


casadi::Function generate_casadi_function(const std::string &data)
{
  return casadi::Function::deserialize(data);
}

namespace py = pybind11;

IDAKLUSolverGroup *create_casadi_solver_group(
  int number_of_states,
  int number_of_parameters,
  const CasadiFunctions::BaseFunctionType &rhs_alg,
  const CasadiFunctions::BaseFunctionType &jac_times_cjmass,
  const np_array_int &jac_times_cjmass_colptrs,
  const np_array_int &jac_times_cjmass_rowvals,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower,
  const int jac_bandwidth_upper,
  const CasadiFunctions::BaseFunctionType &jac_action,
  const CasadiFunctions::BaseFunctionType &mass_action,
  const CasadiFunctions::BaseFunctionType &sens,
  const CasadiFunctions::BaseFunctionType &events,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  const std::vector<CasadiFunctions::BaseFunctionType*>& var_fcns,
  const std::vector<CasadiFunctions::BaseFunctionType*>& dvar_dy_fcns,
  const std::vector<CasadiFunctions::BaseFunctionType*>& dvar_dp_fcns,
  py::dict py_opts,
  const CasadiFunctions::BaseFunctionType &alg_res,
  const CasadiFunctions::BaseFunctionType &alg_jac)
{
  auto setup_opts = SetupOptions(py_opts);
  auto solver_opts = SolverOptions(py_opts);

  std::optional<SerializedCasadiFunctions> serialized_fcns;
  if (setup_opts.num_solvers > 1) {
    serialized_fcns = serialize_casadi_functions(
      rhs_alg,
      jac_times_cjmass,
      jac_action,
      mass_action,
      sens,
      events,
      var_fcns,
      dvar_dy_fcns,
      dvar_dp_fcns);
  }

  std::vector<std::unique_ptr<IDAKLUSolver>> solvers;
  solvers.reserve(setup_opts.num_solvers);
  for (int i = 0; i < setup_opts.num_solvers; i++) {
    auto functions = std::make_unique<CasadiFunctions>(
      rhs_alg,
      jac_times_cjmass,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      jac_times_cjmass_rowvals,
      jac_times_cjmass_colptrs,
      inputs_length,
      jac_action,
      mass_action,
      sens,
      events,
      number_of_states,
      number_of_events,
      number_of_parameters,
      var_fcns,
      dvar_dy_fcns,
      dvar_dp_fcns,
      setup_opts,
      serialized_fcns ? &*serialized_fcns : nullptr,
      alg_res,
      alg_jac
    );
    solvers.emplace_back(
      std::unique_ptr<IDAKLUSolver>(
        create_idaklu_solver(
          std::move(functions),
          number_of_parameters,
          jac_times_cjmass_colptrs,
          jac_times_cjmass_rowvals,
          jac_times_cjmass_nnz,
          jac_bandwidth_lower,
          jac_bandwidth_upper,
          number_of_events,
          rhs_alg_id,
          atol_np,
          rel_tol,
          inputs_length,
          solver_opts,
          setup_opts
        )
      )
    );
  }

  return new IDAKLUSolverGroup(
    std::move(solvers), number_of_states, number_of_parameters);
}

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
    py::arg("logger") = py::none(),
    py::return_value_policy::take_ownership);

  m.def("create_casadi_solver_group", &create_casadi_solver_group,
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
    py::arg("alg_res"),
    py::arg("alg_jac"),
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

  m.def("generate_function", &generate_casadi_function,
    "Generate a casadi function",
    py::arg("string"),
    py::return_value_policy::take_ownership);

  m.def("sundials_error_message", &sundials_error_message,
    "Get a human-readable message for a SUNDIALS error code",
    py::arg("flag"),
    py::return_value_policy::copy);

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

  m.def("reduce_knots", &reduce_knots,
    "Streaming knot reduction on multi-segment solution data",
    py::arg("ts"), py::arg("ys"), py::arg("yps"),
    py::arg("atols"), py::arg("t_evals"),
    py::arg("rtol"),
    py::arg("hermite_reduction_factor"));

  py::class_<StandaloneNewtonSolver>(m, "StandaloneNewtonSolver")
    .def(py::init<casadi::Function, casadi::Function,
                  std::vector<sunrealtype>, sunrealtype, sunrealtype,
                  int, int, sunrealtype, bool>(),
         py::arg("residual"), py::arg("jacobian"),
         py::arg("atol"), py::arg("rtol"), py::arg("step_tol"),
         py::arg("max_iter"), py::arg("max_backtracks"),
         py::arg("eps_newt"), py::arg("use_sparse"))
    .def("solve", &StandaloneNewtonSolver::solve,
         py::arg("t"), py::arg("y0"), py::arg("inputs"),
         py::return_value_policy::move)
    .def("solve_batch", &StandaloneNewtonSolver::solve_batch,
         py::arg("t_eval"), py::arg("y0_alg"), py::arg("inputs"),
         py::return_value_policy::move);

  py::class_<casadi::Function>(m, "Function");

  py::class_<Solution>(m, "solution")
    .def_readwrite("t", &Solution::t)
    .def_readwrite("y", &Solution::y)
    .def_readwrite("yp", &Solution::yp)
    .def_readwrite("yS", &Solution::yS)
    .def_readwrite("ypS", &Solution::ypS)
    .def_readwrite("y_term", &Solution::y_term)
    .def_readwrite("flag", &Solution::flag);

}

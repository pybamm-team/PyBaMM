
#include "idaklu.hpp" 

#include "idaklu_casadi.hpp" 
#include "idaklu_python.hpp" 

#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

Function generate_function(const std::string& data) {
  return Function::deserialize(data);
}

Solution solve_casadi_wrapper(np_array t_np, np_array y0_np, np_array yp0_np,
               const std::string &rhs_alg, 
               const std::string &jac_times_cjmass, 
               const np_array &jac_times_cjmass_colptrs, 
               const np_array &jac_times_cjmass_rowvals, 
               const int jac_times_cjmass_nnz,
               const std::string &jac_action, 
               const std::string &mass_action, 
               const std::string &sens, 
               const std::string &event, 
               const int number_of_events, 
               int use_jacobian, 
               np_array rhs_alg_id,
               np_array atol_np, double rel_tol, int number_of_parameters) {

  return solve_casadi(t_np, y0_np, yp0_np,
                      generate_function(rhs_alg), 
                      generate_function(jac_times_cjmass), 
                      jac_times_cjmass_colptrs, 
                      jac_times_cjmass_rowvals, 
                      jac_times_cjmass_nnz,
                      generate_function(jac_action), 
                      generate_function(mass_action), 
                      generate_function(sens), 
                      generate_function(event), 
                      number_of_events, 
                      use_jacobian, 
                      rhs_alg_id,
                      atol_np, rel_tol, number_of_parameters);

  std::cout << "exiting wrapper" << std::endl;

}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<np_array>);

PYBIND11_MODULE(idaklu, m)
{
  m.doc() = "sundials solvers"; // optional module docstring

  py::bind_vector<std::vector<np_array>>(m, "VectorNdArray");

  m.def("solve_python", &solve_python, "The solve function for python evaluators", 
        py::arg("t"), py::arg("y0"),
        py::arg("yp0"), py::arg("res"), py::arg("jac"), py::arg("sens"), 
        py::arg("get_jac_data"),
        py::arg("get_jac_row_vals"), py::arg("get_jac_col_ptr"), py::arg("nnz"),
        py::arg("events"), py::arg("number_of_events"), py::arg("use_jacobian"),
        py::arg("rhs_alg_id"), py::arg("atol"), py::arg("rtol"),
        py::arg("number_of_sensitivity_parameters"),
        py::return_value_policy::take_ownership);

  m.def("solve_casadi", &solve_casadi, "The solve function for casadi evaluators", 
        py::arg("t"), py::arg("y0"), py::arg("yp0"), 
        py::arg("rhs_alg"), 
        py::arg("jac_times_cjmass"), 
        py::arg("jac_times_cjmass_colptrs"), 
        py::arg("jac_times_cjmass_rowvals"), 
        py::arg("jac_times_cjmass_nnz"), 
        py::arg("jac_action"), 
        py::arg("mass_action"), 
        py::arg("sens"), 
        py::arg("events"), py::arg("number_of_events"), 
        py::arg("use_jacobian"),
        py::arg("rhs_alg_id"),
        py::arg("atol"), py::arg("rtol"),
        py::arg("number_of_sensitivity_parameters"),
        py::return_value_policy::take_ownership);

  m.def("generate_function", &generate_function, "Generate a casadi function", 
        py::arg("string"),
        py::return_value_policy::take_ownership);

  py::class_<Function>(m, "Function");

  py::class_<Solution>(m, "solution")
      .def_readwrite("t", &Solution::t)
      .def_readwrite("y", &Solution::y)
      .def_readwrite("yS", &Solution::yS)
      .def_readwrite("flag", &Solution::flag);
}



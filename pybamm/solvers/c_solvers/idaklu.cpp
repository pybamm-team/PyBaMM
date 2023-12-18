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
#include <functional>

#include "pybind11_kernel_helpers.h"

Function generate_function(const std::string &data)
{
  return Function::deserialize(data);
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<np_array>);

#define PYBIND11_DETAILED_ERROR_MESSAGES

int global_var = 0;
using Handler = std::function<np_array(realtype, realtype, realtype)>;

Handler handler;

void cpu_idaklu(void *out_tuple, const void **in) {
  // Parse the inputs --- note that these come from jax lowering and are NOT np_array's
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t vars = *reinterpret_cast<const std::int64_t *>(in[1]);
  const realtype *t = reinterpret_cast<const realtype *>(in[2]);
  const realtype *in1 = reinterpret_cast<const realtype *>(in[3]);
  const realtype *in2 = reinterpret_cast<const realtype *>(in[4]);
  void *out = reinterpret_cast<realtype *>(out_tuple);

  /*
  py::capsule free_when_done(t, [](void *f) {
    std::cout << "freeing memory" << std::endl;
  });

  // Create a Python object that will free the allocated memory when destroyed
  np_array t_np = py::array_t<realtype>(
    {size}, // shape
    {sizeof(realtype)}, // C-style contiguous strides for double
    t, // the data pointer
    free_when_done
  );
  */
  realtype t_np = *t;

  /*size_t shape[1] = {1};
  size_t strides[1] = {sizeof(realtype)};
  auto t_np = py::array_t<realtype>(shape, strides);
  auto view = t_np.mutable_unchecked<1>();
  for (size_t i = 0; i < size; ++i) {
    view(i) = t[i];
  }*/
  
  std::cout << "size: " << size << std::endl;
  std::cout << "vars: " << vars << std::endl;
  for (std::int64_t n = 0; n < size; ++n) {
    std::cout << "t: " << t[n] << std::endl;
    //std::cout << "a: " << t_np.at(n) << std::endl;
    std::cout << "in1: " << in1[n] << std::endl;
    std::cout << "in2: " << in2[n] << std::endl;
  }

  /*
  np_array out_np = py::array_t<realtype>(
    {size, vars}, // shape
    {vars*sizeof(realtype), sizeof(realtype)}, // C-style contiguous strides
    out, // the data pointer
    free_when_done
  );
  */
  
  np_array a = handler(t_np, in1[0], in2[0]);
  auto buf = a.request();
  std::cout << "ndim: " << a.ndim() << std::endl;
  std::cout << "shape: " << buf.shape[0] << " " << buf.shape[1] << std::endl;
  // TODO: Insert shape checks here
  // c-style pointer from numpy use row-first indexing (across vars)
  realtype* ptr = (realtype*) buf.ptr;
  for (std::int64_t n = 0; n < size; ++n) {
    for (std::int64_t tn = 0; tn < vars; ++tn) {
      std::cout << " item: " << n << " " << tn << ": ";
      std::cout << ptr[n*vars + tn] << std::endl;
    }
  }

  // We have a single output, which is a multi-dimensional array; recurse over the dimensions
  std::int64_t i = 0;
  for (std::int64_t n = 0; n < vars; ++n) {
    for (std::int64_t tn = 0; tn < size; ++tn) {
      reinterpret_cast<realtype *>(out)[i] = n*vars + tn;
      // reinterpret_cast<realtype *>(out)[i] = ptr[n*vars + tn];
      std::cout << " out " << reinterpret_cast<realtype *>(out)[i] << std::endl;
      ++i;
    }
  }
  std::cout << " done" << std::endl;
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_idaklu_f64"] = EncapsulateFunction(cpu_idaklu);
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
  m.def("add_python_callback",
    [](Handler h) { handler = h; },
    py::return_value_policy::take_ownership);

  py::class_<Function>(m, "Function");

  py::class_<Solution>(m, "solution")
  .def_readwrite("t", &Solution::t)
  .def_readwrite("y", &Solution::y)
  .def_readwrite("yS", &Solution::yS)
  .def_readwrite("flag", &Solution::flag);
}

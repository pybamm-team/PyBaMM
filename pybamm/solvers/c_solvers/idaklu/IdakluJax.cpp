#include "IdakluJax.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <vector>
#include <iostream>
#include <functional>

// Initialise static variable
std::int64_t IdakluJax::universal_count = 0;

// Repository of instantiated IdakluJax objects
std::map<std::int64_t, IdakluJax*> idaklu_jax_instances;

// Create a new IdakluJax object, assign identifier, add to the objects list and return as pointer
IdakluJax *create_idaklu_jax() {
  IdakluJax *p = new IdakluJax();
  idaklu_jax_instances[p->get_index()] = p;
  return p;
}

IdakluJax::IdakluJax() {
  index = universal_count++;
}

IdakluJax::~IdakluJax() {
  idaklu_jax_instances.erase(index);
}

void IdakluJax::register_callback_eval(CallbackEval h) {
  callback_eval = h;
}

void IdakluJax::register_callback_jvp(CallbackJvp h) {
  callback_jvp = h;
}

void IdakluJax::register_callback_vjp(CallbackVjp h) {
  callback_vjp = h;
}

void IdakluJax::register_callbacks(CallbackEval h_eval, CallbackJvp h_jvp, CallbackVjp h_vjp) {
  register_callback_eval(h_eval);
  register_callback_jvp(h_jvp);
  register_callback_vjp(h_vjp);
}

void IdakluJax::cpu_idaklu_eval(void *out_tuple, const void **in) {
  // Parse the inputs --- note that these come from jax lowering and are NOT np_array's
  int k = 1;  // Start indexing at 1 to skip idaklu_jax index
  const std::int64_t n_t = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_vars = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_inputs = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const realtype *t = reinterpret_cast<const realtype *>(in[k++]);
  realtype *inputs = new realtype(n_inputs);
  for (int i = 0; i < n_inputs; i++)
    inputs[i] = reinterpret_cast<const realtype *>(in[k++])[0];
  void *out = reinterpret_cast<realtype *>(out_tuple);

  // Log
  DEBUG("cpu_idaklu");
  DEBUG_n(index);
  DEBUG_n(n_t);
  DEBUG_n(n_vars);
  DEBUG_n(n_inputs);
  DEBUG_v(t, n_t);
  DEBUG_v(inputs, n_inputs);

  // Acquire GIL (since this function is called as a capsule)
  py::gil_scoped_acquire acquire;
  PyGILState_STATE state = PyGILState_Ensure();

  // Convert time vector to an np_array
  py::capsule t_capsule(t, "t_capsule");
  np_array t_np = np_array({n_t}, {sizeof(realtype)}, t, t_capsule);

  // Convert inputs to an np_array
  py::capsule in_capsule(inputs, "in_capsule");
  np_array in_np = np_array({n_inputs}, {sizeof(realtype)}, inputs, in_capsule);

  // Call solve function in python to obtain an np_array
  np_array out_np = callback_eval(t_np, in_np);
  auto out_buf = out_np.request();
  const realtype *out_ptr = reinterpret_cast<realtype *>(out_buf.ptr);

  // Arrange into 'out' array
  memcpy(out, out_ptr, n_t * n_vars * sizeof(realtype));

  // Release GIL
  PyGILState_Release(state);
}

void IdakluJax::cpu_idaklu_jvp(void *out_tuple, const void **in) {
  // Parse the inputs --- note that these come from jax lowering and are NOT np_array's
  int k = 1;  // Start indexing at 1 to skip idaklu_jax index
  const std::int64_t n_t = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_vars = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_inputs = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const realtype *primal_t = reinterpret_cast<const realtype *>(in[k++]);
  realtype *primal_inputs = new realtype(n_inputs);
  for (int i = 0; i < n_inputs; i++)
    primal_inputs[i] = reinterpret_cast<const realtype *>(in[k++])[0];
  const realtype *tangent_t = reinterpret_cast<const realtype *>(in[k++]);
  realtype *tangent_inputs = new realtype(n_inputs);
  for (int i = 0; i < n_inputs; i++)
    tangent_inputs[i] = reinterpret_cast<const realtype *>(in[k++])[0];
  void *out = reinterpret_cast<realtype *>(out_tuple);

  // Log
  DEBUG("cpu_idaklu_jvp");
  DEBUG_n(n_t);
  DEBUG_n(n_vars);
  DEBUG_n(n_inputs);
  DEBUG_v(primal_t, n_t);
  DEBUG_v(primal_inputs, n_inputs);
  DEBUG_v(tangent_t, n_t);
  DEBUG_v(tangent_inputs, n_inputs);

  // Acquire GIL (since this function is called as a capsule)
  py::gil_scoped_acquire acquire;
  PyGILState_STATE state = PyGILState_Ensure();

  // Form primals time vector as np_array
  py::capsule primal_t_capsule(primal_t, "primal_t_capsule");
  np_array primal_t_np = np_array(
    {n_t},
    {sizeof(realtype)},
    primal_t,
    primal_t_capsule
  );

  // Pack primals as np_array
  py::capsule primal_inputs_capsule(primal_inputs, "primal_inputs_capsule");
  np_array primal_inputs_np = np_array(
    {n_inputs},
    {sizeof(realtype)},
    primal_inputs,
    primal_inputs_capsule
  );

  // Form tangents time vector as np_array
  py::capsule tangent_t_capsule(tangent_t, "tangent_t_capsule");
  np_array tangent_t_np = np_array(
    {n_t},
    {sizeof(realtype)},
    tangent_t,
    tangent_t_capsule
  );

  // Pack tangents as np_array
  py::capsule tangent_inputs_capsule(tangent_inputs, "tangent_inputs_capsule");
  np_array tangent_inputs_np = np_array(
    {n_inputs},
    {sizeof(realtype)},
    tangent_inputs,
    tangent_inputs_capsule
  );

  // Call JVP function in python to obtain an np_array
  np_array y_dot = callback_jvp(
    primal_t_np, primal_inputs_np,
    tangent_t_np, tangent_inputs_np
  );
  auto buf = y_dot.request();
  const realtype *ptr = reinterpret_cast<realtype *>(buf.ptr);

  // Arrange into 'out' array
  memcpy(out, ptr, n_t * n_vars * sizeof(realtype));

  // Release GIL
  PyGILState_Release(state);
}

void IdakluJax::cpu_idaklu_vjp(void *out_tuple, const void **in) {
  int k = 1;  // Start indexing at 1 to skip idaklu_jax index
  const std::int64_t n_t = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_inputs = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_y_bar0 = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_y_bar1 = *reinterpret_cast<const std::int64_t *>(in[k++]);
  const std::int64_t n_y_bar = (n_y_bar1 > 0) ? (n_y_bar0*n_y_bar1) : n_y_bar0;
  const realtype *y_bar = reinterpret_cast<const realtype *>(in[k++]);
  const std::int64_t *invar = reinterpret_cast<const std::int64_t *>(in[k++]);
  const realtype *t = reinterpret_cast<const realtype *>(in[k++]);
  realtype *inputs = new realtype(n_inputs);
  for (int i = 0; i < n_inputs; i++)
    inputs[i] = reinterpret_cast<const realtype *>(in[k++])[0];
  realtype *out = reinterpret_cast<realtype *>(out_tuple);

  // Log
  DEBUG("cpu_idaklu_vjp");
  DEBUG_n(n_t);
  DEBUG_n(n_inputs);
  DEBUG_n(n_y_bar0);
  DEBUG_n(n_y_bar1);
  DEBUG_v(y_bar, n_y_bar0*n_y_bar1);
  DEBUG_v(invar, 1);
  DEBUG_v(t, n_t);
  DEBUG_v(inputs, n_inputs);

  // Acquire GIL (since this function is called as a capsule)
  py::gil_scoped_acquire acquire;
  PyGILState_STATE state = PyGILState_Ensure();

  // Convert time vector to an np_array
  py::capsule t_capsule(t, "t_capsule");
  np_array t_np = np_array({n_t}, {sizeof(realtype)}, t, t_capsule);

  // Convert y_bar to an np_array
  py::capsule y_bar_capsule(y_bar, "y_bar_capsule");
  np_array y_bar_np = np_array(
      {n_y_bar},
      {sizeof(realtype)},
      y_bar,
      y_bar_capsule
    );

  // Convert inputs to an np_array
  py::capsule in_capsule(inputs, "in_capsule");
  np_array in_np = np_array({n_inputs}, {sizeof(realtype)}, inputs, in_capsule);

  // Call VJP function in python to obtain an np_array
  np_array y_dot = callback_vjp(y_bar_np, n_y_bar0, n_y_bar1, invar[0], t_np, in_np);
  auto buf = y_dot.request();
  const realtype *ptr = reinterpret_cast<realtype *>(buf.ptr);

  // Arrange output
  //memcpy(out, ptr, sizeof(realtype));
  out[0] = ptr[0];  // output is scalar

  // Release GIL
  PyGILState_Release(state);
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(reinterpret_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

void wrap_cpu_idaklu_eval_f64(void *out_tuple, const void **in) {
  const std::int64_t index = *reinterpret_cast<const std::int64_t *>(in[0]);
  idaklu_jax_instances[index]->cpu_idaklu_eval(out_tuple, in);
}

void wrap_cpu_idaklu_jvp_f64(void *out_tuple, const void **in) {
  const std::int64_t index = *reinterpret_cast<const std::int64_t *>(in[0]);
  idaklu_jax_instances[index]->cpu_idaklu_jvp(out_tuple, in);
}

void wrap_cpu_idaklu_vjp_f64(void *out_tuple, const void **in) {
  const std::int64_t index = *reinterpret_cast<const std::int64_t *>(in[0]);
  idaklu_jax_instances[index]->cpu_idaklu_vjp(out_tuple, in);
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_idaklu_f64"] = EncapsulateFunction(wrap_cpu_idaklu_eval_f64);
  dict["cpu_idaklu_jvp_f64"] = EncapsulateFunction(wrap_cpu_idaklu_jvp_f64);
  dict["cpu_idaklu_vjp_f64"] = EncapsulateFunction(wrap_cpu_idaklu_vjp_f64);
  return dict;
}

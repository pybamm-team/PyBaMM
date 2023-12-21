#ifndef PYBAMM_IDAKLU_JAX_SOLVER_HPP
#define PYBAMM_IDAKLU_JAX_SOLVER_HPP

#include "common.hpp"

using Callback = std::function<np_array(np_array, np_array)>;
using CallbackJvp = std::function<np_array(np_array, np_array, np_array, np_array)>;
using CallbackVjp = std::function<np_array(np_array, int, int, realtype, np_array, np_array)>;

class IdakluJax {
public:
  std::int64_t index;
  Callback callback;
  CallbackJvp callback_jvp;
  CallbackVjp callback_vjp;

  void register_callback_jaxsolve(Callback h);
  void register_callback_jvp(CallbackJvp h);
  void register_callback_vjp(CallbackVjp h);
  void register_callbacks(Callback h, CallbackJvp h_jvp, CallbackVjp h_vjp);

  void cpu_idaklu(void *out_tuple, const void **in);
  void cpu_idaklu_jvp(void *out_tuple, const void **in);
  void cpu_idaklu_vjp(void *out_tuple, const void **in);

  std::int64_t get_index() { return index; };
};

IdakluJax *create_idaklu_jax();

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn);
pybind11::dict Registrations();

#endif // PYBAMM_IDAKLU_JAX_SOLVER_HPP

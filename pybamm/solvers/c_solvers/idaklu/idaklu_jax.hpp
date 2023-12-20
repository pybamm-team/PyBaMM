#ifndef PYBAMM_IDAKLU_JAX_SOLVER_HPP
#define PYBAMM_IDAKLU_JAX_SOLVER_HPP

#include "common.hpp"

using Handler = std::function<np_array(np_array, np_array)>;
using HandlerJvp = std::function<np_array(np_array, np_array, np_array, np_array)>;
using HandlerVjp = std::function<np_array(np_array, int, int, realtype, np_array, np_array)>;

void register_callback_jaxsolve(Handler h);
void register_callback_jvp(HandlerJvp h);
void register_callback_vjp(HandlerVjp h);
void register_callbacks(Handler h, HandlerJvp h_jvp, HandlerVjp h_vjp);

void cpu_idaklu(void *out_tuple, const void **in);
void cpu_idaklu_jvp(void *out_tuple, const void **in);
void cpu_idaklu_vjp(void *out_tuple, const void **in);

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn);

pybind11::dict Registrations();

#endif // PYBAMM_IDAKLU_JAX_SOLVER_HPP

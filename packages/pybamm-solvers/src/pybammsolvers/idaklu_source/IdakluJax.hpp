#ifndef PYBAMM_IDAKLU_JAX_SOLVER_HPP
#define PYBAMM_IDAKLU_JAX_SOLVER_HPP

#include "common.hpp"

/**
 * @brief Callback function type for JAX evaluation
 */
using CallbackEval = std::function<np_array(np_array, np_array)>;

/**
 * @brief Callback function type for JVP evaluation
 */
using CallbackJvp = std::function<np_array(np_array, np_array, np_array, np_array)>;

/**
 * @brief Callback function type for VJP evaluation
 */
using CallbackVjp = std::function<np_array(np_array, int, int, realtype, np_array, np_array)>;

/**
 * @brief IDAKLU-JAX interface class.
 *
 * This class provides an interface to the IDAKLU-JAX solver. It is called by the
 * IDAKLUJax class in python and provides the lowering rules for the JAX evaluation,
 * JVP and VJP primitive functions. Each of these make use of the IDAKLU solver via
 * the IDAKLUSolver python class, so this IDAKLUJax class provides a wrapper which
 * redirects calls back to the IDAKLUSolver via python callbacks.
 */
class IdakluJax {
private:
  static std::int64_t universal_count;  // Universal count for IdakluJax objects
  std::int64_t index;  // Instance index
public:
  /**
   * @brief Constructor
   */
  IdakluJax();

  /**
   * @brief Destructor
   */
  ~IdakluJax();

  /**
   * @brief Callback for JAX evaluation
   */
  CallbackEval callback_eval;

  /**
   * @brief Callback for JVP evaluation
   */
  CallbackJvp callback_jvp;

  /**
   * @brief Callback for VJP evaluation
   */
  CallbackVjp callback_vjp;

  /**
   * @brief Register callbacks for JAX evaluation, JVP and VJP
   */
  void register_callback_eval(CallbackEval h);

  /**
   * @brief Register callback for JAX evaluation
   */
  void register_callback_jvp(CallbackJvp h);

  /**
   * @brief Register callback for JVP evaluation
   */
  void register_callback_vjp(CallbackVjp h);

  /**
   * @brief Register callback for VJP evaluation
   */
  void register_callbacks(CallbackEval h, CallbackJvp h_jvp, CallbackVjp h_vjp);

  /**
   * @brief JAX evaluation primitive function
   */
  void cpu_idaklu_eval(void *out_tuple, const void **in);

  /**
   * @brief JVP primitive function
   */
  void cpu_idaklu_jvp(void *out_tuple, const void **in);

  /**
   * @brief VJP primitive function
   */
  void cpu_idaklu_vjp(void *out_tuple, const void **in);

  /**
   * @brief Get the instance index
   */
  std::int64_t get_index() { return index; };
};

/**
 * @brief Non-member function to create a new IdakluJax object
 */
IdakluJax *create_idaklu_jax();

/**
 * @brief (Non-member) encapsulation helper function
 */
template <typename T>
pybind11::capsule EncapsulateFunction(T* fn);

/**
 * @brief (Non-member) function dictionary
 */
pybind11::dict Registrations();

#endif // PYBAMM_IDAKLU_JAX_SOLVER_HPP

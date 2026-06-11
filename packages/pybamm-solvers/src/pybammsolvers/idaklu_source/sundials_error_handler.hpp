#ifndef PYBAMM_SUNDIALS_ERROR_HANDLER_HPP
#define PYBAMM_SUNDIALS_ERROR_HANDLER_HPP

#include <string>
#include <stdexcept>
#include <idas/idas.h>

/**
 * @brief Maps SUNDIALS error codes to human-readable messages
 */
inline const char* sundials_error_message(int flag) {
  switch (flag) {
    // IDA-specific error codes (negative values indicate errors)
    case IDA_TOO_MUCH_WORK:
      return "IDA_TOO_MUCH_WORK: The solver took max internal steps but could not reach tout";
    case IDA_TOO_MUCH_ACC:
      return "IDA_TOO_MUCH_ACC: The solver could not satisfy the accuracy demanded";
    case IDA_ERR_FAIL:
      return "IDA_ERR_FAIL: Error test failures occurred too many times during one step or minimum step size was reached";
    case IDA_CONV_FAIL:
      return "IDA_CONV_FAIL: Convergence test failures occurred too many times during one step or with |h| = hmin";
    case IDA_LINIT_FAIL:
      return "IDA_LINIT_FAIL: The linear solver's initialization function failed";
    case IDA_LSETUP_FAIL:
      return "IDA_LSETUP_FAIL: The linear solver's setup function failed in an unrecoverable manner";
    case IDA_LSOLVE_FAIL:
      return "IDA_LSOLVE_FAIL: The linear solver's solve function failed in an unrecoverable manner";
    case IDA_RES_FAIL:
      return "IDA_RES_FAIL: The user-provided residual function failed in an unrecoverable manner";
    case IDA_REP_RES_ERR:
      return "IDA_REP_RES_ERR: The user's residual function repeatedly returned a recoverable error flag";
    case IDA_RTFUNC_FAIL:
      return "IDA_RTFUNC_FAIL: The rootfinding function failed in an unrecoverable manner";
    case IDA_CONSTR_FAIL:
      return "IDA_CONSTR_FAIL: The inequality constraints were violated and the solver was unable to recover";
    case IDA_FIRST_RES_FAIL:
      return "IDA_FIRST_RES_FAIL: The user's residual function returned a recoverable error flag on the first call";
    case IDA_LINESEARCH_FAIL:
      return "IDA_LINESEARCH_FAIL: The line search failed";
    case IDA_NO_RECOVERY:
      return "IDA_NO_RECOVERY: The residual function, linear solver setup, or linear solver solve had a recoverable failure but IDACalcIC could not recover";
    case IDA_MEM_NULL:
      return "IDA_MEM_NULL: The IDA memory block was not initialized";
    case IDA_MEM_FAIL:
      return "IDA_MEM_FAIL: A memory allocation failed";
    case IDA_ILL_INPUT:
      return "IDA_ILL_INPUT: One of the function inputs is illegal";
    case IDA_NO_MALLOC:
      return "IDA_NO_MALLOC: The IDA memory was not allocated by a call to IDAInit";
    case IDA_BAD_EWT:
      return "IDA_BAD_EWT: Zero value of some error weight component";
    case IDA_BAD_K:
      return "IDA_BAD_K: The k-th derivative is not available";
    case IDA_BAD_T:
      return "IDA_BAD_T: The time t is outside the last step taken";
    case IDA_BAD_DKY:
      return "IDA_BAD_DKY: The vector argument where derivative should be stored is NULL";
    default:
      return "Unknown SUNDIALS error code";
  }
}

/**
 * @brief Throws a runtime error with context and SUNDIALS error code information
 */
inline void throw_sundials_error(int flag, const char* operation_context) {
  // Pre-allocate sufficient buffer for the error message to avoid dynamic allocations
  constexpr size_t buffer_size = 512;
  char buffer[buffer_size];
  
  const char* error_msg = sundials_error_message(flag);
  
  // Format: "Operation context: error message (flag: <code>)"
  int written = snprintf(buffer, buffer_size, "%s: %s (flag: %d)", 
                         operation_context, error_msg, flag);
  
  // Ensure null termination in case of truncation
  if (written >= static_cast<int>(buffer_size)) {
    buffer[buffer_size - 1] = '\0';
  }
  
  throw std::runtime_error(buffer);
}

/**
 * @brief Checks for overflow when multiplying size_t values
 */
inline bool check_size_t_multiply_overflow(size_t a, size_t b, size_t* result) {
  if (a == 0 || b == 0) {
    *result = 0;
    return false;
  }
  
  // Check if a * b would overflow size_t
  if (a > SIZE_MAX / b) {
    return true;  // Overflow would occur
  }
  
  *result = a * b;
  return false;
}

/**
 * @brief Checks for overflow when multiplying three size_t values
 */
inline bool check_size_t_multiply_overflow_3(size_t a, size_t b, size_t c, size_t* result) {
  size_t temp;
  if (check_size_t_multiply_overflow(a, b, &temp)) {
    return true;
  }
  return check_size_t_multiply_overflow(temp, c, result);
}

#endif // PYBAMM_SUNDIALS_ERROR_HANDLER_HPP


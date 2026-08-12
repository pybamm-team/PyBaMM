#ifndef PYBAMM_SOLVER_LOG_HPP
#define PYBAMM_SOLVER_LOG_HPP

#include "common.hpp"
#include <cstdarg>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Debug logger that buffers messages and defers all Python calls to flush().
 *
 * Log methods only format and buffer, so they are safe to call from the GIL-free
 * OpenMP worker threads in IDAKLUSolverGroup::solve. set_logger() and flush() do
 * call into Python and MUST be called with the GIL held, from the serial sections
 * around that OpenMP region. The buffer is cleared only by flush(), so a solver
 * reused for several input sets retains every message until it is drained.
 * In streaming mode the log methods emit as they are called instead, which is
 * only valid when the solve runs on the thread holding the GIL.
 * pybammsolvers has zero knowledge of pybamm.
 */
class SolverLog {
public:
  SolverLog() = default;

  /**
   * @brief Set the Python callable (e.g. pybamm.logger.debug) to log through
   *
   * A null or None callable disables logging. MUST be called with the GIL held.
   */
  void set_logger(py::object logger) {
    logger_ = std::move(logger);
    // A default-constructed py::object is null rather than None; both mean off
    enabled_ = static_cast<bool>(logger_) && !logger_.is_none();
  }

  /**
   * @brief Emit messages as they are logged rather than buffering them
   *
   * Only enable this when the solve runs on the GIL-holding thread, so a long
   * solve reports progress live. Toggling does not drain the current buffer.
   */
  void set_streaming(bool streaming) { streaming_ = streaming; }

  bool enabled() const { return enabled_; }

  bool streaming() const { return streaming_; }

  void log_start(double t0, double tf) {
    if (!enabled_) return;
    append(format("Integrating from t = %.17e to t = %.17e", t0, tf));
  }

  void log_step(int step, double t_val) {
    if (!enabled_) return;
    append(format("Step %5d: t = %.17e", step, t_val));
  }

  void log_consistent_init(double t_val) {
    if (!enabled_) return;
    append(format("Consistent initialization at t = %.17e", t_val));
  }

  void log_breakpoint(double t_val) {
    if (!enabled_) return;
    append(format("Breakpoint at t = %.17e, reinitializing", t_val));
  }

  void log_integration_complete(int n_steps, double t_final) {
    if (!enabled_) return;
    append(format("Integration complete: %d steps, t_final = %.17e", n_steps, t_final));
  }

  void log_newton_start(double t, int n_alg) {
    if (!enabled_) return;
    append(format("Newton solve at t = %.17e, n_alg = %d", t, n_alg));
  }

  void log_newton_iteration(int iter, double res_norm, double step_norm) {
    if (!enabled_) return;
    append(format(" Newton iter %3d: ||g|| = %.4e, ||dy|| = %.4e", iter, res_norm, step_norm));
  }

  void log_newton_converged(int iters, const char* reason) {
    if (!enabled_) return;
    append(format(" Newton converged in %d iterations (%s)", iters, reason));
  }

  void log_newton_failed(int iters, double res_norm, const char* reason) {
    if (!enabled_) return;
    append(format(" Newton FAILED after %d iterations, ||g|| = %.4e (%s)", iters, res_norm, reason));
  }

  /**
   * @brief Emit and discard the buffered messages
   *
   * MUST be called with the GIL held. Never throws: a logger that raises is
   * reported through sys.unraisablehook rather than failing the solve.
   */
  void flush() noexcept {
    for (const auto& msg : buffer_) {
      emit(msg);
    }
    buffer_.clear();
  }

private:
  void append(std::string msg) {
    if (streaming_) {
      emit(msg);
    } else {
      buffer_.push_back(std::move(msg));
    }
  }

  /**
   * @brief Pass one message to the Python logger (GIL held)
   */
  void emit(const std::string& msg) noexcept {
    try {
      logger_(py::str(msg));
    } catch (py::error_already_set& e) {
      e.discard_as_unraisable("pybammsolvers SolverLog::emit");
    } catch (...) {
      // A logging failure must never propagate into the solve
    }
  }

  /**
   * @brief printf-style formatting helper
   */
  static std::string format(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    const int len = std::vsnprintf(nullptr, 0, fmt, args);
    va_end(args);
    if (len < 0) {
      va_end(args_copy);
      return std::string();
    }
    // Sized for the text plus the terminating null, which is then dropped
    std::string out(static_cast<size_t>(len) + 1, '\0');
    std::vsnprintf(&out[0], out.size(), fmt, args_copy);
    va_end(args_copy);
    out.pop_back();
    return out;
  }

  py::object logger_;
  bool enabled_ = false;
  bool streaming_ = false;
  std::vector<std::string> buffer_;
};

#endif // PYBAMM_SOLVER_LOG_HPP

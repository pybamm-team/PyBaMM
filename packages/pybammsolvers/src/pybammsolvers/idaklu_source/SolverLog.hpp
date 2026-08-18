#ifndef PYBAMM_SOLVER_LOG_HPP
#define PYBAMM_SOLVER_LOG_HPP

#include "common.hpp"
#include <cstdarg>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/**
 * @brief Debug logger that only calls Python from the thread holding the GIL.
 *
 * set_logger() records its calling thread, which is the one holding the GIL.
 * Messages logged from that thread are emitted immediately, so a long solve
 * reports progress live; messages logged from anywhere else -- the GIL-free
 * OpenMP worker threads in IDAKLUSolverGroup::solve -- are formatted and
 * buffered, and emitted when that thread calls flush(). The buffer is cleared
 * only by flush(), so a solver reused for several input sets retains every
 * message until it is drained. set_logger() and flush() call into Python and
 * MUST be called with the GIL held.
 * pybammsolvers has zero knowledge of pybamm.
 */
class SolverLog {
public:
  SolverLog() = default;

  /**
   * @brief Set the Python callable (e.g. pybamm.logger.debug) to log through
   *
   * A null or None callable disables logging. MUST be called with the GIL held;
   * the calling thread is recorded as the only one allowed to emit directly.
   */
  void set_logger(py::object logger) {
    logger_ = std::move(logger);
    gil_thread_ = std::this_thread::get_id();
  }

  // A default-constructed py::object is null rather than None; both mean off
  bool enabled() const { return static_cast<bool>(logger_) && !logger_.is_none(); }

  /**
   * @brief Whether this thread holds the GIL, and so may call Python directly
   */
  bool on_gil_thread() const { return std::this_thread::get_id() == gil_thread_; }

  void log_start(double t0, double tf) {
    if (!enabled()) return;
    append(format("Integrating from t = %.17e to t = %.17e", t0, tf));
  }

  void log_step(int step, double t_val) {
    if (!enabled()) return;
    append(format("Step %5d: t = %.17e", step, t_val));
  }

  void log_consistent_init(double t_val) {
    if (!enabled()) return;
    append(format("Consistent initialization at t = %.17e", t_val));
  }

  void log_breakpoint(double t_val) {
    if (!enabled()) return;
    append(format("Breakpoint at t = %.17e, reinitializing", t_val));
  }

  void log_integration_complete(int n_steps, double t_final) {
    if (!enabled()) return;
    append(format("Integration complete: %d steps, t_final = %.17e", n_steps, t_final));
  }

  void log_newton_start(double t, int n_alg) {
    if (!enabled()) return;
    append(format("Newton solve at t = %.17e, n_alg = %d", t, n_alg));
  }

  void log_newton_iteration(int iter, double res_norm, double step_norm) {
    if (!enabled()) return;
    append(format(" Newton iter %3d: ||g|| = %.4e, ||dy|| = %.4e", iter, res_norm, step_norm));
  }

  void log_newton_converged(int iters, const char* reason) {
    if (!enabled()) return;
    append(format(" Newton converged in %d iterations (%s)", iters, reason));
  }

  void log_newton_failed(int iters, double res_norm, const char* reason) {
    if (!enabled()) return;
    append(format(" Newton FAILED after %d iterations, ||g|| = %.4e (%s)", iters, res_norm, reason));
  }

  /**
   * @brief Emit and discard the buffered messages
   *
   * MUST be called with the GIL held. A logger that raises is reported through
   * sys.unraisablehook rather than failing the solve; KeyboardInterrupt is the
   * exception, and propagates.
   */
  void flush() {
    for (const auto& msg : buffer_) {
      emit(msg);
    }
    // Release the capacity too, which a long buffered sweep grows into megabytes
    std::vector<std::string>().swap(buffer_);
  }

  /**
   * @brief Run a Python-calling action, swallowing any exception it raises
   *
   * MUST be called with the GIL held. Shared by every diagnostic sink so that
   * failing output can never turn into a failed solve. KeyboardInterrupt is
   * control flow, not a diagnostics failure, so it propagates.
   */
  template <class Action>
  static void guarded(Action&& action, const char* context) {
    try {
      action();
    } catch (py::error_already_set& e) {
      if (e.matches(PyExc_KeyboardInterrupt)) throw;
      e.discard_as_unraisable(context);
    } catch (...) {
      // A diagnostics failure must never propagate into the solve
    }
  }

private:
  void append(std::string msg) {
    if (on_gil_thread()) {
      emit(msg);
    } else {
      buffer_.push_back(std::move(msg));
    }
  }

  /**
   * @brief Pass one message to the Python logger (GIL held)
   */
  void emit(const std::string& msg) {
    guarded([&] { logger_(py::str(msg)); }, "pybammsolvers SolverLog::emit");
  }

  /**
   * @brief printf-style formatting helper
   */
  static std::string format(const char* fmt, ...) {
    // Every message above fits, so the resize-and-retry path is a formality
    char buf[256];
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    const int len = std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (len < 0 || static_cast<size_t>(len) < sizeof(buf)) {
      va_end(args_copy);
      return len < 0 ? std::string() : std::string(buf, len);
    }
    // Sized for the text plus the terminating null, which is then dropped
    std::string out(static_cast<size_t>(len) + 1, '\0');
    std::vsnprintf(&out[0], out.size(), fmt, args_copy);
    va_end(args_copy);
    out.pop_back();
    return out;
  }

  py::object logger_;
  std::thread::id gil_thread_;
  std::vector<std::string> buffer_;
};

#endif // PYBAMM_SOLVER_LOG_HPP

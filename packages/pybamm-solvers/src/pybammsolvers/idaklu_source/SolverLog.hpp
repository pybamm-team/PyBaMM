#ifndef PYBAMM_SOLVER_LOG_HPP
#define PYBAMM_SOLVER_LOG_HPP

#include "common.hpp"

/**
 * @brief Simple debug logger that wraps an optional Python callable.
 *
 * If a callable (e.g. pybamm.logger.debug) is provided, log methods
 * format and forward messages through it. If None/null, all log
 * methods are no-ops. pybammsolvers has zero knowledge of pybamm.
 */
class SolverLog {
public:
  SolverLog() : enabled_(false) {}

  explicit SolverLog(py::object logger)
    : logger_(std::move(logger)),
      enabled_(!logger_.is_none())
  {}

  bool enabled() const { return enabled_; }

  void log_start(double t0, double tf) {
    if (!enabled_) return;
    logger_(
      py::str("Integrating from t = %.17e to t = %.17e")
        .attr("__mod__")(py::make_tuple(t0, tf))
    );
  }

  void log_step(int step, double t_val) {
    if (!enabled_) return;
    logger_(
      py::str("Step %5d: t = %.17e")
        .attr("__mod__")(py::make_tuple(step, t_val))
    );
  }

  void log_consistent_init(double t_val) {
    if (!enabled_) return;
    logger_(
      py::str("Consistent initialization at t = %.17e")
        .attr("__mod__")(py::make_tuple(t_val))
    );
  }

  void log_breakpoint(double t_val) {
    if (!enabled_) return;
    logger_(
      py::str("Breakpoint at t = %.17e, reinitializing")
        .attr("__mod__")(py::make_tuple(t_val))
    );
  }

  void log_integration_complete(int n_steps, double t_final) {
    if (!enabled_) return;
    logger_(
      py::str("Integration complete: %d steps, t_final = %.17e")
        .attr("__mod__")(py::make_tuple(n_steps, t_final))
    );
  }

private:
  py::object logger_;
  bool enabled_;
};

#endif // PYBAMM_SOLVER_LOG_HPP

#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_HPP

#include "common.hpp"
#include "SolutionData.hpp"
#include "SolverLog.hpp"


/**
 * Abstract base class for solutions that can use different solvers and vector
 * implementations.
 * @brief An abstract base class for the Idaklu solver
 */
class IDAKLUSolver
{
public:

  /**
   * @brief Default constructor
   */
  IDAKLUSolver() = default;

  /**
   * @brief Default destructor
   */
  virtual ~IDAKLUSolver() = default;

  /**
   * @brief Abstract solver method that executes the solver
   */
  virtual SolutionData solve(
    const std::vector<sunrealtype> &t_eval,
    const std::vector<sunrealtype> &t_interp,
    const sunrealtype *y0,
    const sunrealtype *yp0,
    const sunrealtype *inputs,
    bool save_adaptive_steps,
    bool save_interp_steps
  ) = 0;

  /**
   * Abstract method to initialize the solver, once vectors and solver classes
   * are set
   * @brief Abstract initialization method
   */
  virtual void Initialize() = 0;

  /**
   * Install the Python callable that debug messages are logged through. MUST be
   * called with the GIL held, i.e. in the serial section of
   * IDAKLUSolverGroup::solve before the OpenMP region, because copying a
   * py::object touches Python reference counts.
   * @brief Set the logger used for debug output
   */
  void set_logger(py::object logger) { log_.set_logger(std::move(logger)); }

  /**
   * Emit any buffered log and statistics output. MUST be called with the GIL
   * held, i.e. in a serial section of IDAKLUSolverGroup::solve.
   * @brief Flush buffered diagnostic output
   */
  virtual void flush_log() = 0;

  // Common to every implementation, and knows for itself which thread may
  // write to Python, so subclasses need no logging state of their own
  SolverLog log_;
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP

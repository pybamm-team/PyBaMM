#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_HPP

#include "common.hpp"
#include "SolutionData.hpp"


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
    const std::vector<realtype> &t_eval,
    const std::vector<realtype> &t_interp,
    const realtype *y0,
    const realtype *yp0,
    const realtype *inputs,
    bool save_adaptive_steps,
    bool save_interp_steps
  ) = 0;

  /**
   * Abstract method to initialize the solver, once vectors and solver classes
   * are set
   * @brief Abstract initialization method
   */
  virtual void Initialize() = 0;
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP

#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_HPP

#include "common.hpp"
#include "Solution.hpp"

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
  ~IDAKLUSolver() = default;

  /**
   * @brief Abstract solver method that returns a Solution class
   */
  virtual Solution solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs) = 0;

  /**
   * Abstract method to initialize the solver, once vectors and solver classes
   * are set
   * @brief Abstract initialization method
   */
  virtual void Initialize() = 0;
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP

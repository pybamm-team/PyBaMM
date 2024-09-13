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
   * @brief Abstract solver method that executes the solver
   */
  virtual void solve_individual(
    const realtype *t_eval,
    const int number_of_evals,
    const realtype *t_interp,
    const int number_of_interps,
    const realtype *y0,
    const realtype *yp0,
    const realtype *inputs,
    const int length_of_return_vector,
    realtype *y_return,
    realtype *yS_return,
    realtype *t_return,
    int &t_i,
    int &retval
    bool save_adaptive_steps,
  ) = 0;

  /**
   * Abstract method to initialize the solver, once vectors and solver classes
   * are set
   * @brief Abstract initialization method
   */
  virtual void Initialize() = 0;
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_HPP

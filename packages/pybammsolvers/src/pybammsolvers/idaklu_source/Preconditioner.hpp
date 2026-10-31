#ifndef PYBAMM_IDAKLU_PRECONDITIONER_HPP
#define PYBAMM_IDAKLU_PRECONDITIONER_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>

#include <sundials/sundials_context.h>
#include <sundials/sundials_linearsolver.h>

namespace py = pybind11;

/**
 * @brief A preconditioner for the iteration matrix dF/dy - cj M, given as the
 * data of IDAKLU's CSC Jacobian. Each backend implements this interface.
 */
class Preconditioner {
public:
  virtual ~Preconditioner() = default;

  /**
   * @brief Factor the preconditioner from the iteration matrix's CSC data
   * @return 0 on success, 1 on a recoverable failure (e.g. a singular block)
   */
  virtual int setup(const double *matrix_data) = 0;

  /**
   * @brief Apply the preconditioner: z = P^-1 r (z and r may not alias)
   */
  virtual void solve(const double *r, double *z) = 0;
};

/**
 * @brief Creates one preconditioner per solver in a group
 */
using PreconditionerFactory = std::function<std::unique_ptr<Preconditioner>()>;

/**
 * @brief A SUNDIALS direct linear solver that factors IDA's sparse (CSC)
 * iteration matrix with a preconditioner and applies it, so it is exact when
 * the preconditioner is.
 *
 * @param preconditioner Not owned; must outlive the linear solver
 */
SUNLinearSolver SUNLinSol_Preconditioner(Preconditioner *preconditioner, SUNContext sunctx);

#endif // PYBAMM_IDAKLU_PRECONDITIONER_HPP

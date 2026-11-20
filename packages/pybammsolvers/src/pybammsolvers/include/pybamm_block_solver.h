/*
 * Interface for block solvers that plug into the block preconditioner of
 * pybammsolvers' IDAKLU solver.
 *
 * A plugin exposes a static pybamm_block_solver through a Python capsule named
 * PYBAMM_BLOCK_SOLVER_CAPSULE. For each block, IDAKLU calls create once with
 * the block's CSC pattern, setup whenever the iteration matrix changes, and
 * solve on every preconditioner application, from one thread per context.
 */
#ifndef PYBAMM_BLOCK_SOLVER_H
#define PYBAMM_BLOCK_SOLVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PYBAMM_BLOCK_SOLVER_ABI_VERSION 1
#define PYBAMM_BLOCK_SOLVER_CAPSULE "pybamm_block_solver"

typedef struct pybamm_block_solver {
  /* PYBAMM_BLOCK_SOLVER_ABI_VERSION of the header the plugin was built with */
  int abi_version;
  /* A context for a size x size block with this CSC pattern (sorted rows) and
     the solver's options as a JSON object; NULL on failure */
  void *(*create)(int64_t size, const int64_t *indptr, const int64_t *indices,
                  const char *options_json);
  /* Set up from the block's values, in the pattern's order; 0 on success,
     nonzero on a recoverable failure */
  int (*setup)(void *context, const double *values);
  /* Overwrite x with the approximate solution of A x = x */
  void (*solve)(void *context, double *x);
  void (*destroy)(void *context);
} pybamm_block_solver;

#ifdef __cplusplus
}
#endif

#endif /* PYBAMM_BLOCK_SOLVER_H */

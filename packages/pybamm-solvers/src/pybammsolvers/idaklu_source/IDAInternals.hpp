#ifndef PYBAMM_IDAKLU_IDA_INTERNALS_HPP
#define PYBAMM_IDAKLU_IDA_INTERNALS_HPP

#include <idas/idas_impl.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_nvector.h>

static_assert(SUNDIALS_VERSION_MAJOR == 7 && SUNDIALS_VERSION_MINOR == 6,
  "IDAInternals is pinned to SUNDIALS 7.6.x; bumping requires re-auditing IDAMem layout");

extern "C" {
  int IDAInitialSetup(IDAMem IDA_mem);
}

/**
 * @brief Thin adapter wrapping all IDAMem field access.
 *
 * Centralizes every use of the IDA internal struct so that:
 *  1. Only one file includes idas_impl.h (auditable).
 *  2. A SUNDIALS major-version bump triggers a compile error via static_assert.
 *  3. The rest of the codebase never touches IDAMem directly.
 */
struct IDAInternals {
  IDAMem mem;

  explicit IDAInternals(void* ida_mem) : mem(static_cast<IDAMem>(ida_mem)) {}

  /** Run IDAInitialSetup if not already done; otherwise recompute ewt. */
  int EnsureSetup() {
    if (!mem->ida_SetupDone) {
      return IDAInitialSetup(mem);
    }
    mem->ida_efun(mem->ida_phi[0], mem->ida_ewt, mem->ida_edata);
    return IDA_SUCCESS;
  }

  /** Copy IDA's IC convergence tolerance into the Newton tolerance. */
  void SetEpsNewt() {
    mem->ida_epsNewt = mem->ida_epiccon;
  }

  /** Set cj and reset cjratio (needed before lsetup/lsolve). */
  void SetCj(sunrealtype cj) {
    mem->ida_cj = cj;
    mem->ida_cjratio = SUN_RCONST(1.0);
  }

  bool HasLinearSetup() const { return mem->ida_lsetup != nullptr; }

  int LinearSetup(N_Vector y, N_Vector yp, N_Vector r) {
    if (!mem->ida_lsetup) return 0;
    return mem->ida_lsetup(mem, y, yp, r,
                           mem->ida_tempv1, mem->ida_tempv2, mem->ida_tempv3);
  }

  int LinearSolve(N_Vector b, N_Vector y, N_Vector yp, N_Vector r) {
    return mem->ida_lsolve(mem, b, mem->ida_ewt, y, yp, r);
  }
};

#endif // PYBAMM_IDAKLU_IDA_INTERNALS_HPP

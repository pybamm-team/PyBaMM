#ifndef PYBAMM_NONLINEAR_SOLVER_HPP
#define PYBAMM_NONLINEAR_SOLVER_HPP

#include "common.hpp"
#include "BlockPartition.hpp"
#include "SolverLog.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
enum class NonlinearResult {
  CONVERGED_WRMS_AND_STEPTOL,
  CONVERGED_WRMS_STEP_DIVERGED,
  CONVERGED_WRMS_AT_MAX_ITER,
  LSETUP_FAIL,
  LSOLVE_FAIL,
  MAX_ITER_NO_CONVERGE,
};

inline bool nonlinear_success(NonlinearResult r) {
  return r == NonlinearResult::CONVERGED_WRMS_AND_STEPTOL ||
         r == NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED ||
         r == NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER;
}

inline const char* nonlinear_result_reason(NonlinearResult r) {
  switch (r) {
    case NonlinearResult::CONVERGED_WRMS_AND_STEPTOL:    return "wrms+steptol";
    case NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED:  return "wrms (step diverged, reverted)";
    case NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER:    return "wrms (max_iter)";
    case NonlinearResult::LSETUP_FAIL:                   return "lsetup fail";
    case NonlinearResult::LSOLVE_FAIL:                   return "lsolve fail";
    case NonlinearResult::MAX_ITER_NO_CONVERGE:          return "max_iter, no convergence";
  }
  return "unknown";
}

/**
 * @brief Abstract interface for the nonlinear system solved by NonlinearSolver.
 *
 * Concrete implementations provide the residual evaluation and linear solve
 * for a specific algebraic IC mode (sub-block or full-system).
 */
class NonlinearSystem {
public:
  virtual ~NonlinearSystem() = default;
  virtual void eval_residual(sunrealtype t, const sunrealtype* y, sunrealtype* res) = 0;
  virtual int solve_linear(sunrealtype t, const sunrealtype* y,
                           sunrealtype* res, sunrealtype* delta) = 0;
};

/**
 * @brief Newton solver for consistent initial conditions.
 *
 * The BlockPartition decides which states move together. Each block carries its own
 * linesearch damping factor, so an ill-scaled equation damps only its own block, and
 * a block that converges retires while the others keep iterating. Blocks are
 * independent subsystems, so retiring one cannot disturb another.
 *
 * Zero allocations in the hotpath.
 */
class NonlinearSolver {
public:
  NonlinearSolver(
    NonlinearSystem& system,
    int n_vars,
    const sunrealtype* atol_data,
    sunrealtype rtol,
    sunrealtype step_tol,
    int max_iter,
    int max_backtracks,
    sunrealtype epsNewt,
    BlockPartition partition
  );

  ~NonlinearSolver() = default;

  NonlinearSolver(const NonlinearSolver&) = delete;
  NonlinearSolver& operator=(const NonlinearSolver&) = delete;

  /**
   * @brief Find y such that F(t, y) = 0.
   * y is read as initial guess and overwritten with solution (in-place).
   */
  NonlinearResult solve_single(sunrealtype t, sunrealtype* y);

  void set_log(SolverLog* log) { log_ = log; }

  // Algebraic residual inf-norm of the guess and final iterate from the last solve.
  sunrealtype initial_res_norm() const { return initial_res_norm_; }
  sunrealtype final_res_norm() const { return final_res_norm_; }
  int num_iterations() const { return last_num_iterations_; }

  const BlockPartition& partition() const { return part_; }

  // True if every accepted step of the last solve lowered each active block's
  // residual inf-norm. False only where a block exhausted its backtracks.
  bool residual_monotone() const { return residual_monotone_; }

private:
  NonlinearResult RunNewtonLoop(sunrealtype t);

  sunrealtype EvalResidualAndNorm(sunrealtype t);
  int SetupAndSolveLinearSystem(sunrealtype t);

  sunrealtype InfNorm(const sunrealtype* vals) const;
  sunrealtype BlockInfNorm(const sunrealtype* vals, int block) const;
  sunrealtype BlockWrmsNorm(const sunrealtype* vals, int block) const;

  void ComputeEwt();
  void SaveIterate();
  void ActivateAllBlocks();
  void RefreshActiveMask();
  void MaskInactive(sunrealtype* v) const;
  void ApplyBlockSteps();
  sunrealtype WholeSystemResNorm(sunrealtype t);

  int n_vars_;
  sunrealtype rtol_;
  sunrealtype step_tol_;
  int max_iter_;
  int max_backtracks_;
  sunrealtype epsNewt_;

  NonlinearSystem& system_;

  BlockPartition part_;
  std::vector<int> active_blocks_;        // blocks still iterating
  std::vector<int> next_active_;          // scratch for the survivors of an iteration
  std::vector<char> active_idx_;          // n_vars; 1 if the state may move now
  std::vector<char> full_step_blk_;       // block takes an undamped step and finishes
  std::vector<char> converged_blk_;       // block has met the WRMS step test at least once
  std::vector<sunrealtype> alpha_;        // damping factor per block
  std::vector<sunrealtype> res_norm_blk_;
  std::vector<sunrealtype> prev_res_norm_blk_;
  std::vector<sunrealtype> delnorm_blk_;
  std::vector<sunrealtype> trial_res_blk_;

  std::vector<sunrealtype> x_;
  std::vector<sunrealtype> res_;
  std::vector<sunrealtype> delta_;
  std::vector<sunrealtype> x_save_;
  std::vector<sunrealtype> ewt_;
  std::vector<sunrealtype> atol_;

  SolverLog* log_ = nullptr;

  int last_num_iterations_ = 0;
  bool residual_monotone_ = true;

  sunrealtype initial_res_norm_ = std::numeric_limits<sunrealtype>::infinity();
  sunrealtype final_res_norm_ = std::numeric_limits<sunrealtype>::infinity();
};

#include "NonlinearSolver.inl"

#endif // PYBAMM_NONLINEAR_SOLVER_HPP

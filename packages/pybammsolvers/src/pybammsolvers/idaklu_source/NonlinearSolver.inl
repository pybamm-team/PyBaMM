#pragma once

// ────────────────────── Constructor ──────────────────────

inline NonlinearSolver::NonlinearSolver(
  NonlinearSystem& system,
  int n_vars,
  const sunrealtype* atol_data,
  sunrealtype rtol,
  sunrealtype step_tol,
  int max_iter,
  int max_backtracks,
  sunrealtype epsNewt,
  BlockPartition partition
) : n_vars_(n_vars),
    rtol_(rtol),
    step_tol_(step_tol),
    max_iter_(max_iter),
    max_backtracks_(max_backtracks),
    epsNewt_(epsNewt),
    system_(system),
    part_(std::move(partition)),
    last_num_iterations_(0)
{
  if (part_.n_vars != n_vars_)
    throw std::invalid_argument("BlockPartition size does not match n_vars");

  atol_.resize(n_vars_);
  std::memcpy(atol_.data(), atol_data, n_vars_ * sizeof(sunrealtype));

  x_.resize(n_vars_);
  res_.resize(n_vars_);
  delta_.resize(n_vars_);
  x_save_.resize(n_vars_);
  ewt_.resize(n_vars_);

  active_idx_.resize(n_vars_);
  full_step_blk_.resize(part_.n_blocks());
  converged_blk_.resize(part_.n_blocks());
  alpha_.resize(part_.n_blocks());
  res_norm_blk_.resize(part_.n_blocks());
  prev_res_norm_blk_.resize(part_.n_blocks());
  delnorm_blk_.resize(part_.n_blocks());
  trial_res_blk_.resize(part_.n_blocks());
}

// ────────────────────── Helpers ──────────────────────

inline void NonlinearSolver::ActivateAllBlocks() {
  active_blocks_.resize(part_.n_blocks());
  std::iota(active_blocks_.begin(), active_blocks_.end(), 0);
  RefreshActiveMask();
}

inline void NonlinearSolver::RefreshActiveMask() {
  std::fill(active_idx_.begin(), active_idx_.end(), static_cast<char>(0));
  for (int b : active_blocks_)
    for (int i : part_.blocks[b]) active_idx_[i] = 1;
}

inline void NonlinearSolver::MaskInactive(sunrealtype* v) const {
  for (int i = 0; i < n_vars_; i++)
    if (!active_idx_[i]) v[i] = SUN_RCONST(0.0);
}

inline sunrealtype NonlinearSolver::InfNorm(const sunrealtype* vals) const {
  sunrealtype mx = SUN_RCONST(0.0);
  for (int i = 0; i < n_vars_; i++) {
    sunrealtype a = std::abs(vals[i]);
    if (a > mx) mx = a;
  }
  return mx;
}

inline sunrealtype NonlinearSolver::BlockInfNorm(
  const sunrealtype* vals, int block) const {
  sunrealtype mx = SUN_RCONST(0.0);
  for (int i : part_.blocks[block]) {
    sunrealtype a = std::abs(vals[i]);
    if (a > mx) mx = a;
  }
  return mx;
}

// Normalised by n_vars_ so that the sum of squares over the active blocks is the
// whole-vector WRMS norm, and a single block reproduces it exactly.
inline sunrealtype NonlinearSolver::BlockWrmsNorm(
  const sunrealtype* vals, int block) const {
  sunrealtype sum = SUN_RCONST(0.0);
  for (int i : part_.blocks[block]) {
    sunrealtype w = vals[i] * ewt_[i];
    sum += w * w;
  }
  return std::sqrt(sum / n_vars_);
}

inline void NonlinearSolver::ComputeEwt() {
  for (int i = 0; i < n_vars_; i++) {
    ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(x_[i]) + atol_[i]);
  }
}

inline void NonlinearSolver::SaveIterate() {
  std::memcpy(x_save_.data(), x_.data(), n_vars_ * sizeof(sunrealtype));
}

inline void NonlinearSolver::ApplyBlockSteps() {
  for (int i = 0; i < n_vars_; i++) {
    int b = part_.block_of[i];
    x_[i] = (b >= 0) ? x_save_[i] - alpha_[b] * delta_[i] : x_save_[i];
  }
}

// ────────────────────── Evaluate residual ──────────────────────

inline sunrealtype NonlinearSolver::EvalResidualAndNorm(sunrealtype t) {
  system_.eval_residual(t, x_.data(), res_.data());
  MaskInactive(res_.data());
  return InfNorm(res_.data());
}

// ────────────────────── Jacobian setup + linear solve ──────────────────────

inline int NonlinearSolver::SetupAndSolveLinearSystem(sunrealtype t) {
  int flag;
  try {
    flag = system_.solve_linear(t, x_.data(), res_.data(), delta_.data());
  } catch (...) {
    return 1;  // LSETUP_FAIL
  }
  if (flag != 0) return (flag > 0) ? 1 : -1;

  MaskInactive(delta_.data());
  return 0;
}

// ────────────────────── Newton loop ──────────────────────

inline NonlinearResult NonlinearSolver::RunNewtonLoop(sunrealtype t) {
  ActivateAllBlocks();

  // Threshold per block, fixed for the solve, so that the sum of squares over all
  // blocks never exceeds epsNewt^2 - the whole-vector test, unchanged.
  const sunrealtype eps_blk =
    epsNewt_ / std::sqrt(static_cast<sunrealtype>(active_blocks_.size()));

  sunrealtype delnorm = std::numeric_limits<sunrealtype>::infinity();
  NonlinearResult result = NonlinearResult::CONVERGED_WRMS_AND_STEPTOL;

  for (int b : active_blocks_) {
    prev_res_norm_blk_[b] = std::numeric_limits<sunrealtype>::infinity();
    converged_blk_[b] = 0;
  }

  ComputeEwt();

  for (int iter = 0; iter < max_iter_; iter++) {
    sunrealtype res_norm = EvalResidualAndNorm(t);
    for (int b : active_blocks_)
      res_norm_blk_[b] = BlockInfNorm(res_.data(), b);
    if (log_) log_->log_newton_iteration(iter, res_norm, delnorm);

    int lsflag = SetupAndSolveLinearSystem(t);
    if (lsflag != 0) {
      NonlinearResult fail = (lsflag > 0) ? NonlinearResult::LSETUP_FAIL
                                          : NonlinearResult::LSOLVE_FAIL;
      last_num_iterations_ += iter + 1;
      if (log_)
        log_->log_newton_failed(iter + 1, res_norm,
                                nonlinear_result_reason(fail));
      return fail;
    }

    sunrealtype delnorm_sq = SUN_RCONST(0.0);
    for (int b : active_blocks_) {
      delnorm_blk_[b] = BlockWrmsNorm(delta_.data(), b);
      delnorm_sq += delnorm_blk_[b] * delnorm_blk_[b];
    }
    delnorm = std::sqrt(delnorm_sq);

    // Classify each block: finish on an undamped step, finish by rolling back a
    // step that stopped reducing the residual, or keep searching.
    next_active_.clear();
    bool all_full_step = true;
    for (int b : active_blocks_) {
      full_step_blk_[b] = 0;
      if (delnorm_blk_[b] > eps_blk) {
        next_active_.push_back(b);
        all_full_step = false;
        continue;
      }
      converged_blk_[b] = 1;
      if (delnorm_blk_[b] <= step_tol_) {
        full_step_blk_[b] = 1;
        next_active_.push_back(b);
        continue;
      }
      if (iter > 0 && res_norm_blk_[b] >= prev_res_norm_blk_[b]) {
        for (int i : part_.blocks[b]) x_[i] = x_save_[i];  // back to the last iterate
        result = NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED;
        continue;                                          // block is finished
      }
      next_active_.push_back(b);
      all_full_step = false;
    }

    // Blocks that rolled back are dropped before the iterate is saved, so their
    // final value is what the caller sees.
    active_blocks_ = next_active_;
    if (active_blocks_.empty()) {
      last_num_iterations_ += iter + 1;
      if (log_) log_->log_newton_converged(iter + 1,
                                           nonlinear_result_reason(result));
      return result;
    }
    RefreshActiveMask();

    for (int b : active_blocks_) prev_res_norm_blk_[b] = res_norm_blk_[b];
    SaveIterate();
    for (int b : active_blocks_) alpha_[b] = SUN_RCONST(1.0);

    if (all_full_step) {
      ApplyBlockSteps();
      last_num_iterations_ += iter + 1;
      if (log_) log_->log_newton_converged(iter + 1,
                                           nonlinear_result_reason(result));
      return result;
    }

    // Armijo-style linesearch: halve the step of each block that fails the
    // sufficient-decrease test. The 0.5 factor is the standard Armijo parameter
    // (c1 = 0.5) used in SUNDIALS IDA's own Newton iteration (see ida_ic.c).
    // Blocks are independent, so one residual evaluation scores all of them.
    for (int ls = 0; ls < max_backtracks_; ls++) {
      ApplyBlockSteps();
      EvalResidualAndNorm(t);
      bool halved = false;
      for (int b : active_blocks_) {
        trial_res_blk_[b] = BlockInfNorm(res_.data(), b);
        if (full_step_blk_[b]) continue;
        if (trial_res_blk_[b] <=
            (SUN_RCONST(1.0) - alpha_[b] * SUN_RCONST(0.5)) * res_norm_blk_[b])
          continue;
        if (alpha_[b] * delnorm_blk_[b] <= step_tol_)
          continue;
        alpha_[b] *= SUN_RCONST(0.5);
        halved = true;
      }
      if (!halved) break;
    }

    next_active_.clear();
    for (int b : active_blocks_) {
      if (trial_res_blk_[b] > res_norm_blk_[b]) residual_monotone_ = false;
      if (!full_step_blk_[b]) next_active_.push_back(b);
    }
    active_blocks_ = next_active_;
    if (active_blocks_.empty()) {
      last_num_iterations_ += iter + 1;
      if (log_) log_->log_newton_converged(iter + 1,
                                           nonlinear_result_reason(result));
      return result;
    }
    RefreshActiveMask();
  }

  // Only claim convergence at max_iter if every block that is still running has
  // met the WRMS step test. Latching on any one block would report success while
  // the others sit at a large residual.
  last_num_iterations_ += max_iter_;
  bool converged = true;
  for (int b : active_blocks_)
    if (!converged_blk_[b]) converged = false;
  if (converged) {
    if (log_) log_->log_newton_converged(
      max_iter_,
      nonlinear_result_reason(NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER));
    return NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER;
  }

  if (log_) log_->log_newton_failed(
    max_iter_, InfNorm(res_.data()),
    nonlinear_result_reason(NonlinearResult::MAX_ITER_NO_CONVERGE));
  return NonlinearResult::MAX_ITER_NO_CONVERGE;
}

// ────────────────────── Whole-system residual ──────────────────────

// Residual inf-norm over every solved state, ignoring which blocks have retired.
// This is the number the caller judges the solve by, so it must not depend on
// which blocks happened to be active last.
inline sunrealtype NonlinearSolver::WholeSystemResNorm(sunrealtype t) {
  ActivateAllBlocks();
  return EvalResidualAndNorm(t);
}

// ────────────────────── solve_single ──────────────────────

inline NonlinearResult NonlinearSolver::solve_single(
  sunrealtype t, sunrealtype* y
) {
  std::memcpy(x_.data(), y, n_vars_ * sizeof(sunrealtype));

  if (log_) log_->log_newton_start(t, n_vars_, block_mode_name(part_.mode),
                                   part_.n_blocks());

  residual_monotone_ = true;
  last_num_iterations_ = 0;
  initial_res_norm_ = WholeSystemResNorm(t);

  NonlinearResult result = RunNewtonLoop(t);

  final_res_norm_ = WholeSystemResNorm(t);
  std::memcpy(y, x_.data(), n_vars_ * sizeof(sunrealtype));

  return result;
}

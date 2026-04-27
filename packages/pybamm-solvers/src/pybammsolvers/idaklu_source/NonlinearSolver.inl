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
  const std::vector<int>& diff_idx
) : n_vars_(n_vars),
    rtol_(rtol),
    step_tol_(step_tol),
    max_iter_(max_iter),
    max_backtracks_(max_backtracks),
    epsNewt_(epsNewt),
    system_(system),
    diff_idx_(diff_idx),
    last_num_iterations_(0)
{
  atol_.resize(n_vars_);
  std::memcpy(atol_.data(), atol_data, n_vars_ * sizeof(sunrealtype));

  x_.resize(n_vars_);
  res_.resize(n_vars_);
  delta_.resize(n_vars_);
  x_save_.resize(n_vars_);
  ewt_.resize(n_vars_);
}

// ────────────────────── Helpers ──────────────────────

inline void NonlinearSolver::ZeroDiffComponents(sunrealtype* v) const {
  for (int i : diff_idx_) v[i] = SUN_RCONST(0.0);
}

inline sunrealtype NonlinearSolver::WrmsNorm(const sunrealtype* vals) const {
  sunrealtype sum = SUN_RCONST(0.0);
  for (int i = 0; i < n_vars_; i++) {
    sunrealtype w = vals[i] * ewt_[i];
    sum += w * w;
  }
  return std::sqrt(sum / n_vars_);
}

inline sunrealtype NonlinearSolver::InfNorm(const sunrealtype* vals) const {
  sunrealtype mx = SUN_RCONST(0.0);
  for (int i = 0; i < n_vars_; i++) {
    sunrealtype a = std::abs(vals[i]);
    if (a > mx) mx = a;
  }
  return mx;
}

inline void NonlinearSolver::ComputeEwt() {
  for (int i = 0; i < n_vars_; i++) {
    ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(x_[i]) + atol_[i]);
  }
}

inline void NonlinearSolver::SaveIterate() {
  std::memcpy(x_save_.data(), x_.data(), n_vars_ * sizeof(sunrealtype));
}

inline void NonlinearSolver::RevertAndApply(sunrealtype alpha) {
  for (int i = 0; i < n_vars_; i++) {
    x_[i] = x_save_[i] - alpha * delta_[i];
  }
}

// ────────────────────── Evaluate residual ──────────────────────

inline sunrealtype NonlinearSolver::EvalResidualAndNorm(sunrealtype t) {
  system_.eval_residual(t, x_.data(), res_.data());
  if (!diff_idx_.empty()) ZeroDiffComponents(res_.data());
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

  if (!diff_idx_.empty()) ZeroDiffComponents(delta_.data());
  return 0;
}

// ────────────────────── Newton loop ──────────────────────

inline NonlinearResult NonlinearSolver::RunNewtonLoop(sunrealtype t) {
  sunrealtype delnorm = std::numeric_limits<sunrealtype>::infinity();
  sunrealtype prev_res_norm = std::numeric_limits<sunrealtype>::infinity();
  bool converged = false;

  ComputeEwt();

  for (int iter = 0; iter < max_iter_; iter++) {
    sunrealtype res_norm = EvalResidualAndNorm(t);
    if (log_) log_->log_newton_iteration(iter, res_norm, delnorm);

    int lsflag = SetupAndSolveLinearSystem(t);
    if (lsflag > 0) {
      last_message_ = nonlinear_result_reason(NonlinearResult::LSETUP_FAIL);
      last_num_iterations_ = iter + 1;
      if (log_) log_->log_newton_failed(iter + 1, res_norm, last_message_.c_str());
      return NonlinearResult::LSETUP_FAIL;
    }
    if (lsflag < 0) {
      last_message_ = nonlinear_result_reason(NonlinearResult::LSOLVE_FAIL);
      last_num_iterations_ = iter + 1;
      if (log_) log_->log_newton_failed(iter + 1, res_norm, last_message_.c_str());
      return NonlinearResult::LSOLVE_FAIL;
    }

    delnorm = WrmsNorm(delta_.data());

    if (delnorm <= epsNewt_) {
      converged = true;
      if (delnorm <= step_tol_) {
        SaveIterate();
        RevertAndApply(SUN_RCONST(1.0));
        last_message_ = nonlinear_result_reason(NonlinearResult::CONVERGED_WRMS_AND_STEPTOL);
        last_num_iterations_ = iter + 1;
        if (log_) log_->log_newton_converged(iter + 1, last_message_.c_str());
        return NonlinearResult::CONVERGED_WRMS_AND_STEPTOL;
      }
      if (iter > 0 && res_norm >= prev_res_norm) {
        RevertAndApply(SUN_RCONST(0.0));
        last_message_ = nonlinear_result_reason(NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED);
        last_num_iterations_ = iter + 1;
        if (log_) log_->log_newton_converged(iter + 1, last_message_.c_str());
        return NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED;
      }
    }

    prev_res_norm = res_norm;
    SaveIterate();

    // Armijo-style linesearch: halve step until sufficient decrease.
    // The 0.5 factor is the standard Armijo parameter (c1 = 0.5) used
    // in SUNDIALS IDA's own Newton iteration (see ida_ic.c).
    sunrealtype alpha = SUN_RCONST(1.0);
    for (int ls = 0; ls < max_backtracks_; ls++) {
      RevertAndApply(alpha);
      sunrealtype trial_norm = EvalResidualAndNorm(t);
      if (trial_norm <= (SUN_RCONST(1.0) - alpha * SUN_RCONST(0.5)) * res_norm)
        break;
      if (alpha * delnorm <= step_tol_)
        break;
      alpha *= SUN_RCONST(0.5);
    }
  }

  if (converged) {
    last_message_ = nonlinear_result_reason(NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER);
    last_num_iterations_ = max_iter_;
    if (log_) log_->log_newton_converged(max_iter_, last_message_.c_str());
    return NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER;
  }

  last_message_ = nonlinear_result_reason(NonlinearResult::MAX_ITER_NO_CONVERGE);
  last_num_iterations_ = max_iter_;
  if (log_) log_->log_newton_failed(max_iter_, InfNorm(res_.data()), last_message_.c_str());
  return NonlinearResult::MAX_ITER_NO_CONVERGE;
}

// ────────────────────── solve_single ──────────────────────

inline NonlinearResult NonlinearSolver::solve_single(
  sunrealtype t, sunrealtype* y
) {
  std::memcpy(x_.data(), y, n_vars_ * sizeof(sunrealtype));

  if (log_) log_->log_newton_start(t, n_vars_);
  NonlinearResult result = RunNewtonLoop(t);

  std::memcpy(y, x_.data(), n_vars_ * sizeof(sunrealtype));

  return result;
}

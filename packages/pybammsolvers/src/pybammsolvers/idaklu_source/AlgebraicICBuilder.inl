#pragma once

// Algebraic IC solver construction and helpers.
// Included from IDAKLUSolverOpenMP.hpp after the class definition.

#include "Expressions/Expressions.hpp"
#include "sundials_functions.hpp"

// ────────────────────── Newton-IC linear-solve helper ──────────────────────

template <class ExprSet>
int IDAKLUSolverOpenMP<ExprSet>::SolveICLinearSystem(
    N_Vector yy_ptr, N_Vector yyp_ptr,
    const sunrealtype* y_in, sunrealtype* res, sunrealtype* delta) {

  // Reuses IDA's direct LS and J (refilled at cj=1), with no extra allocation;
  // BuildAlgebraicSolver leaves iterative LSs to IDACalcIC instead
  auto& fs = *this->alg_state_->full;

  sunrealtype* yy_data = NV_DATA(yy_ptr);
  std::memcpy(yy_data, y_in, number_of_states * sizeof(sunrealtype));

  sunrealtype* res_data = N_VGetArrayPointer(fs.res_nvec);
  sunrealtype* delta_data = N_VGetArrayPointer(fs.delta_nvec);
  std::memcpy(res_data, res, number_of_states * sizeof(sunrealtype));

  // jacobian_eval ignores tempv1/2/3 in the casadi path; pass y_cache
  // since the signature requires valid N_Vectors.
  int flag = jacobian_eval<ExprSet>(
    SUN_RCONST(0.0), SUN_RCONST(1.0), yy_ptr, yyp_ptr, fs.res_nvec, J,
    functions.get(), y_cache, y_cache, y_cache);
  if (flag != 0) return 1;

  flag = SUNLinSolSetup(LS, J);
  std::memcpy(delta_data, res, number_of_states * sizeof(sunrealtype));
  if (flag == 0) {
    flag = SUNLinSolSolve(LS, J, fs.delta_nvec, fs.res_nvec, SUN_RCONST(0.0));
  }

  if (flag != 0) return 1;
  std::memcpy(delta, delta_data, number_of_states * sizeof(sunrealtype));
  return 0;
}

// ────────────────────── NonlinearSystem implementations ──────────────────────

template <class ExprSet>
class SubBlockSystem : public NonlinearSystem {
public:
  SubBlockSystem(
    ExprSet* funcs, N_Vector yy,
    IDAKLUSolverOpenMP<ExprSet>& solver)
    : funcs_(funcs), yy_(yy), solver_(solver) {}

  void eval_residual(sunrealtype t, const sunrealtype* x_alg, sunrealtype* res_out) override {
    sunrealtype* yy_data = NV_DATA(yy_);
    std::memcpy(yy_data + solver_.len_rhs_, x_alg,
                solver_.len_alg_ * sizeof(sunrealtype));
    funcs_->alg_res->m_arg[0] = &t;
    funcs_->alg_res->m_arg[1] = yy_data;
    funcs_->alg_res->m_arg[2] = funcs_->inputs.data();
    funcs_->alg_res->m_res[0] = res_out;
    (*funcs_->alg_res)();
  }

  int solve_linear(sunrealtype t, const sunrealtype* x_alg,
                   sunrealtype* res, sunrealtype* delta) override {
    sunrealtype* yy_data = NV_DATA(yy_);
    std::memcpy(yy_data + solver_.len_rhs_, x_alg,
                solver_.len_alg_ * sizeof(sunrealtype));

    if (solver_.SetupAlgebraicJacobian(t) != 0) return 1;
    return solver_.SolveAlgebraicJacobian(res, delta);
  }

private:
  ExprSet* funcs_;
  N_Vector yy_;
  IDAKLUSolverOpenMP<ExprSet>& solver_;
};

// ────────────────────── Algebraic sub-block linear solves ──────────────────────

template <class ExprSet>
int IDAKLUSolverOpenMP<ExprSet>::SetupAlgebraicJacobian(sunrealtype t) {
  auto& sb = *alg_state_->sub;
  Expression* alg_jac = functions->alg_jac;
  alg_jac->m_arg[0] = &t;
  alg_jac->m_arg[1] = N_VGetArrayPointer(yy);
  alg_jac->m_arg[2] = functions->inputs.data();
  alg_jac->m_res[0] = sb.full_jac_buf.data();
  (*alg_jac)();

  if (sb.use_sparse) {
    sunrealtype* sub_data = SUNSparseMatrix_Data(sb.J);
    for (int i = 0; i < sb.nnz; i++)
      sub_data[i] = sb.full_jac_buf[sb.data_indices[i]];
  } else {
    sunrealtype* dense = SUNDenseMatrix_Data(sb.J);
    std::memset(dense, 0, len_alg_ * len_alg_ * sizeof(sunrealtype));
    for (int col = 0; col < len_alg_; col++) {
      for (auto k = sb.colptrs[col]; k < sb.colptrs[col + 1]; k++) {
        int row = sb.rowvals[k];
        dense[col * len_alg_ + row] = sb.full_jac_buf[sb.data_indices[k]];
      }
    }
  }

  return SUNLinSolSetup(sb.LS, sb.J);
}

template <class ExprSet>
int IDAKLUSolverOpenMP<ExprSet>::SolveAlgebraicJacobian(
    const sunrealtype* rhs, sunrealtype* solution) {
  auto& sb = *alg_state_->sub;
  std::memcpy(N_VGetArrayPointer(sb.res_nvec), rhs, len_alg_ * sizeof(sunrealtype));
  int const flag = SUNLinSolSolve(sb.LS, sb.J, sb.delta_nvec, sb.res_nvec, SUN_RCONST(0.0));
  std::memcpy(solution, N_VGetArrayPointer(sb.delta_nvec), len_alg_ * sizeof(sunrealtype));
  return flag;
}

// ────────────────────── Full initial-condition matrix ──────────────────────

template <class ExprSet>
int IDAKLUSolverOpenMP<ExprSet>::SetupInitialConditionMatrix(sunrealtype t) {
  const auto& colptrs = functions->jac_times_cjmass_colptrs;
  const auto& rowvals = functions->jac_times_cjmass_rowvals;
  if (!ic_matrix_) {
    ic_matrix_ = std::make_unique<InitialConditionMatrix>();
    auto& m = *ic_matrix_;
    m.A = SUNSparseMatrix(number_of_states, number_of_states,
                          jac_times_cjmass_nnz, CSC_MAT, sunctx);
    m.rhs = N_VNew_Serial(number_of_states, sunctx);
    m.solution = N_VNew_Serial(number_of_states, sunctx);
    m.LS = SUNLinSol_KLU(m.solution, m.A, sunctx);
    configure_klu(m.LS, setup_opts.klu_pivot_tolerance);
    m.jac_at_cj0.resize(jac_times_cjmass_nnz);
    m.jac_at_cj1.resize(jac_times_cjmass_nnz);
  }
  auto& m = *ic_matrix_;

  sunindextype* indexptrs = SUNSparseMatrix_IndexPointers(m.A);
  sunindextype* indexvals = SUNSparseMatrix_IndexValues(m.A);
  for (size_t i = 0; i < colptrs.size(); i++) indexptrs[i] = colptrs[i];
  for (size_t i = 0; i < rowvals.size(); i++) indexvals[i] = rowvals[i];

  // jac_times_cjmass is F_y - cj M, so the difference across cj isolates -M
  Expression* jac = functions->jac_times_cjmass;
  for (int cj_int = 0; cj_int < 2; cj_int++) {
    sunrealtype cj = cj_int;
    jac->m_arg[0] = &t;
    jac->m_arg[1] = N_VGetArrayPointer(yy);
    jac->m_arg[2] = functions->inputs.data();
    jac->m_arg[3] = &cj;
    jac->m_res[0] = (cj_int == 0) ? m.jac_at_cj0.data() : m.jac_at_cj1.data();
    (*jac)();
  }

  sunrealtype* data = SUNSparseMatrix_Data(m.A);
  const sunrealtype* id_val = N_VGetArrayPointer(id);
  for (int col = 0; col < number_of_states; col++) {
    const bool differential = is_differential(id_val[col]);
    for (auto k = colptrs[col]; k < colptrs[col + 1]; k++) {
      data[k] = differential ? m.jac_at_cj1[k] - m.jac_at_cj0[k] : m.jac_at_cj0[k];
    }
  }

  int flag = SUNLinSolSetup(m.LS, m.A);
  if (flag != 0) {
    // A refactorisation reuses the previous pivots, which may now be zero
    CheckErrors(SUNLinSol_KLUReInit(m.LS, m.A, jac_times_cjmass_nnz,
                                    SUNKLU_REINIT_PARTIAL),
                "SUNLinSol_KLUReInit");
    flag = SUNLinSolSetup(m.LS, m.A);
  }
  return flag;
}

template <class ExprSet>
int IDAKLUSolverOpenMP<ExprSet>::SolveInitialConditionMatrix(sunrealtype* rhs_solution) {
  auto& m = *ic_matrix_;
  std::memcpy(N_VGetArrayPointer(m.rhs), rhs_solution, number_of_states * sizeof(sunrealtype));
  int const flag = SUNLinSolSolve(m.LS, m.A, m.solution, m.rhs, SUN_RCONST(0.0));
  if (flag != 0) {
    return flag;
  }
  std::memcpy(rhs_solution, N_VGetArrayPointer(m.solution), number_of_states * sizeof(sunrealtype));
  return 0;
}

template <class ExprSet>
class FullSystem : public NonlinearSystem {
public:
  FullSystem(
    ExprSet* funcs, N_Vector yy, N_Vector yyp,
    IDAKLUSolverOpenMP<ExprSet>& solver)
    : funcs_(funcs), yy_(yy), yyp_(yyp), solver_(solver) {}

  void eval_residual(sunrealtype t, const sunrealtype* y, sunrealtype* res_out) override {
    sunrealtype* yy_data = NV_DATA(yy_);
    std::memcpy(yy_data, y, solver_.number_of_states * sizeof(sunrealtype));
    funcs_->rhs_alg->m_arg[0] = &t;
    funcs_->rhs_alg->m_arg[1] = yy_data;
    funcs_->rhs_alg->m_arg[2] = funcs_->inputs.data();
    funcs_->rhs_alg->m_res[0] = res_out;
    (*funcs_->rhs_alg)();
  }

  int solve_linear(sunrealtype t, const sunrealtype* y,
                   sunrealtype* res, sunrealtype* delta) override {
    return solver_.SolveICLinearSystem(yy_, yyp_, y, res, delta);
  }

private:
  ExprSet* funcs_;
  N_Vector yy_;
  N_Vector yyp_;
  IDAKLUSolverOpenMP<ExprSet>& solver_;
};

// ────────────────────── Sub-block sparsity pre-computation ──────────────────────

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::PrecomputeSubBlockSparsity() {
  auto& sb = *alg_state_->sub;
  Expression* jac_expr = functions->alg_jac;
  const auto& rows = jac_expr->get_row();
  const auto& cols = jac_expr->get_col();
  int nnz_total = jac_expr->nnz_out();

  // Build column pointers from COO data in O(nnz) instead of O(n_alg * nnz).
  // The full Jacobian has number_of_states columns; we only need the algebraic
  // columns [len_rhs_, len_rhs_ + len_alg_).
  int n_full_cols = number_of_states;
  std::vector<int> col_count(n_full_cols + 1, 0);
  for (int k = 0; k < nnz_total; k++)
    col_count[static_cast<int>(cols[k]) + 1]++;

  // Prefix sum -> column pointers
  for (int c = 1; c <= n_full_cols; c++)
    col_count[c] += col_count[c - 1];

  // Scatter into column-sorted order
  std::vector<int> sorted_row(nnz_total);
  std::vector<int> sorted_orig_idx(nnz_total);
  std::vector<int> pos(col_count);  // copy for scatter
  for (int k = 0; k < nnz_total; k++) {
    int c = static_cast<int>(cols[k]);
    int dest = pos[c]++;
    sorted_row[dest] = static_cast<int>(rows[k]);
    sorted_orig_idx[dest] = k;
  }

  // Extract only the algebraic columns
  sb.colptrs.resize(len_alg_ + 1);
  sb.rowvals.clear();
  sb.data_indices.clear();

  for (int alg_col = 0; alg_col < len_alg_; alg_col++) {
    sb.colptrs[alg_col] = static_cast<sunindextype>(sb.rowvals.size());
    int full_col = alg_col + len_rhs_;
    int start = col_count[full_col];
    int end = col_count[full_col + 1];
    for (int p = start; p < end; p++) {
      sb.rowvals.push_back(static_cast<sunindextype>(sorted_row[p]));
      sb.data_indices.push_back(sorted_orig_idx[p]);
    }
  }
  sb.colptrs[len_alg_] = static_cast<sunindextype>(sb.rowvals.size());
  sb.nnz = static_cast<int>(sb.rowvals.size());
}

// ────────────────────── BuildAlgebraicSolver ──────────────────────

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::BuildAlgebraicSolver(const sunrealtype* id_val) {
  DEBUG("IDAKLUSolverOpenMP::BuildAlgebraicSolver");

  // Discretisation makes M zero in the algebraic rows (id 0); the sub-block
  // functions are only passed for standard-form DAEs
  auto* funcs = functions.get();
  bool has_alg_fns = (funcs->alg_res->nnz_out() > 0 && funcs->alg_jac->nnz_out() > 0);

  // FULL mode drives IDA's linear solver outside IDA, which an iterative one
  // cannot survive (its ATimes reads IDA's unset ycur); IDACalcIC is used instead
  if (!has_alg_fns && setup_opts.using_iterative_solver) {
    return;
  }

  alg_state_ = std::make_unique<AlgSolverState>();
  auto& as = *alg_state_;

  for (int i = 0; i < number_of_states; i++) {
    if (is_algebraic(id_val[i]))
      as.alg_idx.push_back(i);
    else
      as.diff_idx.push_back(i);
  }

  auto* yy_ptr = yy;
  auto* yyp_ptr = yyp;
  const sunrealtype* atol_data = N_VGetArrayPointer(avtol);

  using Mode = typename AlgSolverState::Mode;
  as.mode = has_alg_fns ? Mode::SUBBLOCK : Mode::FULL;

  int n_solve_vars;
  std::vector<int> solve_diff_idx;

  if (as.mode == Mode::SUBBLOCK) {
    // ── SUBBLOCK: solve only algebraic variables with own LS/J ──
    n_solve_vars = len_alg_;

    bool use_sparse = (setup_opts.jacobian == "sparse" &&
                       setup_opts.linear_solver == "SUNLinSol_KLU");

    as.sub = std::make_unique<SubBlockResources>();
    auto& sb = *as.sub;
    sb.use_sparse = use_sparse;

    SUNContext_Create(SUN_COMM_NULL, &sb.sunctx);
    PrecomputeSubBlockSparsity();

    int alg_jac_nnz = funcs->alg_jac->nnz_out();
    sb.full_jac_buf.resize(alg_jac_nnz > 0
      ? alg_jac_nnz : len_alg_ * number_of_states);

    sb.res_nvec = N_VNew_Serial(len_alg_, sb.sunctx);
    sb.delta_nvec = N_VNew_Serial(len_alg_, sb.sunctx);

    if (use_sparse) {
      sb.J = SUNSparseMatrix(len_alg_, len_alg_, sb.nnz, CSC_MAT, sb.sunctx);
      sunindextype* jp = SUNSparseMatrix_IndexPointers(sb.J);
      sunindextype* jr = SUNSparseMatrix_IndexValues(sb.J);
      for (int i = 0; i <= len_alg_; i++) jp[i] = sb.colptrs[i];
      for (int i = 0; i < sb.nnz; i++) jr[i] = sb.rowvals[i];
      sb.LS = SUNLinSol_KLU(sb.delta_nvec, sb.J, sb.sunctx);
      configure_klu(sb.LS, setup_opts.klu_pivot_tolerance);
    } else {
      sb.J = SUNDenseMatrix(len_alg_, len_alg_, sb.sunctx);
      sb.LS = SUNLinSol_Dense(sb.delta_nvec, sb.J, sb.sunctx);
    }

    as.system = std::make_unique<SubBlockSystem<ExprSet>>(funcs, yy_ptr, *this);

  } else {
    // ── FULL: full-system via IDA's linear solve interface ──
    n_solve_vars = number_of_states;
    solve_diff_idx = as.diff_idx;

    as.full = std::make_unique<FullSystemResources>();
    auto& fs = *as.full;
    fs.res_nvec = N_VNew_Serial(number_of_states, sunctx);
    fs.delta_nvec = N_VNew_Serial(number_of_states, sunctx);

    as.system = std::make_unique<FullSystem<ExprSet>>(
      funcs, yy_ptr, yyp_ptr, *this);
  }

  // Build atol for the solve dimension
  std::vector<sunrealtype> solve_atol(n_solve_vars);
  if (as.mode == Mode::SUBBLOCK) {
    for (int i = 0; i < len_alg_; i++)
      solve_atol[i] = atol_data[as.alg_idx[i]];
  } else {
    std::memcpy(solve_atol.data(), atol_data, number_of_states * sizeof(sunrealtype));
  }

  as.solver = std::make_unique<NonlinearSolver>(
    *as.system,
    n_solve_vars,
    solve_atol.data(),
    rtol,
    solver_opts.newton_step_tol,
    solver_opts.max_num_iterations_ic,
    solver_opts.max_linesearch_backtracks_ic,
    solver_opts.nonlinear_convergence_coefficient_ic,
    solve_diff_idx
  );
  as.solver->set_log(&log_);

  as.y_backup.resize(number_of_states);
  as.yp_backup.resize(number_of_states);
}

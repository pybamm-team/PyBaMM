#ifndef PYBAMM_STANDALONE_NEWTON_SOLVER_HPP
#define PYBAMM_STANDALONE_NEWTON_SOLVER_HPP

#include "BlockPartition.hpp"
#include "NonlinearSolver.hpp"
#include "Expressions/Casadi/CasadiFunctions.hpp"
#include "common.hpp"
#include <memory>
#include <string>
#include <utility>

/**
 * @brief NonlinearSystem backed by standalone CasadiFunctions + SUNLinSol.
 *
 * Self-contained: no dependency on IDAKLUSolverOpenMP or IDA memory.
 * Residual signature: F(t, y_alg, inputs) -> res   (n_vars outputs)
 * Jacobian signature: J(t, y_alg, inputs) -> data   (COO or dense)
 */
class StandaloneAlgebraicSystem : public NonlinearSystem {
public:
  StandaloneAlgebraicSystem(
    casadi::Function res_fn,
    casadi::Function jac_fn,
    bool use_sparse);

  ~StandaloneAlgebraicSystem();

  int n_vars() const { return n_vars_; }

  const std::vector<sunindextype>& colptrs() const { return colptrs_; }
  const std::vector<sunindextype>& rowvals() const { return rowvals_; }

  void set_inputs(const sunrealtype* inputs_data, int inputs_len) {
    inputs_.assign(inputs_data, inputs_data + inputs_len);
  }

  void eval_residual(sunrealtype t, const sunrealtype* y,
                     sunrealtype* res) override;

  int solve_linear(sunrealtype t, const sunrealtype* y,
                   sunrealtype* res, sunrealtype* delta) override;

private:
  void BuildSparseResources(int jac_nnz);
  void BuildDenseResources();

  CasadiFunction res_cf_;
  CasadiFunction jac_cf_;
  int n_vars_;
  bool use_sparse_;

  SUNContext sunctx_;
  SUNMatrix J_;
  SUNLinearSolver LS_;
  N_Vector res_nvec_;
  N_Vector delta_nvec_;

  int nnz_ = 0;
  std::vector<sunindextype> colptrs_;
  std::vector<sunindextype> rowvals_;
  std::vector<int> data_indices_;
  std::vector<sunrealtype> jac_buf_;
  std::vector<sunrealtype> inputs_;
};

/**
 * @brief Standalone Newton solver exposed to Python via pybind11.
 *
 * Owns a StandaloneAlgebraicSystem and NonlinearSolver. No IDA dependency.
 * Solves F(t, y, inputs) = 0 for y given an initial guess.
 */
class StandaloneNewtonSolver {
public:
  StandaloneNewtonSolver(
    casadi::Function residual_fn,
    casadi::Function jacobian_fn,
    const std::vector<sunrealtype>& atol,
    sunrealtype rtol,
    sunrealtype step_tol,
    int max_iter,
    int max_backtracks,
    sunrealtype epsNewt,
    bool use_sparse,
    const std::string& block_mode = "decoupled");

  /** @brief Resolved block mode and block count of the partition. */
  std::string block_mode() const {
    return block_mode_name(solver_.partition().mode);
  }
  int num_blocks() const { return solver_.partition().n_blocks(); }
  int num_iterations() const { return solver_.num_iterations(); }
  sunrealtype final_res_norm() const { return solver_.final_res_norm(); }
  bool residual_monotone() const { return solver_.residual_monotone(); }

  /**
   * @brief Solve F(t, y, inputs) = 0 starting from y0.
   * @return (success, y_solution) with zero-copy numpy output.
   */
  std::pair<bool, np_array> solve(
    sunrealtype t,
    const np_array& y0_np,
    const np_array& inputs_np);

  /**
   * @brief Batch solve over multiple time points in a single C++ call.
   *
   * Each solve reuses the previous solution as the initial guess.
   * Stops early on the first failure.
   *
   * @return (all_success, y_matrix) where y_matrix has shape (n_vars, n_times).
   */
  std::pair<bool, py::array_t<sunrealtype, py::array::f_style>> solve_batch(
    const np_array& t_eval_np,
    const np_array& y0_alg_np,
    const np_array& inputs_np);

private:
  StandaloneAlgebraicSystem system_;
  NonlinearSolver solver_;
  int n_vars_;
  std::vector<sunrealtype> y_work_;
};

#endif // PYBAMM_STANDALONE_NEWTON_SOLVER_HPP

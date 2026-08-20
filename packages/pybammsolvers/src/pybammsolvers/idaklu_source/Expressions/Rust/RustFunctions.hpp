#ifndef PYBAMM_IDAKLU_RUST_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_RUST_FUNCTIONS_HPP

#include "../Base/Expression.hpp"
#include "../Base/ExpressionSet.hpp"
#include "../../Options.hpp"
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

// Rust FFI declarations - single source of truth
#include "pybamm_rust_ffi.h"

/**
 * @brief Common base for the Rust FFI Expression adapters.
 *
 * Every adapter writes a single fixed-length output block and differs only in
 * which FFI entry point it forwards to, so the shape and sparsity interface
 * lives here. `m_rows`/`m_cols` stay empty unless the adapter emits COO data,
 * matching the CasADi adapters' contract for dense-vector outputs.
 */
class RustExpression : public Expression {
public:
    /**
     * @param model   Opaque Rust model handle
     * @param out_len Elements written per evaluation (nnz for COO adapters)
     * @param n_args  m_arg slots the FFI signature uses
     * @param n_res   m_res slots the FFI signature writes
     */
    RustExpression(void* model, int out_len, int n_args = 3, int n_res = 1)
        : m_model(model), m_out_len(out_len) {
        m_arg.resize(n_args);
        m_res.resize(n_res);
    }

    expr_int out_shape(int k) override { return m_out_len; }
    expr_int nnz() override { return m_out_len; }
    expr_int nnz_out() override { return m_out_len; }
    const std::vector<expr_int>& get_row() override { return m_rows; }
    const std::vector<expr_int>& get_col() override { return m_cols; }

protected:
    void* m_model;
    int m_out_len;
    std::vector<expr_int> m_rows;
    std::vector<expr_int> m_cols;
};

/**
 * @brief RustRhsExpression: evaluates the RHS function via Rust FFI
 *
 * ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_res[0]=output
 */
class RustRhsExpression : public RustExpression {
public:
    RustRhsExpression(void* model, int n_states)
        : RustExpression(model, n_states) {}

    void operator()() override {
        PYBAMM_RUST_CALL(eval_rhs, *m_arg[0], m_arg[1], m_arg[2], m_res[0], m_model);
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        PYBAMM_RUST_CALL(eval_rhs, *inputs[0], inputs[1], inputs[2], results[0], m_model);
    }
};

/**
 * @brief RustJacExpression: assembles Jacobian matrix via Rust FFI
 *
 * ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_arg[3]=cj, m_res[0]=jac_data
 */
class RustJacExpression : public RustExpression {
public:
    RustJacExpression(void* model, int nnz,
                      std::vector<expr_int> rows,
                      std::vector<expr_int> cols)
        : RustExpression(model, nnz, 4) {
        m_rows = std::move(rows);
        m_cols = std::move(cols);
    }

    void operator()() override {
        PYBAMM_RUST_CALL(jac_assemble, *m_arg[0], m_arg[1], m_arg[2], *m_arg[3], m_res[0], m_model);
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        // cj is injected by the SUNDIALS callback via m_arg[3] and is not part
        // of the inputs vector, so the cj read stays as *m_arg[3] even here.
        PYBAMM_RUST_CALL(jac_assemble, *inputs[0], inputs[1], inputs[2], *m_arg[3], results[0], m_model);
    }
};

/**
 * @brief RustJacActionExpression: computes Jacobian-vector product via Rust FFI
 *
 * ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_arg[3]=v, m_res[0]=Jv
 */
class RustJacActionExpression : public RustExpression {
public:
    RustJacActionExpression(void* model, int n_states)
        : RustExpression(model, n_states, 4) {}

    void operator()() override {
        PYBAMM_RUST_CALL(jac_action, *m_arg[0], m_arg[1], m_arg[2], m_arg[3], m_res[0], m_model);
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        PYBAMM_RUST_CALL(jac_action, *inputs[0], inputs[1], inputs[2], inputs[3], results[0], m_model);
    }
};

/**
 * @brief RustMassActionExpression: computes mass-matrix-vector product via Rust FFI
 *
 * ABI: m_arg[0]=v, m_res[0]=Mv
 */
class RustMassActionExpression : public RustExpression {
public:
    RustMassActionExpression(void* model, int n_states)
        : RustExpression(model, n_states, 1) {}

    void operator()() override {
        PYBAMM_RUST_CALL(mass_action, m_arg[0], m_res[0], m_model);
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        PYBAMM_RUST_CALL(mass_action, inputs[0], results[0], m_model);
    }
};

/**
 * @brief RustEventsExpression: event evaluator backed by Rust FFI.
 *
 * Evaluates all events and writes concatenated results to output.
 *
 * ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_res[0]=output (length total_event_len)
 */
class RustEventsExpression : public RustExpression {
public:
    /// `total_event_len == 0` (no events) makes this an inert placeholder.
    RustEventsExpression(void* model, int total_event_len)
        : RustExpression(model, total_event_len) {}

    void operator()() override {
        if (m_out_len > 0) {
            PYBAMM_RUST_CALL(events_eval, *m_arg[0], m_arg[1], m_arg[2], m_res[0], m_model);
        }
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        if (m_out_len > 0) {
            PYBAMM_RUST_CALL(events_eval, *inputs[0], inputs[1], inputs[2], results[0], m_model);
        }
    }
};

/**
 * @brief RustSensExpression: forward-sensitivity evaluator backed by Rust FFI.
 *
 * Computes ∂f/∂p_i for every configured sensitivity parameter and writes the
 * result into `m_res[i]`. Constructed with `n_sens_params == 0` it acts as
 * an empty placeholder, matching the previous behaviour for non-sensitivity
 * solves.
 *
 * ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs; m_res[i]=∂f/∂p_i (length n_states)
 */
class RustSensExpression : public RustExpression {
public:
    /// One result slot per sensitivity parameter; sundials_functions.inl
    /// repoints them at the resvalS N_Vectors before each call.
    RustSensExpression(void* model, int n_states, int n_sens_params)
        : RustExpression(model, n_states, 3, n_sens_params),
          m_n_sens_params(n_sens_params),
          m_columns(static_cast<size_t>(n_states) * n_sens_params) {}

    void operator()() override {
        eval_columns(m_arg[0], m_arg[1], m_arg[2], m_res);
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        eval_columns(inputs[0], inputs[1], inputs[2], results);
    }

    // out_shape stays n_states (the per-parameter block), but a model without
    // sensitivities writes nothing at all.
    expr_int nnz() override { return m_n_sens_params == 0 ? 0 : m_out_len; }
    expr_int nnz_out() override { return nnz(); }

private:
    /*
     * `sens_eval_all` runs the tape's shared primal section once and then one
     * tangent-only sweep per parameter, where per-parameter `sens_eval` repeats
     * the primal every time (58% of the DFN tangent tape). It writes the
     * columns contiguously, so they are scattered out to the per-parameter
     * SUNDIALS buffers, which are separate N_Vectors.
     */
    void eval_columns(const sunrealtype* t, const sunrealtype* y,
                      const sunrealtype* inputs,
                      const std::vector<sunrealtype*>& results) {
        if (m_n_sens_params == 0) {
            return;
        }
        PYBAMM_RUST_CALL(sens_eval_all, *t, y, inputs, m_columns.data(), m_model);
        const size_t n_states = static_cast<size_t>(m_out_len);
        for (int i = 0; i < m_n_sens_params; i++) {
            std::memcpy(results[i], m_columns.data() + i * n_states,
                        n_states * sizeof(sunrealtype));
        }
    }

    int m_n_sens_params;
    std::vector<sunrealtype> m_columns;
};

/**
 * @brief RustOutputExpression: evaluates a single output variable via Rust FFI.
 *
 * IDAKLUSolverOpenMP iterates `functions->var_fcns` whenever output variables
 * are saved (save_outputs_only path), invoking the parameterized
 * `operator()(inputs, results)` overload with `inputs = {&t, y, inputs_data}`.
 * We forward straight into `rust_output_eval` for the configured `var_idx`.
 *
 * ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs; m_res[0]=output (length m_out_len)
 */
class RustOutputExpression : public RustExpression {
public:
    RustOutputExpression(void* model, int var_idx, int out_len)
        : RustExpression(model, out_len), m_var_idx(var_idx) {}

    void operator()() override {
        int written = 0;
        PYBAMM_RUST_CALL(output_eval, *m_arg[0], m_arg[1], m_arg[2], m_var_idx,
                         m_res[0], &written, m_model);
        check_written(written);
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        int written = 0;
        PYBAMM_RUST_CALL(output_eval, *inputs[0], inputs[1], inputs[2], m_var_idx,
                         results[0], &written, m_model);
        check_written(written);
    }

private:
    /* The caller sized the buffer from `m_out_len`, so a short write leaves the
       tail holding the previous step's values. */
    void check_written(int written) const {
        if (written != m_out_len) {
            throw std::runtime_error(
                std::string("pybammsolvers: Rust output variable ") +
                std::to_string(m_var_idx) + " wrote " + std::to_string(written) +
                " elements, expected " + std::to_string(m_out_len) + ".");
        }
    }

    int m_var_idx;
};

/**
 * @brief RustAlgResExpression: evaluates algebraic residuals via Rust FFI.
 *
 * Constructed with `n_alg == 0` it behaves like an empty placeholder so the
 * solver naturally falls back to the full-system IC path.
 */
class RustAlgResExpression : public RustExpression {
public:
    RustAlgResExpression(void* model, int n_alg)
        : RustExpression(model, n_alg) {}

    void operator()() override {
        if (m_out_len > 0) {
            PYBAMM_RUST_CALL(alg_res, *m_arg[0], m_arg[1], m_arg[2], m_res[0], m_model);
        }
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        if (m_out_len > 0) {
            PYBAMM_RUST_CALL(alg_res, *inputs[0], inputs[1], inputs[2], results[0], m_model);
        }
    }
};

/**
 * @brief RustAlgJacExpression: assembles algebraic Jacobians via Rust FFI.
 *
 * The output ordering follows the COO `(row, col)` metadata passed in at
 * construction time. With `m_nnz == 0` it acts as an empty placeholder.
 */
class RustAlgJacExpression : public RustExpression {
public:
    RustAlgJacExpression(void* model, int nnz,
                         std::vector<expr_int> rows,
                         std::vector<expr_int> cols)
        : RustExpression(model, nnz) {
        m_rows = std::move(rows);
        m_cols = std::move(cols);
    }

    void operator()() override {
        if (m_out_len > 0) {
            PYBAMM_RUST_CALL(alg_jac_assemble, *m_arg[0], m_arg[1], m_arg[2], m_res[0], m_model);
        }
    }

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        if (m_out_len > 0) {
            PYBAMM_RUST_CALL(alg_jac_assemble, *inputs[0], inputs[1], inputs[2], results[0], m_model);
        }
    }
};

/**
 * @brief Shared base for the standalone algebraic Newton solve adapters.
 *
 * The Newton solver calls with the convention `F(t, y_alg, [y_diff; inputs])`:
 * m_arg[0]=&t, m_arg[1]=y_alg, m_arg[2]=[y_diff; inputs]. The Rust FFI instead
 * expects the full state `g(t, y_full, inputs)`, so both adapters gather
 * y_diff and y_alg into a full-state buffer and forward the inputs slice that
 * follows the y_diff block.
 */
class RustNewtonExpression : public RustExpression {
public:
    RustNewtonExpression(void* model, int out_len, int n_rhs, int n_alg)
        : RustExpression(model, out_len), m_n_rhs(n_rhs), m_n_alg(n_alg),
          m_y_full(static_cast<size_t>(n_rhs) + n_alg, 0.0) {}

    // Declaring the two-argument overload here would otherwise hide the
    // zero-argument one that the subclasses define and this one delegates to.
    using Expression::operator();

    void operator()(const std::vector<sunrealtype*>& inputs,
                    const std::vector<sunrealtype*>& results) override {
        m_arg.assign(inputs.begin(), inputs.end());
        m_res.assign(results.begin(), results.end());
        (*this)();
    }

protected:
    /// Gather [y_diff; y_alg] into m_y_full; returns the trailing inputs slice.
    const double* gather_full_state() {
        std::memcpy(m_y_full.data(), m_arg[2], m_n_rhs * sizeof(double));            // y_diff
        std::memcpy(m_y_full.data() + m_n_rhs, m_arg[1], m_n_alg * sizeof(double));  // y_alg
        return m_arg[2] + m_n_rhs;
    }

    int m_n_rhs;
    int m_n_alg;
    std::vector<double> m_y_full;
};

/**
 * @brief RustNewtonResExpression: Newton residual adapter.
 */
class RustNewtonResExpression : public RustNewtonExpression {
public:
    RustNewtonResExpression(void* model, int n_rhs, int n_alg)
        : RustNewtonExpression(model, n_alg, n_rhs, n_alg) {}

    void operator()() override {
        const double* inputs = gather_full_state();
        PYBAMM_RUST_CALL(alg_res, *m_arg[0], m_y_full.data(), inputs, m_res[0], m_model);
    }
};

/**
 * @brief RustNewtonJacExpression: Newton Jacobian adapter.
 *
 * `alg_jac_assemble` fills the output in the COO order of the (rows, cols)
 * sparsity passed at construction, so BuildSparseResources consumes it
 * unchanged.
 */
class RustNewtonJacExpression : public RustNewtonExpression {
public:
    RustNewtonJacExpression(void* model, int n_rhs, int n_alg, int nnz,
                            std::vector<expr_int> rows,
                            std::vector<expr_int> cols)
        : RustNewtonExpression(model, nnz, n_rhs, n_alg) {
        m_rows = std::move(rows);
        m_cols = std::move(cols);
    }

    void operator()() override {
        const double* inputs = gather_full_state();
        PYBAMM_RUST_CALL(alg_jac_assemble, *m_arg[0], m_y_full.data(), inputs, m_res[0], m_model);
    }
};

/**
 * @brief RustFunctions: ExpressionSet implementation for Rust-based models
 *
 * Wraps Rust FFI functions in the Expression interface expected by IDAKLU solver.
 *
 * Uses direct members (not unique_ptr) to ensure valid pointers can be passed
 * to the base class constructor. In C++, base classes are initialized before
 * members, but taking &member gives a valid address that will be populated
 * by the time it's dereferenced during actual solving.
 */
class RustFunctions : public ExpressionSet<RustRhsExpression> {
public:
    /**
     * @brief Construct RustFunctions from a Rust model handle
     *
     * @param rust_model Opaque pointer to Rust Model
     * @param n_states Number of state variables
     * @param n_inputs Number of input parameters
     * @param n_sens_params Number of forward-sensitivity parameters
     * @param n_alg Number of algebraic states
     * @param n_events Number of event functions
     * @param nnz Number of non-zeros in Jacobian
     * @param colptrs CSC column pointers
     * @param rowvals CSC row indices
     * @param alg_jac_nnz Number of non-zeros in the algebraic Jacobian
     * @param alg_rowvals Algebraic Jacobian COO row indices
     * @param alg_colvals Algebraic Jacobian COO column indices
     * @param output_lens Output length of each output variable
     * @param options Solver setup options
     */
    RustFunctions(
        void* rust_model,
        int n_states,
        int n_inputs,
        int n_sens_params,
        int n_alg,
        int n_events,
        int nnz,
        const std::vector<int64_t>& colptrs,
        const std::vector<int64_t>& rowvals,
        int alg_jac_nnz,
        const std::vector<int64_t>& alg_rowvals,
        const std::vector<int64_t>& alg_colvals,
        const std::vector<int>& output_lens,
        const SetupOptions& options
    ) :
        tmp_state_vector(n_states),
        tmp_sparse_jacobian_data(nnz),
        // Expression members - use direct members, not unique_ptr. Each owns
        // its COO index vectors, so the conversions are passed as temporaries.
        m_rhs(rust_model, n_states),
        m_jac(rust_model, nnz,
              convert_to_expr_int(rowvals),
              csc_rowvals_to_coo_cols(colptrs, rowvals, nnz)),
        m_jac_action(rust_model, n_states),
        m_mass_action(rust_model, n_states),
        m_events(rust_model,
                 n_events > 0 ? PYBAMM_RUST_VALUE(total_event_len, rust_model) : 0),
        m_sens(rust_model, n_states, n_sens_params),
        m_alg_res(rust_model, n_alg),
        m_alg_jac(rust_model, alg_jac_nnz,
                  convert_to_expr_int(alg_rowvals),
                  convert_to_expr_int(alg_colvals)),
        // Base class - addresses of direct members are valid even before init
        ExpressionSet<RustRhsExpression>(
            static_cast<Expression*>(&m_rhs),
            static_cast<Expression*>(&m_jac),
            nnz,
            0,  // jac_bandwidth_lower (not used for sparse)
            0,  // jac_bandwidth_upper (not used for sparse)
            np_array_int(),  // empty, we store directly
            np_array_int(),  // empty, we store directly
            n_inputs,  // inputs_length
            static_cast<Expression*>(&m_jac_action),
            static_cast<Expression*>(&m_mass_action),
            static_cast<Expression*>(&m_sens),
            static_cast<Expression*>(&m_events),
            n_states,
            n_events,  // number of events
            n_sens_params,  // n_parameters (forward sensitivities)
            options,
            static_cast<Expression*>(&m_alg_res),
            static_cast<Expression*>(&m_alg_jac)
        )
    {
        // Retain the Rust model handle for native output-sensitivity projection.
        m_model = rust_model;
        // Store sparsity pattern in base class members
        jac_times_cjmass_colptrs = colptrs;
        jac_times_cjmass_rowvals = rowvals;
        // Allocate the inputs vector that the SUNDIALS callback layer
        // populates each step (sundials_functions.inl sets m_arg[2] to
        // inputs.data()). The base ExpressionSet ctor takes inputs_length
        // but doesn't size the vector itself; we mirror what CasadiFunctions
        // does in its own ctor body.
        inputs.resize(n_inputs);

        // Construct one RustOutputExpression per requested output variable.
        // unique_ptr keeps the Expression heap-stable across vector growth so
        // the raw pointers we hand to ExpressionSet::var_fcns stay valid.
        m_output_fcns.reserve(output_lens.size());
        for (int i = 0; i < static_cast<int>(output_lens.size()); ++i) {
            m_output_fcns.emplace_back(
                std::make_unique<RustOutputExpression>(
                    rust_model, i, output_lens[i]
                )
            );
            var_fcns.push_back(static_cast<Expression*>(m_output_fcns.back().get()));
        }
    }

    sunrealtype* get_tmp_state_vector() override {
        return tmp_state_vector.data();
    }

    sunrealtype* get_tmp_sparse_jacobian_data() override {
        return tmp_sparse_jacobian_data.data();
    }

    // Compile-time marker: this ExprSet projects output sensitivities natively,
    // so the solver takes the native branch instead of the CasADi sparse loop.
    static constexpr bool kNativeOutputSensitivities = true;

    // Compile-time marker: output variables can be evaluated over a batch of
    // trajectory points in one call, amortising interpreter dispatch.
    static constexpr bool kNativeBatchedOutputs = true;

    // Batch-evaluate every output variable over k staged points.
    // ts: [k]; ys: [k * n_states], each point contiguous;
    // out: [k * total_output_len], each point's stacked outputs contiguous.
    void eval_outputs_batch(
        const sunrealtype* ts, const sunrealtype* ys, int k, sunrealtype* out) {
        PYBAMM_RUST_CALL(output_eval_batch, ts, ys, k, inputs.data(), out, m_model);
    }

    // Project state sensitivities onto output-variable sensitivities via Rust FFI.
    // yS_flat: [n_sens_params * n_states]; out: [n_sens_params * total_output_len].
    void project_output_sensitivities(
        double t, sunrealtype* y, sunrealtype* inputs,
        const sunrealtype* yS_flat, sunrealtype* out) {
        PYBAMM_RUST_CALL(output_sens_project, t, y, inputs, yS_flat, out, m_model);
    }

private:
    // Rust model handle, retained for native output-sensitivity projection.
    void* m_model = nullptr;

    std::vector<sunrealtype> tmp_state_vector;
    std::vector<sunrealtype> tmp_sparse_jacobian_data;

    // Expression members - direct members (not unique_ptr) so that addresses
    // can be safely passed to base class constructor
    RustRhsExpression m_rhs;
    RustJacExpression m_jac;
    RustJacActionExpression m_jac_action;
    RustMassActionExpression m_mass_action;
    RustEventsExpression m_events;
    RustSensExpression m_sens;
    RustAlgResExpression m_alg_res;
    RustAlgJacExpression m_alg_jac;
    // Output-variable expressions, one per configured output. Stored via
    // unique_ptr so the Expression* references in `var_fcns` (base class)
    // remain valid even if the vector reallocates.
    std::vector<std::unique_ptr<RustOutputExpression>> m_output_fcns;

    /**
     * @brief Convert int64_t vector to expr_int vector
     */
    static std::vector<expr_int> convert_to_expr_int(const std::vector<int64_t>& v) {
        std::vector<expr_int> result;
        result.reserve(v.size());
        for (const auto& val : v) {
            result.push_back(static_cast<expr_int>(val));
        }
        return result;
    }

    /**
     * @brief Convert CSC rowvals to COO column indices
     *
     * In CSC format, colptrs[j] to colptrs[j+1] gives the range of entries in column j.
     * For COO format, we need the column index for each entry.
     */
    static std::vector<expr_int> csc_rowvals_to_coo_cols(
        const std::vector<int64_t>& colptrs,
        const std::vector<int64_t>& rowvals,
        int nnz
    ) {
        std::vector<expr_int> cols(nnz);
        int n_cols = static_cast<int>(colptrs.size()) - 1;
        for (int col = 0; col < n_cols; ++col) {
            for (int64_t k = colptrs[col]; k < colptrs[col + 1]; ++k) {
                cols[k] = static_cast<expr_int>(col);
            }
        }
        return cols;
    }
};

#endif // PYBAMM_IDAKLU_RUST_FUNCTIONS_HPP

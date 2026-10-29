#ifndef PYBAMM_IDAKLU_RUST_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_RUST_FUNCTIONS_HPP

#include "../Base/Expression.hpp"
#include "../Base/ExpressionSet.hpp"
#include "../../Options.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "pybamm_rust_ffi.h"

namespace pybamm_rust_detail {

/* Rust models reach C++ as the integer address Python holds, so the cast is unavoidable. */
inline void* model_from_address(std::uintptr_t address) {
    return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

/* Throw unless `given`, a count C++ sizes buffers from, equals the Rust model's `rust`. */
inline void check_dim(const char* name, int64_t given, int64_t rust) {
    if (given != rust) {
        throw std::invalid_argument(
            std::string("pybammsolvers: ") + name + " is " + std::to_string(given) +
            " but the Rust model has " + std::to_string(rust) + ".");
    }
}

/* Throw unless COO `(rows, cols)` holds `nnz` entries inside an `n_rows` x `n_cols` matrix. */
inline void check_coo_pattern(const std::vector<expr_int>& rows,
                              const std::vector<expr_int>& cols,
                              int nnz, int n_rows, int n_cols) {
    if (rows.size() != static_cast<size_t>(nnz) || cols.size() != static_cast<size_t>(nnz)) {
        throw std::invalid_argument(
            "pybammsolvers: the algebraic Jacobian pattern's rows and cols have lengths " +
            std::to_string(rows.size()) + " and " + std::to_string(cols.size()) +
            ", but nnz is " + std::to_string(nnz) + ".");
    }
    for (size_t k = 0; k < rows.size(); ++k) {
        if (rows[k] < 0 || rows[k] >= n_rows || cols[k] < 0 || cols[k] >= n_cols) {
            throw std::invalid_argument(
                "pybammsolvers: algebraic Jacobian entry " + std::to_string(k) + " at (" +
                std::to_string(rows[k]) + ", " + std::to_string(cols[k]) + ") lies outside its " +
                std::to_string(n_rows) + " x " + std::to_string(n_cols) + " matrix.");
        }
    }
}

}  // namespace pybamm_rust_detail

/** Common base for the Rust FFI adapters; `m_rows`/`m_cols` stay empty unless emitting COO. */
class RustExpression : public Expression {
public:
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

/** RHS; ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_res[0]=output */
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

/** Jacobian; ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_arg[3]=cj, m_res[0]=jac_data */
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
        // cj comes from the SUNDIALS callback via m_arg[3], not from `inputs`.
        PYBAMM_RUST_CALL(jac_assemble, *inputs[0], inputs[1], inputs[2], *m_arg[3], results[0], m_model);
    }
};

/** Jacobian action; ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_arg[3]=v, m_res[0]=Jv */
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

/** Mass-matrix action; ABI: m_arg[0]=v, m_res[0]=Mv */
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

/** Concatenated events; ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs, m_res[0]=output */
class RustEventsExpression : public RustExpression {
public:
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

/** Forward sensitivities; ABI: m_arg[0]=t, m_arg[1]=y, m_arg[2]=inputs; m_res[i]=∂f/∂p_i */
class RustSensExpression : public RustExpression {
public:
    /// One result slot per parameter, repointed at the resvalS N_Vectors before each call.
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

    // out_shape stays the per-parameter block, but without sensitivities nothing is written.
    expr_int nnz() override { return m_n_sens_params == 0 ? 0 : m_out_len; }
    expr_int nnz_out() override { return nnz(); }

private:
    // `sens_eval_all` shares one primal sweep; its contiguous columns go to separate N_Vectors.
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

/** Size of one output variable; RustFunctions evaluates every output in one batched call. */
class RustOutputExpression : public RustExpression {
public:
    explicit RustOutputExpression(int out_len)
        : RustExpression(nullptr, out_len) {}

    void operator()() override { unreachable(); }

    void operator()(const std::vector<sunrealtype*>&,
                    const std::vector<sunrealtype*>&) override {
        unreachable();
    }

private:
    [[noreturn]] static void unreachable() {
        throw std::logic_error(
            "pybammsolvers: unreachable: Rust outputs are evaluated in batch.");
    }
};

/** Algebraic residual; with `n_alg == 0` the solver falls back to the full-system IC path. */
class RustAlgResExpression : public RustExpression {
public:
    RustAlgResExpression(void* model, int n_alg)
        : RustExpression(model, n_alg) {
        // Rust writes its own residual length into the IC solver's `n_alg` buffer.
        if (n_alg != 0) {
            pybamm_rust_detail::check_dim("n_alg", n_alg, PYBAMM_RUST_VALUE(n_algebraic, model));
        }
    }

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

/** Algebraic Jacobian, in the COO order of the `(row, col)` passed at construction. */
class RustAlgJacExpression : public RustExpression {
public:
    RustAlgJacExpression(void* model, int nnz,
                         std::vector<expr_int> rows,
                         std::vector<expr_int> cols)
        : RustExpression(model, nnz) {
        m_rows = std::move(rows);
        m_cols = std::move(cols);
        // Rust writes its own non-zero count, and the IC solver reads the pattern at `nnz`; rows
        // index the algebraic block and columns the full state.
        pybamm_rust_detail::check_coo_pattern(
            m_rows, m_cols, nnz, PYBAMM_RUST_VALUE(n_algebraic, model),
            PYBAMM_RUST_VALUE(n_states, model));
        if (nnz != 0) {
            pybamm_rust_detail::check_dim(
                "alg_jac_nnz", nnz, PYBAMM_RUST_VALUE(alg_jac_nnz, model));
        }
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
 * Base for the Newton adapters: the solver passes `F(t, y_alg, [y_diff; inputs])` but the Rust
 * FFI expects `g(t, y_full, inputs)`, so `[y_diff; y_alg]` is gathered into a full-state buffer.
 */
class RustNewtonExpression : public RustExpression {
public:
    RustNewtonExpression(void* model, int out_len, int n_rhs, int n_alg)
        : RustExpression(model, out_len), m_n_rhs(n_rhs),
          m_n_alg(checked_split(model, n_rhs, n_alg)),
          m_y_full(static_cast<size_t>(n_rhs) + n_alg, 0.0) {}

    // Unhide the zero-argument overload the subclasses define and this one delegates to.
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
        std::memcpy(m_y_full.data(), m_arg[2], m_n_rhs * sizeof(double));
        std::memcpy(m_y_full.data() + m_n_rhs, m_arg[1], m_n_alg * sizeof(double));
        return m_arg[2] + m_n_rhs;
    }

    int m_n_rhs;
    int m_n_alg;
    std::vector<double> m_y_full;

private:
    // Rust reads `m_y_full` at its state count and writes its residual length, so both are
    // checked before `m_y_full` is sized.
    static int checked_split(void* model, int n_rhs, int n_alg) {
        pybamm_rust_detail::check_dim("n_alg", n_alg, PYBAMM_RUST_VALUE(n_algebraic, model));
        pybamm_rust_detail::check_dim("n_rhs + n_alg", static_cast<int64_t>(n_rhs) + n_alg,
                                      PYBAMM_RUST_VALUE(n_states, model));
        return n_alg;
    }
};

class RustNewtonResExpression : public RustNewtonExpression {
public:
    RustNewtonResExpression(void* model, int n_rhs, int n_alg)
        : RustNewtonExpression(model, n_alg, n_rhs, n_alg) {}

    void operator()() override {
        const double* inputs = gather_full_state();
        PYBAMM_RUST_CALL(alg_res, *m_arg[0], m_y_full.data(), inputs, m_res[0], m_model);
    }
};

/** Fills the COO order of `(rows, cols)`, which BuildSparseResources consumes unchanged. */
class RustNewtonJacExpression : public RustNewtonExpression {
public:
    RustNewtonJacExpression(void* model, int n_rhs, int n_alg, int nnz,
                            std::vector<expr_int> rows,
                            std::vector<expr_int> cols)
        : RustNewtonExpression(model, nnz, n_rhs, n_alg) {
        m_rows = std::move(rows);
        m_cols = std::move(cols);
        // The Newton system indexes rows and columns within the algebraic block, and Rust
        // writes its own non-zero count.
        pybamm_rust_detail::check_coo_pattern(m_rows, m_cols, nnz, n_alg, n_alg);
        pybamm_rust_detail::check_dim("alg_jac_nnz", nnz, PYBAMM_RUST_VALUE(alg_jac_nnz, model));
    }

    void operator()() override {
        const double* inputs = gather_full_state();
        PYBAMM_RUST_CALL(alg_jac_assemble, *m_arg[0], m_y_full.data(), inputs, m_res[0], m_model);
    }
};

/**
 * ExpressionSet for Rust-backed models. Expressions are direct members, not unique_ptr: their
 * addresses are valid for the base constructor even though members initialise after it.
 */
class RustFunctions : public ExpressionSet<RustRhsExpression> {
public:
    /** `colptrs`/`rowvals` are the Jacobian's CSC pattern; `alg_rowvals`/`alg_colvals` are COO. */
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
        m_n_sens_params(n_sens_params),
        tmp_state_vector(n_states),
        tmp_sparse_jacobian_data(nnz),
        m_sens_directions(static_cast<size_t>(n_states) * n_sens_params),
        m_sens_jac_actions(m_sens_directions.size()),
        m_rhs(rust_model, n_states),
        m_jac(rust_model, nnz,
              convert_to_expr_int(rowvals),
              csc_rowvals_to_coo_cols(colptrs, rowvals, nnz)),
        m_jac_action(rust_model, n_states),
        m_mass_action(rust_model, n_states),
        m_events(rust_model, n_events),
        m_sens(rust_model, n_states, n_sens_params),
        m_alg_res(rust_model, n_alg),
        m_alg_jac(rust_model, alg_jac_nnz,
                  convert_to_expr_int(alg_rowvals),
                  convert_to_expr_int(alg_colvals)),
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
            // Checked as the base initialises, before any member is sized from these counts.
            checked_dims(rust_model, n_states, n_inputs, nnz, n_sens_params),
            checked_event_count(rust_model, n_events),
            n_sens_params,  // n_parameters (forward sensitivities)
            options,
            static_cast<Expression*>(&m_alg_res),
            static_cast<Expression*>(&m_alg_jac)
        )
    {
        m_model = rust_model;
        jac_times_cjmass_colptrs = colptrs;
        jac_times_cjmass_rowvals = rowvals;
        // The base ctor takes inputs_length but does not size `inputs` itself.
        inputs.resize(n_inputs);

        check_output_lens(rust_model, output_lens);
        m_output_fcns.reserve(output_lens.size());
        for (int i = 0; i < static_cast<int>(output_lens.size()); ++i) {
            m_output_fcns.emplace_back(std::make_unique<RustOutputExpression>(output_lens[i]));
            var_fcns.push_back(static_cast<Expression*>(m_output_fcns.back().get()));
        }
    }

    sunrealtype* get_tmp_state_vector() override {
        return tmp_state_vector.data();
    }

    sunrealtype* get_tmp_sparse_jacobian_data() override {
        return tmp_sparse_jacobian_data.data();
    }

    static constexpr bool kNativeOutputSensitivities = true;
    static constexpr bool kNativeBatchedOutputs = true;
    static constexpr bool kNativeBatchedJacActions = true;

    // J·yS_i for every sensitivity parameter in one call, which shares a single primal pass.
    const sunrealtype* jac_action_sensitivities(
        sunrealtype t, const sunrealtype* y, const N_Vector* yS) {
        const size_t n_states = static_cast<size_t>(number_of_states);
        for (int i = 0; i < m_n_sens_params; ++i) {
            const sunrealtype* direction = N_VGetArrayPointer(yS[i]);
            std::copy(direction, direction + n_states, m_sens_directions.begin() + i * n_states);
        }
        PYBAMM_RUST_CALL(jac_action_batch, t, y, inputs.data(), m_n_sens_params,
                         m_sens_directions.data(), m_sens_jac_actions.data(), m_model);
        return m_sens_jac_actions.data();
    }

    // ts: [k]; ys: [k * n_states] and out: [k * total_output_len], each point contiguous.
    void eval_outputs_batch(
        const sunrealtype* ts, const sunrealtype* ys, int k, sunrealtype* out) {
        PYBAMM_RUST_CALL(output_eval_batch, ts, ys, k, inputs.data(), out, m_model);
    }

    // yS_flat: [n_sens_params * n_states]; out: [n_sens_params * total_output_len].
    void project_output_sensitivities(
        double t, sunrealtype* y, sunrealtype* inputs,
        const sunrealtype* yS_flat, sunrealtype* out) {
        PYBAMM_RUST_CALL(output_sens_project, t, y, inputs, yS_flat, out, m_model);
    }

private:
    void* m_model = nullptr;
    int m_n_sens_params;

    std::vector<sunrealtype> tmp_state_vector;
    std::vector<sunrealtype> tmp_sparse_jacobian_data;
    // n_sens_params contiguous n_states blocks, as jac_action_batch reads and writes them.
    std::vector<sunrealtype> m_sens_directions;
    std::vector<sunrealtype> m_sens_jac_actions;

    RustRhsExpression m_rhs;
    RustJacExpression m_jac;
    RustJacActionExpression m_jac_action;
    RustMassActionExpression m_mass_action;
    RustEventsExpression m_events;
    RustSensExpression m_sens;
    RustAlgResExpression m_alg_res;
    RustAlgJacExpression m_alg_jac;
    // unique_ptr keeps each Expression heap-stable, so the pointers in `var_fcns` stay valid.
    std::vector<std::unique_ptr<RustOutputExpression>> m_output_fcns;

    // Rust borrows every buffer sized from these at its own counts, so a mismatch would overrun.
    static int checked_dims(void* rust_model, int n_states, int n_inputs, int nnz,
                            int n_sens_params) {
        using pybamm_rust_detail::check_dim;
        check_dim("number_of_states", n_states, PYBAMM_RUST_VALUE(n_states, rust_model));
        check_dim("number_of_inputs", n_inputs, PYBAMM_RUST_VALUE(n_inputs, rust_model));
        check_dim("jac_nnz", nnz, PYBAMM_RUST_VALUE(jac_nnz, rust_model));
        check_dim("number_of_sens_params", n_sens_params,
                  PYBAMM_RUST_VALUE(n_sens_params, rust_model));
        return n_states;
    }

    // IDA sizes its root buffer by `n_events`, so each Rust event must be one scalar.
    static int checked_event_count(void* rust_model, int n_events) {
        const int total_event_len = PYBAMM_RUST_VALUE(total_event_len, rust_model);
        if (total_event_len != n_events) {
            throw std::invalid_argument(
                "pybammsolvers: number_of_events is " + std::to_string(n_events) +
                " but the Rust model's events have " + std::to_string(total_event_len) +
                " components; every termination event must be scalar.");
        }
        return n_events;
    }

    // Output buffers are sized from `output_lens`, so each must match what Rust writes.
    static void check_output_lens(void* rust_model, const std::vector<int>& output_lens) {
        const int n_outputs = PYBAMM_RUST_VALUE(n_outputs, rust_model);
        if (static_cast<int>(output_lens.size()) != n_outputs) {
            throw std::invalid_argument(
                "pybammsolvers: " + std::to_string(output_lens.size()) +
                " output lengths were given for a Rust model with " +
                std::to_string(n_outputs) + " output variables.");
        }
        for (int i = 0; i < n_outputs; ++i) {
            const int len = PYBAMM_RUST_VALUE(output_len, rust_model, i);
            if (output_lens[i] != len) {
                throw std::invalid_argument(
                    "pybammsolvers: output_lens[" + std::to_string(i) + "] is " +
                    std::to_string(output_lens[i]) + " but Rust output variable " +
                    std::to_string(i) + " has " + std::to_string(len) + " elements.");
            }
        }
    }

    static std::vector<expr_int> convert_to_expr_int(const std::vector<int64_t>& v) {
        std::vector<expr_int> result;
        result.reserve(v.size());
        for (const auto& val : v) {
            result.push_back(static_cast<expr_int>(val));
        }
        return result;
    }

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

//! Crate-wide error type for recoverable failures that should surface as
//! ordinary `Result::Err` values (and, at the Python boundary, as `ValueError`)
//! rather than panics.
//!
//! Panics remain reserved for broken internal invariants; anything driven by
//! caller-supplied arguments or externally-constructed data is reported here.

/// Errors returned by fallible `pybamm-core` entry points.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// The evaluation time grid was empty; there is nothing to integrate.
    #[error("t_eval must not be empty")]
    EmptyTimePoints,

    /// The evaluation time grid decreased at some point, or held a NaN.
    #[error("t_eval must increase monotonically, but t_eval[{index}] = {got} follows {previous}")]
    UnsortedTimePoints {
        index: usize,
        got: f64,
        previous: f64,
    },

    /// The initial state vector length did not match the model's state count.
    #[error("y0 has length {got} but the model has {expected} states")]
    Y0Length { got: usize, expected: usize },

    /// The packed input array length did not match the model's parameter width.
    #[error("inputs has length {got} but the model expects {expected} packed parameter value(s)")]
    InputsLength { got: usize, expected: usize },

    /// The absolute-tolerance vector length did not match the model's state count.
    #[error("atol has length {got} but the model has {expected} states")]
    AtolLength { got: usize, expected: usize },

    /// A sensitivity solve was requested of a model compiled without any.
    #[error("no sensitivity parameters were requested when the model was compiled")]
    NoSensitivityParams,

    /// An output-variable solve was requested of a model carrying none.
    #[error("no output variables were registered when the model was compiled")]
    NoOutputVariables,

    /// The `dy0/dp` seed was neither empty nor `n_states x n_sens_params`.
    #[error(
        "y0_sens has length {got} but must be empty or {expected} (n_states x n_sens_params, column-major)"
    )]
    Y0SensLength { got: usize, expected: usize },

    /// A batch supplied a different number of initial states and input vectors.
    #[error(
        "a batch needs one entry per input set, but got {y0} initial state(s) and {inputs} input vector(s)"
    )]
    BatchWidths { y0: usize, inputs: usize },

    /// A batch supplied a different number of `dy0/dp` seeds than input sets.
    #[error("a batch of {expected} input set(s) needs {expected} dy0/dp seed(s), but got {got}")]
    BatchSensWidth { got: usize, expected: usize },

    /// The sensitivity atol factor was not a finite, strictly positive number.
    #[error("sens_atol_factor must be finite and > 0, got {got}")]
    SensAtolFactor { got: f64 },

    /// A solver option held a value diffsol would accept and then misbehave on.
    #[error("solver option {name} must be finite and > 0, got {got}")]
    SolverOption { name: String, got: f64 },

    /// Both the error-controlled sensitivity solve and its relaxed retry failed.
    #[error(
        "sensitivity solve failed under error control ({controlled}); the retry with sensitivities excluded from error control also failed ({relaxed})"
    )]
    SensRetryFailed { controlled: String, relaxed: String },

    /// A CSR matrix supplied at the boundary violated its structural
    /// invariants (indptr length/monotonicity, indices/data lengths, or a
    /// column index out of range).
    #[error("invalid CSR matrix: {0}")]
    Csr(String),

    /// A dense array's data length did not match its declared shape.
    #[error("invalid array: {0}")]
    Array(String),

    /// An interpolation table violated its invariants (non-empty, matching
    /// lengths, strictly increasing finite knots, coefficient counts).
    #[error("invalid interpolant: {0}")]
    Interpolant(String),

    /// An underlying diffsol integration error.
    #[cfg(feature = "diffsol")]
    #[error(transparent)]
    Diffsol(#[from] diffsol::DiffsolError),
}

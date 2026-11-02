//! Crate-wide error type for failures driven by caller-supplied data.
//!
//! Panics remain reserved for broken internal invariants.

/// Errors returned by fallible `pybamm-core` entry points.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// A half-open `[start, end)` node extent ran backwards.
    #[error("{node}: start ({start}) must not exceed end ({end})")]
    InvertedExtent {
        node: &'static str,
        start: usize,
        end: usize,
    },

    /// A deserialised node named a child at or after itself, which could cycle or overrun.
    #[error("node {node} names child {child}, which is not allocated before it")]
    ChildNotBeforeParent { node: usize, child: usize },

    #[error("root {root} is out of range for an arena of {len} nodes")]
    RootOutOfRange { root: usize, len: usize },

    #[error("Conditional requires at least one branch")]
    EmptyConditional,

    #[error("InterpolantNd: expected {axes} children (one per axis), got {children}")]
    InterpolantNdArity { axes: usize, children: usize },

    /// A sensitivity index names no registered input parameter.
    #[error(
        "sensitivity index {index} names no input parameter; the model registers {n_registered}"
    )]
    SensParamIndex { index: usize, n_registered: usize },

    /// A sensitivity was requested for a vector-valued input; a column is one scalar direction.
    #[error(
        "sensitivity index {index} names an input of width {width}; only a width-1 input can be differentiated"
    )]
    SensParamVector { index: usize, width: usize },

    #[error("sensitivity index {index} was requested more than once")]
    SensParamRepeated { index: usize },

    /// The declared state count was below the extent the expression reads.
    #[error("n_states={declared} is below the expression's state extent {extent}")]
    StateCountTooSmall { declared: usize, extent: usize },

    /// The declared state count was past the `u32` range tape state indices are stored in.
    #[error("n_states={declared} exceeds the largest state count a tape can index, {max}")]
    StateCountTooLarge { declared: usize, max: u32 },

    /// The residual's length differed from the declared state count.
    #[error("the residual has {got} rows but n_states={n_states}")]
    ResidualLength { got: usize, n_states: usize },

    /// The mass matrix was not `n_states x n_states`.
    #[error("the mass matrix is {rows}x{cols} but n_states={n_states}")]
    MassMatrixShape {
        rows: usize,
        cols: usize,
        n_states: usize,
    },

    /// The declared packed input width was below what the graph's inputs occupy.
    ///
    /// The FFI borrows the declared width from the caller's pointer, so raising it would over-read.
    #[error(
        "n_inputs={declared} is below the packed width {extent} of the inputs the graph registers"
    )]
    InputWidthTooSmall { declared: usize, extent: usize },

    /// A name was registered twice at different widths, which would shift every later offset.
    #[error(
        "input parameter '{name}' re-registered with width {width}, previously registered with width {registered}"
    )]
    ParamWidthConflict {
        name: String,
        width: usize,
        registered: usize,
    },

    /// Two `InputParameter` nodes describe one registration index differently.
    #[error(
        "input parameter index {index} carries two slots: '{name}' at offset {offset} width {width}, and '{other_name}' at offset {other_offset} width {other_width}"
    )]
    ParamSlotConflict {
        index: usize,
        name: String,
        offset: usize,
        width: usize,
        other_name: String,
        other_offset: usize,
        other_width: usize,
    },

    /// One name registered at two indices.
    #[error("input parameter '{name}' is registered at both index {index} and index {other_index}")]
    ParamNameConflict {
        name: String,
        index: usize,
        other_index: usize,
    },

    /// Two parameters claim the same packed element.
    #[error(
        "input parameters '{name}' (offset {offset}, width {width}) and '{other_name}' (offset {other_offset}, width {other_width}) overlap in the packed inputs"
    )]
    ParamSlotOverlap {
        name: String,
        offset: usize,
        width: usize,
        other_name: String,
        other_offset: usize,
        other_width: usize,
    },

    /// A parameter's packed slot ran past the `u32` range packed offsets are stored in.
    #[error(
        "input parameter '{name}' (offset {offset}, width {width}) runs past the packed-input index range"
    )]
    ParamSlotRange {
        name: String,
        offset: usize,
        width: usize,
    },

    /// A parameter node's registration index reached past its arena, which no registration yields.
    #[error("input parameter index {index} is out of range for an arena of {n_nodes} nodes")]
    ParamIndexRange { index: usize, n_nodes: usize },

    /// An observable's graph puts a parameter the model registered at another packed slot.
    #[error(
        "input parameter index {index} sits at another packed slot in the observables' graph than in the model's"
    )]
    ObservableParamSlot { index: usize },

    #[error("algebraic variable index {index} is out of range for n_states={n_states}")]
    AlgebraicVariableIndex { index: usize, n_states: usize },

    /// An algebraic block's variable indices repeated or went backwards.
    #[error(
        "algebraic variable indices must be strictly ascending, but {index} follows {previous}"
    )]
    AlgebraicVariableOrder { previous: usize, index: usize },

    /// An algebraic block named a state the mass matrix makes differential.
    #[error(
        "algebraic variable index {index} names a differential state; its mass-matrix column is non-zero"
    )]
    AlgebraicVariableDifferential { index: usize },

    #[error(
        "algebraic block has {rows} residual rows but {columns} variable columns; it must be square"
    )]
    AlgebraicBlockNotSquare { rows: usize, columns: usize },

    #[error("invalid CSR matrix: {0}")]
    Csr(String),

    #[error("invalid array: {0}")]
    Array(String),

    #[error("invalid interpolant: {0}")]
    Interpolant(String),
}

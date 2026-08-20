//! Expression compiler and evaluator for `PyBaMM` models, filling the role
//! `CasADi` plays on `PyBaMM`'s other backends.
//!
//! Python hands over a discretised model as an expression DAG and this crate
//! turns it into flat instruction tapes it can evaluate, differentiate and
//! solve. The stages, each a module below:
//!
//! 1. **Build**: the bindings allocate [`Node`]s into an [`Arena`]; a node is
//!    referenced by a `u32` [`NodeId`], so sharing a subexpression is repeating
//!    an id.
//! 2. **Rewrite**: `simplify` folds identities and runs CSE/DCE,
//!    `zero_propagate` proves subtrees identically zero.
//! 3. **Differentiate**: `tangent` emits forward-mode JVP DAGs; `adjoint`
//!    fills one wide Jacobian row from a single backward pass instead.
//! 4. **Lower**: [`TypedIr::from_arena`] flattens a DAG into slot-addressed
//!    [`Instruction`]s with constants interned in a [`ConstPool`].
//! 5. **Evaluate**: [`CompiledExpr`] interprets a tape against a
//!    caller-supplied scratch buffer, one time point at a time or `k` lanes
//!    at once (`eval`, `eval_batch`).
//!
//! [`CompiledModel`] assembles those pieces into a DAE `M y' = f(t, y; p)`
//! carrying a symbolic `df/dy`, its sparsity pattern and a column coloring, and
//! two consumers drive it: `solver` runs diffsol in-process, while `ffi`
//! exposes a C ABI for the IDAKLU solver to call.
//!
//! Nothing here is thread-confined by construction, but evaluation writes into
//! scratch buffers, so a [`Workspace`] belongs to one solve at a time.
//! [`CompiledModel`] is immutable and shared via `Arc`; [`ModelEvaluator`]
//! pairs one with an owned [`Workspace`] for callers that want a single
//! `&mut self` handle instead, adding only what owning the workspace buys and
//! dereferencing to the model for the rest.

#![deny(unused_must_use)]

pub mod adjoint;
pub mod arena;
pub mod branch_regions;
pub mod coloring;
pub mod const_entries;
pub mod error;
pub mod eval;
pub mod eval_batch;
pub mod ffi;
pub mod ir;
pub mod jacobian;
pub mod model;
pub mod node;
pub mod observable;
mod row_extract;
pub mod simplify;
#[cfg(feature = "serialize")]
pub mod snapshot;
pub mod sparsity;
pub mod tangent;
pub mod tangent_batch;
pub mod zero_propagate;

#[cfg(feature = "diffsol")]
pub mod solver;

pub use arena::{Arena, NodeId, NodeMap, StateUsage, scan_state_usage};
pub use branch_regions::{
    BranchLabel, Ownership, RegionGroup, RegionSchedule, active_branch, owner_sets,
    privatise_conditionals, schedule_regions, schedule_regions_partitioned,
};
pub use coloring::{ColumnColoring, color_columns};
pub use error::CoreError;
pub use eval::{CompiledExpr, PrimalCache, TangentInputs};
pub use eval_batch::BatchEvalError;
#[cfg(feature = "profile")]
pub use ffi::pybamm_rust_profile_report;
pub use ffi::{
    ERROR_NULL_POINTER, ERROR_PANIC, RUST_ABI_VERSION, SUCCESS, pybamm_rust_abi_version,
    pybamm_rust_eval_rhs, pybamm_rust_jac_mul, pybamm_rust_n_inputs, pybamm_rust_n_states,
    pybamm_rust_residual,
};
pub use ir::{
    BinaryOp, BroadcastKind, ConstPool, Instruction, Slot, SlotStats, SplitEvalInfo, TypedIr,
    UnaryOp, first_invalid, first_unsupported, infer_sizes,
};
pub use jacobian::{CscPattern, JacobianData, JacobianLayout, JacobianScratch};
pub use model::{
    CompiledModel, CompiledModelAlgebraicBlock, CompiledModelOptions, JacobianStats,
    JacobianStrategy, ModelEvaluator, Workspace,
};
pub use node::{
    ArrayData, CsrData, CubicInterpolantData, InterpolantData, NdInterpolantData, Node, Shape,
    structural_hash,
};
pub use observable::{CompiledObservable, ObservableKind, ObservableScratch, ObservableSet};
pub use row_extract::{ScalarRowBlock, extract_scalar_rows};
pub use simplify::{SimplifyMode, cse, dce, simplify, simplify_pipeline, simplify_with_mode};
#[cfg(feature = "serialize")]
pub use snapshot::DagSnapshot;
pub use sparsity::{SparsityPattern, detect_sparsity_per_output};
pub use tangent::{DiffTarget, tangent_wrt_params, tangent_wrt_states, tangent_wrt_subset};
pub use zero_propagate::{ShapeInfo, ZeroStatus, zero_propagate};

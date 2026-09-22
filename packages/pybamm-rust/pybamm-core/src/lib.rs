//! Expression compiler and evaluator for `PyBaMM` models, filling the role
//! `CasADi` plays on `PyBaMM`'s other backends.
//!
//! # Pipeline
//!
//! 1. **Build**: the bindings allocate [`node`]s into an [`arena`], where sharing a
//!    subexpression repeats an id rather than cloning a subtree.
//! 2. **Rewrite**: [`simplify()`], [`zero_propagate()`], [`cse()`] and [`dce()`].
//! 3. **Differentiate**: [`tangent`] emits a forward-mode DAG for JVPs; [`adjoint`] fills one
//!    wide Jacobian row per backward pass.
//! 4. **Lower**: [`ir`] flattens the DAG into fixed-size instructions over one scratch buffer.
//! 5. **Evaluate**: [`eval`] runs a tape at one time point, [`CompiledExpr::eval_batch`] at
//!    several at once.
//!
//! [`CompiledModel`] assembles a DAE `M y' = f(t, y; p)` with its [`jacobian`], which IDAKLU
//! drives through the [`ffi`] C ABI. Assembly colors columns that never share a row, costing
//! one primal pass plus one tangent sweep per color, and leaves rows too wide to color to
//! reverse mode.

#![deny(unused_must_use)]

pub mod adjoint;
pub mod arena;
mod branch_regions;
mod coloring;
pub mod const_entries;
mod error;
pub mod eval;
mod eval_batch;
pub mod ffi;
pub mod ir;
pub mod jacobian;
pub mod model;
mod newtype;
pub mod node;
mod observable;
mod params;
mod row_extract;
mod simplify;
#[cfg(feature = "serialize")]
mod snapshot;
mod sparsity;
pub mod tangent;
pub mod tangent_batch;
mod zero_propagate;

pub use arena::{Arena, Dag, DagRef, NodeId, ReadExtents, resolve_state_count};
pub use error::CoreError;
pub use eval::{CompiledExpr, SplitCompiledExpr, TangentInputs};
pub use eval_batch::BatchEvalError;
pub use ffi::ModelEvaluator;
pub use ir::{GraphError, TypedIr, infer_sizes, validate};
pub use jacobian::{JacobianData, JacobianScratch};
pub use model::{CompiledModel, CompiledModelOptions};
pub use node::{
    ArrayData, CsrData, CubicInterpolantData, InterpolantData, NdInterpolantData, Node, Shape,
};
pub use observable::ObservableKind;
pub use params::{PackedWidth, ParamIdx, ParamRegistry, ParamSlot, resolve_input_width};
pub use row_extract::{ScalarRowBlock, extract_scalar_rows};
pub use simplify::{cse, dce, simplify, simplify_pipeline};
#[cfg(feature = "serialize")]
pub use snapshot::DagSnapshot;
pub use sparsity::{SparsityPattern, detect_sparsity_per_output};
pub use tangent::{DiffTarget, tangent_wrt_params, tangent_wrt_states};
pub use zero_propagate::zero_propagate;

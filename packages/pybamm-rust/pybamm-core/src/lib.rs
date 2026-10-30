//! Expression compiler and evaluator for `PyBaMM` models, filling the role
//! `CasADi` plays on `PyBaMM`'s other backends:
//!
//! 1. **Build**: the bindings allocate [`node`]s into an [`arena`], sharing subexpressions by id.
//! 2. **Rewrite**: [`simplify()`], [`zero_propagate()`], [`cse()`] and [`dce()`].
//! 3. **Differentiate**: [`tangent`] in forward mode, and [`adjoint`] for wide Jacobian rows.
//! 4. **Lower**: [`ir`] flattens the DAG into fixed-size instructions over one scratch buffer.
//! 5. **Evaluate**: [`eval`] runs a tape, [`CompiledExpr::eval_batch`] at several time points.
//! 6. **Assemble**: [`CompiledModel`] and its colored [`jacobian`], driven by IDAKLU via [`ffi`].

#![deny(unused_must_use)]

mod accumulate;
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
pub use ir::{GraphError, TypedIr, infer_sizes, validate};
pub use jacobian::{JacobianData, JacobianScratch};
pub use model::{CompiledModel, CompiledModelOptions, ModelEvaluator};
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

//! Expression compiler and evaluator for `PyBaMM` models, filling the role
//! `CasADi` plays on `PyBaMM`'s other backends:
//!
//! 1. **Build**: the bindings allocate [`node`]s into an [`arena`], sharing subexpressions by id.
//! 2. **Rewrite**: [`simplify()`], [`zero_propagate()`], [`cse()`] and [`dce()`].
//! 3. **Differentiate**: [`tangent`] emits a forward-mode DAG for JVPs.
//! 4. **Lower**: [`ir`] flattens the DAG into fixed-size instructions over one scratch buffer.
//! 5. **Evaluate**: [`eval`] runs a tape, [`CompiledExpr::eval_batch`] at several time points.

#![deny(unused_must_use)]

pub mod arena;
mod branch_regions;
mod error;
pub mod eval;
mod eval_batch;
pub mod ir;
mod newtype;
pub mod node;
mod params;
mod simplify;
pub mod tangent;
pub mod tangent_batch;
mod zero_propagate;

pub use arena::{Arena, Dag, DagRef, NodeId, ReadExtents, resolve_state_count};
pub use error::CoreError;
pub use eval::{CompiledExpr, SplitCompiledExpr, TangentInputs};
pub use eval_batch::BatchEvalError;
pub use ir::{GraphError, TypedIr, infer_sizes, validate};
pub use node::{
    ArrayData, CsrData, CubicInterpolantData, InterpolantData, NdInterpolantData, Node, Shape,
};
pub use params::{PackedWidth, ParamIdx, ParamRegistry, ParamSlot, resolve_input_width};
pub use simplify::{cse, dce, simplify, simplify_pipeline};
pub use tangent::{DiffTarget, tangent_wrt_params, tangent_wrt_states};
pub use zero_propagate::zero_propagate;

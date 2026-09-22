//! No entry point trusts per-caller scratch left by an earlier call, finished or unwound.
//!
//! Every entry point the FFI and diffsol reach runs on a fresh [`Workspace`], then on one that
//! every entry point has dirtied and that was then overwritten whole; both give the same bits.

use super::test_fixtures::{
    algebraic_options, build_trailing_param_sens_model, compile_with, dense_row_model, element,
    identity_mass_matrix, input,
};
use super::{CompiledModel, CompiledModelOptions, Workspace};
use crate::arena::Arena;
use crate::jacobian::tests::banded_nonlinear;
use crate::node::{CsrData, Node, Shape};
use crate::observable::ObservableKind;
use crate::params::ParamIdx;

/// One entry point at a fixed point, returning what it wrote.
type EntryPoint<'a> = Box<dyn Fn(&mut Workspace) -> Vec<f64> + 'a>;

/// Labelled entry points, in the order they dirty a workspace.
struct EntryPoints<'a>(Vec<(String, EntryPoint<'a>)>);

impl<'a> EntryPoints<'a> {
    /// Add `call`, which writes `len` values.
    fn add(
        &mut self,
        label: impl Into<String>,
        len: usize,
        call: impl Fn(&mut Workspace, &mut [f64]) + 'a,
    ) {
        let run = move |ws: &mut Workspace| {
            let mut out = vec![0.0; len];
            call(ws, &mut out);
            out
        };
        self.0.push((label.into(), Box::new(run)));
    }
}

/// A deterministic vector of `len` values in `[-1, 1]`.
fn wave(len: usize, phase: f64) -> Vec<f64> {
    (0..len)
        .map(|i| (i as f64).mul_add(0.37, phase).sin())
        .collect()
}

/// Every entry point `model` supports, at `(0.3, y)`.
fn entry_points<'a>(model: &'a CompiledModel, y: &'a [f64], inputs: &'a [f64]) -> EntryPoints<'a> {
    const T: f64 = 0.3;
    let (n, k, nnz) = (model.n_states(), model.n_sens_params(), model.nnz());
    let mut entries = EntryPoints(Vec::new());
    entries.add("eval_rhs", n, move |ws, out| {
        model.eval_rhs(ws, T, y, inputs, out);
    });
    entries.add("eval_residual", n, move |ws, out| {
        model.eval_residual(ws, T, y, &wave(n, 0.2), inputs, out);
    });
    entries.add("assemble_jacobian_csc_into", nnz, move |ws, out| {
        model.assemble_jacobian_csc_into(ws, T, y, inputs, 0.7, out);
    });
    entries.add("assemble_jacobian_csc_no_mass", nnz, move |ws, out| {
        model.assemble_jacobian_csc_no_mass(ws, T, y, inputs, out);
    });
    entries.add("jac_action", n, move |ws, out| {
        model.jac_action(ws, T, y, inputs, &wave(n, 0.1), out);
    });
    // Scalar, part-empty, full-width and multi-walk batches.
    for count in [1, 3, 5, 8, 11] {
        entries.add(
            format!("apply_lanes x{count}"),
            count * n,
            move |ws, out| {
                let directions = wave(count * n, 0.3);
                model
                    .jac_action_at(ws, T, y, inputs)
                    .apply_lanes(directions.chunks_exact(n), out.chunks_exact_mut(n));
            },
        );
    }
    if k > 0 {
        entries.add("eval_sens_all", k * n, move |ws, out| {
            model.eval_sens_all(ws, T, y, inputs, out);
        });
        entries.add("sens_action", n, move |ws, out| {
            model.sens_action(ws, T, y, inputs, &wave(k, 0.4), out);
        });
    }
    for kind in [ObservableKind::Outputs, ObservableKind::Events] {
        let len = model.observables(kind).total_len();
        if len == 0 {
            continue;
        }
        entries.add(format!("eval_observables {kind:?}"), len, move |ws, out| {
            model.eval_observables(ws, kind, T, y, inputs, out);
        });
        if k > 0 {
            entries.add(
                format!("observable_sens_action {kind:?}"),
                len,
                move |ws, out| {
                    model.observable_sens_action(ws, kind, T, y, inputs, &wave(k, 0.5), out);
                },
            );
            entries.add(
                format!("observable_jac_action {kind:?}"),
                len,
                move |ws, out| {
                    model.observable_jac_action(ws, kind, T, y, inputs, &wave(n, 0.1), out);
                },
            );
        }
    }
    let n_out = model.total_output_len();
    if n_out > 0 {
        let ts = [0.1, 0.2, 0.3];
        entries.add("eval_outputs_batch", ts.len() * n_out, move |ws, out| {
            let y_cols: Vec<f64> = ts.iter().flat_map(|_| y.iter().copied()).collect();
            model.eval_outputs_batch(ws, ts.len(), &ts, &y_cols, inputs, out);
        });
    }
    if n_out > 0 && k > 0 {
        entries.add("output_sens_project", k * n_out, move |ws, out| {
            model.output_sens_project(ws, T, y, inputs, &wave(k * n, 0.6), out);
        });
    }
    if let Some(block) = model.algebraic() {
        entries.add(
            "algebraic eval_residual",
            block.n_variables(),
            move |ws, out| {
                block.eval_residual(ws, T, y, inputs, out);
            },
        );
        let nnz = block.jacobian().nnz();
        entries.add("algebraic assemble_jacobian_into", nnz, move |ws, out| {
            block.assemble_jacobian_into(ws, T, y, inputs, out);
        });
    }
    entries
}

/// Each entry point gives the same bits on a fresh workspace as on one that every entry point
/// has dirtied and that was then poisoned whole.
fn assert_poison_changes_nothing(name: &str, model: &CompiledModel, y: &[f64], inputs: &[f64]) {
    let bits = |values: Vec<f64>| -> Vec<u64> { values.iter().map(|v| v.to_bits()).collect() };
    let EntryPoints(entries) = entry_points(model, y, inputs);
    for (label, entry) in &entries {
        let fresh = bits(entry(&mut model.create_workspace()));
        for value in [1.0, f64::NAN, -3.5e7] {
            let mut ws = model.create_workspace();
            for (_, other) in &entries {
                other(&mut ws);
            }
            ws.poison(value);
            assert!(
                bits(entry(&mut ws)) == fresh,
                "{name}: {label} changed on a workspace poisoned with {value}"
            );
        }
    }
}

#[test]
fn a_batched_banded_model_ignores_a_poisoned_workspace() {
    let (arena, rhs) = banded_nonlinear(64, 3);
    let model = compile_with(
        &arena,
        rhs,
        identity_mass_matrix(64),
        64,
        0,
        CompiledModelOptions::default(),
    );
    assert!(
        model.jacobian_stats().jac_lane_width > 1,
        "the sweep must batch"
    );
    let y: Vec<f64> = (0..64).map(|i| f64::from(i).mul_add(0.03, 0.2)).collect();
    assert_poison_changes_nothing("banded", &model, &y, &[]);
}

#[test]
fn dense_row_models_ignore_a_poisoned_workspace() {
    for first in [true, false] {
        let model = dense_row_model(20, 0..20, first, identity_mass_matrix(20));
        assert_eq!(model.jacobian_stats().n_dense_rows, 1, "the row must split");
        let y: Vec<f64> = (0..20).map(|i| f64::from(i).mul_add(0.1, 0.3)).collect();
        assert_poison_changes_nothing("dense row", &model, &y, &[]);
    }
}

#[test]
fn a_sensitivity_model_with_an_output_ignores_a_poisoned_workspace() {
    let (model, y) = build_trailing_param_sens_model();
    assert_poison_changes_nothing("sens + output", &model, &y, &[5.0, 9.0]);
}

/// Five sensitivities, so output projections walk `dH/dy` batched, over 12 coupled states.
#[test]
fn batched_output_sensitivities_ignore_a_poisoned_workspace() {
    const N: usize = 12;
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: N });
    let p: Vec<_> = (0..5)
        .map(|i| input(&mut arena, &format!("p{i}"), i, i, 1))
        .collect();
    let sin = arena.alloc(Node::Sin(y));
    let square = arena.alloc(Node::Mul(y, y));
    let head = arena.alloc(Node::Index {
        child: y,
        start: 0,
        end: N - 1,
    });
    let tail = arena.alloc(Node::Index {
        child: y,
        start: 1,
        end: N,
    });
    let coupling = arena.alloc(Node::Mul(head, tail));
    let zero = arena.alloc(Node::Scalar(0.0));
    let coupling = arena.alloc(Node::Concat(vec![coupling, zero]));
    let terms = [sin, square, coupling];
    let mut rhs = arena.alloc(Node::Mul(p[0], terms[0]));
    for (&param, &term) in p[1..].iter().zip(&terms[1..]) {
        let scaled = arena.alloc(Node::Mul(param, term));
        rhs = arena.alloc(Node::Add(rhs, scaled));
    }
    let rhs = arena.alloc(Node::Add(rhs, p[3]));
    let rhs = arena.alloc(Node::Sub(rhs, p[4]));
    let sens: Vec<_> = (0..5_u32).map(ParamIdx::new).collect();
    let options = CompiledModelOptions {
        sensitivities: &sens,
        algebraic: None,
    };
    let mut model = compile_with(&arena, rhs, identity_mass_matrix(N), N, 5, options);

    let y0 = element(&mut arena, y, 0);
    let scaled = arena.alloc(Node::Mul(p[0], y0));
    let shifted = arena.alloc(Node::Add(tail, p[1]));
    let half = arena.alloc(Node::Scalar(0.5));
    let crossing = arena.alloc(Node::Sub(y0, half));
    model
        .set_observables(ObservableKind::Outputs, &mut arena, &[scaled, sin, shifted])
        .expect("valid outputs");
    model
        .set_observables(ObservableKind::Events, &mut arena, &[crossing])
        .expect("a valid event");
    assert!(
        model.observables(ObservableKind::Outputs).widest_jac_walk() > 1,
        "the output projection must batch"
    );
    let y = wave(N, 0.8);
    assert_poison_changes_nothing("batched outputs", &model, &y, &[0.9, -0.4, 1.3, 0.2, 2.0]);
}

/// Fused outputs and events, a sensitivity and an algebraic block, behind a selector mass.
#[test]
fn a_dae_with_every_artifact_ignores_a_poisoned_workspace() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let k = input(&mut arena, "k", 0, 0, 1);
    let [y0, y1, y2] = [0, 1, 2].map(|i| element(&mut arena, y, i));
    let sin = arena.alloc(Node::Sin(y));
    let rhs = arena.alloc(Node::Mul(k, sin));
    let product = arena.alloc(Node::Mul(y1, y2));
    let g0 = arena.alloc(Node::Add(y0, product));
    let scaled = arena.alloc(Node::Mul(k, y2));
    let g1 = arena.alloc(Node::Add(y1, scaled));
    let residual = arena.alloc(Node::Concat(vec![g0, g1]));
    let mass = CsrData {
        shape: Shape::matrix(3, 3),
        indptr: vec![0, 1, 1, 1],
        indices: vec![0],
        data: vec![1.0],
    };
    let options = CompiledModelOptions {
        sensitivities: &[ParamIdx::new(0)],
        ..algebraic_options(residual, &[1, 2])
    };
    let mut model = compile_with(&arena, rhs, mass, 3, 1, options);
    let output = arena.alloc(Node::Mul(y0, y1));
    let half = arena.alloc(Node::Scalar(0.5));
    let crossing = arena.alloc(Node::Sub(y0, half));
    model
        .set_observables(ObservableKind::Outputs, &mut arena, &[output, scaled, y])
        .expect("valid outputs");
    model
        .set_observables(ObservableKind::Events, &mut arena, &[crossing, output])
        .expect("valid events");
    assert_poison_changes_nothing("every artifact", &model, &[0.4, 1.3, -0.8], &[2.5]);
}

mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::{build_coupled, identity_mass_matrix};
use pybamm_core::{
    Arena, CompiledExpr, ModelEvaluator, Node, NodeId, SimplifyMode, TangentInputs, TypedIr,
    simplify_with_mode, tangent_wrt_states,
};

fn build_simple_expression(arena: &mut Arena, n_states: usize) -> NodeId {
    let y = arena.alloc(Node::StateVector {
        start: 0,
        end: n_states,
    });
    let two = arena.alloc(Node::Scalar(2.0));
    let y_sq = arena.alloc(Node::Pow(y, two));
    let sin_y = arena.alloc(Node::Sin(y));
    arena.alloc(Node::Add(y_sq, sin_y))
}

fn bench_primal_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("primal_eval");

    for n in [10, 50, 100, 200, 500, 1000] {
        let mut arena = Arena::new();
        let root = build_coupled(&mut arena, n);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let compiled = CompiledExpr::new(&arena, root);
        let mut scratch = vec![0.0; compiled.scratch_len()];

        group.bench_with_input(BenchmarkId::new("coupled", n), &n, |b, _| {
            b.iter(|| {
                let result = compiled.eval(
                    black_box(&mut scratch),
                    black_box(0.0),
                    black_box(&y),
                    black_box(&[]),
                    black_box(&[]),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_jvp(c: &mut Criterion) {
    let mut group = c.benchmark_group("jvp");

    for n in [50, 100, 200, 500, 1000, 2000] {
        let mut arena = Arena::new();
        let root = build_coupled(&mut arena, n);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let v: Vec<f64> = vec![1.0; n];
        let compiled = CompiledExpr::new(&arena, root);
        let mut scratch = vec![0.0; compiled.scratch_len()];

        group.bench_with_input(BenchmarkId::new("primal", n), &n, |b, _| {
            b.iter(|| {
                let result = compiled.eval(
                    black_box(&mut scratch),
                    black_box(0.0),
                    black_box(&y),
                    black_box(&[]),
                    black_box(&[]),
                );
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("forward", n), &n, |b, _| {
            b.iter(|| {
                let tangent = TangentInputs {
                    dy: Some(&v),
                    dp: None,
                };
                let result = compiled.eval_with_tangent(
                    black_box(&mut scratch),
                    black_box(0.0),
                    black_box(&y),
                    black_box(&[]),
                    black_box(&[]),
                    black_box(&tangent),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_jacobian_symbolic(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobian_symbolic");

    for n in [10, 50, 100, 200, 500] {
        let mut arena = Arena::new();
        let root = build_simple_expression(&mut arena, n);

        group.bench_with_input(BenchmarkId::new("build", n), &n, |b, _| {
            b.iter(|| {
                let mut a = Arena::new();
                let r = build_simple_expression(&mut a, black_box(n));
                let jac = tangent_wrt_states(&mut a, r);
                let _jac = simplify_with_mode(&mut a, jac, SimplifyMode::Aggressive);
            });
        });

        let jac_root = tangent_wrt_states(&mut arena, root);
        let jac_root = simplify_with_mode(&mut arena, jac_root, SimplifyMode::Aggressive);
        let ir = TypedIr::from_arena(&arena, jac_root);
        let expr = CompiledExpr::from_ir(ir);
        let mut scratch = vec![0.0; expr.scratch_len()];
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let v: Vec<f64> = vec![1.0; n];

        group.bench_with_input(BenchmarkId::new("eval", n), &n, |b, _| {
            b.iter(|| {
                let tangent = TangentInputs {
                    dy: Some(black_box(&v)),
                    dp: None,
                };
                let result = expr.eval_with_tangent(
                    black_box(&mut scratch),
                    black_box(0.0),
                    black_box(&y),
                    &[],
                    &[],
                    &tangent,
                );
                black_box(result.len());
            });
        });
    }

    group.finish();
}

fn bench_jacobian_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobian_assembly");

    for n in [50, 100, 200, 500] {
        let mut arena = Arena::new();
        let root = build_coupled(&mut arena, n);
        let mass = identity_mass_matrix(n);
        let mut model = ModelEvaluator::new(&arena, root, mass, n, 0);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let mut jac_data = vec![0.0; model.nnz()];

        group.bench_with_input(BenchmarkId::new("csc_into", n), &n, |b, _| {
            b.iter(|| {
                model.set_cj(1.0);
                model.assemble_jacobian_csc_into(black_box(0.0), black_box(&y), &[], &mut jac_data);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "serialize")]
fn bench_real_model_eval(c: &mut Criterion) {
    use pybamm_core::DagSnapshot;

    let mut group = c.benchmark_group("real_model_eval");
    group.sample_size(20);

    for fixture in helpers::FIXTURES {
        let snap = DagSnapshot::from_bytes(fixture.bytes);
        // Unsimplified on purpose: `CompiledModel::new` skips the rhs simplify
        // pass, so this is the tape the solver actually walks.
        let compiled = CompiledExpr::new(&snap.arena, snap.root);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let (y, inputs) = helpers::fixture_state(snap.n_states, snap.n_params);

        group.bench_with_input(
            BenchmarkId::new("rhs", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| {
                    let result = compiled.eval(
                        black_box(&mut scratch),
                        black_box(0.0),
                        black_box(&y),
                        black_box(&[]),
                        black_box(&inputs),
                    );
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "serialize"))]
const fn bench_real_model_eval(_c: &mut Criterion) {}

/// Per-call assembly on the real fixtures, which is the number any change to
/// the batched sweep or the constant table has to move.
#[cfg(feature = "serialize")]
fn bench_real_model_assembly(c: &mut Criterion) {
    use pybamm_core::DagSnapshot;

    let mut group = c.benchmark_group("real_model_assembly");
    group.sample_size(20);

    for fixture in helpers::FIXTURES {
        let snap = DagSnapshot::from_bytes(fixture.bytes);
        let mass = identity_mass_matrix(snap.n_states);
        let mut model = ModelEvaluator::new(&snap.arena, snap.root, mass, snap.n_states, 0);
        let (y, inputs) = helpers::fixture_state(snap.n_states, snap.n_params);
        let mut jac_data = vec![0.0; model.nnz()];

        group.bench_with_input(
            BenchmarkId::new("csc_into", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| {
                    model.set_cj(1.0);
                    model.assemble_jacobian_csc_into(
                        black_box(0.0),
                        black_box(&y),
                        black_box(&inputs),
                        &mut jac_data,
                    );
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "serialize"))]
const fn bench_real_model_assembly(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_primal_eval,
    bench_jvp,
    bench_jacobian_symbolic,
    bench_jacobian_assembly,
    bench_real_model_eval,
    bench_real_model_assembly,
);
criterion_main!(benches);

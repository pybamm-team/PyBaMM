mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::{build_coupled, identity_mass_matrix};
use pybamm_core::{
    Arena, CompiledExpr, CompiledModel, CompiledModelOptions, DagRef, Node, NodeId, PackedWidth,
    SplitCompiledExpr, TangentInputs, simplify, tangent_wrt_states,
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
        let compiled = CompiledExpr::new(DagRef::new(&arena, root));
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

/// Eight JVPs of `build_coupled(n)`, one per colour-like seed: `mono` runs the whole
/// tangent tape each time, `split` runs the primal once and eight tangent sweeps over it.
fn bench_jvp(c: &mut Criterion) {
    const DIRECTIONS: usize = 8;
    let mut group = c.benchmark_group("jvp");

    for n in [50, 100, 200, 500, 1000, 2000] {
        let mut arena = Arena::new();
        let root = build_coupled(&mut arena, n);
        let jvp_root = tangent_wrt_states(&mut arena, root);
        let dag = DagRef::new(&arena, jvp_root);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let seeds: Vec<Vec<f64>> = (0..DIRECTIONS)
            .map(|d| {
                (0..n)
                    .map(|i| f64::from(u8::from(i % DIRECTIONS == d)))
                    .collect()
            })
            .collect();

        let mono = CompiledExpr::new(dag);
        let mut scratch = vec![0.0; mono.scratch_len()];
        group.bench_with_input(BenchmarkId::new("mono", n), &n, |b, _| {
            b.iter(|| {
                for seed in &seeds {
                    let tangent = TangentInputs {
                        dy: Some(seed),
                        dp: None,
                    };
                    let result = mono.eval_with_tangent(
                        black_box(&mut scratch),
                        black_box(0.0),
                        black_box(&y),
                        &[],
                        &[],
                        &tangent,
                    );
                    black_box(result);
                }
            });
        });

        let split = SplitCompiledExpr::compile(dag);
        let mut scratch = vec![0.0; split.scratch_len()];
        group.bench_with_input(BenchmarkId::new("split", n), &n, |b, _| {
            b.iter(|| {
                let mut primal = split.eval_primal(
                    black_box(&mut scratch),
                    black_box(0.0),
                    black_box(&y),
                    &[],
                    &[],
                );
                for seed in &seeds {
                    let tangent = TangentInputs {
                        dy: Some(seed),
                        dp: None,
                    };
                    black_box(primal.eval_tangent(&tangent));
                }
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
                let _jac = simplify(&mut a, jac);
            });
        });

        let jac_root = tangent_wrt_states(&mut arena, root);
        let jac_root = simplify(&mut arena, jac_root);
        let expr = CompiledExpr::new(DagRef::new(&arena, jac_root));
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
        let model = CompiledModel::new(
            &arena,
            root,
            mass,
            n,
            PackedWidth::new(0),
            CompiledModelOptions::default(),
        )
        .expect("a valid fixture");
        let mut ws = model.create_workspace();
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let mut jac_data = vec![0.0; model.nnz()];

        group.bench_with_input(BenchmarkId::new("csc_into", n), &n, |b, _| {
            b.iter(|| {
                model.assemble_jacobian_csc_into(
                    &mut ws,
                    black_box(0.0),
                    black_box(&y),
                    &[],
                    1.0,
                    &mut jac_data,
                );
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
        let snap = DagSnapshot::from_bytes(fixture.bytes).expect("fixture decodes");
        // Unsimplified on purpose: `CompiledModel::new` skips the rhs simplify pass.
        let compiled = CompiledExpr::new(snap.dag());
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let (y, inputs) = helpers::fixture_state(snap.n_states, snap.packed_input_width.get());

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

        // `k` lanes per call, to compare against `k` scalar `rhs` calls.
        for k in [8, 32] {
            let ts = vec![0.0; k];
            let y_cols = y.repeat(k);
            let mut batch_scratch = vec![0.0; compiled.scratch_len() * k];
            group.bench_with_input(
                BenchmarkId::new(format!("batch_k{k}"), fixture.name),
                &fixture.name,
                |b, _| {
                    b.iter(|| {
                        let result = compiled
                            .eval_batch(
                                black_box(&mut batch_scratch),
                                k,
                                black_box(&ts),
                                black_box(&y_cols),
                                y.len(),
                                black_box(&inputs),
                            )
                            .expect("the fixture rhs is a primal tape");
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "serialize"))]
const fn bench_real_model_eval(_c: &mut Criterion) {}

/// Per-call assembly on the real fixtures.
#[cfg(feature = "serialize")]
fn bench_real_model_assembly(c: &mut Criterion) {
    use pybamm_core::DagSnapshot;

    let mut group = c.benchmark_group("real_model_assembly");
    group.sample_size(20);

    for fixture in helpers::FIXTURES {
        let snap = DagSnapshot::from_bytes(fixture.bytes).expect("fixture decodes");
        let model = helpers::compile_fixture(&snap);
        let mut ws = model.create_workspace();
        let (y, inputs) = helpers::fixture_state(snap.n_states, snap.packed_input_width.get());
        let mut jac_data = vec![0.0; model.nnz()];

        group.bench_with_input(
            BenchmarkId::new("csc_into", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| {
                    model.assemble_jacobian_csc_into(
                        &mut ws,
                        black_box(0.0),
                        black_box(&y),
                        black_box(&inputs),
                        1.0,
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

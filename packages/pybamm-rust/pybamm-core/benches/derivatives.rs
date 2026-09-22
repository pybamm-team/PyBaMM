//! Derivative paths the other benches miss: output sensitivities, dense-row reverse fill, and
//! colour sweeps and actions at the lane-width boundaries.

mod helpers;

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use pybamm_core::ffi::{pybamm_rust_jac_action_batch, pybamm_rust_output_sens_project};
use pybamm_core::{
    Arena, CompiledModel, CompiledModelOptions, CsrData, DagSnapshot, InterpolantData,
    JacobianData, JacobianScratch, ModelEvaluator, Node, NodeId, ObservableKind, PackedWidth,
    ParamIdx, Shape,
};

/// Sensitivity parameters the output benches differentiate by.
const N_SENS: usize = 4;

/// `p0 f + p1 sin f + p2 cos f + p3` over a fixture's rhs `f`, with its parameters.
fn parametrised_rhs(arena: &mut Arena, rhs: NodeId) -> (NodeId, Vec<NodeId>) {
    let params: Vec<NodeId> = (0..N_SENS)
        .map(|i| {
            arena.alloc(Node::InputParameter {
                name: format!("p{i}"),
                index: i,
                offset: i,
                width: 1,
            })
        })
        .collect();
    let sin = arena.alloc(Node::Sin(rhs));
    let cos = arena.alloc(Node::Cos(rhs));
    let scaled = arena.alloc(Node::Mul(params[0], rhs));
    let sin_term = arena.alloc(Node::Mul(params[1], sin));
    let cos_term = arena.alloc(Node::Mul(params[2], cos));
    let partial = arena.alloc(Node::Add(scaled, sin_term));
    let sum = arena.alloc(Node::Add(partial, cos_term));
    (arena.alloc(Node::Add(sum, params[3])), params)
}

/// Outputs sharing most of their cone; `edge` swaps in ones whose `dH/dp` or `dH/dy` is zero.
fn outputs(arena: &mut Arena, rhs: NodeId, params: &[NodeId], n: usize, edge: bool) -> Vec<NodeId> {
    let ones = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, n],
            (0..n).collect(),
            vec![1.0; n],
            Shape::matrix(1, n),
        )
        .expect("valid row matrix"),
    )));
    let total = arena.alloc(Node::MatMul(ones, rhs));
    let last = arena.alloc(Node::Index {
        child: rhs,
        start: n - 1,
        end: n,
    });
    if edge {
        let slice = arena.alloc(Node::StateVector { start: 2, end: 9 });
        let product = arena.alloc(Node::Mul(params[2], params[3]));
        let zero = arena.alloc(Node::Scalar(-0.0));
        let negated = arena.alloc(Node::Neg(last));
        let hypot = arena.alloc(Node::Hypot(negated, params[0]));
        return vec![slice, product, zero, hypot, total];
    }
    let head = arena.alloc(Node::Index {
        child: rhs,
        start: 0,
        end: n / 3,
    });
    let middle = arena.alloc(Node::Index {
        child: rhs,
        start: n / 3,
        end: 2 * n / 3,
    });
    let scaled_head = arena.alloc(Node::Mul(head, params[0]));
    let sin_middle = arena.alloc(Node::Sin(middle));
    let shifted_last = arena.alloc(Node::Add(last, params[1]));
    vec![total, scaled_head, sin_middle, shifted_last]
}

/// A fixture compiled with sensitivities, behind the C ABI's evaluator.
fn sensitivity_evaluator(snap: &DagSnapshot, edge: bool) -> (ModelEvaluator, usize) {
    let n = snap.n_states;
    let mut arena = snap.dag().arena().clone();
    let (rhs, params) = parametrised_rhs(&mut arena, snap.dag().root());
    let sens: Vec<ParamIdx> = (0..N_SENS)
        .map(|i| ParamIdx::new(u32::try_from(i).expect("few parameters")))
        .collect();
    let mut model = CompiledModel::new(
        &arena,
        rhs,
        helpers::identity_mass_matrix(n),
        n,
        PackedWidth::new(N_SENS),
        CompiledModelOptions {
            sensitivities: &sens,
            algebraic: None,
        },
    )
    .expect("a valid fixture");
    let roots = outputs(&mut arena, rhs, &params, n, edge);
    model
        .set_observables(ObservableKind::Outputs, &mut arena, &roots)
        .expect("valid outputs");
    let n_out = model.total_output_len();
    (ModelEvaluator::from_compiled(Arc::new(model)), n_out)
}

/// `output_sens_project` at one output point: what every sensitivity solve pays per point.
fn bench_output_sens_project(c: &mut Criterion) {
    let mut group = c.benchmark_group("output_sens_project");
    group.sample_size(30);

    for fixture in &helpers::FIXTURES[1..] {
        let snap = DagSnapshot::from_bytes(fixture.bytes).expect("fixture decodes");
        let n = snap.n_states;
        let (y, _) = helpers::fixture_state(n, 0);
        let inputs: Vec<f64> = (0..N_SENS).map(|i| 0.1f64.mul_add(i as f64, 0.3)).collect();
        let y_sens: Vec<f64> = (0..N_SENS * n)
            .map(|k| ((k * 7919) % 101) as f64 / 101.0 - 0.5)
            .collect();
        for (label, edge) in [("shared", false), ("edge", true)] {
            let (mut evaluator, n_out) = sensitivity_evaluator(&snap, edge);
            let mut out = vec![0.0; N_SENS * n_out];
            let user_data = std::ptr::from_mut(&mut evaluator).cast::<std::ffi::c_void>();
            group.bench_with_input(
                BenchmarkId::new(label, fixture.name),
                &fixture.name,
                |b, _| {
                    b.iter(|| {
                        // The C ABI is the only public route to the projection.
                        // SAFETY: `user_data` is a live evaluator, and each buffer holds its count.
                        #[allow(unsafe_code)]
                        let status = unsafe {
                            pybamm_rust_output_sens_project(
                                black_box(0.1),
                                y.as_ptr(),
                                inputs.as_ptr(),
                                y_sens.as_ptr(),
                                out.as_mut_ptr(),
                                user_data,
                            )
                        };
                        assert_eq!(status, 0);
                    });
                },
            );
        }
    }

    group.finish();
}

/// `sin(y)` over `n` states, then `ROWS` fully dense rows `A f(y)`.
fn dense_block(n: usize, interpolated: bool) -> (Arena, NodeId) {
    const ROWS: usize = 16;
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let diagonal = arena.alloc(Node::Sin(y));
    let coupled = if interpolated {
        let knots: Vec<f64> = (0..50).map(|i| f64::from(i) / 49.0 * 2.0).collect();
        let values = knots.iter().map(|x| x * x).collect();
        arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(InterpolantData::try_new(knots, values).expect("valid table")),
            child: y,
        })
    } else {
        arena.alloc(Node::Tanh(y))
    };
    let matrix = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            (0..=ROWS).map(|row| row * n).collect(),
            (0..ROWS).flat_map(|_| 0..n).collect(),
            (0..ROWS * n)
                .map(|k| 0.1f64.mul_add((k % 7) as f64, 0.5))
                .collect(),
            Shape::matrix(ROWS, n),
        )
        .expect("valid block"),
    )));
    let block = arena.alloc(Node::MatMul(matrix, coupled));
    let root = arena.alloc(Node::Concat(vec![diagonal, block]));
    (arena, root)
}

/// Building and assembling a Jacobian with a dense block, whose rows reverse mode fills.
fn bench_dense_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_block");
    group.sample_size(10);

    for n in [200, 1000] {
        for (label, interpolated) in [("tanh", false), ("interpolant", true)] {
            let (arena, root) = dense_block(n, interpolated);
            let n_rows = n + 16;
            let y: Vec<f64> = (0..n).map(|i| 0.5 + i as f64 / n as f64).collect();
            group.bench_with_input(BenchmarkId::new(format!("build_{label}"), n), &n, |b, _| {
                b.iter(|| JacobianData::new_wrt_states(&arena, root, n_rows, n));
            });

            let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);
            let layout = jac.layout();
            let mut scratch = JacobianScratch::new(&jac);
            let mut out = vec![0.0; layout.n_slots()];
            group.bench_with_input(
                BenchmarkId::new(format!("assemble_{label}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        jac.assemble_into(
                            &mut scratch,
                            layout,
                            black_box(0.0),
                            black_box(&y),
                            &[],
                            &[],
                            &mut out,
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

/// Rows over `n` states, each reading the `below + above + 1` consecutive states around it.
fn band(n: usize, below: usize, above: usize) -> (Arena, NodeId) {
    let rows = n - below - above;
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let window = |arena: &mut Arena, start: usize| {
        arena.alloc(Node::Index {
            child: y,
            start,
            end: start + rows,
        })
    };
    let centre = window(&mut arena, below);
    let mut sum = arena.alloc(Node::Exp(centre));
    for start in (0..=below + above).filter(|&start| start != below) {
        let neighbour = window(&mut arena, start);
        let term = if start < below {
            arena.alloc(Node::Sin(neighbour))
        } else {
            arena.alloc(Node::Cos(neighbour))
        };
        let product = arena.alloc(Node::Mul(term, centre));
        sum = arena.alloc(Node::Add(sum, product));
    }
    (arena, sum)
}

/// One assembly of a band over `n` states needing `k` colours, around the lane widths 4 and 8.
///
/// Short bands are dispatch-bound and batch; long ones are element-wise work, which does not.
fn bench_colour_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_sweep");

    for n in [20, 200, 2000] {
        for (k, below, above) in [(3, 1, 1), (5, 2, 2), (6, 2, 3), (7, 3, 3)] {
            let (arena, root) = band(n, below, above);
            let jac = JacobianData::new_wrt_states(&arena, root, n - below - above, n);
            assert_eq!(jac.n_colors(), k, "the band must need {k} colours");
            let layout = jac.layout();
            let mut scratch = JacobianScratch::new(&jac);
            let mut out = vec![0.0; layout.n_slots()];
            let y: Vec<f64> = (0..n).map(|i| 0.3 + i as f64 / n as f64).collect();
            group.bench_with_input(BenchmarkId::new(format!("k{k}"), n), &n, |b, _| {
                b.iter(|| {
                    jac.assemble_into(
                        &mut scratch,
                        layout,
                        black_box(0.0),
                        black_box(&y),
                        &[],
                        &[],
                        &mut out,
                    );
                });
            });
        }
    }

    group.finish();
}

/// `J v` in `k` directions at one point, as IDAKLU's sensitivity residual asks for them.
fn bench_jac_action_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("jac_action_batch");
    group.sample_size(30);

    for fixture in &helpers::FIXTURES[1..] {
        let snap = DagSnapshot::from_bytes(fixture.bytes).expect("fixture decodes");
        let n = snap.n_states;
        let model = CompiledModel::new(
            snap.dag().arena(),
            snap.dag().root(),
            helpers::identity_mass_matrix(n),
            n,
            PackedWidth::new(0),
            CompiledModelOptions::default(),
        )
        .expect("a valid fixture");
        let mut evaluator = ModelEvaluator::from_compiled(Arc::new(model));
        let user_data = std::ptr::from_mut(&mut evaluator).cast::<std::ffi::c_void>();
        let (y, _) = helpers::fixture_state(n, 0);
        for k in [3, 5, 6, 7] {
            let directions: Vec<f64> = (0..k * n)
                .map(|i| ((i * 31) % 17) as f64 / 17.0 - 0.5)
                .collect();
            let mut out = vec![0.0; k * n];
            let n_dirs = i32::try_from(k).expect("few directions");
            group.bench_with_input(
                BenchmarkId::new(format!("k{k}"), fixture.name),
                &fixture.name,
                |b, _| {
                    b.iter(|| {
                        // The C ABI is the only public route to the batched action.
                        // SAFETY: `user_data` is a live evaluator, and each buffer holds its count.
                        #[allow(unsafe_code)]
                        let status = unsafe {
                            pybamm_rust_jac_action_batch(
                                black_box(0.0),
                                y.as_ptr(),
                                std::ptr::null(),
                                n_dirs,
                                directions.as_ptr(),
                                out.as_mut_ptr(),
                                user_data,
                            )
                        };
                        assert_eq!(status, 0);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_output_sens_project,
    bench_dense_block,
    bench_colour_sweep,
    bench_jac_action_batch,
);
criterion_main!(benches);

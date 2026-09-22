//! Benchmark: the saving from integrating only the requested sensitivity columns.

mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::identity_mass_matrix;
use pybamm_core::solver::solve::{InputSet, PreparedSolver, SolveRequest};
use pybamm_core::{Arena, CompiledModel, CompiledModelOptions, Node, PackedWidth, ParamIdx};

const N_STATES: usize = 8;
const N_PARAMS: usize = 16;

/// `dy_i/dt = -(sum_j p_j) * y_i`: every state depends on every parameter.
fn build_dense_decay_model(sens: &[ParamIdx]) -> CompiledModel {
    let mut arena = Arena::new();
    let mut sum = arena.alloc(Node::Scalar(0.0));
    for index in 0..N_PARAMS {
        let p = arena.alloc(Node::InputParameter {
            name: format!("p{index}"),
            index,
            offset: index,
            width: 1,
        });
        sum = arena.alloc(Node::Add(sum, p));
    }
    let neg = arena.alloc(Node::Scalar(-1.0));
    let rate = arena.alloc(Node::Mul(neg, sum));

    let rows: Vec<_> = (0..N_STATES)
        .map(|i| {
            let y_i = arena.alloc(Node::StateVector {
                start: i,
                end: i + 1,
            });
            arena.alloc(Node::Mul(rate, y_i))
        })
        .collect();
    let rhs = arena.alloc(Node::Concat(rows));

    CompiledModel::new(
        &arena,
        rhs,
        identity_mass_matrix(N_STATES),
        N_STATES,
        PackedWidth::new(N_PARAMS),
        CompiledModelOptions {
            sensitivities: sens,
            algebraic: None,
        },
    )
    .expect("a valid fixture")
}

fn bench_sens_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("sens_params");
    let all: Vec<ParamIdx> = (0..N_PARAMS).map(ParamIdx::from_usize).collect();
    let y0 = vec![1.0; N_STATES];
    let inputs = vec![1.0 / N_PARAMS as f64; N_PARAMS];
    let atol = vec![1e-8; N_STATES];
    let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.05).collect();

    for k in [1usize, 4, N_PARAMS] {
        let prepared = PreparedSolver::new(build_dense_decay_model(&all[..k]), 1e-8, &atol)
            .expect("PreparedSolver failed");
        group.bench_function(BenchmarkId::from_parameter(k), |b| {
            b.iter(|| {
                black_box(
                    prepared
                        .solve(
                            SolveRequest::new(&t_eval).with_sensitivities(),
                            InputSet::new(&y0, &inputs),
                        )
                        .expect("sensitivity solve failed"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sens_params);
criterion_main!(benches);

//! Benchmark: prepare-once cost, and reuse versus no reuse of that handle.
//! Per-solve diffsol setup runs inside `solve`; read it off `SolverStatistics`.

mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::{FIXTURES, compile_fixture, fixture_state};
use pybamm_core::DagSnapshot;
use pybamm_core::solver::solve::{InputSet, PreparedSolver, SolveRequest};

/// Benchmark A: `PreparedSolver::new()` only, excluding model compilation.
fn bench_solver_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_setup");
    group.sample_size(100);

    for case in FIXTURES {
        let snap = DagSnapshot::from_bytes(case.bytes).expect("fixture decodes");

        group.bench_with_input(
            BenchmarkId::from_parameter(case.name),
            &case.name,
            |b, _| {
                b.iter_batched(
                    || compile_fixture(&snap),
                    |model| {
                        let n_states = model.n_states();
                        let atol = vec![1e-6; n_states];
                        let prepared = PreparedSolver::new(model, 1e-6, &atol).unwrap();
                        black_box(prepared);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark B: `solve()` on an already-constructed `PreparedSolver` (reuse path)
fn bench_solve_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_reuse");
    group.sample_size(50);

    for case in FIXTURES {
        let snap = DagSnapshot::from_bytes(case.bytes).expect("fixture decodes");
        let model = compile_fixture(&snap);
        let n_states = model.n_states();
        let (y0, inputs) = fixture_state(n_states, model.input_width().get());
        let atol = vec![1e-6; n_states];

        let prepared = PreparedSolver::new(model, 1e-6, &atol).unwrap();

        let t_eval: Vec<f64> = (0..100).map(|i| f64::from(i) * 36.0).collect();

        if prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
            .is_err()
        {
            eprintln!(
                "Skipping {} solve_reuse: dummy y0 doesn't converge",
                case.name
            );
            continue;
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(case.name),
            &case.name,
            |b, _| {
                b.iter(|| {
                    let result = prepared
                        .solve(
                            SolveRequest::new(black_box(&t_eval)),
                            InputSet::new(black_box(&y0), black_box(&inputs)),
                        )
                        .unwrap();
                    black_box(&result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark C: fresh `PreparedSolver::new()` + `solve()` (no-reuse path).
fn bench_fresh_setup_and_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("fresh_setup_and_solve");
    group.sample_size(50);

    for case in FIXTURES {
        let snap = DagSnapshot::from_bytes(case.bytes).expect("fixture decodes");
        let n_states = snap.n_states;
        let (y0, inputs) = fixture_state(n_states, snap.packed_input_width.get());
        let t_eval: Vec<f64> = (0..100).map(|i| f64::from(i) * 36.0).collect();

        {
            let model = compile_fixture(&snap);
            let atol = vec![1e-6; n_states];
            let prepared = PreparedSolver::new(model, 1e-6, &atol).unwrap();
            if prepared
                .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
                .is_err()
            {
                eprintln!(
                    "Skipping {} fresh_setup_and_solve: dummy y0 doesn't converge",
                    case.name
                );
                continue;
            }
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(case.name),
            &case.name,
            |b, _| {
                b.iter_batched(
                    || compile_fixture(&snap),
                    |model| {
                        let atol = vec![1e-6; n_states];
                        let prepared = PreparedSolver::new(model, 1e-6, &atol).unwrap();
                        let result = prepared
                            .solve(
                                SolveRequest::new(black_box(&t_eval)),
                                InputSet::new(black_box(&y0), black_box(&inputs)),
                            )
                            .unwrap();
                        black_box(&result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_solver_setup,
    bench_solve_reuse,
    bench_fresh_setup_and_solve,
);
criterion_main!(benches);

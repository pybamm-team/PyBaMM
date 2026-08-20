//! Benchmark: solver setup cost vs solve cost.
//!
//! Measures `PreparedSolver::new()` separately from `solve()` to confirm
//! that building a fresh diffsol solver per solve is cheap relative to the
//! solve itself. This justifies the immutable `PreparedSolver` design, which
//! constructs a fresh `Workspace` and BDF solver on every `solve` rather than
//! holding a mutable, reused solver across calls.

#![cfg(feature = "serialize")]

mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::identity_mass_matrix;
use pybamm_core::solver::solve::{InputSet, PreparedSolver, SolveRequest};
use pybamm_core::{DagSnapshot, ModelEvaluator};

const SPM: &[u8] = include_bytes!("fixtures/spm.bin");
const SPME: &[u8] = include_bytes!("fixtures/spme.bin");
const DFN: &[u8] = include_bytes!("fixtures/dfn.bin");

struct TestCase {
    name: &'static str,
    bytes: &'static [u8],
}

const CASES: &[TestCase] = &[
    TestCase {
        name: "SPM",
        bytes: SPM,
    },
    TestCase {
        name: "SPMe",
        bytes: SPME,
    },
    TestCase {
        name: "DFN",
        bytes: DFN,
    },
];

fn build_compiled_model(snap: &DagSnapshot) -> ModelEvaluator {
    let mass = snap
        .mass_matrix
        .clone()
        .unwrap_or_else(|| identity_mass_matrix(snap.n_states));
    ModelEvaluator::new(&snap.arena, snap.root, mass, snap.n_states, snap.n_params)
}

/// Benchmark A: `PreparedSolver::new()` only (BDF solver construction).
///
/// Uses `iter_batched` so that `build_compiled_model()` runs per iteration
/// but is NOT included in the timing. This isolates the diffsol/BDF setup
/// cost from the expression compilation cost.
fn bench_solver_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_setup");
    group.sample_size(100);

    for case in CASES {
        let snap = DagSnapshot::from_bytes(case.bytes);

        group.bench_with_input(
            BenchmarkId::from_parameter(case.name),
            &case.name,
            |b, _| {
                b.iter_batched(
                    || build_compiled_model(&snap),
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

/// `ModelEvaluator::new()` only: differentiation, simplification, sparsity
/// detection, coloring and bytecode compilation.
fn bench_model_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_compilation");
    group.sample_size(50);

    for case in CASES {
        let snap = DagSnapshot::from_bytes(case.bytes);

        group.bench_with_input(
            BenchmarkId::from_parameter(case.name),
            &case.name,
            |b, _| {
                b.iter(|| {
                    let model = build_compiled_model(&snap);
                    black_box(model);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark B: `solve()` on an already-constructed `PreparedSolver` (reuse path)
fn bench_solve_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_reuse");
    group.sample_size(50);

    for case in CASES {
        let snap = DagSnapshot::from_bytes(case.bytes);
        let model = build_compiled_model(&snap);
        let n_states = model.n_states();
        let n_params = model.n_params();
        let atol = vec![1e-6; n_states];

        let prepared = PreparedSolver::new(model, 1e-6, &atol).unwrap();

        // Use a small perturbation around 0.5 as initial state, physically
        // more reasonable than all-ones for battery models.
        let y0: Vec<f64> = (0..n_states)
            .map(|i| 0.01f64.mul_add(i as f64 / n_states as f64, 0.5))
            .collect();
        let inputs: Vec<f64> = vec![0.0; n_params];
        let t_eval: Vec<f64> = (0..100).map(|i| f64::from(i) * 36.0).collect();

        // Warmup: skip this model if the dummy y0 doesn't converge
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

/// Benchmark C: fresh `PreparedSolver::new()` + `solve()` (no-reuse path)
fn bench_fresh_setup_and_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("fresh_setup_and_solve");
    group.sample_size(50);

    for case in CASES {
        let snap = DagSnapshot::from_bytes(case.bytes);
        let n_states = snap.n_states;
        let n_params = snap.n_params;

        let y0: Vec<f64> = (0..n_states)
            .map(|i| 0.01f64.mul_add(i as f64 / n_states as f64, 0.5))
            .collect();
        let inputs: Vec<f64> = vec![0.0; n_params];
        let t_eval: Vec<f64> = (0..100).map(|i| f64::from(i) * 36.0).collect();

        // Skip models where dummy y0 doesn't converge
        {
            let model = build_compiled_model(&snap);
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
                    || build_compiled_model(&snap),
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
    bench_model_compilation,
    bench_solver_setup,
    bench_solve_reuse,
    bench_fresh_setup_and_solve,
);
criterion_main!(benches);

//! Solving with faer's process-global parallelism disabled, as a host embedding faer may.
//!
//! A binary of its own: the setting is process-wide, and once it is disabled any faer call
//! that still reads it panics.

use std::sync::Arc;

use faer::Par;
use pybamm_core::solver::solve::{InputSet, PreparedSolver, SolveOutcome, SolveRequest};
use pybamm_core::{Arena, CompiledModel, CompiledModelOptions, CsrData, Node, PackedWidth, Shape};

/// Disables faer's global parallelism until dropped, then restores what it was.
struct GlobalParallelismDisabled(Par);

impl GlobalParallelismDisabled {
    fn new() -> Self {
        let previous = faer::get_global_parallelism();
        faer::disable_global_parallelism();
        Self(previous)
    }
}

impl Drop for GlobalParallelismDisabled {
    fn drop(&mut self) {
        faer::set_global_parallelism(self.0);
    }
}

fn compile(arena: &Arena, rhs: pybamm_core::NodeId, mass: CsrData, inputs: usize) -> CompiledModel {
    let n_states = mass.shape().rows;
    let options = CompiledModelOptions::default();
    CompiledModel::new(
        arena,
        rhs,
        mass,
        n_states,
        PackedWidth::new(inputs),
        options,
    )
    .expect("a valid fixture")
}

/// `dy/dt = -a*y` with `0 = 2y - z - z^3`: each held derivative factors the algebraic block.
fn nonlinear_dae() -> CompiledModel {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let z = arena.alloc(Node::StateVector { start: 1, end: 2 });
    let a = arena.alloc(Node::InputParameter {
        name: "a".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let neg = arena.alloc(Node::Scalar(-1.0));
    let neg_a = arena.alloc(Node::Mul(neg, a));
    let dydt = arena.alloc(Node::Mul(neg_a, y));
    let two = arena.alloc(Node::Scalar(2.0));
    let two_y = arena.alloc(Node::Mul(two, y));
    let z2 = arena.alloc(Node::Mul(z, z));
    let z3 = arena.alloc(Node::Mul(z2, z));
    let z_plus_z3 = arena.alloc(Node::Add(z, z3));
    let constraint = arena.alloc(Node::Sub(two_y, z_plus_z3));
    let rhs = arena.alloc(Node::Concat(vec![dydt, constraint]));
    let mass = CsrData::try_new(vec![0, 1, 1], vec![0], vec![1.0], Shape::matrix(2, 2))
        .expect("diag(1, 0)");
    compile(&arena, rhs, mass, 1)
}

/// `M dy/dt = -a*y` with `M = [[2, 1], [0, 1]]`: each held derivative solves against `M`.
fn ode_with_a_non_diagonal_mass() -> CompiledModel {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let a = arena.alloc(Node::InputParameter {
        name: "a".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let neg = arena.alloc(Node::Scalar(-1.0));
    let neg_a = arena.alloc(Node::Mul(neg, a));
    let rhs = arena.alloc(Node::Mul(neg_a, y));
    let mass = CsrData::try_new(
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![2.0, 1.0, 1.0],
        Shape::matrix(2, 2),
    )
    .expect("an upper-triangular M");
    compile(&arena, rhs, mass, 1)
}

// Bit-identical is the property under test.
#[allow(clippy::float_cmp)]
fn assert_same_trajectory(got: &SolveOutcome, want: &SolveOutcome, context: &str) {
    assert_eq!(got.t, want.t, "{context}: times differ");
    assert_eq!(got.y, want.y, "{context}: trajectories differ");
    assert_eq!(got.yp, want.yp, "{context}: yp differs");
}

#[test]
fn every_lu_honours_the_requested_parallelism_not_the_global_one() {
    let t_eval: Vec<f64> = (0..=8).map(|i| f64::from(i) * 0.25).collect();
    // A stop at every time holds a consistent derivative at each one.
    let request = SolveRequest::new(&t_eval).with_stop_times(&t_eval);
    let rates: Vec<[f64; 1]> = (1..=4).map(|i| [0.5 * f64::from(i)]).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool build failed");

    for (label, model, y0) in [
        ("DAE", nonlinear_dae(), [1.0, 1.0]),
        ("non-diagonal M", ode_with_a_non_diagonal_mass(), [1.0, 0.5]),
    ] {
        let model = Arc::new(model);
        let prepare = || {
            PreparedSolver::new(Arc::clone(&model), 1e-8, &[1e-8, 1e-8])
                .expect("prepare")
                .with_store_yp(true)
        };
        let sets: Vec<InputSet<'_>> = rates.iter().map(|a| InputSet::new(&y0, a)).collect();
        let serial: Vec<SolveOutcome> = {
            let prepared = prepare();
            sets.iter()
                .map(|set| prepared.solve(request, *set).expect("serial solve"))
                .collect()
        };

        let _disabled = GlobalParallelismDisabled::new();
        let prepared = prepare();
        let batch = pool.install(|| prepared.solve_batch(request, &sets));
        for (i, (got, want)) in batch.into_iter().zip(&serial).enumerate() {
            let got = got.expect("batched solve");
            assert_same_trajectory(&got, want, &format!("{label}, set {i} of the batch"));
        }
        let threaded = pool
            .install(|| prepared.solve(request.with_lu_threads(4), sets[0]))
            .expect("threaded solve");
        assert_same_trajectory(&threaded, &serial[0], &format!("{label}, 4 LU threads"));
    }
}

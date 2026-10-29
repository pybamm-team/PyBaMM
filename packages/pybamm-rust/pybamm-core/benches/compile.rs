mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::{build_coupled, identity_mass_matrix, slot_stats};
use pybamm_core::{
    Arena, CompiledModel, CompiledModelOptions, DagRef, Node, NodeId, PackedWidth, TypedIr,
};

fn build_linear_chain(arena: &mut Arena, depth: usize, vec_len: usize) -> NodeId {
    let y = arena.alloc(Node::StateVector {
        start: 0,
        end: vec_len,
    });
    let mut current = y;
    for _ in 0..depth {
        current = arena.alloc(Node::Sin(current));
        current = arena.alloc(Node::Neg(current));
    }
    current
}

fn build_wide_fanout(arena: &mut Arena, width: usize) -> NodeId {
    let mut terms = Vec::with_capacity(width);
    for i in 0..width {
        let y = arena.alloc(Node::StateVector {
            start: i * 10,
            end: (i + 1) * 10,
        });
        let a = arena.alloc(Node::Sin(y));
        let b = arena.alloc(Node::Exp(a));
        terms.push(b);
    }
    arena.alloc(Node::Concat(terms))
}

fn build_high_fanout(arena: &mut Arena, fanout: usize) -> NodeId {
    let y = arena.alloc(Node::StateVector { start: 0, end: 50 });
    let mut terms = Vec::with_capacity(fanout);
    for _ in 0..fanout {
        let a = arena.alloc(Node::Sin(y));
        let b = arena.alloc(Node::Exp(a));
        terms.push(b);
    }
    arena.alloc(Node::Concat(terms))
}

fn build_diamond_dag(arena: &mut Arena, depth: usize) -> NodeId {
    let y = arena.alloc(Node::StateVector { start: 0, end: 10 });
    let two = arena.alloc(Node::Scalar(2.0));

    let mut layer = vec![y];
    for _ in 0..depth {
        let mut next_layer = Vec::new();
        for &node in &layer {
            let a = arena.alloc(Node::Sin(node));
            let b = arena.alloc(Node::Mul(node, two));
            next_layer.push(a);
            next_layer.push(b);
        }
        let cap = next_layer.len().min(16);
        next_layer.truncate(cap);
        layer = next_layer;
    }

    let mut acc = layer[0];
    for &node in &layer[1..] {
        acc = arena.alloc(Node::Add(acc, node));
    }
    acc
}

/// One synthetic topology: its label, builder and the sizes it is benchmarked at.
type Topology = (
    &'static str,
    fn(&mut Arena, usize) -> NodeId,
    &'static [usize],
);

/// Lowering time per topology, printing each case's slot reuse first.
fn bench_dag_to_ir(c: &mut Criterion) {
    let topologies: [Topology; 5] = [
        (
            "linear_chain",
            |arena, depth| build_linear_chain(arena, depth, 10),
            &[10, 50, 100, 500],
        ),
        ("wide_fanout", build_wide_fanout, &[10, 50, 100, 500]),
        ("diamond", build_diamond_dag, &[4, 6, 8, 10]),
        ("high_fanout", build_high_fanout, &[20, 100]),
        ("coupled", build_coupled, &[50, 100, 200, 500]),
    ];

    let mut group = c.benchmark_group("dag_to_ir");
    group.sample_size(20);

    println!(
        "\n{:>25} {:>8} {:>8} {:>8} {:>6}",
        "topology", "naive", "actual", "saved", "ratio"
    );
    println!("{:-<25} {:-<8} {:-<8} {:-<8} {:-<6}", "", "", "", "", "");

    for (label, build, sizes) in topologies {
        for &size in sizes {
            let mut arena = Arena::new();
            let root = build(&mut arena, size);
            let (naive, buffer, ratio) = slot_stats(&arena, root);
            println!(
                "{:>25} {naive:>8} {buffer:>8} {:>8} {ratio:>6.3}",
                format!("{label}({size})"),
                naive.saturating_sub(buffer)
            );
            group.bench_with_input(BenchmarkId::new(label, size), &size, |b, _| {
                b.iter(|| TypedIr::from_dag(black_box(DagRef::new(&arena, root))));
            });
        }
    }

    group.finish();
}

fn bench_model_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_model_new");
    group.sample_size(10);

    for n in [50, 100, 200, 500] {
        group.bench_with_input(BenchmarkId::new("coupled", n), &n, |b, &n| {
            b.iter(|| {
                let mut arena = Arena::new();
                let root = build_coupled(&mut arena, n);
                let mass = identity_mass_matrix(n);
                CompiledModel::new(
                    black_box(&arena),
                    root,
                    mass,
                    n,
                    PackedWidth::new(0),
                    CompiledModelOptions::default(),
                )
                .expect("a valid fixture")
            });
        });
    }

    group.finish();
}

#[cfg(feature = "serialize")]
fn bench_real_models(c: &mut Criterion) {
    use pybamm_core::{DagSnapshot, SplitCompiledExpr};

    let mut group = c.benchmark_group("real_model_compile");
    group.sample_size(20);

    println!(
        "\n{:>6} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "model", "states", "nodes", "naive", "actual", "ratio"
    );
    println!(
        "{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<6}",
        "", "", "", "", "", ""
    );

    for fixture in helpers::FIXTURES {
        let snap = DagSnapshot::from_bytes(fixture.bytes).expect("fixture decodes");
        let (naive, buffer, ratio) = slot_stats(snap.dag().arena(), snap.dag().root());
        let eval_order = snap.dag().arena().topological_order(snap.dag().root());

        println!(
            "{:>6} {:>8} {:>8} {naive:>8} {buffer:>8} {ratio:>6.3}",
            fixture.name,
            snap.n_states,
            eval_order.len(),
        );

        group.bench_with_input(
            BenchmarkId::new("dag_to_ir", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| TypedIr::from_dag(black_box(snap.dag())));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dag_to_split_ir", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| SplitCompiledExpr::compile(black_box(snap.dag())));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compiled_model_new", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| helpers::compile_fixture(black_box(&snap)));
            },
        );
    }

    group.finish();
}

/// A model build with outputs and events, the observable compile cost included.
#[cfg(feature = "serialize")]
fn bench_real_models_with_observables(c: &mut Criterion) {
    use pybamm_core::{DagSnapshot, ObservableKind};

    const N_OBSERVABLES: usize = 10;

    let mut group = c.benchmark_group("real_model_compile_observables");
    group.sample_size(20);

    for fixture in helpers::FIXTURES {
        let snap = DagSnapshot::from_bytes(fixture.bytes).expect("fixture decodes");
        let mut arena = snap.dag().arena().clone();
        let root = snap.dag().root();
        let n = snap.n_states;
        let threshold = arena.alloc(Node::Scalar(0.5));
        let rows: Vec<NodeId> = (0..N_OBSERVABLES)
            .map(|i| {
                let start = i * n / N_OBSERVABLES;
                arena.alloc(Node::Index {
                    child: root,
                    start,
                    end: start + 1,
                })
            })
            .collect();
        let events: Vec<NodeId> = rows
            .iter()
            .map(|&row| arena.alloc(Node::Sub(row, threshold)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("outputs_and_events", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter_batched(
                    || arena.clone(),
                    |mut arena| {
                        // The observables extend a clone of the snapshot's arena, whose root
                        // reaches none of them, so the snapshot compiles the same model.
                        let mut model = helpers::compile_fixture(black_box(&snap));
                        model
                            .set_observables(ObservableKind::Outputs, &mut arena, &rows)
                            .expect("valid outputs");
                        model
                            .set_observables(ObservableKind::Events, &mut arena, &events)
                            .expect("valid events");
                        model
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "serialize"))]
const fn bench_real_models(_c: &mut Criterion) {}

#[cfg(not(feature = "serialize"))]
const fn bench_real_models_with_observables(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_dag_to_ir,
    bench_model_compile,
    bench_real_models,
    bench_real_models_with_observables,
);
criterion_main!(benches);

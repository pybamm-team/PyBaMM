mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::{build_coupled, identity_mass_matrix};
use pybamm_core::{Arena, ModelEvaluator, Node, NodeId, TypedIr};

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

fn bench_dag_to_ir(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_to_ir");
    group.sample_size(20);

    for depth in [10, 50, 100, 500] {
        let mut arena = Arena::new();
        let root = build_linear_chain(&mut arena, depth, 10);
        group.bench_with_input(BenchmarkId::new("linear_chain", depth), &depth, |b, _| {
            b.iter(|| TypedIr::from_arena(black_box(&arena), black_box(root)));
        });
    }

    for width in [10, 50, 100, 500] {
        let mut arena = Arena::new();
        let root = build_wide_fanout(&mut arena, width);
        group.bench_with_input(BenchmarkId::new("wide_fanout", width), &width, |b, _| {
            b.iter(|| TypedIr::from_arena(black_box(&arena), black_box(root)));
        });
    }

    for depth in [4, 6, 8, 10] {
        let mut arena = Arena::new();
        let root = build_diamond_dag(&mut arena, depth);
        group.bench_with_input(BenchmarkId::new("diamond", depth), &depth, |b, _| {
            b.iter(|| TypedIr::from_arena(black_box(&arena), black_box(root)));
        });
    }

    for fanout in [20, 100] {
        let mut arena = Arena::new();
        let root = build_high_fanout(&mut arena, fanout);
        group.bench_with_input(BenchmarkId::new("high_fanout", fanout), &fanout, |b, _| {
            b.iter(|| TypedIr::from_arena(black_box(&arena), black_box(root)));
        });
    }

    for n in [50, 100, 200, 500] {
        let mut arena = Arena::new();
        let root = build_coupled(&mut arena, n);
        group.bench_with_input(BenchmarkId::new("coupled", n), &n, |b, _| {
            b.iter(|| TypedIr::from_arena(black_box(&arena), black_box(root)));
        });
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
                let _model = ModelEvaluator::new(black_box(&arena), root, mass, n, 0);
            });
        });
    }

    group.finish();
}

struct Case {
    label: &'static str,
    param: usize,
}

fn bench_slot_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("slot_stats");
    group.sample_size(20);

    let cases = [
        Case {
            label: "linear_chain",
            param: 100,
        },
        Case {
            label: "linear_chain",
            param: 500,
        },
        Case {
            label: "wide_fanout",
            param: 50,
        },
        Case {
            label: "wide_fanout",
            param: 200,
        },
        Case {
            label: "high_fanout",
            param: 20,
        },
        Case {
            label: "high_fanout",
            param: 100,
        },
        Case {
            label: "coupled",
            param: 100,
        },
        Case {
            label: "coupled",
            param: 500,
        },
    ];

    println!(
        "\n{:>25} {:>8} {:>8} {:>8} {:>6}",
        "topology", "naive", "actual", "saved", "ratio"
    );
    println!("{:-<25} {:-<8} {:-<8} {:-<8} {:-<6}", "", "", "", "", "");

    for case in &cases {
        let mut arena = Arena::new();
        let root = match case.label {
            "linear_chain" => build_linear_chain(&mut arena, case.param, 10),
            "wide_fanout" => build_wide_fanout(&mut arena, case.param),
            "high_fanout" => build_high_fanout(&mut arena, case.param),
            "coupled" => build_coupled(&mut arena, case.param),
            _ => unreachable!(),
        };

        let stats = TypedIr::slot_stats(&arena, root);
        let saved = stats.naive_size.saturating_sub(stats.buffer_size);
        let tag = format!("{}({})", case.label, case.param);
        println!(
            "{tag:>25} {:>8} {:>8} {:>8} {:>6.3}",
            stats.naive_size, stats.buffer_size, saved, stats.reuse_ratio
        );

        group.bench_with_input(BenchmarkId::new("compile", &tag), &case.param, |b, _| {
            b.iter(|| TypedIr::from_arena(black_box(&arena), black_box(root)));
        });
    }

    group.finish();
}

#[cfg(feature = "serialize")]
fn bench_real_models(c: &mut Criterion) {
    use pybamm_core::DagSnapshot;

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
        let snap = DagSnapshot::from_bytes(fixture.bytes);
        let stats = TypedIr::slot_stats(&snap.arena, snap.root);
        let eval_order = snap.arena.topological_order(snap.root);

        println!(
            "{:>6} {:>8} {:>8} {:>8} {:>8} {:>6.3}",
            fixture.name,
            snap.n_states,
            eval_order.len(),
            stats.naive_size,
            stats.buffer_size,
            stats.reuse_ratio,
        );

        group.bench_with_input(
            BenchmarkId::new("dag_to_ir", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| TypedIr::from_arena(black_box(&snap.arena), black_box(snap.root)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dag_to_split_ir", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| {
                    TypedIr::from_arena_split_eval(black_box(&snap.arena), black_box(snap.root))
                });
            },
        );

        // Surfaces any memory-driven regression in the whole of
        // `ModelEvaluator::new` that the focused detect_sparsity bench misses.
        group.bench_with_input(
            BenchmarkId::new("compiled_model_new", fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| {
                    let mass = snap
                        .mass_matrix
                        .clone()
                        .unwrap_or_else(|| identity_mass_matrix(snap.n_states));
                    ModelEvaluator::new(
                        black_box(&snap.arena),
                        black_box(snap.root),
                        mass,
                        snap.n_states,
                        snap.n_params,
                    )
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "serialize"))]
const fn bench_real_models(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_dag_to_ir,
    bench_model_compile,
    bench_slot_stats,
    bench_real_models,
);
criterion_main!(benches);

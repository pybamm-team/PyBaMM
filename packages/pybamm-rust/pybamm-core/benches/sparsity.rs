//! Focused microbench for sparsity analysis on real model fixtures.

mod helpers;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helpers::FIXTURES;
use pybamm_core::{DagSnapshot, TypedIr, detect_sparsity_per_output};

fn bench_detect_sparsity(c: &mut Criterion) {
    let mut group = c.benchmark_group("detect_sparsity_per_output");
    group.sample_size(50);

    for fixture in FIXTURES {
        let snap = DagSnapshot::from_bytes(fixture.bytes).expect("fixture decodes");
        let ir = TypedIr::from_dag(snap.dag());
        let n_outputs = ir.output_len();

        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.name),
            &fixture.name,
            |b, _| {
                b.iter(|| {
                    detect_sparsity_per_output(
                        black_box(snap.dag().arena()),
                        black_box(snap.dag().root()),
                        black_box(n_outputs),
                        black_box(snap.n_states),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_detect_sparsity);
criterion_main!(benches);

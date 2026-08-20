//! Focused microbench for sparsity analysis on real model fixtures.

#![cfg(feature = "serialize")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use pybamm_core::{DagSnapshot, TypedIr, detect_sparsity_per_output};

const SPM: &[u8] = include_bytes!("fixtures/spm.bin");
const SPME: &[u8] = include_bytes!("fixtures/spme.bin");
const DFN: &[u8] = include_bytes!("fixtures/dfn.bin");

fn bench_detect_sparsity(c: &mut Criterion) {
    let mut group = c.benchmark_group("detect_sparsity_per_output");
    group.sample_size(50);

    for (name, bytes) in [("SPM", SPM), ("SPMe", SPME), ("DFN", DFN)] {
        let snap = DagSnapshot::from_bytes(bytes);
        let ir = TypedIr::from_arena(&snap.arena, snap.root);
        let n_outputs = ir.output_len();

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                detect_sparsity_per_output(
                    black_box(&snap.arena),
                    black_box(snap.root),
                    black_box(n_outputs),
                    black_box(snap.n_states),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_detect_sparsity);
criterion_main!(benches);

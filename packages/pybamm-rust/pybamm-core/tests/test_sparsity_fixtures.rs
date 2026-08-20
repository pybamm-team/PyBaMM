//! Snapshot regression tests for sparsity patterns on real models.
//!
//! Run with: cargo test --features serialize --test `test_sparsity_fixtures`
//!
//! To regenerate the .bin fixtures (e.g., after a deliberate, audited
//! semantic change), run the #[ignore]'d test:
//!   cargo test --features serialize --test `test_sparsity_fixtures` \
//!     -- --ignored `regenerate_golden_fixtures` --nocapture

#![cfg(feature = "serialize")]

use pybamm_core::{DagSnapshot, JacobianData, TypedIr, detect_sparsity_per_output};
use std::fs;
use std::path::Path;

const SPM: &[u8] = include_bytes!("../benches/fixtures/spm.bin");
const SPME: &[u8] = include_bytes!("../benches/fixtures/spme.bin");
const DFN: &[u8] = include_bytes!("../benches/fixtures/dfn.bin");

/// `(constant entries, colours)` of each fixture's compiled state Jacobian.
///
/// Inline rather than in the `.bin` fixtures: these are two integers a human
/// has to weigh on every classifier change, so a change should read as a diff
/// here instead of as three rewritten binary blobs.
const JACOBIAN_COUNTS: [(&str, usize, usize); 3] =
    [("spm", 116, 0), ("spme", 116, 3), ("dfn", 2316, 9)];

fn compute_pattern(bytes: &[u8]) -> (usize, usize, Vec<usize>, Vec<usize>) {
    let snap = DagSnapshot::from_bytes(bytes);
    let ir = TypedIr::from_arena(&snap.arena, snap.root);
    let n_outputs = ir.output_len();
    let p = detect_sparsity_per_output(&snap.arena, snap.root, n_outputs, snap.n_states);
    (p.nrows, p.ncols, p.indptr.clone(), p.indices)
}

#[test]
#[ignore = "regeneration only — produces .bin fixture files"]
fn regenerate_golden_fixtures() {
    let dir = Path::new("tests/fixtures");
    fs::create_dir_all(dir).expect("create fixtures dir");
    for (name, bytes) in [("spm", SPM), ("spme", SPME), ("dfn", DFN)] {
        let payload = compute_pattern(bytes);
        let encoded = bincode::serialize(&payload).expect("serialize");
        let path = dir.join(format!("sparsity_{name}.bin"));
        fs::write(&path, &encoded).expect("write fixture");
        println!(
            "wrote {} ({} bytes, nrows={}, ncols={}, nnz={})",
            path.display(),
            encoded.len(),
            payload.0,
            payload.1,
            payload.3.len()
        );
    }
}

fn load_golden(name: &str) -> (usize, usize, Vec<usize>, Vec<usize>) {
    let path = format!("tests/fixtures/sparsity_{name}.bin");
    let bytes = fs::read(&path).unwrap_or_else(|e| {
        panic!("failed to read {path}: {e}. Run --ignored regenerate_golden_fixtures.")
    });
    bincode::deserialize(&bytes).expect("deserialize golden")
}

fn check_against_golden(name: &str, bytes: &[u8]) {
    let actual = compute_pattern(bytes);
    let golden = load_golden(name);
    assert_eq!(actual.0, golden.0, "{name}: nrows");
    assert_eq!(actual.1, golden.1, "{name}: ncols");
    assert_eq!(actual.2, golden.2, "{name}: indptr");
    assert_eq!(actual.3, golden.3, "{name}: indices");
}

/// A classifier that silently proved nothing would cost only sweeps, so the
/// pattern fixtures above would still pass. These pin what it resolved.
#[test]
fn jacobian_counts_match_the_pinned_table() {
    for (name, constants, colors) in JACOBIAN_COUNTS {
        let bytes = match name {
            "spm" => SPM,
            "spme" => SPME,
            _ => DFN,
        };
        let snap = DagSnapshot::from_bytes(bytes);
        let n_outputs = TypedIr::from_arena(&snap.arena, snap.root).output_len();
        let jac = JacobianData::new_wrt_states(&snap.arena, snap.root, n_outputs, snap.n_states);
        assert_eq!(
            jac.constant_csr_entries().len(),
            constants,
            "{name}: constant entries"
        );
        assert_eq!(jac.coloring().n_colors, colors, "{name}: colours");
    }
}

#[test]
fn sparsity_spm_matches_golden() {
    check_against_golden("spm", SPM);
}

#[test]
fn sparsity_spme_matches_golden() {
    check_against_golden("spme", SPME);
}

#[test]
fn sparsity_dfn_matches_golden() {
    check_against_golden("dfn", DFN);
}

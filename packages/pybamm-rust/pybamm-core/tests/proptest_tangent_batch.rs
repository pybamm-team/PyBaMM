mod common;

use common::cases::{TangentCase, arb_split_eval_case, targeted_split_eval_cases};
use common::numeric_eq::assert_bitwise_eq;
use proptest::prelude::*;
use pybamm_core::tangent_batch::{is_batchable, run_tangent_batch, tangent_scratch_len};
use pybamm_core::{CompiledExpr, TangentInputs, TypedIr, tangent_wrt_states};

const LANES: usize = 4;

/// A batched sweep must reproduce `LANES` independent scalar tangent sweeps bit
/// for bit: each lane accumulates in the same order, so nothing may drift.
fn check_batched_matches_scalar(case: &TangentCase) {
    let mut arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut arena, case.root);
    let ir = TypedIr::from_arena_split_eval(&arena, tangent_root);
    if !is_batchable(&ir) {
        return;
    }
    let split = ir.split_eval_info().expect("split-eval tape");
    let primal_len = split.primal_buffer_size;
    let n_states = case.y.len();

    // Pad to a full block so short cases still exercise every lane.
    let seeds: Vec<Vec<f64>> = (0..LANES)
        .map(|lane| {
            case.seeds
                .get(lane)
                .cloned()
                .unwrap_or_else(|| vec![0.0; n_states])
        })
        .collect();

    let compiled = CompiledExpr::from_ir(ir);
    let mut scratch = vec![0.0; compiled.scratch_len()];

    let mut cache = compiled.eval_primal(&mut scratch, 0.0, &case.y, &[], &[]);
    let scalar: Vec<Vec<f64>> = seeds
        .iter()
        .map(|seed| {
            cache
                .eval_tangent(&TangentInputs {
                    dy: Some(seed),
                    dp: None,
                })
                .to_vec()
        })
        .collect();

    let mut lane_seeds = vec![0.0; n_states * LANES];
    for (lane, seed) in seeds.iter().enumerate() {
        for (state, &value) in seed.iter().enumerate() {
            lane_seeds[state * LANES + lane] = value;
        }
    }

    let ir = compiled.ir();
    let mut tan = vec![0.0; tangent_scratch_len(ir, LANES)];
    compiled.run_primal_section(&mut scratch, 0.0, &case.y, &[], &[]);
    let batched = run_tangent_batch::<LANES>(ir, &scratch[..primal_len], &mut tan, &lane_seeds);

    for (lane, want) in scalar.iter().enumerate() {
        let got: Vec<f64> = (0..want.len()).map(|e| batched[e * LANES + lane]).collect();
        assert_bitwise_eq(want, &got);
    }
}

#[test]
fn targeted_batched_tangent_matches_scalar() {
    for case in targeted_split_eval_cases() {
        check_batched_matches_scalar(&case);
    }
}

proptest! {
    #[test]
    fn batched_tangent_matches_scalar(case in arb_split_eval_case()) {
        check_batched_matches_scalar(&case);
    }
}

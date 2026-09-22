mod common;

use common::cases::{
    ALL_EXTRAS, TangentCase, arb_split_eval_case, arb_split_eval_case_with,
    targeted_split_eval_cases,
};
use common::numeric_eq::assert_bitwise_eq;
use proptest::prelude::*;
use pybamm_core::tangent_batch::{is_batchable, run_tangent_batch, tangent_scratch_len};
use pybamm_core::{Arena, DagRef, Node, SplitCompiledExpr, TangentInputs, tangent_wrt_states};

/// A batched tangent sweep reproduces its scalar sweeps bit for bit, at every lane count.
fn check_batched_matches_scalar(case: &TangentCase) {
    for lanes in [4, 8] {
        check_lanes(case, lanes);
    }
}

fn check_lanes(case: &TangentCase, lanes: usize) {
    let mut arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut arena, case.root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&arena, tangent_root));
    if !is_batchable(&compiled) {
        return;
    }
    let primal_len = compiled.split_info().primal_buffer_size;
    let n_states = case.y.len();

    // Fill every lane, reusing the case's seeds rescaled, so no lane rides along as zeros.
    let seeds: Vec<Vec<f64>> = (0..lanes)
        .map(|lane| match case.seeds.len() {
            0 => vec![0.0; n_states],
            len => {
                let scale = (lane / len + 1) as f64;
                case.seeds[lane % len].iter().map(|v| v * scale).collect()
            },
        })
        .collect();

    let mut scratch = vec![0.0; compiled.scratch_len()];

    let mut cache = compiled.eval_primal(&mut scratch, 0.0, &case.y, &[], &case.inputs);
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

    let mut lane_seeds = vec![0.0; n_states * lanes];
    for (lane, seed) in seeds.iter().enumerate() {
        for (state, &value) in seed.iter().enumerate() {
            lane_seeds[state * lanes + lane] = value;
        }
    }

    // Scalar sweeps write only tangent slots, so the primal region is still this point's.
    let mut tan = vec![0.0; tangent_scratch_len(&compiled, lanes)];
    let primal = &scratch[..primal_len];
    let batched = run_tangent_batch(&compiled, lanes, primal, &mut tan, &lane_seeds);

    for (lane, want) in scalar.iter().enumerate() {
        let got: Vec<f64> = (0..want.len()).map(|e| batched[e * lanes + lane]).collect();
        assert_bitwise_eq(want, &got);
    }
}

#[test]
fn targeted_batched_tangent_matches_scalar() {
    for case in targeted_split_eval_cases() {
        check_batched_matches_scalar(&case);
    }
}

/// Zero-width loads, slices and concatenations sit at offset 0, inside the primal region.
#[test]
fn zero_width_tangent_nodes_sweep_batched() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let sin = arena.alloc(Node::Sin(y));
    let empty_state = arena.alloc(Node::StateVector { start: 2, end: 2 });
    let empty_slice = arena.alloc(Node::Index {
        child: sin,
        start: 1,
        end: 1,
    });
    let empty_concat = arena.alloc(Node::Concat(vec![empty_state]));
    let root = arena.alloc(Node::Concat(vec![
        sin,
        empty_state,
        empty_slice,
        empty_concat,
    ]));
    let case = TangentCase {
        arena,
        root,
        y: vec![0.3, 0.7],
        seeds: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, -2.0]],
        n_states: 2,
        inputs: Vec::new(),
    };

    let mut tangent_arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut tangent_arena, case.root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&tangent_arena, tangent_root));
    assert!(
        is_batchable(&compiled),
        "the case must take the batched sweep"
    );
    check_batched_matches_scalar(&case);
}

proptest! {
    #[test]
    fn batched_tangent_matches_scalar(case in arb_split_eval_case()) {
        check_batched_matches_scalar(&case);
    }

    /// Steps, interpolants and reductions, whose tangents fold to primal roots and constants.
    #[test]
    fn batched_tangent_matches_scalar_with_every_extra(
        case in arb_split_eval_case_with(ALL_EXTRAS),
    ) {
        check_batched_matches_scalar(&case);
    }
}

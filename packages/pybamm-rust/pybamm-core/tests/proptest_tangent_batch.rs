mod common;

use common::cases::{
    ALL_EXTRAS, TangentCase, arb_split_eval_case, arb_split_eval_case_with,
    targeted_split_eval_cases,
};
use common::numeric_eq::assert_bitwise_eq;
use proptest::prelude::*;
use pybamm_core::ir::Instruction;
use pybamm_core::tangent_batch::tangent_scratch_len;
use pybamm_core::{
    Arena, ArrayData, DagRef, Node, Shape, SplitCompiledExpr, TangentInputs, tangent_wrt_states,
};

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
    if !compiled.is_batchable() {
        return;
    }
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
    let batched = cache.eval_tangent_batch(lanes, &mut tan, &lane_seeds);

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
        compiled.is_batchable(),
        "the case must take the batched sweep"
    );
    check_batched_matches_scalar(&case);
}

/// Short-circuited branches sweep batched, each lane taking the branch the scalar sweep takes;
/// the literal branch's tangent is primal, and out-of-range selectors pick no branch.
#[test]
fn dispatched_conditional_tangents_sweep_batched() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let mut sines = y;
    for _ in 0..3 {
        sines = arena.alloc(Node::Sin(sines));
    }
    let mut exps = y;
    for _ in 0..4 {
        exps = arena.alloc(Node::Exp(exps));
    }
    let literal = ArrayData::try_new(vec![2.0, -3.0], Shape::vector(2)).expect("valid array");
    let literal = arena.alloc(Node::Array(Box::new(literal)));
    let root = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![sines, exps, literal],
    });

    let mut tangent_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut tangent_arena, root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&tangent_arena, tangent_root));
    let split = compiled.split_info();
    assert!(
        compiled.ir().instructions()[split.primal_end..]
            .iter()
            .any(|instr| matches!(instr, Instruction::Dispatch { .. })),
        "the tangent section must short-circuit its branches"
    );
    assert!(
        compiled.is_batchable(),
        "the case must take the batched sweep"
    );

    for selector in [1.0, 2.0, 3.0, 0.0, 4.0, f64::NAN] {
        let case = TangentCase {
            arena: arena.clone(),
            root,
            y: vec![0.3, -0.2],
            seeds: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, -2.0]],
            n_states: 2,
            inputs: vec![selector],
        };
        check_batched_matches_scalar(&case);
    }
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

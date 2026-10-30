//! Only-active-branch execution, asserted from the executed instruction count.

use pybamm_core::arena::{Arena, NodeId};
use pybamm_core::eval::{CompiledExpr, TangentInputs};
use pybamm_core::ir::{Instruction, TypedIr};
use pybamm_core::node::{CsrData, Node, Shape};
use pybamm_core::{DagRef, SplitCompiledExpr, simplify, tangent_wrt_states};

/// `cond(sel, [chain(y, 1), chain(y, 8), chain(y, 16)])` over an `InputParameter` selector.
fn uneven_branches() -> (Arena, NodeId, Vec<usize>) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let mut branches = Vec::new();
    let mut lens = Vec::new();
    for depth in [1_usize, 8, 16] {
        let mut node = y;
        for _ in 0..depth {
            node = arena.alloc(Node::Sin(node));
        }
        branches.push(node);
        lens.push(depth);
    }
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches,
    });
    (arena, cond, lens)
}

#[test]
fn only_the_active_branch_executes() {
    let (arena, root, lens) = uneven_branches();
    let expr = CompiledExpr::new(DagRef::new(&arena, root));
    let mut scratch = vec![0.0; expr.scratch_len()];

    // common = y load + selector load + Conditional + Dispatch = 4.
    let common = 4;
    assert_eq!(expr.ir().common_instruction_count(), common);
    assert_eq!(
        expr.ir().instructions().len(),
        common + lens.iter().sum::<usize>()
    );
    for (i, &len) in lens.iter().enumerate() {
        let sel = (i + 1) as f64;
        let no_seed = TangentInputs::default();
        let (executed, _) =
            expr.eval_counted_with_tangent(&mut scratch, 0.0, &[0.3], &[], &[sel], &no_seed);
        assert_eq!(
            executed,
            common + len,
            "selector {sel} executed {executed} instructions, expected {}",
            common + len
        );
    }

    // No match: the dispatch runs, no block does.
    let no_seed = TangentInputs::default();
    let (executed, out) =
        expr.eval_counted_with_tangent(&mut scratch, 0.0, &[0.3], &[], &[0.0], &no_seed);
    assert_eq!(executed, common);
    assert_eq!(out, &[0.0]);
}

#[test]
fn executed_count_is_independent_of_the_other_branches() {
    let cost_of_branch_one = |depth_of_last: usize| {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let b1 = arena.alloc(Node::Sin(y));
        let mut b2 = y;
        for _ in 0..depth_of_last {
            b2 = arena.alloc(Node::Cos(b2));
        }
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2],
        });
        let expr = CompiledExpr::new(DagRef::new(&arena, cond));
        let mut scratch = vec![0.0; expr.scratch_len()];
        let no_seed = TangentInputs::default();
        expr.eval_counted_with_tangent(&mut scratch, 0.0, &[0.3], &[], &[1.0], &no_seed)
            .0
    };
    assert_eq!(cost_of_branch_one(2), cost_of_branch_one(64));
}

#[test]
fn short_circuited_results_are_bitwise_identical() {
    let (arena, root, _) = uneven_branches();
    let short = CompiledExpr::new(DagRef::new(&arena, root));
    let mut s = vec![0.0; short.scratch_len()];
    for sel in [
        0.0_f64,
        0.49,
        0.5,
        0.51,
        1.0,
        1.49,
        1.5,
        2.0,
        2.5,
        3.0,
        3.49,
        3.5,
        4.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
    ] {
        let got = short.eval(&mut s, 0.0, &[0.3], &[], &[sel])[0];
        let expected = expected_conditional(0.3, sel);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "selector {sel}: got {got}, expected {expected}"
        );
    }
}

/// The semantics contract, written out independently of the evaluator.
fn expected_conditional(y: f64, sel: f64) -> f64 {
    branch_window(sel, 3).map_or(0.0, |i| {
        let mut v = y;
        for _ in 0..[1_usize, 8, 16][i] {
            v = v.sin();
        }
        v
    })
}

#[test]
fn nested_conditionals_degrade_without_miscompiling() {
    // The inner cone is forced common, so only the outer conditional short-circuits.
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let outer_sel = arena.alloc(Node::InputParameter {
        name: "o".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let inner_sel = arena.alloc(Node::InputParameter {
        name: "i".to_string(),
        index: 1,
        offset: 1,
        width: 1,
    });
    let ia = arena.alloc(Node::Sin(y));
    let ib = arena.alloc(Node::Cos(y));
    let inner = arena.alloc(Node::Conditional {
        selector: inner_sel,
        branches: vec![ia, ib],
    });
    let ob = arena.alloc(Node::Neg(y));
    let outer = arena.alloc(Node::Conditional {
        selector: outer_sel,
        branches: vec![inner, ob],
    });

    let ir = TypedIr::from_dag(DagRef::new(&arena, outer));
    assert_eq!(
        ir.dispatch_count(),
        1,
        "only the outer conditional is blockable"
    );
    assert_eq!(ir.branch_block_lens(), vec![1, 1]);

    let expr = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; expr.scratch_len()];
    let y0 = 0.3_f64;
    for (o, i, expected) in [
        (1.0, 1.0, y0.sin()),
        (1.0, 2.0, y0.cos()),
        (1.0, 0.0, 0.0),
        (2.0, 1.0, -y0),
        (0.0, 1.0, 0.0),
    ] {
        let got = expr.eval(&mut s, 0.0, &[y0], &[], &[o, i])[0];
        assert_eq!(got.to_bits(), expected.to_bits(), "outer {o} inner {i}");
    }
}

#[test]
fn a_branch_owning_no_nodes_gets_an_empty_block() {
    // Branch 1 is a bare shared state load (common), branch 2 owns one node.
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let b2 = arena.alloc(Node::Sin(y));
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![y, b2],
    });
    let ir = TypedIr::from_dag(DagRef::new(&arena, cond));
    assert_eq!(ir.branch_block_lens(), vec![0, 1]);

    let expr = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; expr.scratch_len()];
    assert_eq!(expr.eval(&mut s, 0.0, &[0.3], &[], &[1.0]), &[0.3]);
    assert_eq!(
        expr.eval(&mut s, 0.0, &[0.3], &[], &[2.0]),
        &[0.3_f64.sin()]
    );
}

/// Reuse-slot and pinned layouts vs a closed-form oracle on a graph mixing every block hazard.
#[test]
fn scheduled_tape_matches_the_pinned_tape_and_a_closed_form_oracle() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let sel = arena.alloc(Node::StateVector { start: 3, end: 4 });
    let shared_inner = arena.alloc(Node::Exp(y));
    let shared = arena.alloc(Node::Sqrt(shared_inner));

    let b0 = arena.alloc(Node::Sin(shared));
    let csr = arena.alloc(Node::SparseMatrix(
        CsrData::try_new(
            vec![0, 1, 2, 3],
            vec![1, 2, 0],
            vec![2.0, -1.5, 0.75],
            Shape { rows: 3, cols: 3 },
        )
        .expect("valid csr")
        .into(),
    ));
    let mm = arena.alloc(Node::MatMul(csr, shared));
    let b1 = arena.alloc(Node::Cos(mm));

    let inner_sel = arena.alloc(Node::Time);
    let ia = arena.alloc(Node::Tanh(y));
    let ib = arena.alloc(Node::Abs(y));
    let b2 = arena.alloc(Node::Conditional {
        selector: inner_sel,
        branches: vec![ia, ib],
    });

    let outer = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b0, b1, b2],
    });
    // `shared_inner` is also read outside the conditional, so it must stay `Common`.
    let root = arena.alloc(Node::Add(outer, shared_inner));

    let scheduled = CompiledExpr::new(DagRef::new(&arena, root));
    let pinned = CompiledExpr::from_ir(TypedIr::from_dag_pinned(DagRef::new(&arena, root)));
    // Block lengths count instructions, not nodes: branch 1's `SparseMatrix` emits none.
    assert_eq!(scheduled.ir().dispatch_count(), 1);
    assert_eq!(scheduled.ir().branch_block_lens(), vec![2, 3, 1]);
    assert_eq!(pinned.ir().dispatch_count(), 1);
    assert_eq!(pinned.ir().branch_block_lens(), vec![2, 3, 1]);

    let mut sched_scratch = vec![0.0; scheduled.scratch_len()];
    let mut pinned_scratch = vec![0.0; pinned.scratch_len()];
    for sel_val in [0.0_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, f64::NAN] {
        for t in [1.0_f64, 2.0, 0.0] {
            let state = [0.35, -0.2, 0.9, sel_val];
            let got = scheduled.eval(&mut sched_scratch, t, &state, &[], &[]);
            let want = pinned.eval(&mut pinned_scratch, t, &state, &[], &[]);
            let oracle = expected_hazard_graph([state[0], state[1], state[2]], sel_val, t);
            assert_eq!(got.len(), want.len());
            assert_eq!(got.len(), oracle.len());
            for (i, ((g, w), o)) in got.iter().zip(want).zip(&oracle).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "selector {sel_val}, t {t}, element {i}: scheduled {g}, pinned {w}"
                );
                assert!(
                    (g - o).abs() <= 1e-15 * o.abs(),
                    "selector {sel_val}, t {t}, element {i}: got {g}, oracle {o}"
                );
            }
        }
    }
}

/// The hazard graph's semantics, written out independently of the evaluator.
fn expected_hazard_graph(y: [f64; 3], sel: f64, t: f64) -> Vec<f64> {
    let exp_y = y.map(f64::exp);
    let shared = exp_y.map(f64::sqrt);
    let outer = match branch_window(sel, 3) {
        Some(0) => shared.map(f64::sin),
        Some(1) => [2.0 * shared[1], -1.5 * shared[2], 0.75 * shared[0]].map(f64::cos),
        Some(2) => match branch_window(t, 2) {
            Some(0) => y.map(f64::tanh),
            Some(1) => y.map(f64::abs),
            _ => [0.0; 3],
        },
        _ => [0.0; 3],
    };
    (0..3).map(|i| outer[i] + exp_y[i]).collect()
}

/// The 1-based round-to-nearest branch window of the semantics contract.
fn branch_window(selector: f64, n_branches: usize) -> Option<usize> {
    (0..n_branches).find(|&i| {
        let idx = (i + 1) as f64;
        selector > idx - 0.5 && selector < idx + 0.5
    })
}

/// Lanes select different branches, so every dispatched block stays live.
#[test]
fn batch_eval_over_dispatched_blocks_matches_per_lane_eval() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    // A state-derived selector: `InputParameter` broadcasts to every lane.
    let sel = arena.alloc(Node::StateVector { start: 1, end: 2 });
    let mut branches = Vec::new();
    for depth in [1_usize, 4, 9] {
        let mut node = y;
        for _ in 0..depth {
            node = arena.alloc(Node::Sin(node));
        }
        branches.push(node);
    }
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches,
    });

    let expr = CompiledExpr::new(DagRef::new(&arena, cond));
    assert_eq!(
        expr.ir().branch_block_lens(),
        vec![1, 4, 9],
        "the tape must actually carry blocks for this to test anything"
    );

    // Four lanes: branch 1, branch 3, branch 2, and no match.
    let k = 4;
    let ts = vec![0.0; k];
    let y_cols = [0.3, 1.0, 0.4, 3.0, 0.5, 2.0, 0.6, 0.0];
    let n_states = 2;

    let mut per_lane = vec![0.0_f64; k];
    let mut s = vec![0.0; expr.scratch_len()];
    for l in 0..k {
        let y_l = &y_cols[l * n_states..(l + 1) * n_states];
        per_lane[l] = expr.eval(&mut s, ts[l], y_l, &[], &[])[0];
    }

    let mut batch_scratch = vec![0.0; expr.scratch_len() * k];
    let batched = expr
        .eval_batch(&mut batch_scratch, k, &ts, &y_cols, n_states, &[])
        .expect("primal tape batches");
    for l in 0..k {
        assert_eq!(
            batched[l].to_bits(),
            per_lane[l].to_bits(),
            "lane {l}: batch {} vs scalar {}",
            batched[l],
            per_lane[l]
        );
    }
}

/// Two sibling top-level conditionals, the shape `PyBaMM`'s unified experiment model builds.
#[test]
fn two_independent_conditionals_get_one_dispatch_each() {
    const DEPTHS_A: [usize; 2] = [1, 3];
    const DEPTHS_B: [usize; 2] = [2, 5];

    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let sel_a = arena.alloc(Node::InputParameter {
        name: "a".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let sel_b = arena.alloc(Node::InputParameter {
        name: "b".to_string(),
        index: 1,
        offset: 1,
        width: 1,
    });
    let mut a_branches = Vec::new();
    for depth in DEPTHS_A {
        let mut node = y;
        for _ in 0..depth {
            node = arena.alloc(Node::Sin(node));
        }
        a_branches.push(node);
    }
    let mut b_branches = Vec::new();
    for depth in DEPTHS_B {
        let mut node = y;
        for _ in 0..depth {
            node = arena.alloc(Node::Cos(node));
        }
        b_branches.push(node);
    }
    let cond_a = arena.alloc(Node::Conditional {
        selector: sel_a,
        branches: a_branches,
    });
    let cond_b = arena.alloc(Node::Conditional {
        selector: sel_b,
        branches: b_branches,
    });
    let root = arena.alloc(Node::Add(cond_a, cond_b));

    let ir = TypedIr::from_dag(DagRef::new(&arena, root));
    assert_eq!(ir.dispatch_count(), 2);
    // `b`'s dispatch is emitted first: the topological walk reaches its cone first.
    assert_eq!(ir.branch_block_lens(), vec![2, 5, 1, 3]);
    // y + two selectors + two Conditionals + Add + two Dispatches.
    let common = 8;
    assert_eq!(ir.common_instruction_count(), common);
    assert_eq!(ir.instructions().len(), 19);

    let expr = CompiledExpr::from_ir(ir);
    let mut scratch = vec![0.0; expr.scratch_len()];
    let y0 = 0.3_f64;
    for (a_index, sel_a_val) in [(None, 0.0_f64), (Some(0), 1.0), (Some(1), 2.0), (None, 3.0)] {
        for (b_index, sel_b_val) in [(None, 0.0_f64), (Some(0), 1.0), (Some(1), 2.0), (None, 3.0)] {
            let a_cost = a_index.map_or(0, |i| DEPTHS_A[i]);
            let b_cost = b_index.map_or(0, |i| DEPTHS_B[i]);
            let no_seed = TangentInputs::default();
            let inputs = [sel_a_val, sel_b_val];
            let (executed, out) =
                expr.eval_counted_with_tangent(&mut scratch, 0.0, &[y0], &[], &inputs, &no_seed);
            assert_eq!(
                executed,
                common + a_cost + b_cost,
                "selectors ({sel_a_val}, {sel_b_val}) executed {executed}"
            );

            let chain = |depth: usize, f: fn(f64) -> f64| {
                let mut v = y0;
                for _ in 0..depth {
                    v = f(v);
                }
                v
            };
            let expected = a_index.map_or(0.0, |i| chain(DEPTHS_A[i], f64::sin))
                + b_index.map_or(0.0, |i| chain(DEPTHS_B[i], f64::cos));
            assert_eq!(
                out[0].to_bits(),
                expected.to_bits(),
                "selectors ({sel_a_val}, {sel_b_val}): got {}, expected {expected}",
                out[0]
            );
        }
    }
}

#[test]
fn jvp_over_dispatched_blocks_matches_the_split_eval_tape() {
    let mut arena = Arena::new();
    let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    // Nonlinear branches, so each adjoint genuinely needs its own primal value.
    let b0 = arena.alloc(Node::Sin(x));
    let b1 = arena.alloc(Node::Exp(x));
    let b2 = arena.alloc(Node::Tanh(x));
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b0, b1, b2],
    });
    let jac = tangent_wrt_states(&mut arena, cond);
    let jac = simplify(&mut arena, jac);

    let scheduled = CompiledExpr::new(DagRef::new(&arena, jac));
    let split = SplitCompiledExpr::compile(DagRef::new(&arena, jac));
    assert_eq!(scheduled.ir().dispatch_count(), 1);
    assert_eq!(scheduled.ir().branch_block_lens(), vec![2, 2, 5]);
    assert_eq!(
        split.ir().dispatch_count(),
        2,
        "split-eval splits this cone into one block per partition half"
    );

    let mut sched_scratch = vec![0.0; scheduled.scratch_len()];
    let mut split_scratch = vec![0.0; split.scratch_len()];
    let tangent = TangentInputs {
        dy: Some(&[1.0]),
        dp: None,
    };
    let x0 = 0.4_f64;
    for sel_val in [
        0.0_f64,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
    ] {
        let got =
            scheduled.eval_with_tangent(&mut sched_scratch, 0.0, &[x0], &[], &[sel_val], &tangent)
                [0];
        let want =
            split.eval_with_tangent(&mut split_scratch, 0.0, &[x0], &[], &[sel_val], &tangent)[0];
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "selector {sel_val}: scheduled {got}, split {want}"
        );
    }

    // Closed form, so the two tapes can't pass by a shared mistake.
    for (sel_val, expected) in [
        (1.0_f64, x0.cos()),
        (2.0, x0.exp()),
        (3.0, x0.tanh().mul_add(-x0.tanh(), 1.0)),
        (0.0, 0.0),
    ] {
        let got =
            scheduled.eval_with_tangent(&mut sched_scratch, 0.0, &[x0], &[], &[sel_val], &tangent)
                [0];
        assert!(
            (got - expected).abs() < 1e-12,
            "selector {sel_val}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn split_eval_blocks_respect_the_primal_tangent_partition() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let mut b1 = y;
    for _ in 0..3 {
        b1 = arena.alloc(Node::Sin(b1));
    }
    let mut b2 = y;
    for _ in 0..12 {
        b2 = arena.alloc(Node::Exp(b2));
    }
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b1, b2],
    });
    let tangent_root = tangent_wrt_states(&mut arena, cond);

    let compiled = SplitCompiledExpr::compile(DagRef::new(&arena, tangent_root));
    let (ir, split) = (compiled.ir(), compiled.split_info());
    // One block per half, so the cone really is split rather than degraded.
    assert_eq!(ir.dispatch_count(), 2);

    for (i, instr) in ir.instructions().iter().enumerate() {
        if let Instruction::Dispatch {
            blocks_idx,
            blocks_len,
            span_len,
            ..
        } = *instr
        {
            // The blocks tile the span after the `Dispatch`, which evaluation then skips whole.
            let mut next = i + 1;
            for b in 0..blocks_len.index() {
                let block = ir.consts().branch_blocks[blocks_idx.index() + b].bounds(i);
                assert_eq!(block.start, next, "block {b} of the Dispatch at {i}");
                next = block.end;
                assert!(
                    split.primal_end <= block.start || split.primal_end >= block.end,
                    "primal_end {} splits block [{}, {})",
                    split.primal_end,
                    block.start,
                    block.end
                );
            }
            assert_eq!(
                i + span_len.index(),
                next,
                "the Dispatch at {i} spans its blocks"
            );
        }
    }

    let expr = CompiledExpr::from_ir(ir.clone());
    let mut s = vec![0.0; expr.scratch_len()];
    let seed = [1.0, 0.0];
    let tangent = TangentInputs {
        dy: Some(&seed),
        dp: None,
    };
    let mut counts = Vec::new();
    for sel_val in [1.0_f64, 2.0] {
        let (executed, _) =
            expr.eval_counted_with_tangent(&mut s, 0.0, &[0.3, 0.4], &[], &[sel_val], &tangent);
        counts.push(executed);
    }
    assert!(
        counts[0] < counts[1],
        "branch 1 ({}) should cost less than branch 2 ({})",
        counts[0],
        counts[1]
    );
}

#[test]
fn split_eval_matches_monolithic_for_every_selector() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let b1 = arena.alloc(Node::Sin(y));
    let b2 = arena.alloc(Node::Exp(y));
    let b3 = arena.alloc(Node::Cos(y));
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b1, b2, b3],
    });
    let tangent_root = tangent_wrt_states(&mut arena, cond);

    let mono = CompiledExpr::new(DagRef::new(&arena, tangent_root));
    let split = SplitCompiledExpr::compile(DagRef::new(&arena, tangent_root));
    let seed = [1.0, 1.0];
    let tangent = TangentInputs {
        dy: Some(&seed),
        dp: None,
    };
    let mut sm = vec![0.0; mono.scratch_len()];
    let mut ss = vec![0.0; split.scratch_len()];
    for sel_val in [0.0_f64, 1.0, 2.0, 3.0, 4.0, 1.5, f64::NAN] {
        let a = mono
            .eval_with_tangent(&mut sm, 0.0, &[0.3, 0.4], &[], &[sel_val], &tangent)
            .to_vec();
        let mut cache = split.eval_primal(&mut ss, 0.0, &[0.3, 0.4], &[], &[sel_val]);
        let b = cache.eval_tangent(&tangent).to_vec();
        assert_eq!(a.len(), b.len());
        for (x, z) in a.iter().zip(&b) {
            assert_eq!(x.to_bits(), z.to_bits(), "selector {sel_val}");
        }
    }
}

/// A tangent-tainted selector must degrade: a primal `Dispatch` would read an unwritten slot.
#[test]
fn a_tangent_selector_degrades_rather_than_dispatching_on_an_unwritten_slot() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let dy = arena.alloc(Node::TangentStateVector { start: 0, end: 1 });
    let one = arena.alloc(Node::Scalar(1.0));
    let sel = arena.alloc(Node::Add(dy, one));
    let b1 = arena.alloc(Node::Sin(y));
    let b2 = arena.alloc(Node::Exp(y));
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b1, b2],
    });

    let mono = CompiledExpr::new(DagRef::new(&arena, cond));
    let split = SplitCompiledExpr::compile(DagRef::new(&arena, cond));
    let mut sm = vec![0.0; mono.scratch_len()];
    let mut ss = vec![0.0; split.scratch_len()];
    let y0 = 0.4_f64;
    for seed in [-1.0_f64, 0.0, 0.5, 1.0, 2.0, f64::NAN] {
        let seeds = [seed];
        let tangent = TangentInputs {
            dy: Some(&seeds),
            dp: None,
        };
        let want = mono.eval_with_tangent(&mut sm, 0.0, &[y0], &[], &[], &tangent)[0];
        let mut cache = split.eval_primal(&mut ss, 0.0, &[y0], &[], &[]);
        let got = cache.eval_tangent(&tangent)[0];
        let expected = match seed + 1.0 {
            s if s > 0.5 && s < 1.5 => y0.sin(),
            s if s > 1.5 && s < 2.5 => y0.exp(),
            _ => 0.0,
        };
        assert_eq!(got.to_bits(), want.to_bits(), "seed {seed}");
        assert_eq!(got.to_bits(), expected.to_bits(), "seed {seed}");
    }
}

#[test]
fn pinned_layout_short_circuits_and_stays_ssa() {
    let (arena, root, lens) = uneven_branches();
    let ir = TypedIr::from_dag_pinned(DagRef::new(&arena, root));
    assert_eq!(ir.branch_block_lens(), vec![1, 8, 16]);
    // Still SSA (reverse AD needs it): `y`, the selector and the `Conditional` are the 3.
    assert_eq!(ir.buffer_size(), 3 + lens.iter().sum::<usize>());

    let expr = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; expr.scratch_len()];
    for (i, &len) in lens.iter().enumerate() {
        let sel = (i + 1) as f64;
        let no_seed = TangentInputs::default();
        let (executed, out) =
            expr.eval_counted_with_tangent(&mut s, 0.0, &[0.3], &[], &[sel], &no_seed);
        assert_eq!(executed, 4 + len);
        assert_eq!(out[0].to_bits(), expected_conditional(0.3, sel).to_bits());
    }
}

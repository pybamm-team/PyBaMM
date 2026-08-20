//! Only-active-branch execution: the invariant is asserted from the executed
//! instruction count, not from a reported tape length, because a count metric
//! alone cannot tell "the work is skipped" from "the work is not reported".

use pybamm_core::adjoint::AdjointTape;
use pybamm_core::arena::{Arena, NodeId};
use pybamm_core::eval::{CompiledExpr, TangentInputs};
use pybamm_core::ir::{Instruction, TypedIr};
use pybamm_core::node::{CsrData, Node, Shape};
use pybamm_core::{simplify, tangent_wrt_states};

/// `cond(sel, [chain(y, 1), chain(y, 8), chain(y, 16)])` — three exclusive
/// branches of very different sizes over an `InputParameter` selector.
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
    let expr = CompiledExpr::from_ir(TypedIr::from_arena(&arena, root));
    let mut scratch = vec![0.0; expr.scratch_len()];

    // common = y load + selector load + Conditional + Dispatch = 4.
    let common = 4;
    for (i, &len) in lens.iter().enumerate() {
        let sel = (i + 1) as f64;
        let (executed, _) = expr.eval_counted(&mut scratch, 0.0, &[0.3], &[], &[sel]);
        assert_eq!(
            executed,
            common + len,
            "selector {sel} executed {executed} instructions, expected {}",
            common + len
        );
    }

    // No match: the dispatch runs, no block does.
    let (executed, out) = expr.eval_counted(&mut scratch, 0.0, &[0.3], &[], &[0.0]);
    assert_eq!(executed, common);
    assert_eq!(out, &[0.0]);
}

#[test]
fn executed_count_is_independent_of_the_other_branches() {
    // Growing branch 3 must not change what branch 1 costs. This is the
    // invariant `test_unified_active_branch_independent_of_other_modes` checks.
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
        let expr = CompiledExpr::from_ir(TypedIr::from_arena(&arena, cond));
        let mut scratch = vec![0.0; expr.scratch_len()];
        expr.eval_counted(&mut scratch, 0.0, &[0.3], &[], &[1.0]).0
    };
    assert_eq!(cost_of_branch_one(2), cost_of_branch_one(64));
}

#[test]
fn short_circuited_results_are_bitwise_identical() {
    let (arena, root, _) = uneven_branches();
    let short = CompiledExpr::from_ir(TypedIr::from_arena(&arena, root));
    // Checked against `expected_conditional`, an oracle written out from the
    // semantics contract rather than derived from the evaluator.
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
    for (i, depth) in [1_usize, 8, 16].iter().enumerate() {
        let idx = (i + 1) as f64;
        if sel > idx - 0.5 && sel < idx + 0.5 {
            let mut v = y;
            for _ in 0..*depth {
                v = v.sin();
            }
            return v;
        }
    }
    0.0
}

#[test]
fn nested_conditionals_degrade_without_miscompiling() {
    // Inner conditional inside outer branch 1. The inner cone is forced common,
    // so only the outer conditional short-circuits — and the values are right.
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

    let ir = TypedIr::from_arena(&arena, outer);
    // Values alone would pass even if the pass emitted no blocks, so pin the
    // outcome: only the outer conditional blocks, one instruction per branch.
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
    let ir = TypedIr::from_arena(&arena, cond);
    assert_eq!(ir.branch_block_lens(), vec![0, 1]);

    let expr = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; expr.scratch_len()];
    assert_eq!(expr.eval(&mut s, 0.0, &[0.3], &[], &[1.0]), &[0.3]);
    assert_eq!(
        expr.eval(&mut s, 0.0, &[0.3], &[], &[2.0]),
        &[0.3_f64.sin()]
    );
}

/// Differential test between the two block-forming layouts, reuse slots
/// (`from_arena`) and pinned SSA slots (`from_arena_pinned`), plus a closed-form
/// oracle, on a graph mixing every hazard: a cone shared by a strict subset of
/// branches, a `Common` node also read outside the conditional, a nested
/// conditional, and a `SparseMatrix` inside a block.
///
/// The worst failure mode is a node wrongly placed in a block whose value is read
/// from outside. That gives a *different* wrong value per layout, so the two tapes
/// disagree, and the oracle catches a mistake they share.
#[test]
fn scheduled_tape_matches_the_pinned_tape_and_a_closed_form_oracle() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let sel = arena.alloc(Node::StateVector { start: 3, end: 4 });
    let shared_inner = arena.alloc(Node::Exp(y));
    let shared = arena.alloc(Node::Sqrt(shared_inner));

    // Branch 0 and branch 1 share `shared` (a strict subset of three branches).
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

    // Branch 2 is itself a conditional: its cone degrades to `Common`.
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
    // `shared_inner` is also read outside the conditional, so it must stay
    // `Common` and be emitted outside every block.
    let root = arena.alloc(Node::Add(outer, shared_inner));

    let scheduled = CompiledExpr::from_ir(TypedIr::from_arena(&arena, root));
    let pinned = CompiledExpr::from_ir(TypedIr::from_arena_pinned(&arena, root));
    // Block lengths count instructions, not nodes: branch 1's `SparseMatrix` emits
    // none, and slot allocation cannot change the count, so both layouts must agree.
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

/// The hazard graph's semantics, written out independently of the evaluator:
/// `cond(sel, [sin(s), cos(A @ s), cond(t, [tanh(y), abs(y)])]) + exp(y)` where
/// `s = sqrt(exp(y))` and `A` is the test's 3x3 CSR matrix.
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

/// `eval_batch` dispatches the union of branches any lane selects, then lets
/// `Conditional` pick per lane within that union. Lanes select different
/// branches here, so every block stays live — the scenario a batch evaluator
/// that skipped based on a single lane's choice would get wrong.
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

    let expr = CompiledExpr::from_ir(TypedIr::from_arena(&arena, cond));
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
        .eval_batch(&mut batch_scratch, k, &ts, &y_cols, &[])
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

/// Two sibling top-level conditionals — the shape `PyBaMM`'s unified experiment
/// model actually builds (a control residual plus a fused termination event).
/// This is the only shape that exercises per-`Dispatch` `blocks_idx` bookkeeping,
/// the `group_at` anchor table, and summing `branch_block_lens` over more than
/// one dispatch.
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

    let ir = TypedIr::from_arena(&arena, root);
    assert_eq!(ir.dispatch_count(), 2);
    // `b`'s dispatch is emitted first: the topological walk reaches `cond_b`'s
    // cone before `cond_a`'s. Both dispatches' blocks are listed in tape order.
    assert_eq!(ir.branch_block_lens(), vec![2, 5, 1, 3]);
    // y + two selectors + two Conditionals + Add + two Dispatches.
    let common = 8;
    assert_eq!(ir.common_instruction_count(), common);
    assert_eq!(ir.instructions().len(), 19);

    let expr = CompiledExpr::from_ir(ir);
    let mut scratch = vec![0.0; expr.scratch_len()];
    let y0 = 0.3_f64;
    // A selector of 0 or 3 matches no branch of a two-branch conditional, so the
    // matrix covers a live block, the other live block, and no-match on each side.
    for (a_index, sel_a_val) in [(None, 0.0_f64), (Some(0), 1.0), (Some(1), 2.0), (None, 3.0)] {
        for (b_index, sel_b_val) in [(None, 0.0_f64), (Some(0), 1.0), (Some(1), 2.0), (None, 3.0)] {
            let a_cost = a_index.map_or(0, |i| DEPTHS_A[i]);
            let b_cost = b_index.map_or(0, |i| DEPTHS_B[i]);
            let (executed, out) =
                expr.eval_counted(&mut scratch, 0.0, &[y0], &[], &[sel_a_val, sel_b_val]);
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

/// Forward-mode JVP over a dispatched tape. `eval_with_tangent` on a non-split
/// `from_arena` tape is production-live; the split-eval tape reaches the same
/// derivative graph through a different layout — one `Dispatch` per half instead
/// of one for the whole tape — so the two are independent of each other. The
/// closed-form check at the end stops them passing by a shared mistake.
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

    let scheduled = CompiledExpr::from_ir(TypedIr::from_arena(&arena, jac));
    let split = CompiledExpr::from_ir(TypedIr::from_arena_split_eval(&arena, jac));
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

    // And the active branch's derivative is the mathematically right one, so the
    // two tapes are not merely agreeing on a shared mistake.
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
fn common_instruction_count_excludes_branch_blocks() {
    let (arena, root, lens) = uneven_branches();
    let ir = TypedIr::from_arena(&arena, root);
    let total: usize = lens.iter().sum();
    assert_eq!(
        ir.instructions().len(),
        ir.common_instruction_count() + total
    );
    // common = y + selector + Dispatch + Conditional
    assert_eq!(ir.common_instruction_count(), 4);
}

/// A conditional whose cone spans the primal/tangent partition still
/// short-circuits in both halves, and `primal_end` remains a valid split point.
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
    // Two branches whose primal work is deliberately unequal.
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

    let ir = TypedIr::from_arena_split_eval(&arena, tangent_root);
    let split = ir.split_eval_info().expect("split-eval info");
    // One block per half, so the cone really is split rather than degraded.
    assert_eq!(ir.dispatch_count(), 2);

    // primal_end must not land inside a block span.
    for (i, instr) in ir.instructions().iter().enumerate() {
        if let Instruction::Dispatch {
            blocks_idx,
            blocks_len,
            ..
        } = *instr
        {
            for b in 0..blocks_len as usize {
                let (rel, len) = ir.consts().branch_blocks[blocks_idx as usize + b];
                let (start, end) = (i + rel as usize, i + rel as usize + len as usize);
                assert!(
                    split.primal_end <= start || split.primal_end >= end,
                    "primal_end {} splits block [{start}, {end})",
                    split.primal_end
                );
            }
        }
    }

    // Both halves short-circuit: the executed count differs by branch.
    let expr = CompiledExpr::from_ir(ir);
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

/// The split tape must match the monolithic tape bitwise for every selector,
/// including no-match.
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

    let mono = CompiledExpr::from_ir(TypedIr::from_arena(&arena, tangent_root));
    let split = CompiledExpr::from_ir(TypedIr::from_arena_split_eval(&arena, tangent_root));
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

/// A `Dispatch` reads its selector's slot, so a group anchored at the end of one
/// partition half is only safe when that half also computes the selector. A
/// tangent-tainted selector puts it in the *tangent* pool while the branch-owned
/// primal nodes stay in the primal half — the primal `Dispatch` would then read
/// an unwritten slot and skip a block whose value the tangent half still reads.
/// `assert_block_slots_private` cannot see this (there is no writer to blame),
/// so the scheduler must degrade instead.
#[test]
fn a_tangent_selector_degrades_rather_than_dispatching_on_an_unwritten_slot() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let dy = arena.alloc(Node::TangentStateVector { start: 0, end: 1 });
    let one = arena.alloc(Node::Scalar(1.0));
    // Tangent-tainted, so the selector lands in the tangent slot pool.
    let sel = arena.alloc(Node::Add(dy, one));
    let b1 = arena.alloc(Node::Sin(y));
    let b2 = arena.alloc(Node::Exp(y));
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b1, b2],
    });

    let mono = CompiledExpr::from_ir(TypedIr::from_arena(&arena, cond));
    let split = CompiledExpr::from_ir(TypedIr::from_arena_split_eval(&arena, cond));
    let mut sm = vec![0.0; mono.scratch_len()];
    let mut ss = vec![0.0; split.scratch_len()];
    let y0 = 0.4_f64;
    // `dy + 1` sweeps no-match, both branches and a half-integer boundary.
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
    let ir = TypedIr::from_arena_pinned(&arena, root);
    assert_eq!(ir.branch_block_lens(), vec![1, 8, 16]);
    // Blocking must not cost the pinned layout its SSA property, which reverse AD
    // needs: with no reuse, `y`, the selector and the `Conditional` account for the 3.
    assert_eq!(ir.buffer_size(), 3 + lens.iter().sum::<usize>());

    let expr = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; expr.scratch_len()];
    for (i, &len) in lens.iter().enumerate() {
        let sel = (i + 1) as f64;
        let (executed, out) = expr.eval_counted(&mut s, 0.0, &[0.3], &[], &[sel]);
        assert_eq!(executed, 4 + len);
        assert_eq!(out[0].to_bits(), expected_conditional(0.3, sel).to_bits());
    }
}

/// The backward pass must not walk an inactive branch's block. Counted, not
/// inferred: `assemble` reports what the adjoint replay actually touched.
#[test]
fn reverse_pass_skips_inactive_branch_blocks() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let b1 = arena.alloc(Node::Sin(y));
    // `Tanh`, not `Exp`: iterated `exp` overflows to inf by depth 5, which would
    // make the gradient check vacuous (inf == inf).
    let mut b2 = y;
    for _ in 0..20 {
        b2 = arena.alloc(Node::Tanh(b2));
    }
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b1, b2],
    });

    let tape = AdjointTape::new(&arena, cond, 1);
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![0.0; 1];

    let walked_1 = tape.assemble(&mut scratch, &mut bar, &mut grad, 0.0, &[0.3], &[], &[1.0]);
    let g1 = grad[0];
    let walked_2 = tape.assemble(&mut scratch, &mut bar, &mut grad, 0.0, &[0.3], &[], &[2.0]);
    let g2 = grad[0];

    assert!(
        walked_1 < walked_2,
        "branch 1 backward walk ({walked_1}) should be shorter than branch 2 ({walked_2})"
    );
    // Values still correct: d/dy sin(y) and d/dy tanh^20(y).
    assert!((g1 - 0.3_f64.cos()).abs() < 1e-12);
    let mut expected = 1.0_f64;
    let mut v = 0.3_f64;
    for _ in 0..20 {
        v = v.tanh();
        expected *= v.mul_add(-v, 1.0);
    }
    assert!(
        (g2 - expected).abs() / expected < 1e-9,
        "got {g2}, want {expected}"
    );
}

/// Two sibling conditionals in one adjoint tape: the backward walk crosses two
/// span ends, so it must resolve each independently rather than treating the
/// first one it meets as the only one. The reverse-side counterpart of
/// `two_independent_conditionals_get_one_dispatch_each`.
#[test]
fn reverse_pass_skips_blocks_of_two_sibling_conditionals() {
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
    let chain = |arena: &mut Arena, depth: usize, tanh: bool| {
        let mut node = y;
        for _ in 0..depth {
            node = arena.alloc(if tanh {
                Node::Tanh(node)
            } else {
                Node::Sin(node)
            });
        }
        node
    };
    let a_branches = DEPTHS_A.map(|d| chain(&mut arena, d, false)).to_vec();
    let b_branches = DEPTHS_B.map(|d| chain(&mut arena, d, true)).to_vec();
    let cond_a = arena.alloc(Node::Conditional {
        selector: sel_a,
        branches: a_branches,
    });
    let cond_b = arena.alloc(Node::Conditional {
        selector: sel_b,
        branches: b_branches,
    });
    let root = arena.alloc(Node::Add(cond_a, cond_b));

    let tape = AdjointTape::new(&arena, root, 1);
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![0.0; 1];
    // y + two selectors + two Conditionals + Add + two Dispatches.
    let common = 8;
    let y0 = 0.3_f64;

    // d/dy of an iterated sin / tanh chain, by the chain rule.
    let d_chain = |depth: usize, tanh: bool| {
        let mut derivative = 1.0_f64;
        let mut v = y0;
        for _ in 0..depth {
            if tanh {
                v = v.tanh();
                derivative *= v.mul_add(-v, 1.0);
            } else {
                derivative *= v.cos();
                v = v.sin();
            }
        }
        derivative
    };

    for (a_index, sel_a_val) in [(None, 0.0_f64), (Some(0), 1.0), (Some(1), 2.0)] {
        for (b_index, sel_b_val) in [(None, 0.0_f64), (Some(0), 1.0), (Some(1), 2.0)] {
            let walked = tape.assemble(
                &mut scratch,
                &mut bar,
                &mut grad,
                0.0,
                &[y0],
                &[],
                &[sel_a_val, sel_b_val],
            );
            let a_cost = a_index.map_or(0, |i| DEPTHS_A[i]);
            let b_cost = b_index.map_or(0, |i| DEPTHS_B[i]);
            assert_eq!(
                walked,
                common + a_cost + b_cost,
                "selectors ({sel_a_val}, {sel_b_val}) walked {walked}"
            );

            let expected = a_index.map_or(0.0, |i| d_chain(DEPTHS_A[i], false))
                + b_index.map_or(0.0, |i| d_chain(DEPTHS_B[i], true));
            assert!(
                (grad[0] - expected).abs() < 1e-12,
                "selectors ({sel_a_val}, {sel_b_val}): got {}, want {expected}",
                grad[0]
            );
        }
    }
}

/// No match: no block is walked and the gradient is all zeros. The count is what
/// distinguishes skipping from not skipping — an all-zero gradient alone was
/// already true before blocks existed, because the `Conditional` adjoint seeds no
/// branch when nothing matches.
#[test]
fn reverse_pass_on_no_match_walks_no_block() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let b1 = arena.alloc(Node::Sin(y));
    let b2 = arena.alloc(Node::Cos(y));
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b1, b2],
    });
    let tape = AdjointTape::new(&arena, cond, 1);
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![9.0; 1];
    let walked = tape.assemble(&mut scratch, &mut bar, &mut grad, 0.0, &[0.3], &[], &[0.0]);
    assert_eq!(grad, vec![0.0]);
    // y + selector + Conditional + Dispatch, and neither one-instruction block:
    // entering both would be 6, and a blockless tape would be 5.
    assert_eq!(walked, 4, "no-match walk touched a branch block");

    // The contrast: matching a branch walks exactly one more instruction.
    let matched = tape.assemble(&mut scratch, &mut bar, &mut grad, 0.0, &[0.3], &[], &[1.0]);
    assert_eq!(matched, 5);
    assert!((grad[0] - 0.3_f64.cos()).abs() < 1e-12);
}

/// The semantics contract's edge cases through the backward walk: a half-integer
/// window boundary, NaN, a negative and an infinite selector all match no branch,
/// so no block is walked and the gradient is zero.
#[test]
fn reverse_pass_on_boundary_and_nan_selectors_walks_no_block() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let sel = arena.alloc(Node::InputParameter {
        name: "s".to_string(),
        index: 0,
        offset: 0,
        width: 1,
    });
    let b1 = arena.alloc(Node::Sin(y));
    let b2 = arena.alloc(Node::Cos(y));
    let cond = arena.alloc(Node::Conditional {
        selector: sel,
        branches: vec![b1, b2],
    });
    let tape = AdjointTape::new(&arena, cond, 1);
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![9.0; 1];

    for sel_val in [0.5_f64, 1.5, 2.5, -1.0, f64::NAN, f64::INFINITY] {
        let walked = tape.assemble(
            &mut scratch,
            &mut bar,
            &mut grad,
            0.0,
            &[0.3],
            &[],
            &[sel_val],
        );
        // y + selector + Conditional + Dispatch, and neither one-instruction block.
        assert_eq!(walked, 4, "selector {sel_val} walked a branch block");
        assert_eq!(grad, vec![0.0], "selector {sel_val}");
    }
}

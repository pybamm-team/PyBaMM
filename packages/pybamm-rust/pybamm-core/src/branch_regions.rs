//! Conditional branch region analysis: which nodes each `Conditional` branch owns exclusively.
//!
//! Ambiguous cases degrade to [`Ownership::Common`], costing speed but never correctness.

use crate::arena::{Arena, Dag, DagRef, NodeId};
use crate::node::Node;

/// One branch of one conditional: `branches[index]` of `cond`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchLabel {
    pub cond: NodeId,
    pub index: usize,
}

/// Which conditional branches exclusively need a node's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ownership {
    /// No consumer seen (unreachable from the root).
    Unreached,
    /// Needed only by the listed branches of one conditional (sorted, deduplicated).
    Branches { cond: NodeId, indices: Vec<usize> },
    /// Needed regardless of which branch runs.
    Common,
}

impl Ownership {
    /// Join in the lattice `Unreached < Branches < Common`; different conditionals give `Common`.
    fn join(&mut self, other: &Self) {
        match (&mut *self, other) {
            (_, Self::Unreached) | (Self::Common, _) => {},
            (Self::Unreached, _) => *self = other.clone(),
            (_, Self::Common) => *self = Self::Common,
            (
                Self::Branches { cond, indices },
                Self::Branches {
                    cond: other_cond,
                    indices: other_indices,
                },
            ) => {
                if cond == other_cond {
                    for &i in other_indices {
                        if let Err(pos) = indices.binary_search(&i) {
                            indices.insert(pos, i);
                        }
                    }
                } else {
                    *self = Self::Common;
                }
            },
        }
    }
}

/// The single branch that exclusively owns this node, if exactly one does.
#[must_use]
fn sole_owner(ownership: &Ownership) -> Option<BranchLabel> {
    match ownership {
        Ownership::Branches { cond, indices } if indices.len() == 1 => Some(BranchLabel {
            cond: *cond,
            index: indices[0],
        }),
        _ => None,
    }
}

/// Cheap over-approximate check for any conditional that could become blocks.
#[must_use]
fn has_multi_branch_conditional(arena: &Arena) -> bool {
    arena
        .nodes()
        .iter()
        .any(|node| matches!(node, Node::Conditional { branches, .. } if branches.len() >= 2))
}

/// Whether `cond`'s branches may become instruction blocks (it is `Common`, i.e. top-level).
///
/// A deeply nested conditional degraded to `Common` also qualifies, but its runs are empty.
#[must_use]
fn is_conditional_blockable(arena: &Arena, owners: &[Ownership], cond: NodeId) -> bool {
    matches!(arena.get(cond), Node::Conditional { branches, .. } if branches.len() >= 2)
        && owners[cond.index()] == Ownership::Common
}

/// Label every node in `eval_order` with the branches that exclusively need it.
///
/// Indexed by `NodeId::index()`; nodes outside `eval_order` stay [`Ownership::Unreached`].
#[must_use]
pub fn owner_sets(arena: &Arena, eval_order: &[NodeId]) -> Vec<Ownership> {
    let mut owners = vec![Ownership::Unreached; arena.len()];
    if let Some(&root) = eval_order.last() {
        owners[root.index()] = Ownership::Common;
    }

    for &id in eval_order.iter().rev() {
        let label = owners[id.index()].clone();
        match arena.get(id) {
            Node::Conditional { selector, branches } if branches.len() >= 2 => {
                owners[selector.index()].join(&Ownership::Common);
                for (i, &b) in branches.iter().enumerate() {
                    let edge = Ownership::Branches {
                        cond: id,
                        indices: vec![i],
                    };
                    owners[b.index()].join(&edge);
                }
            },
            node => node.for_each_child(|c| owners[c.index()].join(&label)),
        }
    }

    close_full_branch_coverage(arena, &mut owners);
    force_nested_conditionals_common(arena, eval_order, &mut owners);
    owners
}

/// Degrade a node to `Common` once its `Branches` label spans every branch.
fn close_full_branch_coverage(arena: &Arena, owners: &mut [Ownership]) {
    for owner in owners.iter_mut() {
        if let Ownership::Branches { cond, indices } = owner
            && let Node::Conditional { branches, .. } = arena.get(*cond)
            && indices.len() >= branches.len()
        {
            *owner = Ownership::Common;
        }
    }
}

/// Degrade the cone of every non-top-level conditional to `Common`.
///
/// Keeps blocks flat and disjoint, which the evaluator's block skipping relies on.
fn force_nested_conditionals_common(
    arena: &Arena,
    eval_order: &[NodeId],
    owners: &mut [Ownership],
) {
    let mut nested = vec![false; arena.len()];
    let mut any_nested = false;
    for &id in eval_order {
        if matches!(arena.get(id), Node::Conditional { branches, .. } if branches.len() >= 2)
            && owners[id.index()] != Ownership::Common
        {
            nested[id.index()] = true;
            any_nested = true;
        }
    }
    if !any_nested {
        return;
    }
    for &id in eval_order {
        if let Ownership::Branches { cond, .. } = &owners[id.index()]
            && nested[cond.index()]
        {
            owners[id.index()] = Ownership::Common;
        }
    }
}

/// Ceiling on the privatised arena's node count, as a multiple of the input's.
///
/// Chain-nested cones grow the arena quadratically in branch count; this caps them.
const CLONE_BUDGET_MULTIPLE: usize = 4;

/// Clone subgraphs shared by a strict subset of one conditional's branches, one copy per branch.
///
/// Returns `None` when nothing needs cloning or the clone budget is exceeded.
/// Runs inside lowering; no hash-consing pass may follow it, or `cse` would merge the
/// per-branch clones back.
#[must_use]
pub fn privatise_conditionals(dag: DagRef<'_>) -> Option<Dag> {
    let (arena, root) = (dag.arena(), dag.root());
    if !has_multi_branch_conditional(arena) {
        return None;
    }
    let eval_order = arena.topological_order(root);
    let owners = owner_sets(arena, &eval_order);

    // Privatisable iff owned by 2..n of one blockable conditional's branches; a label covering
    // all n was already degraded to `Common`.
    let privatisable = |id: NodeId| match &owners[id.index()] {
        Ownership::Branches { cond, indices }
            if indices.len() >= 2 && is_conditional_blockable(arena, &owners, *cond) =>
        {
            Some(indices.as_slice())
        },
        _ => None,
    };

    let mut projected_nodes = 0_usize;
    let mut any_privatisable = false;
    for &id in &eval_order {
        match privatisable(id) {
            Some(indices) => {
                any_privatisable = true;
                projected_nodes += indices.len();
            },
            None => projected_nodes += 1,
        }
    }
    if !any_privatisable || projected_nodes > CLONE_BUDGET_MULTIPLE * arena.len() {
        return None;
    }

    let mut new_arena = Arena::new();
    let mut common: Vec<Option<NodeId>> = vec![None; arena.len()];
    let mut per_branch: Vec<Vec<(usize, NodeId)>> = vec![Vec::new(); arena.len()];

    for &id in &eval_order {
        let node = arena.get(id);
        match privatisable(id) {
            None => {
                let remapped =
                    remap_children(node, owner_branch_of(&owners, id), &common, &per_branch);
                common[id.index()] = Some(new_arena.alloc(remapped));
            },
            Some(indices) => {
                for &i in indices {
                    let remapped = remap_children(node, Some(i), &common, &per_branch);
                    let clone = new_arena.alloc(remapped);
                    per_branch[id.index()].push((i, clone));
                }
            },
        }
    }

    let new_root = common[root.index()].expect("the root is never privatised");
    Some(Dag::new(new_arena, new_root))
}

/// Rebuild `node` against the privatised copies, reading each child from `branch`'s copy.
///
/// A multi-branch `Conditional` resolves per slot, since `cse` can alias two branch slots
/// onto one node; a single-branch one is owned like any node, as in [`owner_sets`].
fn remap_children(
    node: &Node,
    branch: Option<usize>,
    common: &[Option<NodeId>],
    per_branch: &[Vec<(usize, NodeId)>],
) -> Node {
    match node {
        Node::Conditional { selector, branches } if branches.len() >= 2 => Node::Conditional {
            selector: resolve_child(common, per_branch, *selector, branch),
            branches: branches
                .iter()
                .enumerate()
                .map(|(i, &b)| resolve_child(common, per_branch, b, Some(i)))
                .collect(),
        },
        other => other.map_children(|c| resolve_child(common, per_branch, c, branch)),
    }
}

fn owner_branch_of(owners: &[Ownership], id: NodeId) -> Option<usize> {
    sole_owner(&owners[id.index()]).map(|label| label.index)
}

/// One conditional's branch nodes, as a contiguous range of the emission order.
#[derive(Clone, Debug)]
pub struct RegionGroup {
    pub cond: NodeId,
    /// Index in [`RegionSchedule::order`] where this group's block nodes start.
    pub anchor: usize,
    /// Node count per branch; branches sit back to back from `anchor`.
    pub branch_lens: Vec<usize>,
}

/// A block-contiguous emission order plus the groups to wrap in a `Dispatch`.
#[derive(Clone, Debug)]
pub struct RegionSchedule {
    /// Emission order for every reachable node.
    pub order: Vec<NodeId>,
    pub groups: Vec<RegionGroup>,
}

/// [`owner_sets`] over `eval_order`, or `None` when no conditional can become blocks.
///
/// Computed once per lowering and shared by every schedule of that arena.
#[must_use]
pub fn region_owners(arena: &Arena, eval_order: &[NodeId]) -> Option<Vec<Ownership>> {
    has_multi_branch_conditional(arena).then(|| owner_sets(arena, eval_order))
}

/// Reorder topological `base_order` so each blockable conditional's branch-owned nodes are
/// contiguous, grouped by branch, and immediately precede the `Conditional`.
#[must_use]
pub fn schedule_regions(arena: &Arena, base_order: &[NodeId]) -> RegionSchedule {
    let owners = region_owners(arena, base_order);
    schedule_regions_partitioned(arena, base_order, base_order, owners.as_deref())
}

/// [`schedule_regions`] for `partition`, an ordered subset of `base_order`, whose
/// [`region_owners`] are `owners`.
///
/// A group whose `Conditional` is outside `partition` anchors at the end, and only
/// forms when `partition` holds the selector.
#[must_use]
pub fn schedule_regions_partitioned(
    arena: &Arena,
    partition: &[NodeId],
    base_order: &[NodeId],
    owners: Option<&[Ownership]>,
) -> RegionSchedule {
    let unscheduled = || RegionSchedule {
        order: partition.to_vec(),
        groups: Vec::new(),
    };
    let Some(owners) = owners else {
        return unscheduled();
    };
    let mut in_partition = vec![false; arena.len()];
    for &id in partition {
        in_partition[id.index()] = true;
    }

    // One run per blockable conditional that can group here, keyed by its node.
    let mut run_of: Vec<Option<usize>> = vec![None; arena.len()];
    let mut runs: Vec<(NodeId, Vec<Vec<NodeId>>)> = Vec::new();
    for &id in base_order {
        if let Node::Conditional { selector, branches } = arena.get(id)
            && is_conditional_blockable(arena, owners, id)
            && (in_partition[id.index()] || in_partition[selector.index()])
        {
            run_of[id.index()] = Some(runs.len());
            runs.push((id, vec![Vec::new(); branches.len()]));
        }
    }
    if runs.is_empty() {
        return unscheduled();
    }
    let owning_run = |id: NodeId| {
        let label = sole_owner(&owners[id.index()])?;
        run_of[label.cond.index()].map(|run| (run, label.index))
    };
    for &id in partition {
        if let Some((run, branch)) = owning_run(id) {
            runs[run].1[branch].push(id);
        }
    }

    let mut order = Vec::with_capacity(partition.len());
    let mut groups = Vec::with_capacity(runs.len());
    let mut pending = vec![true; runs.len()];

    for &id in partition {
        if owning_run(id).is_some() {
            continue;
        }
        if let Some(pos) = run_of[id.index()] {
            let branch_runs = std::mem::take(&mut runs[pos].1);
            flush_group(&mut order, &mut groups, id, &branch_runs);
            pending[pos] = false;
        }
        order.push(id);
    }
    for (pos, run) in runs.iter_mut().enumerate() {
        if pending[pos] {
            let branch_runs = std::mem::take(&mut run.1);
            flush_group(&mut order, &mut groups, run.0, &branch_runs);
        }
    }

    // Release-mode: a dropped node lowers to silently wrong values, not a crash.
    assert_eq!(
        order.len(),
        partition.len(),
        "region scheduling dropped or duplicated a node"
    );
    RegionSchedule { order, groups }
}

/// Append one group's branch runs to `order`; an empty group is dropped to keep anchors distinct.
fn flush_group(
    order: &mut Vec<NodeId>,
    groups: &mut Vec<RegionGroup>,
    cond: NodeId,
    branch_runs: &[Vec<NodeId>],
) {
    if branch_runs.iter().all(Vec::is_empty) {
        return;
    }
    let anchor = order.len();
    let branch_lens = branch_runs.iter().map(Vec::len).collect();
    for run in branch_runs {
        order.extend_from_slice(run);
    }
    groups.push(RegionGroup {
        cond,
        anchor,
        branch_lens,
    });
}

/// The active branch for `selector` (1-based, round-to-nearest), or `None` when nothing matches.
///
/// Shared by every evaluator so their branch choices cannot drift apart.
#[inline]
#[must_use]
pub fn active_branch(selector: f64, n_branches: usize) -> Option<usize> {
    for i in 0..n_branches {
        let idx = (i + 1) as f64;
        if selector > idx - 0.5 && selector < idx + 0.5 {
            return Some(i);
        }
    }
    None
}

/// Resolve `child` to its `branch` copy when one was made, otherwise the common copy.
fn resolve_child(
    common: &[Option<NodeId>],
    per_branch: &[Vec<(usize, NodeId)>],
    child: NodeId,
    branch: Option<usize>,
) -> NodeId {
    if let Some(b) = branch
        && let Some(&(_, id)) = per_branch[child.index()].iter().find(|&&(i, _)| i == b)
    {
        return id;
    }
    common[child.index()].unwrap_or_else(|| {
        panic!("child {child:?} has no copy reachable from its consumer's branch")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::CompiledExpr;
    use crate::ir::{TypedIr, UnaryOp};
    use crate::node::Node;

    /// Assert `before` and `after` agree bitwise at `y = 0.3` for every selector value.
    #[track_caller]
    fn assert_same_for_selectors(before: &CompiledExpr, after: &CompiledExpr, selectors: &[f64]) {
        for &sel_val in selectors {
            let mut s1 = vec![0.0; before.scratch_len()];
            let mut s2 = vec![0.0; after.scratch_len()];
            let a = before.eval(&mut s1, 0.0, &[0.3], &[], &[sel_val]);
            let b = after.eval(&mut s2, 0.0, &[0.3], &[], &[sel_val]);
            assert_eq!(a[0].to_bits(), b[0].to_bits(), "selector {sel_val}");
        }
    }

    #[test]
    fn exclusive_branch_cones_are_labelled_per_branch() {
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

        let order = arena.topological_order(cond);
        let owners = owner_sets(&arena, &order);

        assert_eq!(
            sole_owner(&owners[b1.index()]),
            Some(BranchLabel { cond, index: 0 })
        );
        assert_eq!(
            sole_owner(&owners[b2.index()]),
            Some(BranchLabel { cond, index: 1 })
        );
        assert_eq!(owners[y.index()], Ownership::Common);
        assert_eq!(owners[sel.index()], Ownership::Common);
        assert_eq!(owners[cond.index()], Ownership::Common);
    }

    #[test]
    fn strict_subset_sharing_records_every_sharing_branch() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::Scalar(1.0));
        let shared = arena.alloc(Node::Exp(y));
        let b1 = arena.alloc(Node::Sin(shared));
        let b2 = arena.alloc(Node::Cos(shared));
        let b3 = arena.alloc(Node::Neg(y));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2, b3],
        });

        let order = arena.topological_order(cond);
        let owners = owner_sets(&arena, &order);

        assert_eq!(
            owners[shared.index()],
            Ownership::Branches {
                cond,
                indices: vec![0, 1]
            }
        );
        assert_eq!(sole_owner(&owners[shared.index()]), None);
    }

    #[test]
    fn nodes_used_outside_the_conditional_stay_common() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::Scalar(1.0));
        let shared = arena.alloc(Node::Exp(y));
        let b1 = arena.alloc(Node::Sin(shared));
        let b2 = arena.alloc(Node::Cos(y));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2],
        });
        let root = arena.alloc(Node::Add(cond, shared));

        let order = arena.topological_order(root);
        let owners = owner_sets(&arena, &order);

        assert_eq!(owners[shared.index()], Ownership::Common);
        assert_eq!(
            sole_owner(&owners[b1.index()]),
            Some(BranchLabel { cond, index: 0 })
        );
    }

    #[test]
    fn selector_cone_is_common_even_when_otherwise_exclusive() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::Floor(y));
        let b1 = arena.alloc(Node::Sin(y));
        let b2 = arena.alloc(Node::Cos(y));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2],
        });

        let order = arena.topological_order(cond);
        let owners = owner_sets(&arena, &order);

        assert_eq!(owners[sel.index()], Ownership::Common);
    }

    #[test]
    fn nested_conditional_cone_is_forced_common() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let outer_sel = arena.alloc(Node::Scalar(1.0));
        let inner_sel = arena.alloc(Node::Scalar(2.0));
        let inner_b1 = arena.alloc(Node::Sin(y));
        let inner_b2 = arena.alloc(Node::Cos(y));
        let inner = arena.alloc(Node::Conditional {
            selector: inner_sel,
            branches: vec![inner_b1, inner_b2],
        });
        let outer_b2 = arena.alloc(Node::Neg(y));
        let outer = arena.alloc(Node::Conditional {
            selector: outer_sel,
            branches: vec![inner, outer_b2],
        });

        let order = arena.topological_order(outer);
        let owners = owner_sets(&arena, &order);

        assert!(is_conditional_blockable(&arena, &owners, outer));
        assert!(!is_conditional_blockable(&arena, &owners, inner));
        assert_eq!(owners[inner_b1.index()], Ownership::Common);
        assert_eq!(owners[inner_b2.index()], Ownership::Common);
        assert_eq!(
            sole_owner(&owners[inner.index()]),
            Some(BranchLabel {
                cond: outer,
                index: 0
            })
        );
    }

    #[test]
    fn sharing_across_two_conditionals_degrades_to_common() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::Scalar(1.0));
        let shared = arena.alloc(Node::Exp(y));
        let a1 = arena.alloc(Node::Sin(shared));
        let a2 = arena.alloc(Node::Neg(y));
        let c1 = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![a1, a2],
        });
        let b1 = arena.alloc(Node::Cos(shared));
        let b2 = arena.alloc(Node::Abs(y));
        let c2 = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2],
        });
        let root = arena.alloc(Node::Add(c1, c2));

        let order = arena.topological_order(root);
        let owners = owner_sets(&arena, &order);

        assert_eq!(owners[shared.index()], Ownership::Common);
    }

    #[test]
    fn plain_expression_is_all_common() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let root = arena.alloc(Node::Sin(y));
        let order = arena.topological_order(root);
        let owners = owner_sets(&arena, &order);
        assert!(order.iter().all(|n| owners[n.index()] == Ownership::Common));
    }

    #[test]
    fn privatise_is_a_no_op_without_strict_subset_sharing() {
        let mut plain = Arena::new();
        let y = plain.alloc(Node::StateVector { start: 0, end: 1 });
        let root = plain.alloc(Node::Sin(y));
        assert!(privatise_conditionals(DagRef::new(&plain, root)).is_none());

        let mut all = Arena::new();
        let y = all.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = all.alloc(Node::Scalar(1.0));
        let shared = all.alloc(Node::Exp(y));
        let b1 = all.alloc(Node::Sin(shared));
        let b2 = all.alloc(Node::Cos(shared));
        let cond = all.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2],
        });
        assert!(privatise_conditionals(DagRef::new(&all, cond)).is_none());
    }

    #[test]
    fn privatise_clones_a_strict_subset_cone_per_branch() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::Scalar(1.0));
        let inner = arena.alloc(Node::Exp(y));
        let shared = arena.alloc(Node::Sqrt(inner));
        let b1 = arena.alloc(Node::Sin(shared));
        let b2 = arena.alloc(Node::Cos(shared));
        let b3 = arena.alloc(Node::Neg(y));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2, b3],
        });

        let private = privatise_conditionals(DagRef::new(&arena, cond))
            .expect("strict subset must privatise");
        assert!(private.arena().len() <= CLONE_BUDGET_MULTIPLE * arena.len());

        let order = private.arena().topological_order(private.root());
        let n_sqrt = order
            .iter()
            .filter(|&&id| matches!(private.arena().get(id), Node::Sqrt(_)))
            .count();
        let n_exp = order
            .iter()
            .filter(|&&id| matches!(private.arena().get(id), Node::Exp(_)))
            .count();
        assert_eq!((n_sqrt, n_exp), (2, 2));

        let owners = owner_sets(private.arena(), &order);
        for &id in &order {
            if matches!(private.arena().get(id), Node::Sqrt(_) | Node::Exp(_)) {
                assert!(
                    sole_owner(&owners[id.index()]).is_some(),
                    "cone node {id:?} is still shared after privatisation"
                );
            }
        }
    }

    #[test]
    fn privatise_preserves_values_for_every_selector() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let shared = arena.alloc(Node::Exp(y));
        let b1 = arena.alloc(Node::Sin(shared));
        let b2 = arena.alloc(Node::Cos(shared));
        let b3 = arena.alloc(Node::Neg(y));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b1, b2, b3],
        });

        let private = privatise_conditionals(DagRef::new(&arena, cond)).expect("privatises");
        let before = CompiledExpr::from_ir(TypedIr::from_dag_raw(DagRef::new(&arena, cond)));
        let after = CompiledExpr::from_ir(TypedIr::from_dag_raw(private.as_ref()));
        let selectors = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 0.5, 1.5, f64::NAN];
        assert_same_for_selectors(&before, &after, &selectors);
    }

    /// Regression: `cse` aliasing two branch slots onto one node used to panic.
    #[test]
    fn aliased_branch_slot_privatises_per_position() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let shared = arena.alloc(Node::Scalar(1.0));
        let other = arena.alloc(Node::Sin(y));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![shared, other, shared],
        });

        let order = arena.topological_order(cond);
        let owners = owner_sets(&arena, &order);
        assert_eq!(
            owners[shared.index()],
            Ownership::Branches {
                cond,
                indices: vec![0, 2]
            },
            "an aliased branches slot is owned by exactly the positions that name it"
        );

        let before = CompiledExpr::from_ir(TypedIr::from_dag_raw(DagRef::new(&arena, cond)));
        let private =
            privatise_conditionals(DagRef::new(&arena, cond)).expect("the aliased slot privatises");
        let Node::Conditional { branches, .. } = private.arena().get(private.root()) else {
            panic!("the root is still the conditional")
        };
        assert_ne!(
            branches[0], branches[2],
            "each aliased position must resolve to its own clone"
        );
        let after = CompiledExpr::from_ir(TypedIr::from_dag_raw(private.as_ref()));
        let selectors = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 0.5, 1.5, f64::NAN];
        assert_same_for_selectors(&before, &after, &selectors);
    }

    /// Regression: a single-branch conditional is no conditional to `owner_sets`, so its
    /// branch must resolve through the enclosing branch's copy, not slot position 0.
    #[test]
    fn a_single_branch_conditional_in_a_shared_cone_privatises() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let always = arena.alloc(Node::Scalar(1.0));
        let sin = arena.alloc(Node::Sin(y));
        let single = arena.alloc(Node::Conditional {
            selector: always,
            branches: vec![sin],
        });
        let b0 = arena.alloc(Node::Neg(y));
        let b1 = arena.alloc(Node::Exp(single));
        let b2 = arena.alloc(Node::Cos(single));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![b0, b1, b2],
        });

        let dag = DagRef::new(&arena, cond);
        assert!(
            privatise_conditionals(dag).is_some(),
            "the single-branch cone is shared by branches 1 and 2"
        );
        let raw = CompiledExpr::from_ir(TypedIr::from_dag_raw(dag));
        let private = CompiledExpr::new(dag);
        assert_same_for_selectors(&raw, &private, &[1.0, 2.0, 3.0, 0.5, f64::NAN]);
    }

    /// `cond(sel, [tanh(y), tanh²(y), ..., tanhⁿ(y)])`: each cone contains the previous.
    fn chain_nested_conditional(n: usize) -> (Arena, NodeId) {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let mut node = y;
        let mut branches = Vec::with_capacity(n);
        for _ in 0..n {
            node = arena.alloc(Node::Tanh(node));
            branches.push(node);
        }
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches,
        });
        (arena, cond)
    }

    fn expected_chain(y: f64, n: usize, selector: f64) -> f64 {
        active_branch(selector, n).map_or(0.0, |i| {
            let mut v = y;
            for _ in 0..=i {
                v = UnaryOp::Tanh.apply(v);
            }
            v
        })
    }

    #[test]
    fn a_chain_nested_conditional_bails_out_of_the_clone_budget() {
        const N: usize = 32;
        let (arena, cond) = chain_nested_conditional(N);
        assert!(
            privatise_conditionals(DagRef::new(&arena, cond)).is_none(),
            "the projected clone count must exceed the budget"
        );

        let ir = TypedIr::from_dag(DagRef::new(&arena, cond));
        let budget = CLONE_BUDGET_MULTIPLE * arena.len();
        assert!(
            ir.instructions().len() <= budget,
            "raw tape of {} instructions exceeds the {budget}-instruction budget",
            ir.instructions().len()
        );

        let expr = CompiledExpr::from_ir(ir);
        let mut scratch = vec![0.0; expr.scratch_len()];
        let y0 = 0.4_f64;
        for sel_val in [
            0.0_f64,
            1.0,
            2.0,
            17.0,
            32.0,
            33.0,
            0.5,
            1.5,
            -1.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            let got = expr.eval(&mut scratch, 0.0, &[y0], &[], &[sel_val])[0];
            let want = expected_chain(y0, N, sel_val);
            assert_eq!(got.to_bits(), want.to_bits(), "selector {sel_val}");
        }
    }

    #[test]
    fn a_short_chain_still_privatises_within_the_budget() {
        const N: usize = 4;
        let (arena, cond) = chain_nested_conditional(N);
        let private = privatise_conditionals(DagRef::new(&arena, cond))
            .expect("a 4-branch chain fits the budget");
        assert!(
            private.arena().len() > arena.len(),
            "cones must have been cloned"
        );

        let expr = CompiledExpr::new(DagRef::new(&arena, cond));
        let mut scratch = vec![0.0; expr.scratch_len()];
        let y0 = 0.4_f64;
        for sel_val in [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, f64::NAN] {
            let got = expr.eval(&mut scratch, 0.0, &[y0], &[], &[sel_val])[0];
            let want = expected_chain(y0, N, sel_val);
            assert_eq!(got.to_bits(), want.to_bits(), "selector {sel_val}");
        }
    }

    #[test]
    fn aliased_and_disjoint_shared_cones_privatise_independently() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sel = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let x = arena.alloc(Node::Scalar(1.0));
        let inner = arena.alloc(Node::Exp(y));
        let q = arena.alloc(Node::Sqrt(inner));
        let b1 = arena.alloc(Node::Sin(q));
        let b3 = arena.alloc(Node::Cos(q));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![x, b1, x, b3],
        });

        let order = arena.topological_order(cond);
        let owners = owner_sets(&arena, &order);
        assert_eq!(
            owners[x.index()],
            Ownership::Branches {
                cond,
                indices: vec![0, 2]
            },
            "the aliased slot is owned by exactly the positions that name it"
        );
        assert_eq!(
            owners[q.index()],
            Ownership::Branches {
                cond,
                indices: vec![1, 3]
            },
            "a disjoint shared cone privatises on its own index-set"
        );

        let private = privatise_conditionals(DagRef::new(&arena, cond))
            .expect("the disjoint shared cone must privatise");

        let new_order = private.arena().topological_order(private.root());
        let n_sqrt = new_order
            .iter()
            .filter(|&&id| matches!(private.arena().get(id), Node::Sqrt(_)))
            .count();
        let n_exp = new_order
            .iter()
            .filter(|&&id| matches!(private.arena().get(id), Node::Exp(_)))
            .count();
        assert_eq!(
            (n_sqrt, n_exp),
            (2, 2),
            "q and inner must be cloned once per sharing branch"
        );
        let n_aliased_scalar = new_order
            .iter()
            .filter(|&&id| matches!(private.arena().get(id), Node::Scalar(v) if v.to_bits() == 1.0_f64.to_bits()))
            .count();
        assert_eq!(
            n_aliased_scalar, 2,
            "the aliased slot is cloned once per position that names it"
        );

        let before = CompiledExpr::from_ir(TypedIr::from_dag_raw(DagRef::new(&arena, cond)));
        let after = CompiledExpr::from_ir(TypedIr::from_dag_raw(private.as_ref()));
        let selectors = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, f64::NAN];
        assert_same_for_selectors(&before, &after, &selectors);
    }
}

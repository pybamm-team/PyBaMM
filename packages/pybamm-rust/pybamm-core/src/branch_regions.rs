//! Conditional branch region analysis.
//!
//! A `Node::Conditional` lowers every branch into one flat instruction tape. To
//! execute only the active branch, the compiler must first know which nodes
//! belong exclusively to which branch. This module answers that with a
//! consumer-propagation pass over the DAG, and uses the answer twice: to
//! privatise subgraphs `cse` made shared between a strict subset of branches
//! ([`privatise_conditionals`]), and to schedule each branch's nodes into one
//! contiguous instruction block.
//!
//! Every ambiguous case degrades to [`Ownership::Common`], "compute it
//! always", which costs performance but never correctness.

use crate::arena::{Arena, NodeId};
use crate::ir::ConstPool;
use crate::node::Node;

/// One branch of one conditional: `branches[index]` of `cond`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchLabel {
    /// The `Node::Conditional` this branch belongs to.
    pub cond: NodeId,
    /// Position of the branch within that conditional, 0-based.
    pub index: u32,
}

/// Which conditional branches exclusively need a node's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ownership {
    /// No consumer seen (unreachable from the root).
    Unreached,
    /// Needed only by the listed branches of one conditional. `indices` is
    /// sorted and deduplicated.
    Branches { cond: NodeId, indices: Vec<u32> },
    /// Needed regardless of which branch runs: reachable from outside the
    /// conditional, needed by branches of more than one conditional, or the
    /// selector's own cone.
    Common,
}

impl Ownership {
    /// Join two labels in the lattice `Unreached < Branches < Common`. Two
    /// `Branches` over the same conditional merge their index sets; over
    /// different conditionals they degrade to `Common`.
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
pub fn sole_owner(ownership: &Ownership) -> Option<BranchLabel> {
    match ownership {
        Ownership::Branches { cond, indices } if indices.len() == 1 => Some(BranchLabel {
            cond: *cond,
            index: indices[0],
        }),
        _ => None,
    }
}

/// Whether `arena` holds any conditional that could become blocks at all.
///
/// A linear, allocation-free scan that skips [`owner_sets`] on the overwhelming
/// majority of compiles, which carry no conditional. Answers over the whole arena
/// rather than one root's cone, so it can only over-approximate.
#[must_use]
pub fn has_multi_branch_conditional(arena: &Arena) -> bool {
    arena
        .nodes()
        .iter()
        .any(|node| matches!(node, Node::Conditional { branches, .. } if branches.len() >= 2))
}

/// Whether `cond`'s branches may become instruction blocks.
///
/// True for a top-level conditional: one whose own value is needed regardless of
/// which branch of anything else runs. It is also true for a conditional nested
/// three or more deep, whose own label the nested-conditional pass has degraded to
/// [`Ownership::Common`]. That costs nothing and cannot nest blocks: the same pass
/// degraded every node its branches could own, so its branch runs come out empty
/// and scheduling drops the group.
#[must_use]
pub fn is_conditional_blockable(arena: &Arena, owners: &[Ownership], cond: NodeId) -> bool {
    matches!(arena.get(cond), Node::Conditional { branches, .. } if branches.len() >= 2)
        && owners[cond.index()] == Ownership::Common
}

/// Label every node in `eval_order` with the branches that exclusively need it.
///
/// Propagates labels from consumers to producers over the reverse of the
/// topological order. The root is `Common`; a `Conditional`'s branch edge
/// carries that branch's label, its selector edge carries `Common`, and every
/// other edge carries the consumer's own label.
///
/// Returns a vector indexed by `NodeId::index()`, length `arena.len()`; nodes
/// outside `eval_order` stay [`Ownership::Unreached`].
#[must_use]
// A branch index never approaches u32::MAX; `Vec::len()` is `usize` only by convention.
#[allow(clippy::cast_possible_truncation)]
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
                        indices: vec![i as u32],
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

/// Degrade a node to `Common` once its `Branches` label already spans every
/// branch of its conditional.
///
/// Consumer-propagation merges branch labels one edge at a time, so a node fed
/// by all of a conditional's branches only reveals that once every branch has
/// contributed its edge. Full coverage means "needed no matter which branch
/// runs", i.e. `Common`, not a `Branches` set naming all of them.
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
/// Keeps blocks flat and pairwise disjoint, which is what makes the evaluator's
/// jump-over-inactive-block correct without a nesting-aware skip table. The
/// rewrite is closed: a node labelled `Branches { cond: inner, .. }` can only be
/// consumed by nodes with that same label or `Common`, so no further propagation
/// is needed.
fn force_nested_conditionals_common(
    arena: &Arena,
    eval_order: &[NodeId],
    owners: &mut [Ownership],
) {
    let nested: Vec<NodeId> = eval_order
        .iter()
        .copied()
        .filter(|&id| {
            matches!(arena.get(id), Node::Conditional { branches, .. } if branches.len() >= 2)
                && owners[id.index()] != Ownership::Common
        })
        .collect();
    if nested.is_empty() {
        return;
    }
    for &id in eval_order {
        if let Ownership::Branches { cond, .. } = &owners[id.index()]
            && nested.contains(cond)
        {
            owners[id.index()] = Ownership::Common;
        }
    }
}

/// Ceiling on the privatised arena's node count, as a multiple of the input's.
///
/// Cloning is bounded only when sharing is all-or-nothing. In chain-nested cones,
/// where branch *i*'s cone strictly contains branch *i-1*'s, every interior node is
/// shared with a strict *subset*, so the arena grows quadratically in branch count:
/// a `tanh` chain measures 1.4x at 4 branches, 2.9x at 8 and 14.3x at 32.
///
/// 4x admits every shape `PyBaMM` builds today (widest measured 1.97x) while
/// still capping the chain shape from 16 branches up. Over budget costs only the
/// short-circuit, never correctness.
const CLONE_BUDGET_MULTIPLE: usize = 4;

/// Clone subgraphs shared between a strict subset of one conditional's branches
/// so each sharing branch owns its own copy.
///
/// Returns `None` when nothing needs cloning, meaning there is no conditional or
/// every shared cone node is `Common` or shared by *all* branches. Also returns
/// `None` when the projected clone count would exceed the clone budget. `None` is
/// only slower, never wrong: the caller lowers the unprivatised arena.
///
/// Must run **after** `cse`, which created the sharing, and immediately before
/// lowering, which is why `IRBuilder` calls it.
#[must_use]
pub fn privatise_conditionals(arena: &Arena, root: NodeId) -> Option<(Arena, NodeId)> {
    if !has_multi_branch_conditional(arena) {
        return None;
    }
    let eval_order = arena.topological_order(root);
    let owners = owner_sets(arena, &eval_order);

    // A node is privatisable iff it is owned by 2..n of one *blockable*
    // conditional's branches.
    let privatisable = |id: NodeId| -> Option<(NodeId, &[u32])> {
        let Ownership::Branches { cond, indices } = &owners[id.index()] else {
            return None;
        };
        if indices.len() < 2 || !is_conditional_blockable(arena, &owners, *cond) {
            return None;
        }
        let Node::Conditional { branches, .. } = arena.get(*cond) else {
            return None;
        };
        (indices.len() < branches.len()).then_some((*cond, indices.as_slice()))
    };

    // Projected new arena size, in the pass that already decides whether anything
    // is privatisable: one copy per kept node, one per sharing branch otherwise.
    let mut projected_nodes = 0_usize;
    let mut any_privatisable = false;
    for &id in &eval_order {
        match privatisable(id) {
            Some((_, indices)) => {
                any_privatisable = true;
                projected_nodes += indices.len();
            },
            None => projected_nodes += 1,
        }
    }
    if !any_privatisable || projected_nodes > CLONE_BUDGET_MULTIPLE * arena.len() {
        return None;
    }

    // Rebuild in topological order: a common node is copied once, a privatised
    // node once per sharing branch with children resolved to that branch's copy.
    let mut new_arena = Arena::new();
    let mut common: Vec<Option<NodeId>> = vec![None; arena.len()];
    let mut per_branch: Vec<Vec<(u32, NodeId)>> = vec![Vec::new(); arena.len()];

    for &id in &eval_order {
        let node = arena.get(id);
        match privatisable(id) {
            None => {
                let remapped =
                    remap_children(node, owner_branch_of(&owners, id), &common, &per_branch);
                common[id.index()] = Some(new_arena.alloc(remapped));
            },
            Some((_, indices)) => {
                for &i in indices {
                    let remapped = remap_children(node, Some(i), &common, &per_branch);
                    let clone = new_arena.alloc(remapped);
                    per_branch[id.index()].push((i, clone));
                }
            },
        }
    }

    let new_root = common[root.index()].expect("the root is never privatised");
    Some((new_arena, new_root))
}

/// Rebuild `node` against the privatised copies, reading each child from
/// `branch`'s copy where one exists.
///
/// `Conditional` is the one node whose child *positions* carry per-branch
/// meaning: `cse` can alias two branch slots onto the same node, and each slot
/// must still resolve to its own branch's clone. Every other node's children
/// inherit the consumer's own branch, so one uniform closure suffices.
///
/// A slot index is unambiguous even though [`resolve_child`] keys clones by index
/// alone: a node needed by two conditionals is `Ownership::Common`, which is not
/// privatisable.
// A branch index never approaches u32::MAX; `Vec::len()` is `usize` only by convention.
#[allow(clippy::cast_possible_truncation)]
fn remap_children(
    node: &Node,
    branch: Option<u32>,
    common: &[Option<NodeId>],
    per_branch: &[Vec<(u32, NodeId)>],
) -> Node {
    match node {
        Node::Conditional { selector, branches } => Node::Conditional {
            selector: resolve_child(common, per_branch, *selector, branch),
            branches: branches
                .iter()
                .enumerate()
                .map(|(i, &b)| resolve_child(common, per_branch, b, Some(i as u32)))
                .collect(),
        },
        other => other.map_children(|c| resolve_child(common, per_branch, c, branch)),
    }
}

/// The branch index a node was assigned to, if exactly one owns it. Picks which
/// copy of a privatised child a consumer should read.
fn owner_branch_of(owners: &[Ownership], id: NodeId) -> Option<u32> {
    sole_owner(&owners[id.index()]).map(|label| label.index)
}

/// One conditional's branch nodes, as a contiguous range of the emission order.
#[derive(Clone, Debug)]
pub struct RegionGroup {
    /// The conditional whose branches this group covers.
    pub cond: NodeId,
    /// Index in [`RegionSchedule::order`] where this group's block nodes start.
    pub anchor: usize,
    /// Node count per branch, in branch order. The branches occupy
    /// `order[anchor .. anchor + branch_lens.iter().sum()]` back to back.
    pub branch_lens: Vec<usize>,
}

/// A block-contiguous emission order plus the groups to wrap in a `Dispatch`.
///
/// `order` holds **every** node, group nodes sit at their block position, so
/// slot allocation can be driven by it with no allocator change.
#[derive(Clone, Debug)]
pub struct RegionSchedule {
    /// Emission order for every reachable node.
    pub order: Vec<NodeId>,
    /// The conditionals whose branch runs came out contiguous, and so can be
    /// guarded by a `Dispatch`.
    pub groups: Vec<RegionGroup>,
}

/// Reorder `base_order` so each blockable conditional's branch-owned nodes are
/// contiguous, grouped by branch, and immediately precede the `Conditional`.
///
/// Valid because a branch-`i`-owned node can only depend on `Common` nodes
/// (all emitted before the `Conditional`, hence before the group) and on other
/// branch-`i`-owned nodes (kept in `base_order` order within the run). A
/// cross-branch dependency is impossible: if a branch-`i` node consumed a node,
/// that node's owner set would include `i`.
///
/// `base_order` must be a topological order (children first).
#[must_use]
pub fn schedule_regions(arena: &Arena, base_order: &[NodeId]) -> RegionSchedule {
    let unscheduled = || RegionSchedule {
        order: base_order.to_vec(),
        groups: Vec::new(),
    };
    if !has_multi_branch_conditional(arena) {
        return unscheduled();
    }
    let owners = owner_sets(arena, base_order);
    let Some(mut runs) = collect_branch_runs(arena, &owners, base_order, base_order) else {
        return unscheduled();
    };

    let mut order = Vec::with_capacity(base_order.len());
    let mut groups = Vec::with_capacity(runs.len());
    for &id in base_order {
        if owned_by_a_run(&owners, &runs, id) {
            continue; // deferred into its group
        }
        if let Some(pos) = runs.iter().position(|(c, _)| *c == id) {
            let branch_runs = std::mem::take(&mut runs[pos].1);
            flush_group(&mut order, &mut groups, id, &branch_runs);
        }
        order.push(id);
    }

    // A dropped node leaves its slot at `(0, 0)`, which lowers to silently wrong
    // values rather than a crash, so this ships rather than being debug-only.
    assert_eq!(
        order.len(),
        base_order.len(),
        "region scheduling dropped or duplicated a node"
    );
    RegionSchedule { order, groups }
}

/// [`schedule_regions`] for one half of a primal/tangent partition.
///
/// `partition` is the subset of `base_order`, in order, that this half emits. A
/// group whose `Conditional` lands in the *other* half anchors at the end of this
/// partition: nothing here consumes it and its `Common` dependencies are all
/// earlier, which lets one branch yield two blocks with a valid split point.
///
/// Such a group forms only when this partition holds the selector, since a
/// `Dispatch` reads the selector's slot. Otherwise it is dropped and its nodes go
/// inline, costing the short-circuit but never correctness.
#[must_use]
pub fn schedule_regions_partitioned(
    arena: &Arena,
    partition: &[NodeId],
    base_order: &[NodeId],
) -> RegionSchedule {
    let unscheduled = || RegionSchedule {
        order: partition.to_vec(),
        groups: Vec::new(),
    };
    if !has_multi_branch_conditional(arena) {
        return unscheduled();
    }
    let owners = owner_sets(arena, base_order);
    // Runs are collected from `partition` only, so a branch's primal nodes and
    // its tangent nodes end up in different halves' groups.
    let Some(mut runs) = collect_branch_runs(arena, &owners, base_order, partition) else {
        return unscheduled();
    };

    let mut in_partition = vec![false; arena.len()];
    for &id in partition {
        in_partition[id.index()] = true;
    }
    runs.retain(|(cond, _)| {
        in_partition[cond.index()]
            || matches!(arena.get(*cond), Node::Conditional { selector, .. }
                if in_partition[selector.index()])
    });

    let mut order = Vec::with_capacity(partition.len());
    let mut groups = Vec::with_capacity(runs.len());
    let mut pending = vec![true; runs.len()];

    for &id in partition {
        if owned_by_a_run(&owners, &runs, id) {
            continue; // deferred into its group
        }
        if let Some(pos) = runs.iter().position(|(c, _)| *c == id) {
            let branch_runs = std::mem::take(&mut runs[pos].1);
            flush_group(&mut order, &mut groups, id, &branch_runs);
            pending[pos] = false;
        }
        order.push(id);
    }
    // Groups whose `Conditional` lives in the other half: nothing here consumes
    // them, so they go last.
    for (pos, run) in runs.iter_mut().enumerate() {
        if pending[pos] {
            let branch_runs = std::mem::take(&mut run.1);
            flush_group(&mut order, &mut groups, run.0, &branch_runs);
        }
    }

    // Ships for the same reason as in `schedule_regions`: a dropped node leaves
    // its slot at `(0, 0)`, which lowers to silently wrong values, not a crash.
    assert_eq!(
        order.len(),
        partition.len(),
        "partitioned region scheduling dropped or duplicated a node"
    );
    RegionSchedule { order, groups }
}

/// Per-branch runs for every blockable conditional, in `source` order.
/// `None` when there is no blockable conditional at all.
fn collect_branch_runs(
    arena: &Arena,
    owners: &[Ownership],
    base_order: &[NodeId],
    source: &[NodeId],
) -> Option<Vec<(NodeId, Vec<Vec<NodeId>>)>> {
    let mut runs: Vec<(NodeId, Vec<Vec<NodeId>>)> = Vec::new();
    for &id in base_order {
        if is_conditional_blockable(arena, owners, id) {
            let Node::Conditional { branches, .. } = arena.get(id) else {
                unreachable!("is_conditional_blockable checked the node kind")
            };
            runs.push((id, vec![Vec::new(); branches.len()]));
        }
    }
    if runs.is_empty() {
        return None;
    }
    for &id in source {
        if let Some(label) = sole_owner(&owners[id.index()])
            && let Some((_, branch_runs)) = runs.iter_mut().find(|(c, _)| *c == label.cond)
        {
            branch_runs[label.index as usize].push(id);
        }
    }
    Some(runs)
}

/// Whether `id` was deferred into one of `runs`' groups.
fn owned_by_a_run(owners: &[Ownership], runs: &[(NodeId, Vec<Vec<NodeId>>)], id: NodeId) -> bool {
    sole_owner(&owners[id.index()]).is_some_and(|label| runs.iter().any(|(c, _)| *c == label.cond))
}

/// Append one group's branch runs to `order` and record the annotated range.
/// A group whose branches own nothing is dropped: a `Dispatch` guarding no
/// instructions is pure overhead, and dropping it keeps anchors distinct.
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

/// The active branch for `selector`, or `None` when nothing matches.
///
/// 1-based round-to-nearest windows, first match wins: branch `i` is active iff
/// `selector > (i+1) - 0.5 && selector < (i+1) + 0.5`. The single home for the
/// window, shared by the scalar, batch and reverse evaluators so their branch
/// choices cannot drift apart.
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

/// One past the last instruction guarded by the `Dispatch` at `pc`.
///
/// The companion of [`active_branch`], and the single home for the *other* half
/// of the `Dispatch` contract: forward evaluation resumes here, and the backward
/// walk skips everything in `(pc, end)` bar the active block. Two copies of this
/// reduction could desynchronise the two directions silently.
///
/// `blocks_len` of 0 falls back to `pc + 1`, so a span always covers at least
/// the `Dispatch` itself.
#[inline]
#[must_use]
pub(crate) fn dispatch_span_end(
    consts: &ConstPool,
    pc: usize,
    blocks_idx: u32,
    blocks_len: u32,
) -> usize {
    (0..blocks_len as usize)
        .map(|b| {
            let (rel, len) = consts.branch_blocks[blocks_idx as usize + b];
            pc + rel as usize + len as usize
        })
        .max()
        .unwrap_or(pc + 1)
}

/// Resolve `child` to the copy the consumer should read: its `branch` copy when
/// one was made, otherwise the single common copy.
fn resolve_child(
    common: &[Option<NodeId>],
    per_branch: &[Vec<(u32, NodeId)>],
    child: NodeId,
    branch: Option<u32>,
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
    use crate::node::Node;

    /// `cond(sel, [sin(y0), cos(y0)])` with `y0` shared: the two unary nodes are
    /// branch-exclusive, `y0` and `sel` are common.
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

    /// A node consumed by two of three branches is `Branches` with both indices,
    /// not `Common`, which is exactly the shape privatisation targets.
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

    /// Reachable from outside the conditional => never owned by a branch.
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
        // `shared` also feeds the root outside the conditional.
        let root = arena.alloc(Node::Add(cond, shared));

        let order = arena.topological_order(root);
        let owners = owner_sets(&arena, &order);

        assert_eq!(owners[shared.index()], Ownership::Common);
        assert_eq!(
            sole_owner(&owners[b1.index()]),
            Some(BranchLabel { cond, index: 0 })
        );
    }

    /// The selector edge carries `Common`: the dispatch reads it before the blocks.
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

    /// An inner conditional inside an outer branch cone is not blockable, and its
    /// own cone is forced `Common`.
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
        // The inner cone degraded to Common; the inner Conditional itself is
        // still owned by the outer branch it sits in.
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

    /// A node needed by branches of two different conditionals degrades.
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

    /// No conditional at all: everything is `Common`, nothing panics.
    #[test]
    fn plain_expression_is_all_common() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let root = arena.alloc(Node::Sin(y));
        let order = arena.topological_order(root);
        let owners = owner_sets(&arena, &order);
        assert!(order.iter().all(|n| owners[n.index()] == Ownership::Common));
    }

    /// Nothing to privatise: no conditional, or no strict-subset sharing.
    #[test]
    fn privatise_is_a_no_op_without_strict_subset_sharing() {
        let mut plain = Arena::new();
        let y = plain.alloc(Node::StateVector { start: 0, end: 1 });
        let root = plain.alloc(Node::Sin(y));
        assert!(privatise_conditionals(&plain, root).is_none());

        // Shared by ALL branches: needed whichever branch runs, so cloning
        // would be pure waste.
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
        assert!(privatise_conditionals(&all, cond).is_none());
    }

    /// Shared by 2 of 3 branches: cloned once per sharing branch, so each
    /// branch's cone becomes exclusive.
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

        let (new_arena, new_root) =
            privatise_conditionals(&arena, cond).expect("strict subset must privatise");
        // The clone budget is a cap on pathological shapes, not on this one.
        assert!(new_arena.len() <= CLONE_BUDGET_MULTIPLE * arena.len());

        let order = new_arena.topological_order(new_root);
        // Both nodes of the shared cone now exist twice.
        let n_sqrt = order
            .iter()
            .filter(|&&id| matches!(new_arena.get(id), Node::Sqrt(_)))
            .count();
        let n_exp = order
            .iter()
            .filter(|&&id| matches!(new_arena.get(id), Node::Exp(_)))
            .count();
        assert_eq!((n_sqrt, n_exp), (2, 2));

        // And every cone node is now exclusively owned by one branch.
        let owners = owner_sets(&new_arena, &order);
        for &id in &order {
            if matches!(new_arena.get(id), Node::Sqrt(_) | Node::Exp(_)) {
                assert!(
                    sole_owner(&owners[id.index()]).is_some(),
                    "cone node {id:?} is still shared after privatisation"
                );
            }
        }
    }

    /// Privatisation is value-preserving: the cloned graph evaluates identically
    /// for every selector, including no-match.
    #[test]
    fn privatise_preserves_values_for_every_selector() {
        use crate::eval::CompiledExpr;
        use crate::ir::TypedIr;

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

        let (new_arena, new_root) = privatise_conditionals(&arena, cond).expect("privatises");
        let before = CompiledExpr::from_ir(TypedIr::from_arena_raw(&arena, cond));
        let after = CompiledExpr::from_ir(TypedIr::from_arena_raw(&new_arena, new_root));

        for sel_val in [0.0_f64, 1.0, 2.0, 3.0, 4.0, 0.5, 1.5, f64::NAN] {
            let mut s1 = vec![0.0; before.scratch_len()];
            let mut s2 = vec![0.0; after.scratch_len()];
            let a = before.eval(&mut s1, 0.0, &[0.3], &[], &[sel_val]);
            let b = after.eval(&mut s2, 0.0, &[0.3], &[], &[sel_val]);
            assert_eq!(a[0].to_bits(), b[0].to_bits(), "selector {sel_val}");
        }
    }

    /// Regression for a real unified-experiment-model panic: two branches of a
    /// 3-way `Conditional` (a current-control step and a power-control step,
    /// both with no extra termination condition) canonicalise to the literal
    /// same `Scalar(1.0)` node under `cse`, aliasing `branches[0]` and
    /// `branches[2]`. The slot stays privatisable, `remap_children` resolves
    /// each branch position to its own clone, so the aliased node is cloned
    /// once per sharing branch and the values are unchanged.
    #[test]
    fn aliased_branch_slot_privatises_per_position() {
        use crate::eval::CompiledExpr;
        use crate::ir::TypedIr;

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
            // branches[0] and branches[2] are the literal same NodeId.
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

        let before = CompiledExpr::from_ir(TypedIr::from_arena_raw(&arena, cond));
        // Must not panic: this call is the regression this test guards.
        let (new_arena, new_root) =
            privatise_conditionals(&arena, cond).expect("the aliased slot privatises");
        let Node::Conditional { branches, .. } = new_arena.get(new_root) else {
            panic!("the root is still the conditional")
        };
        assert_ne!(
            branches[0], branches[2],
            "each aliased position must resolve to its own clone"
        );
        let after = CompiledExpr::from_ir(TypedIr::from_arena_raw(&new_arena, new_root));

        for sel_val in [0.0_f64, 1.0, 2.0, 3.0, 4.0, 0.5, 1.5, f64::NAN] {
            let mut s1 = vec![0.0; before.scratch_len()];
            let mut s2 = vec![0.0; after.scratch_len()];
            let a = before.eval(&mut s1, 0.0, &[0.3], &[], &[sel_val]);
            let b = after.eval(&mut s2, 0.0, &[0.3], &[], &[sel_val]);
            assert_eq!(a[0].to_bits(), b[0].to_bits(), "selector {sel_val}");
        }
    }

    /// `cond(sel, [tanh(y), tanh²(y), ..., tanhⁿ(y)])`: branch `i`'s cone strictly
    /// contains branch `i-1`'s, the shape `cse` makes of a progressive expression.
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

    /// The semantics contract for [`chain_nested_conditional`], written out
    /// independently of the evaluator.
    fn expected_chain(y: f64, n: usize, selector: f64) -> f64 {
        active_branch(selector, n).map_or(0.0, |i| {
            let mut v = y;
            for _ in 0..=i {
                v = v.tanh();
            }
            v
        })
    }

    /// Chain-nested cones share every interior node with a strict *subset* of
    /// branches, so privatising clones each once per sharing branch: quadratic in
    /// the branch count (~14x the arena at 32 branches). The budget must bail out,
    /// leaving the tape the size of the arena and the values untouched.
    #[test]
    fn a_chain_nested_conditional_bails_out_of_the_clone_budget() {
        use crate::eval::CompiledExpr;
        use crate::ir::TypedIr;

        const N: usize = 32;
        let (arena, cond) = chain_nested_conditional(N);
        assert!(
            privatise_conditionals(&arena, cond).is_none(),
            "the projected clone count must exceed the budget"
        );

        let ir = TypedIr::from_arena(&arena, cond);
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

    /// The same shape within budget still privatises, and agrees with the same
    /// oracle: the cap changes how a chain is scheduled, never what it computes.
    #[test]
    fn a_short_chain_still_privatises_within_the_budget() {
        use crate::eval::CompiledExpr;
        use crate::ir::TypedIr;

        const N: usize = 4;
        let (arena, cond) = chain_nested_conditional(N);
        let (new_arena, _) =
            privatise_conditionals(&arena, cond).expect("a 4-branch chain fits the budget");
        assert!(new_arena.len() > arena.len(), "cones must have been cloned");

        let expr = CompiledExpr::from_ir(TypedIr::from_arena(&arena, cond));
        let mut scratch = vec![0.0; expr.scratch_len()];
        let y0 = 0.4_f64;
        for sel_val in [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, f64::NAN] {
            let got = expr.eval(&mut scratch, 0.0, &[y0], &[], &[sel_val])[0];
            let want = expected_chain(y0, N, sel_val);
            assert_eq!(got.to_bits(), want.to_bits(), "selector {sel_val}");
        }
    }

    /// A conditional can carry BOTH an aliased slot pair AND a genuinely shared,
    /// disjoint subgraph at the same time: `branches[0] == branches[2] == x`
    /// (index-set `{0,2}`) while `branches[1]` and `branches[3]` independently
    /// share `q` (index-set `{1,3}`). Both privatise, independently: the two
    /// index-sets never interfere.
    #[test]
    fn aliased_and_disjoint_shared_cones_privatise_independently() {
        use crate::eval::CompiledExpr;
        use crate::ir::TypedIr;

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
            // branches[0] and branches[2] alias the same node `x`;
            // branches[1] and branches[3] independently share `q`.
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

        let (new_arena, new_root) =
            privatise_conditionals(&arena, cond).expect("the disjoint shared cone must privatise");

        let new_order = new_arena.topological_order(new_root);
        let n_sqrt = new_order
            .iter()
            .filter(|&&id| matches!(new_arena.get(id), Node::Sqrt(_)))
            .count();
        let n_exp = new_order
            .iter()
            .filter(|&&id| matches!(new_arena.get(id), Node::Exp(_)))
            .count();
        assert_eq!(
            (n_sqrt, n_exp),
            (2, 2),
            "q and inner must be cloned once per sharing branch"
        );
        let n_aliased_scalar = new_order
            .iter()
            .filter(|&&id| matches!(new_arena.get(id), Node::Scalar(v) if v.to_bits() == 1.0_f64.to_bits()))
            .count();
        assert_eq!(
            n_aliased_scalar, 2,
            "the aliased slot is cloned once per position that names it"
        );

        let before = CompiledExpr::from_ir(TypedIr::from_arena_raw(&arena, cond));
        let after = CompiledExpr::from_ir(TypedIr::from_arena_raw(&new_arena, new_root));
        for sel_val in [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, f64::NAN] {
            let mut s1 = vec![0.0; before.scratch_len()];
            let mut s2 = vec![0.0; after.scratch_len()];
            let a = before.eval(&mut s1, 0.0, &[0.3], &[], &[sel_val]);
            let b = after.eval(&mut s2, 0.0, &[0.3], &[], &[sel_val]);
            assert_eq!(a[0].to_bits(), b[0].to_bits(), "selector {sel_val}");
        }
    }
}

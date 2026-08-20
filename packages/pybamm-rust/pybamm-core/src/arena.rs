//! Flat storage for the expression DAG.
//!
//! Every [`Node`] of a model lives in one `Vec` inside an [`Arena`] and is named
//! by a `u32` [`NodeId`]. Ids are `Copy` and stable for the arena's lifetime, so
//! a shared subexpression is one node referenced twice rather than a cloned
//! subtree, and passes can annotate nodes out-of-band through [`NodeMap`]
//! instead of touching the DAG.
//!
//! Ids are only meaningful against the arena that issued them: rewriting passes
//! that build a new arena (`cse`, `dce`, `privatise_conditionals`) return that
//! arena with a new root, and mixing the two id spaces indexes the wrong node.
//! `simplify` instead appends to the arena it is given, so its root stays valid
//! against the same arena.

use std::ops::Index;

use crate::node::Node;

/// Handle to a node in one [`Arena`], valid only against that arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(u32);

impl NodeId {
    /// Position in the arena, for indexing side tables.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The underlying `u32`, as the instruction tape and FFI carry it.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(val: u32) -> Self {
        Self(val)
    }
}

impl From<usize> for NodeId {
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn from(val: usize) -> Self {
        Self(val as u32)
    }
}

/// Owner of an expression DAG: nodes in allocation order, addressed by
/// [`NodeId`].
///
/// An arena only grows, so ids stay valid while it is alive. Because a node names
/// its children by id, a DAG is acyclic by construction as long as callers keep
/// allocating children before parents, which every builder here does.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    /// An empty arena.
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Append `node` and return its id.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn alloc(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Borrow a node.
    ///
    /// # Panics
    ///
    /// Panics if `id` came from a different arena and is out of range here.
    #[inline]
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    /// Nodes allocated so far, reachable or not.
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Every allocated node, in allocation order, including nodes unreachable
    /// from any particular root. For cheap whole-arena scans.
    #[inline]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Whether nothing has been allocated yet.
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Nodes reachable from `root`, children before parents and each visited once.
    ///
    /// This is the evaluation order every pass works in, and the reason a shared
    /// subexpression is computed once rather than per parent.
    pub fn topological_order(&self, root: NodeId) -> Vec<NodeId> {
        // Iterative post-order DFS. Two frame variants avoid the per-node
        // children buffer an explicit (id, flag) stack would need.
        let n = self.len();
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut stack: Vec<TopoFrame> = Vec::with_capacity(n);
        stack.push(TopoFrame::Pending(root));
        while let Some(frame) = stack.pop() {
            match frame {
                TopoFrame::Visit(id) => {
                    if visited[id.index()] {
                        continue;
                    }
                    visited[id.index()] = true;
                    order.push(id);
                },
                TopoFrame::Pending(id) => {
                    if visited[id.index()] {
                        continue;
                    }
                    stack.push(TopoFrame::Visit(id));
                    self.get(id).for_each_child(|c| {
                        if !visited[c.index()] {
                            stack.push(TopoFrame::Pending(c));
                        }
                    });
                },
            }
        }
        order
    }
}

/// State-input usage of the sub-DAG rooted at `root`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateUsage {
    /// 1 + max state index referenced across both `y` and `y_dot`
    /// (0 if no state nodes are reachable).
    pub n_states: usize,
    /// Whether any `StateVectorDot` node is reachable.
    pub uses_y_dot: bool,
}

/// Scan the reachable sub-DAG for `StateVector`/`StateVectorDot` extents.
///
/// Primal DAGs only: tangent nodes (`TangentStateVector` etc.) are
/// intentionally not counted.
pub fn scan_state_usage(arena: &Arena, root: NodeId) -> StateUsage {
    let mut visited = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut usage = StateUsage {
        n_states: 0,
        uses_y_dot: false,
    };
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut visited[id.index()], true) {
            continue;
        }
        let node = arena.get(id);
        match node {
            Node::StateVector { end, .. } => usage.n_states = usage.n_states.max(*end),
            Node::StateVectorDot { end, .. } => {
                usage.uses_y_dot = true;
                usage.n_states = usage.n_states.max(*end);
            },
            _ => {},
        }
        node.for_each_child(|c| stack.push(c));
    }
    usage
}

enum TopoFrame {
    /// Expands into this node's `Visit` frame with its children stacked above.
    Pending(NodeId),
    /// Children are all emitted; emit this node now.
    Visit(NodeId),
}

impl Index<NodeId> for Arena {
    type Output = Node;

    #[inline]
    fn index(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }
}

/// Per-node side table, keyed by [`NodeId`].
///
/// A dense `Vec<Option<T>>` rather than a hash map: keys are arena positions, so
/// lookup is an index and a pass over every node touches memory in order. Ids are
/// dense and small, which is what makes the wasted `None` slots cheaper than
/// hashing.
#[derive(Clone, Debug)]
pub struct NodeMap<T> {
    slots: Vec<Option<T>>,
}

impl<T> NodeMap<T> {
    /// A table sized for an arena of `arena_len` nodes, all entries empty.
    #[inline]
    pub fn new(arena_len: usize) -> Self {
        let mut slots = Vec::with_capacity(arena_len);
        slots.resize_with(arena_len, || None);
        Self { slots }
    }

    /// The value for `id`, or `None` if unset or out of range.
    #[inline]
    pub fn get(&self, id: NodeId) -> Option<&T> {
        self.slots.get(id.index()).and_then(|opt| opt.as_ref())
    }

    /// Set the value for `id`, returning what it replaced. Grows the table when
    /// the arena has outgrown the size passed to [`new`](Self::new).
    #[inline]
    pub fn insert(&mut self, id: NodeId, value: T) -> Option<T> {
        if id.index() >= self.slots.len() {
            self.slots.resize_with(id.index() + 1, || None);
        }
        self.slots[id.index()].replace(value)
    }

    /// Whether `id` has a value set.
    #[inline]
    pub fn contains_key(&self, id: NodeId) -> bool {
        self.slots.get(id.index()).is_some_and(Option::is_some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    #[test]
    fn test_alloc_and_get() {
        let mut arena = Arena::new();
        let id = arena.alloc(Node::Scalar(42.5));
        match arena.get(id) {
            Node::Scalar(v) => assert!((*v - 42.5).abs() < f64::EPSILON),
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn test_sequential_ids() {
        let mut arena = Arena::new();
        let id0 = arena.alloc(Node::Scalar(1.0));
        let id1 = arena.alloc(Node::Scalar(2.0));
        assert_eq!(id0.index(), 0);
        assert_eq!(id1.index(), 1);
    }

    #[test]
    fn test_node_id_copy() {
        let mut arena = Arena::new();
        let id = arena.alloc(Node::Scalar(1.0));
        let id_copy = id;
        assert_eq!(id, id_copy);
        assert_eq!(arena.get(id), arena.get(id_copy));
    }

    #[test]
    fn test_topological_order_visits_children_first() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(1.0));
        let b = arena.alloc(Node::Scalar(2.0));
        let add = arena.alloc(Node::Add(a, b));

        let order = arena.topological_order(add);

        let pos_a = order.iter().position(|&n| n == a).unwrap();
        let pos_b = order.iter().position(|&n| n == b).unwrap();
        let pos_add = order.iter().position(|&n| n == add).unwrap();
        assert!(pos_a < pos_add);
        assert!(pos_b < pos_add);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_topological_order_deduplicates_shared_subexpression() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(1.0));
        let neg_a = arena.alloc(Node::Neg(a));
        let combined = arena.alloc(Node::Add(neg_a, neg_a));

        let order = arena.topological_order(combined);
        assert_eq!(order.len(), 3);
        assert_eq!(order.iter().filter(|&&n| n == a).count(), 1);
        assert_eq!(order.iter().filter(|&&n| n == neg_a).count(), 1);
    }

    #[test]
    fn test_topological_order_skips_unreachable_nodes() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(1.0));
        let _unreachable = arena.alloc(Node::Scalar(99.0));
        let neg_a = arena.alloc(Node::Neg(a));

        let order = arena.topological_order(neg_a);
        assert_eq!(order.len(), 2);
        assert!(order.contains(&a));
        assert!(order.contains(&neg_a));
    }

    #[test]
    fn scan_state_usage_finds_extents_and_ydot() {
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 2, end: 5 });
        let svd = arena.alloc(Node::StateVectorDot { start: 0, end: 1 });
        let sum = arena.alloc(Node::Add(sv, svd));
        let usage = scan_state_usage(&arena, sum);
        assert_eq!(usage.n_states, 5);
        assert!(usage.uses_y_dot);

        let lone = arena.alloc(Node::Scalar(1.0));
        let usage = scan_state_usage(&arena, lone);
        assert_eq!(usage.n_states, 0);
        assert!(!usage.uses_y_dot);
    }

    #[test]
    fn test_node_map_basic_get_insert() {
        let mut map: NodeMap<u32> = NodeMap::new(4);
        let id0: NodeId = 0u32.into();
        let id3: NodeId = 3u32.into();

        assert!(!map.contains_key(id0));
        assert_eq!(map.get(id0), None);

        assert_eq!(map.insert(id0, 42), None);
        assert_eq!(map.get(id0), Some(&42));
        assert!(map.contains_key(id0));

        assert_eq!(map.insert(id0, 99), Some(42));
        assert_eq!(map.get(id0), Some(&99));

        assert!(!map.contains_key(id3));
        map.insert(id3, 7);
        assert_eq!(map.get(id3), Some(&7));
    }
}

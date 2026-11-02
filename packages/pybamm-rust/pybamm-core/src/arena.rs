//! Flat storage for the expression DAG: every [`Node`] lives in one [`Arena`],
//! addressed by a `u32` [`NodeId`] that is only valid against that arena.

use std::ops::Index;

use crate::error::CoreError;
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

    /// The underlying `u32`.
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

/// Owner of an expression DAG: nodes in allocation order, addressed by [`NodeId`].
///
/// Append-only, so ids stay valid. Every child id is below its parent's, so ids are in
/// range and the graph is acyclic; [`alloc`](Self::alloc) and deserialising enforce it.
#[derive(Clone, Debug, Default)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PersistedArena")
)]
pub struct Arena {
    nodes: Vec<Node>,
}

/// The first child of `node` not allocated before position `at`.
fn first_unallocated_child(node: &Node, at: usize) -> Option<NodeId> {
    let mut unallocated = None;
    node.for_each_child(|child| {
        if child.index() >= at && unallocated.is_none() {
            unallocated = Some(child);
        }
    });
    unallocated
}

impl Arena {
    /// An empty arena.
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Append `node` and return its id.
    ///
    /// Panics if a child is not yet allocated here, or past `u32::MAX` nodes.
    #[must_use]
    pub fn alloc(&mut self, node: Node) -> NodeId {
        let at = self.nodes.len();
        if let Some(child) = first_unallocated_child(&node, at) {
            panic!(
                "node {at} names child {}, which is not allocated before it",
                child.raw()
            );
        }
        let id = NodeId(u32::try_from(at).expect("arena exceeded u32::MAX nodes"));
        self.nodes.push(node);
        id
    }

    /// Borrow a node.
    #[inline]
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    /// Nodes allocated so far, reachable or not.
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Every allocated node in allocation order, reachable or not.
    #[inline]
    pub(crate) fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Whether nothing has been allocated yet.
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Nodes reachable from `root`, children before parents and each visited once.
    pub fn topological_order(&self, root: NodeId) -> Vec<NodeId> {
        let n = self.len();
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut stack: Vec<TopoFrame> = Vec::with_capacity(n);
        stack.push(TopoFrame::Pending(root));
        while let Some(frame) = stack.pop() {
            match frame {
                // Pushed once per node: any other `Pending(id)` sits below this frame, since one
                // pushed above it would come from a descendant of `id`, i.e. a cycle.
                TopoFrame::Visit(id) => {
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

    /// One value per node of `order` (children first), returning the last node's.
    ///
    /// `visit` reads its children's values from the map. Each value is dropped once its
    /// last parent has run, so only the frontier stays live.
    pub(crate) fn fold_frontier<T>(
        &self,
        order: &[NodeId],
        mut visit: impl FnMut(NodeId, &NodeMap<T>) -> T,
    ) -> T {
        let mut pending_parents = vec![0usize; self.len()];
        for &id in order {
            self.get(id)
                .for_each_child(|child| pending_parents[child.index()] += 1);
        }
        let mut values = NodeMap::new(self.len());
        for &id in order {
            let value = visit(id, &values);
            values.insert(id, value);
            self.get(id).for_each_child(|child| {
                pending_parents[child.index()] -= 1;
                if pending_parents[child.index()] == 0 {
                    values.take(child);
                }
            });
        }
        let root = order.last().expect("a topological order ends at its root");
        values
            .take(*root)
            .expect("the root has no parent to drop it")
    }
}

/// The on-disk form of an [`Arena`], refused unless every child precedes its parent.
#[cfg(feature = "serialize")]
#[derive(serde::Deserialize)]
struct PersistedArena {
    nodes: Vec<Node>,
}

#[cfg(feature = "serialize")]
impl TryFrom<PersistedArena> for Arena {
    type Error = CoreError;

    fn try_from(persisted: PersistedArena) -> Result<Self, CoreError> {
        for (at, node) in persisted.nodes.iter().enumerate() {
            if let Some(child) = first_unallocated_child(node, at) {
                return Err(CoreError::ChildNotBeforeParent {
                    node: at,
                    child: child.index(),
                });
            }
        }
        Ok(Self {
            nodes: persisted.nodes,
        })
    }
}

/// An expression DAG together with its root node.
///
/// Fields are private so a rebuilt arena cannot be paired with a stale root.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PersistedDag")
)]
pub struct Dag {
    arena: Arena,
    root: NodeId,
}

/// The on-disk form of a [`Dag`], refused unless the root is a node of the arena.
#[cfg(feature = "serialize")]
#[derive(serde::Deserialize)]
struct PersistedDag {
    arena: Arena,
    root: NodeId,
}

#[cfg(feature = "serialize")]
impl TryFrom<PersistedDag> for Dag {
    type Error = CoreError;

    fn try_from(persisted: PersistedDag) -> Result<Self, CoreError> {
        let (root, len) = (persisted.root.index(), persisted.arena.len());
        if root >= len {
            return Err(CoreError::RootOutOfRange { root, len });
        }
        Ok(Self::new(persisted.arena, persisted.root))
    }
}

impl Dag {
    /// Pair an arena with a root it issued; panics if `root` is out of range.
    pub const fn new(arena: Arena, root: NodeId) -> Self {
        assert!(
            root.index() < arena.len(),
            "a DAG's root must be a node of its arena"
        );
        Self { arena, root }
    }

    /// The DAG's nodes.
    pub const fn arena(&self) -> &Arena {
        &self.arena
    }

    /// The node this DAG evaluates to.
    pub const fn root(&self) -> NodeId {
        self.root
    }

    /// Append to the DAG and move the root onto the appended node.
    ///
    /// Panics if `rewrite` returns a root outside the arena it leaves behind.
    pub fn rewrite_in_place(&mut self, rewrite: impl FnOnce(&mut Arena, NodeId) -> NodeId) {
        let root = rewrite(&mut self.arena, self.root);
        assert!(
            root.index() < self.arena.len(),
            "a DAG's root must be a node of its arena"
        );
        self.root = root;
    }

    /// Borrow the pair.
    pub const fn as_ref(&self) -> DagRef<'_> {
        DagRef {
            arena: &self.arena,
            root: self.root,
        }
    }
}

/// A borrowed [`Dag`].
#[derive(Clone, Copy, Debug)]
pub struct DagRef<'a> {
    arena: &'a Arena,
    root: NodeId,
}

impl<'a> DagRef<'a> {
    /// Pair a borrowed arena with a root it issued; panics if `root` is out of range.
    pub const fn new(arena: &'a Arena, root: NodeId) -> Self {
        assert!(
            root.index() < arena.len(),
            "a DAG's root must be a node of its arena"
        );
        Self { arena, root }
    }

    /// The DAG's nodes.
    pub const fn arena(self) -> &'a Arena {
        self.arena
    }

    /// The node this DAG evaluates to.
    pub const fn root(self) -> NodeId {
        self.root
    }
}

/// How far into `y`, `y_dot` and the packed inputs the sub-DAG at a root reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadExtents {
    /// One past the highest element of `y` read.
    pub y: usize,
    /// One past the highest element of `y_dot` read, or `None` if no `StateVectorDot` is reached.
    pub y_dot: Option<usize>,
    /// One past the highest packed input element read.
    pub inputs: usize,
}

impl ReadExtents {
    /// Scan the nodes `root` reaches; tangent leaves read no input and are not counted.
    pub fn scan(arena: &Arena, root: NodeId) -> Self {
        // Children precede parents, so one descending sweep marks everything `root` reaches.
        let mut reached = vec![false; root.index() + 1];
        reached[root.index()] = true;
        let mut extents = Self::default();
        for (at, node) in arena.nodes()[..=root.index()].iter().enumerate().rev() {
            if !reached[at] {
                continue;
            }
            match node {
                Node::StateVector { end, .. } => extents.y = extents.y.max(*end),
                Node::StateVectorDot { end, .. } => {
                    extents.y_dot = Some(extents.y_dot.unwrap_or(0).max(*end));
                },
                Node::InputParameter { offset, width, .. } => {
                    extents.inputs = extents.inputs.max(offset.saturating_add(*width));
                },
                _ => {},
            }
            node.for_each_child(|child| reached[child.index()] = true);
        }
        extents
    }

    /// The state count both `y` and `y_dot` must cover.
    pub fn n_states(self) -> usize {
        self.y.max(self.y_dot.unwrap_or(0))
    }
}

/// Reconcile a declared state count with the extent an expression reads.
///
/// Declaring fewer than `extent` is an error (it would panic inside evaluation), as is
/// declaring more than the tape's `u32` state indices reach; `None` adopts the extent.
/// Per-state tables are sized by the count, so one near `u32::MAX` can still exhaust memory.
pub const fn resolve_state_count(
    declared: Option<usize>,
    extent: usize,
) -> Result<usize, CoreError> {
    match declared {
        Some(n) if n < extent => Err(CoreError::StateCountTooSmall {
            declared: n,
            extent,
        }),
        Some(n) if n > u32::MAX as usize => Err(CoreError::StateCountTooLarge {
            declared: n,
            max: u32::MAX,
        }),
        Some(n) => Ok(n),
        None => Ok(extent),
    }
}

enum TopoFrame {
    Pending(NodeId),
    /// Children are all emitted; emit this node now.
    Visit(NodeId),
}

impl Index<NodeId> for Arena {
    type Output = Node;

    #[inline]
    fn index(&self, id: NodeId) -> &Node {
        self.get(id)
    }
}

/// Dense per-node side table, keyed by [`NodeId`].
#[derive(Clone, Debug)]
pub(crate) struct NodeMap<T> {
    slots: Vec<Option<T>>,
}

impl<T> NodeMap<T> {
    /// A table sized for an arena of `arena_len` nodes, all entries empty.
    #[inline]
    pub(crate) fn new(arena_len: usize) -> Self {
        let mut slots = Vec::with_capacity(arena_len);
        slots.resize_with(arena_len, || None);
        Self { slots }
    }

    /// The value for `id`, or `None` if unset or out of range.
    #[inline]
    pub(crate) fn get(&self, id: NodeId) -> Option<&T> {
        self.slots.get(id.index()).and_then(|opt| opt.as_ref())
    }

    /// Set the value for `id`, growing the table if needed; returns the old value.
    #[inline]
    pub(crate) fn insert(&mut self, id: NodeId, value: T) -> Option<T> {
        if id.index() >= self.slots.len() {
            self.slots.resize_with(id.index() + 1, || None);
        }
        self.slots[id.index()].replace(value)
    }

    /// Whether `id` has a value set.
    #[inline]
    pub(crate) fn contains_key(&self, id: NodeId) -> bool {
        self.slots.get(id.index()).is_some_and(Option::is_some)
    }

    /// Remove and return the value for `id`, if set.
    #[inline]
    pub(crate) fn take(&mut self, id: NodeId) -> Option<T> {
        self.slots.get_mut(id.index()).and_then(Option::take)
    }
}

impl NodeMap<NodeId> {
    /// `id`'s counterpart in a pass's output, given children are rebuilt before parents.
    #[inline]
    pub(crate) fn rebuilt(&self, id: NodeId) -> NodeId {
        *self
            .get(id)
            .expect("children are rebuilt before their parents")
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
    fn read_extents_cover_only_what_the_root_reaches() {
        let mut arena = Arena::new();
        let wide = arena.alloc(Node::StateVector { start: 0, end: 9 });
        let sv = arena.alloc(Node::StateVector { start: 2, end: 5 });
        let svd = arena.alloc(Node::StateVectorDot { start: 0, end: 1 });
        let p = arena.alloc(Node::InputParameter {
            name: "p".to_string(),
            index: 0,
            offset: 1,
            width: 2,
        });
        let product = arena.alloc(Node::Mul(svd, p));
        let sum = arena.alloc(Node::Add(sv, product));
        let _unreached = arena.alloc(Node::Neg(wide));
        let extents = ReadExtents::scan(&arena, sum);
        assert_eq!(
            extents,
            ReadExtents {
                y: 5,
                y_dot: Some(1),
                inputs: 3
            }
        );
        assert_eq!(extents.n_states(), 5);

        let empty_dot = arena.alloc(Node::StateVectorDot { start: 0, end: 0 });
        let extents = ReadExtents::scan(&arena, empty_dot);
        assert_eq!(
            extents.y_dot,
            Some(0),
            "an empty y_dot read still reads y_dot"
        );
        assert_eq!(extents.n_states(), 0);

        let lone = arena.alloc(Node::Scalar(1.0));
        assert_eq!(ReadExtents::scan(&arena, lone), ReadExtents::default());
    }

    #[test]
    fn a_state_count_past_the_tapes_u32_indices_is_refused() {
        let max = u32::MAX as usize;
        assert_eq!(resolve_state_count(Some(max), 1).ok(), Some(max));
        // Unrefused, a Jacobian sized by this count aborts the process on allocation.
        let too_large = resolve_state_count(Some(1 << 45), 1);
        assert!(matches!(
            too_large,
            Err(CoreError::StateCountTooLarge { declared, max: u32::MAX }) if declared == 1 << 45
        ));
        assert!(resolve_state_count(Some(max + 1), 1).is_err());
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

    #[test]
    #[should_panic(expected = "node 1 names child 1, which is not allocated before it")]
    fn alloc_refuses_a_child_not_yet_allocated() {
        let mut arena = Arena::new();
        let _ = arena.alloc(Node::Scalar(1.0));
        let _ = arena.alloc(Node::Neg(NodeId::from(1u32)));
    }

    #[test]
    #[should_panic(expected = "a DAG's root must be a node of its arena")]
    fn a_rewrite_cannot_leave_the_root_outside_its_arena() {
        let mut arena = Arena::new();
        let root = arena.alloc(Node::Time);
        let mut dag = Dag::new(arena, root);
        dag.rewrite_in_place(|arena, root| {
            *arena = Arena::new();
            root
        });
    }

    #[test]
    #[should_panic(expected = "a DAG's root must be a node of its arena")]
    fn a_dag_refuses_a_root_outside_its_arena() {
        let mut arena = Arena::new();
        let _ = arena.alloc(Node::Scalar(1.0));
        let _ = DagRef::new(&arena, NodeId::from(1u32));
    }
}

#[cfg(all(test, feature = "serialize"))]
mod persisted_tests {
    use super::*;

    #[test]
    fn an_arena_persists_as_its_bare_node_list() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::Time);
        let root = arena.alloc(Node::Neg(x));
        let bytes = bincode::serialize(&Dag::new(arena.clone(), root)).expect("serialise");
        let bare = bincode::serialize(&(arena.nodes(), root.raw())).expect("serialise");
        assert_eq!(bytes, bare);

        let back: Dag = bincode::deserialize(&bytes).expect("deserialise");
        assert_eq!(back.arena().nodes(), arena.nodes());
        assert_eq!(back.root(), root);
    }

    #[test]
    fn a_persisted_child_at_or_after_its_parent_is_refused() {
        let ahead = vec![Node::Neg(NodeId::from(1u32)), Node::Time];
        let looped = vec![Node::Time, Node::Neg(NodeId::from(1u32))];
        for nodes in [ahead, looped] {
            let bytes = bincode::serialize(&nodes).expect("serialise");
            let err = bincode::deserialize::<Arena>(&bytes).expect_err("must be refused");
            assert!(err.to_string().contains("which is not allocated before it"));
        }
    }

    #[test]
    fn a_persisted_root_outside_its_arena_is_refused() {
        let bytes = bincode::serialize(&(vec![Node::Time], 1u32)).expect("serialise");
        let err = bincode::deserialize::<Dag>(&bytes).expect_err("must be refused");
        assert!(err.to_string().contains("root 1 is out of range"));
    }
}

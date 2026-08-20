use pybamm_core::{Arena, Node, NodeId, cse};

#[test]
fn cse_canonicalizes_structurally_equal_nodes() {
    let mut arena = Arena::new();
    let s1 = arena.alloc(Node::Scalar(2.0));
    let s2 = arena.alloc(Node::Scalar(2.0));
    let y1 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let y2 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let m1 = arena.alloc(Node::Mul(s1, y1));
    let m2 = arena.alloc(Node::Mul(s2, y2));
    let root = arena.alloc(Node::Add(m1, m2));

    let (cse_arena, _root) = cse(&arena, root);
    let n_muls = (0..cse_arena.len())
        .filter(|&i| matches!(cse_arena.get(NodeId::from(i)), Node::Mul(_, _)))
        .count();
    assert_eq!(n_muls, 1, "structurally identical Muls must canonicalize");
}

#[test]
fn cse_distinguishes_state_vectors_with_different_ranges() {
    let mut arena = Arena::new();
    let y1 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let y2 = arena.alloc(Node::StateVector { start: 1, end: 2 });
    let root = arena.alloc(Node::Add(y1, y2));

    let (cse_arena, _root) = cse(&arena, root);
    let n_states = (0..cse_arena.len())
        .filter(|&i| matches!(cse_arena.get(NodeId::from(i)), Node::StateVector { .. }))
        .count();
    assert_eq!(
        n_states, 2,
        "StateVectors with different ranges must remain distinct"
    );
}

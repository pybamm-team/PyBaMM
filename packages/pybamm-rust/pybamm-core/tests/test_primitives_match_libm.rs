//! Every primitive evaluates bitwise as the `libm` crate does, on every evaluation path,
//! so a model takes the same values on every target rather than following the platform libm.
//! Where IEEE 754 specifies the result exactly (`sqrt`, `floor`, `ceil`, `abs`, `%`) the
//! tapes use `f64` and the references `libm`, so agreement checks both against that result.
//!
//! IEEE 754 leaves NaN payloads unspecified, so a NaN need only meet a NaN; every other
//! result, signed zeros and infinities included, must match bit for bit.

use std::f64::consts::{FRAC_2_SQRT_PI, FRAC_PI_2, PI};

use pybamm_core::adjoint::AdjointTape;
use pybamm_core::ir::{BinaryOp, UnaryOp};
use pybamm_core::{Arena, CompiledExpr, DagRef, Node, NodeId, simplify};

type UnaryNode = fn(NodeId) -> Node;
type BinaryNode = fn(NodeId, NodeId) -> Node;
type UnaryFn = fn(f64) -> f64;
type BinaryFn = fn(f64, f64) -> f64;
type Partials = fn(f64, f64) -> [f64; 2];

const UNARY_OPS: [UnaryOp; 16] = [
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Sqrt,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Tanh,
    UnaryOp::Sinh,
    UnaryOp::Cosh,
    UnaryOp::Arcsinh,
    UnaryOp::Arctan,
    UnaryOp::Erf,
    UnaryOp::Sign,
    UnaryOp::Floor,
    UnaryOp::Ceiling,
];

const BINARY_OPS: [BinaryOp; 12] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Pow,
    BinaryOp::Minimum,
    BinaryOp::Maximum,
    BinaryOp::Modulo,
    BinaryOp::Hypot,
    BinaryOp::EqualHeaviside,
    BinaryOp::NotEqualHeaviside,
    BinaryOp::Equality,
];

/// The node that lowers to `op`, and the value it must take.
///
/// Exhaustive, so a new primitive does not compile until it has a reference here.
fn unary(op: UnaryOp) -> (UnaryNode, UnaryFn) {
    match op {
        UnaryOp::Neg => (Node::Neg, |x| -x),
        UnaryOp::Abs => (Node::Abs, libm::fabs),
        UnaryOp::Sqrt => (Node::Sqrt, libm::sqrt),
        UnaryOp::Exp => (Node::Exp, libm::exp),
        UnaryOp::Log => (Node::Log, libm::log),
        UnaryOp::Sin => (Node::Sin, libm::sin),
        UnaryOp::Cos => (Node::Cos, libm::cos),
        UnaryOp::Tanh => (Node::Tanh, libm::tanh),
        UnaryOp::Sinh => (Node::Sinh, libm::sinh),
        UnaryOp::Cosh => (Node::Cosh, libm::cosh),
        UnaryOp::Arcsinh => (Node::Arcsinh, libm::asinh),
        UnaryOp::Arctan => (Node::Arctan, libm::atan),
        UnaryOp::Erf => (Node::Erf, libm::erf),
        // `sign(±0) = +0`, unlike `copysign(1, x)`.
        UnaryOp::Sign => (Node::Sign, |x| {
            if x.is_nan() {
                x
            } else if x == 0.0 {
                0.0
            } else {
                libm::copysign(1.0, x)
            }
        }),
        UnaryOp::Floor => (Node::Floor, libm::floor),
        UnaryOp::Ceiling => (Node::Ceiling, libm::ceil),
    }
}

/// The node that lowers to `op`, and the value it must take; the arithmetic is IEEE's own.
///
/// Exhaustive, so a new primitive does not compile until it has a reference here.
fn binary(op: BinaryOp) -> (BinaryNode, BinaryFn) {
    match op {
        BinaryOp::Add => (Node::Add, |x, y| x + y),
        BinaryOp::Sub => (Node::Sub, |x, y| x - y),
        BinaryOp::Mul => (Node::Mul, |x, y| x * y),
        BinaryOp::Div => (Node::Div, |x, y| x / y),
        BinaryOp::Pow => (Node::Pow, libm::pow),
        BinaryOp::Minimum => (Node::Minimum, libm::fminimum_num),
        BinaryOp::Maximum => (Node::Maximum, libm::fmaximum_num),
        BinaryOp::Modulo => (Node::Modulo, libm::fmod),
        BinaryOp::Hypot => (Node::Hypot, libm::hypot),
        BinaryOp::EqualHeaviside => (Node::EqualHeaviside, |x, y| f64::from(u8::from(x <= y))),
        BinaryOp::NotEqualHeaviside => (Node::NotEqualHeaviside, |x, y| f64::from(u8::from(x < y))),
        BinaryOp::Equality => (Node::Equality, |x, y| {
            f64::from(u8::from(libm::fabs(x - y) < 1e-14))
        }),
    }
}

/// Bitwise equal, or both NaN; with `any_zero_sign`, also `+0` against `-0`.
fn same(got: f64, want: f64, any_zero_sign: bool) -> bool {
    got.to_bits() == want.to_bits()
        || (got.is_nan() && want.is_nan())
        || (any_zero_sign && got == 0.0 && want == 0.0)
}

#[track_caller]
fn assert_matches(label: &str, args: &[f64], got: f64, want: f64) {
    assert!(
        same(got, want, false),
        "{label}{args:?} = {got:e} ({:#018x}), libm gives {want:e} ({:#018x})",
        got.to_bits(),
        want.to_bits(),
    );
}

/// xorshift64*, so the sweep is reproducible without a rand dependency.
struct XorShift(u64);

impl XorShift {
    const fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Any bit pattern, so every exponent appears, infinities and NaNs among them.
    const fn any_bits(&mut self) -> f64 {
        f64::from_bits(self.next_u64())
    }

    /// A magnitude log-uniform over `1e-12..1e12`, either sign: the range model values span.
    fn log_uniform(&mut self) -> f64 {
        let sign = if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        sign * libm::pow(10.0, 24.0 * self.unit() - 12.0)
    }

    /// Uniform over `-50..50`, where the functions do most of their switching.
    fn moderate(&mut self) -> f64 {
        100.0 * self.unit() - 50.0
    }

    /// An integer or half-integer in `-10..=10`, where `pow` takes its special paths.
    fn exponent(&mut self) -> f64 {
        (self.next_u64() % 41) as f64 / 2.0 - 10.0
    }
}

/// Signed zeros, subnormals, the extremes, infinities and NaN, plus the points where the
/// functions switch branches, overflow or saturate.
fn special_values() -> Vec<f64> {
    let magnitudes = [
        0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1e-300,
        1e-8,
        0.1,
        0.5,
        0.84375,
        1.0 - f64::EPSILON / 2.0,
        1.0,
        1.0 + f64::EPSILON,
        1.25,
        1.5,
        2.0,
        1.0 / 0.35,
        3.0,
        FRAC_PI_2,
        PI,
        6.0,
        19.0,
        22.0,
        709.782_712_893_384,
        710.475_860_073_943_9,
        745.133_219_101_941_1,
        1e22,
        1e300,
        f64::MAX,
        f64::INFINITY,
    ];
    magnitudes
        .iter()
        .flat_map(|&v| [v, -v])
        .chain([f64::NAN])
        .collect()
}

const SWEEP: usize = 1 << 14;

fn unary_inputs() -> Vec<f64> {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut xs = special_values();
    for _ in 0..SWEEP {
        xs.extend([rng.any_bits(), rng.log_uniform(), rng.moderate()]);
    }
    xs
}

fn binary_inputs() -> (Vec<f64>, Vec<f64>) {
    let special = special_values();
    let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = special
        .iter()
        .flat_map(|&x| special.iter().map(move |&y| (x, y)))
        .unzip();
    let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
    for _ in 0..SWEEP {
        let pairs = [
            (rng.any_bits(), rng.any_bits()),
            (rng.log_uniform(), rng.log_uniform()),
            (rng.moderate(), rng.moderate()),
            (rng.log_uniform(), rng.exponent()),
            (rng.moderate(), rng.exponent()),
        ];
        for (x, y) in pairs {
            xs.push(x);
            ys.push(y);
        }
    }
    (xs, ys)
}

fn state(arena: &mut Arena, start: usize, end: usize) -> NodeId {
    arena.alloc(Node::StateVector { start, end })
}

fn eval(arena: &Arena, root: NodeId, y: &[f64]) -> Vec<f64> {
    let expr = CompiledExpr::new(DagRef::new(arena, root));
    let mut scratch = vec![0.0; expr.scratch_len()];
    expr.eval(&mut scratch, 0.0, y, &[], &[]).to_vec()
}

/// Two lanes, the second holding `lane_two`; returns each lane's root values.
fn eval_two_lanes(arena: &Arena, root: NodeId, y: &[f64], lane_two: &[f64]) -> [Vec<f64>; 2] {
    let expr = CompiledExpr::new(DagRef::new(arena, root));
    let mut scratch = vec![0.0; expr.scratch_len() * 2];
    let y_cols = [y, lane_two].concat();
    let out = expr
        .eval_batch(&mut scratch, 2, &[0.0, 0.0], &y_cols, y.len(), &[])
        .expect("a primal tape batches");
    // Lane-minor: element `e` of lane `l` sits at `2 * e + l`.
    [0, 1].map(|lane| out.iter().skip(lane).step_by(2).copied().collect())
}

/// The literal `simplify` folds `root` to, if it folds.
fn folded(arena: &mut Arena, root: NodeId) -> Option<f64> {
    let simplified = simplify(arena, root);
    match *arena.get(simplified) {
        Node::Scalar(v) => Some(v),
        _ => None,
    }
}

#[test]
fn unary_primitives_evaluate_bitwise_as_libm() {
    let xs = unary_inputs();
    let negated: Vec<f64> = xs.iter().map(|x| -x).collect();
    for op in UNARY_OPS {
        let (node, reference) = unary(op);
        let label = format!("{op:?}");
        let mut arena = Arena::new();
        let y = state(&mut arena, 0, xs.len());
        let root = arena.alloc(node(y));

        for (&x, got) in xs.iter().zip(eval(&arena, root, &xs)) {
            assert_matches(&label, &[x], got, reference(x));
        }
        let lanes = eval_two_lanes(&arena, root, &xs, &negated);
        for (lane, inputs) in lanes.iter().zip([&xs, &negated]) {
            for (&x, &got) in inputs.iter().zip(lane) {
                assert_matches(&format!("{label} (batched)"), &[x], got, reference(x));
            }
        }
        for x in special_values() {
            let mut arena = Arena::new();
            let literal = arena.alloc(Node::Scalar(x));
            let root = arena.alloc(node(literal));
            if let Some(got) = folded(&mut arena, root) {
                assert_matches(&format!("{label} (folded)"), &[x], got, reference(x));
            }
        }
    }
}

#[test]
fn binary_primitives_evaluate_bitwise_as_libm() {
    let (xs, ys) = binary_inputs();
    let n = xs.len();
    let (forward, swapped) = ([xs.as_slice(), &ys].concat(), [ys.as_slice(), &xs].concat());
    for op in BINARY_OPS {
        let (node, reference) = binary(op);
        let label = format!("{op:?}");
        let mut arena = Arena::new();
        let (lhs, rhs) = (state(&mut arena, 0, n), state(&mut arena, n, 2 * n));
        let root = arena.alloc(node(lhs, rhs));

        for ((&x, &y), got) in xs.iter().zip(&ys).zip(eval(&arena, root, &forward)) {
            assert_matches(&label, &[x, y], got, reference(x, y));
        }
        let [lane_one, lane_two] = eval_two_lanes(&arena, root, &forward, &swapped);
        for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
            let batched = format!("{label} (batched)");
            assert_matches(&batched, &[x, y], lane_one[i], reference(x, y));
            assert_matches(&batched, &[y, x], lane_two[i], reference(y, x));
        }
    }
}

/// A one-element operand against a vector, either side, as the broadcast kernels run it.
#[test]
fn broadcast_binary_primitives_evaluate_bitwise_as_libm() {
    let vector: Vec<f64> = unary_inputs().into_iter().step_by(24).collect();
    let n = vector.len();
    for op in BINARY_OPS {
        let (node, reference) = binary(op);
        for s in special_values() {
            let y = [&[s], vector.as_slice()].concat();
            let mut arena = Arena::new();
            let (scalar, wide) = (state(&mut arena, 0, 1), state(&mut arena, 1, n + 1));
            let scalar_left = arena.alloc(node(scalar, wide));
            let scalar_right = arena.alloc(node(wide, scalar));
            let left = eval(&arena, scalar_left, &y);
            let right = eval(&arena, scalar_right, &y);
            for (i, &v) in vector.iter().enumerate() {
                assert_matches(
                    &format!("{op:?} (scalar left)"),
                    &[s, v],
                    left[i],
                    reference(s, v),
                );
                assert_matches(
                    &format!("{op:?} (scalar right)"),
                    &[v, s],
                    right[i],
                    reference(v, s),
                );
            }
        }
    }
}

#[test]
fn constant_folding_takes_libm_values() {
    let special = special_values();
    for op in BINARY_OPS {
        let (node, reference) = binary(op);
        for &x in &special {
            for &y in &special {
                let mut arena = Arena::new();
                let (lhs, rhs) = (arena.alloc(Node::Scalar(x)), arena.alloc(Node::Scalar(y)));
                let root = arena.alloc(node(lhs, rhs));
                if let Some(got) = folded(&mut arena, root) {
                    assert_matches(&format!("{op:?} (folded)"), &[x, y], got, reference(x, y));
                }
            }
        }
    }
}

/// Reverse-mode partials of `root` (reading `y`) at `y`, one per state.
fn adjoint_partials(arena: &Arena, root: NodeId, y: &[f64]) -> Vec<f64> {
    let tape = AdjointTape::new(DagRef::new(arena, root), y.len());
    let (mut scratch, mut bar) = (vec![0.0; tape.scratch_len()], vec![0.0; tape.scratch_len()]);
    let mut grad = vec![0.0; y.len()];
    tape.assemble(&mut scratch, &mut bar, &mut grad, 0.0, y, &[], &[]);
    grad
}

/// The reverse-mode rules take their transcendental values from libm too. Partials land
/// by accumulation into zero, so a `-0` partial reads back as `+0`.
#[test]
fn adjoint_partials_take_libm_values() {
    let unary_rules: [(UnaryNode, &str, UnaryFn); 6] = [
        (Node::Sin, "sin", libm::cos),
        (Node::Cos, "cos", |x| -libm::sin(x)),
        (Node::Sinh, "sinh", libm::cosh),
        (Node::Cosh, "cosh", libm::sinh),
        (Node::Arcsinh, "arcsinh", |x| 1.0 / libm::sqrt(1.0 + x * x)),
        (Node::Erf, "erf", |x| FRAC_2_SQRT_PI * libm::exp(-x * x)),
    ];
    let xs: Vec<f64> = unary_inputs().into_iter().step_by(48).collect();
    for (node, label, rule) in unary_rules {
        let mut arena = Arena::new();
        let y = state(&mut arena, 0, 1);
        let root = arena.alloc(node(y));
        for &x in &xs {
            let got = adjoint_partials(&arena, root, &[x])[0];
            let want = rule(x);
            assert!(
                same(got, want, true),
                "d{label}({x:e}) = {got:e}, libm gives {want:e}"
            );
        }
    }

    let binary_rules: [(BinaryNode, &str, Partials); 2] = [
        (Node::Pow, "pow", |x, y| {
            let by_exponent = if x > 0.0 {
                libm::pow(x, y) * libm::log(x)
            } else {
                0.0
            };
            [y * libm::pow(x, y - 1.0), by_exponent]
        }),
        (Node::Hypot, "hypot", |x, y| {
            [x / libm::hypot(x, y), y / libm::hypot(x, y)]
        }),
    ];
    let (xs, ys) = binary_inputs();
    for (node, label, rule) in binary_rules {
        let mut arena = Arena::new();
        let (lhs, rhs) = (state(&mut arena, 0, 1), state(&mut arena, 1, 2));
        let root = arena.alloc(node(lhs, rhs));
        for (&x, &y) in xs.iter().zip(&ys).step_by(48) {
            let got = adjoint_partials(&arena, root, &[x, y]);
            let want = rule(x, y);
            assert!(
                got.iter().zip(want).all(|(&g, w)| same(g, w, true)),
                "d{label}({x:e}, {y:e}) = {got:?}, libm gives {want:?}"
            );
        }
    }
}

/// FNV-1a over every reference value the sweeps above produce, NaNs canonicalised.
const LIBM_FINGERPRINT: u64 = 0x17b6_6c10_a1a2_4284;

/// The tests above hold the tapes to whatever `libm` computes, so a `libm` that moved bits
/// would move every model's values unnoticed. Pinning its outputs makes such a bump fail
/// here, and running this on each CI target shows the values agree across targets.
#[test]
fn libm_values_are_pinned() {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    let mut absorb = |v: f64| {
        let bits = if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        };
        for byte in bits.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    let xs = unary_inputs();
    for op in UNARY_OPS {
        let (_, reference) = unary(op);
        for &x in &xs {
            absorb(reference(x));
        }
    }
    let (xs, ys) = binary_inputs();
    for op in BINARY_OPS {
        let (_, reference) = binary(op);
        for (&x, &y) in xs.iter().zip(&ys) {
            absorb(reference(x, y));
        }
    }
    assert_eq!(
        hash, LIBM_FINGERPRINT,
        "libm's values moved, and every model's values with them; if that is intended, pin \
         {hash:#018x}"
    );
}

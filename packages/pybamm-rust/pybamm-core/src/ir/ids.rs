//! Typed `u32` indices for the instruction tape and its side tables, so buffer
//! offsets, element counts and pool indices cannot be swapped by accident.

/// Define a `u32` index over the instruction tape, with `usize` conversions.
macro_rules! tape_index {
    ($(#[$attr:meta])* $name:ident) => {
        $crate::newtype::index_newtype! { $(#[$attr])* $name(u32) }

        impl $name {
            /// Widened for indexing.
            pub const fn index(self) -> usize {
                self.get() as usize
            }

            /// Narrow a `usize` count or offset; panics past `u32::MAX` rather than truncating.
            pub fn from_usize(value: usize) -> Self {
                Self::new(
                    u32::try_from(value)
                        .expect(concat!(stringify!($name), " exceeded u32::MAX")),
                )
            }
        }
    };
}

pub(crate) use tape_index;

/// Define a tape index that also indexes a constant-pool table.
macro_rules! pool_index {
    ($(#[$attr:meta])* $name:ident) => {
        tape_index! { $(#[$attr])* $name }

        impl $name {
            /// The index the next `push` onto `table` will occupy.
            pub(crate) fn of_next<T>(table: &[T]) -> Self {
                Self::from_usize(table.len())
            }
        }
    };
}

tape_index! {
    /// Element offset of a [`Slot`](super::Slot)'s start in the evaluation buffer.
    BufSlot
}

tape_index! {
    /// A count of `f64` elements: a slot's length, or an input's packed width.
    Width
}

tape_index! {
    /// Element offset relative to the start of a slot, not to the buffer.
    SlotOffset
}

impl BufSlot {
    /// The buffer index `within` elements into this slot.
    #[must_use]
    pub(crate) const fn offset_by(self, within: SlotOffset) -> usize {
        self.index() + within.index()
    }
}

tape_index! {
    /// A count of consecutive entries in one constant-pool side table.
    EntryCount
}

tape_index! {
    /// A count of consecutive instructions on the tape.
    InstructionCount
}

tape_index! {
    /// Row count of a dense matrix operand.
    Rows
}

tape_index! {
    /// Column count of a dense matrix operand.
    Cols
}

impl Rows {
    /// Element count of a column with this many rows.
    pub(crate) const fn to_width(self) -> Width {
        Width::new(self.get())
    }

    /// Element count of a dense `rows x cols` matrix.
    pub(crate) fn times(self, cols: Cols) -> Width {
        Width::from_usize(self.index() * cols.index())
    }
}

impl Cols {
    /// Element count of a row with this many columns.
    pub(crate) const fn to_width(self) -> Width {
        Width::new(self.get())
    }
}

tape_index! {
    /// Axis of an N-D interpolant, in `0..ndim`.
    Axis
}

tape_index! {
    /// Index into the state vector `y` (or its derivative `y'`).
    StateIdx
}

tape_index! {
    /// Element offset into the packed input vector, not a
    /// [`ParamIdx`](crate::params::ParamIdx) registration index.
    InputOffset
}

pool_index! {
    /// Index into [`ConstPool::array_offsets`](super::ConstPool::array_offsets).
    ArrayIdx
}

pool_index! {
    /// Index into [`ConstPool::csr_data`](super::ConstPool::csr_data).
    CsrIdx
}

pool_index! {
    /// Index into one of the [`Slot`](super::Slot) side tables; the instruction fixes which.
    SlotTableIdx
}

pool_index! {
    /// Index into
    /// [`ConstPool::branch_blocks`](super::ConstPool::branch_blocks).
    BlockSpanIdx
}

pool_index! {
    /// Index into [`ConstPool::interpolants`](super::ConstPool::interpolants).
    InterpolantIdx
}

pool_index! {
    /// Index into
    /// [`ConstPool::cubic_interpolants`](super::ConstPool::cubic_interpolants).
    CubicInterpolantIdx
}

pool_index! {
    /// Index into
    /// [`ConstPool::nd_interpolants`](super::ConstPool::nd_interpolants).
    NdInterpolantIdx
}

/// A contiguous run of instructions one `Dispatch` branch owns.
///
/// Offsets are relative to the owning `Dispatch`, since split tapes execute as sub-slices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockSpan {
    rel_start: u32,
    len: u32,
}

impl BlockSpan {
    /// A run of `len` instructions starting `rel_start` past its `Dispatch`.
    pub(crate) const fn new(rel_start: u32, len: u32) -> Self {
        Self { rel_start, len }
    }

    /// A span from the `usize` instruction positions lowering counts in.
    pub(crate) fn from_usize(rel_start: usize, len: usize) -> Self {
        Self {
            rel_start: u32::try_from(rel_start).expect("branch block offset exceeded u32::MAX"),
            len: u32::try_from(len).expect("branch block length exceeded u32::MAX"),
        }
    }

    /// Instruction count; `0` when this branch owns none here.
    pub const fn len(self) -> usize {
        self.len as usize
    }

    /// Whether this branch owns no instructions here.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Absolute instruction bounds, given the `Dispatch`'s index in the executing tape.
    pub const fn bounds(self, dispatch_pc: usize) -> std::ops::Range<usize> {
        let start = dispatch_pc + self.rel_start as usize;
        start..start + self.len as usize
    }
}

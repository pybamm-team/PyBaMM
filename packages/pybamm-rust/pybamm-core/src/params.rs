//! The two index spaces a model's input parameters live in.
//!
//! A parameter's *registration index* and its *packed offset* into `inputs`
//! diverge once any input is vector-valued, so each is its own type.

use std::collections::HashMap;

use crate::arena::Arena;
use crate::error::CoreError;
use crate::ir::ids::{InputOffset, Width, tape_index};
use crate::newtype::index_newtype;
use crate::node::Node;

tape_index! {
    /// Registration index of an input parameter; the space `dp` is indexed in.
    ParamIdx
}

index_newtype! {
    /// Length of the packed `inputs` slice: the sum of every parameter's width.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
    PackedWidth(usize)
}

/// Where one registered input parameter's values sit in the packed vector.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParamSlot {
    /// Element offset into the packed `inputs` slice.
    pub offset: InputOffset,
    /// How many packed elements the parameter occupies; `> 1` for a vector input.
    pub width: Width,
}

impl ParamSlot {
    /// The slot `[offset, offset + width)` of `name`, refused if it ends past `u32::MAX`.
    fn try_new(name: &str, offset: usize, width: usize) -> Result<Self, CoreError> {
        if offset
            .checked_add(width)
            .is_none_or(|end| u32::try_from(end).is_err())
        {
            return Err(CoreError::ParamSlotRange {
                name: name.to_string(),
                offset,
                width,
            });
        }
        Ok(Self {
            offset: InputOffset::from_usize(offset),
            width: Width::from_usize(width),
        })
    }
}

/// The one place a parameter's registration index and packed slot are decided.
///
/// Not owned by [`Arena`], since rewriters build fresh arenas and would drop it;
/// [`from_arena`](Self::from_arena) recovers it from the nodes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParamRegistry {
    by_name: HashMap<String, ParamIdx>,
    /// Indexed by [`ParamIdx`]; `None` at an index naming no parameter.
    entries: Vec<Option<ParamEntry>>,
    packed_width: PackedWidth,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct ParamEntry {
    name: String,
    slot: ParamSlot,
}

impl Default for ParamRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamRegistry {
    pub fn new() -> Self {
        Self {
            by_name: HashMap::new(),
            entries: Vec::new(),
            packed_width: PackedWidth::new(0),
        }
    }

    /// Register `name` at the end of the packed vector, or look it up if already registered.
    ///
    /// A repeat must repeat the width, since later offsets are running totals of widths.
    pub fn register(
        &mut self,
        name: &str,
        width: usize,
    ) -> Result<(ParamIdx, ParamSlot), CoreError> {
        if let Some(&index) = self.by_name.get(name) {
            let slot = self
                .slot(index)
                .expect("a name in the registry indexes a registered parameter");
            if slot.width.index() != width {
                return Err(CoreError::ParamWidthConflict {
                    name: name.to_string(),
                    width,
                    registered: slot.width.index(),
                });
            }
            return Ok((index, slot));
        }
        let slot = ParamSlot::try_new(name, self.packed_width.get(), width)?;
        let index = ParamIdx::from_usize(self.entries.len());
        self.insert(index, name, slot);
        self.packed_width = PackedWidth::new(self.packed_width.get() + width);
        Ok((index, slot))
    }

    /// Recover the registry an arena's nodes describe.
    ///
    /// Whole-arena, so a parameter only an output or event reads stays registered.
    /// Conflicting or overlapping slots are refused; holes below the arena's length are
    /// tolerated, and an index at or past it is refused before anything is allocated.
    pub fn from_arena(arena: &Arena) -> Result<Self, CoreError> {
        // Each registration allocates a node, so a wider index can only come from forged bytes.
        let in_range = |index: usize| {
            if index < arena.len() {
                Ok(index)
            } else {
                Err(CoreError::ParamIndexRange {
                    index,
                    n_nodes: arena.len(),
                })
            }
        };
        let mut registry = Self::new();
        for node in arena.nodes() {
            match node {
                Node::InputParameter {
                    name,
                    index,
                    offset,
                    width,
                } => registry.absorb(name, in_range(*index)?, *offset, *width)?,
                Node::TangentParameter { index } => registry.reserve(in_range(*index)?),
                _ => {},
            }
        }
        registry.packed_width = registry.checked_packed_extent()?;
        Ok(registry)
    }

    /// The registration index `name` holds, or `None` if it holds none.
    pub fn index_of(&self, name: &str) -> Option<ParamIdx> {
        self.by_name.get(name).copied()
    }

    /// Where parameter `idx` sits in the packed vector, if registered.
    pub fn slot(&self, idx: ParamIdx) -> Option<ParamSlot> {
        self.entry(idx).map(|entry| entry.slot)
    }

    /// Every registered parameter in registration order, as `(index, name, slot)`.
    ///
    /// Unslotted indices are skipped; size tables by [`index_extent`](Self::index_extent).
    pub fn registered(&self) -> impl Iterator<Item = (ParamIdx, &str, ParamSlot)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                entry
                    .as_ref()
                    .map(|entry| (ParamIdx::from_usize(index), entry.name.as_str(), entry.slot))
            })
    }

    /// Registered names in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.registered().map(|(_, name, _)| name).collect()
    }

    /// One past the highest registration index; not the registered count.
    pub const fn index_extent(&self) -> usize {
        self.entries.len()
    }

    /// How many indices actually carry a packed slot.
    pub fn n_registered(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    /// Length of the packed `inputs` slice these parameters need.
    pub const fn packed_width(&self) -> PackedWidth {
        self.packed_width
    }

    fn entry(&self, idx: ParamIdx) -> Option<&ParamEntry> {
        self.entries.get(idx.index())?.as_ref()
    }

    /// Widen the index space so `index` is addressable, naming no parameter.
    fn reserve(&mut self, index: usize) {
        if self.entries.len() <= index {
            self.entries.resize(index + 1, None);
        }
    }

    /// Take in one `InputParameter` node's fields, checking them against earlier nodes.
    fn absorb(
        &mut self,
        name: &str,
        index: usize,
        offset: usize,
        width: usize,
    ) -> Result<(), CoreError> {
        let slot = ParamSlot::try_new(name, offset, width)?;
        self.reserve(index);
        if let Some(existing) = self.entries[index].as_ref() {
            if existing.name != name || existing.slot != slot {
                return Err(CoreError::ParamSlotConflict {
                    index,
                    name: name.to_string(),
                    offset,
                    width,
                    other_name: existing.name.clone(),
                    other_offset: existing.slot.offset.index(),
                    other_width: existing.slot.width.index(),
                });
            }
            return Ok(());
        }
        if let Some(&other_index) = self.by_name.get(name) {
            return Err(CoreError::ParamNameConflict {
                name: name.to_string(),
                index,
                other_index: other_index.index(),
            });
        }
        self.insert(ParamIdx::from_usize(index), name, slot);
        Ok(())
    }

    /// Record `name` at `index`, widening the index space to reach it.
    fn insert(&mut self, index: ParamIdx, name: &str, slot: ParamSlot) {
        self.reserve(index.index());
        self.entries[index.index()] = Some(ParamEntry {
            name: name.to_string(),
            slot,
        });
        self.by_name.insert(name.to_string(), index);
    }

    /// One past the highest occupied packed element, refusing overlapping slots.
    fn checked_packed_extent(&self) -> Result<PackedWidth, CoreError> {
        let mut ordered: Vec<(&str, ParamSlot)> = self
            .registered()
            .map(|(_, name, slot)| (name, slot))
            .collect();
        ordered.sort_by_key(|(_, slot)| slot.offset);

        let mut extent = 0usize;
        let mut previous: Option<(&str, ParamSlot)> = None;
        for (name, slot) in ordered {
            if let Some((other_name, other)) = previous
                && slot.offset.index() < other.offset.index() + other.width.index()
            {
                return Err(CoreError::ParamSlotOverlap {
                    name: name.to_string(),
                    offset: slot.offset.index(),
                    width: slot.width.index(),
                    other_name: other_name.to_string(),
                    other_offset: other.offset.index(),
                    other_width: other.width.index(),
                });
            }
            extent = slot.offset.index() + slot.width.index();
            previous = Some((name, slot));
        }
        Ok(PackedWidth::new(extent))
    }
}

/// Resolve a declared packed input width against `extent`; `None` adopts the extent.
///
/// Below the extent is refused, not raised: the FFI borrows a buffer of the declared width.
pub const fn resolve_input_width(
    declared: Option<PackedWidth>,
    extent: PackedWidth,
) -> Result<PackedWidth, CoreError> {
    match declared {
        Some(width) if width.get() < extent.get() => Err(CoreError::InputWidthTooSmall {
            declared: width.get(),
            extent: extent.get(),
        }),
        Some(width) => Ok(width),
        None => Ok(extent),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A width-3 input then a scalar, so the two index spaces diverge.
    fn wide_then_scalar() -> Arena {
        let mut arena = Arena::new();
        let _wide = arena.alloc(Node::InputParameter {
            name: "wide".to_string(),
            index: 0,
            offset: 0,
            width: 3,
        });
        let _scalar = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 1,
            offset: 3,
            width: 1,
        });
        arena
    }

    fn registry_of(arena: &Arena) -> ParamRegistry {
        ParamRegistry::from_arena(arena).expect("a well-formed arena")
    }

    fn input_node(registry: &mut ParamRegistry, name: &str, width: usize) -> Node {
        let (index, slot) = registry.register(name, width).expect("a fresh name");
        Node::InputParameter {
            name: name.to_string(),
            index: index.index(),
            offset: slot.offset.index(),
            width: slot.width.index(),
        }
    }

    #[test]
    fn a_vector_input_separates_the_two_spaces() {
        let registry = registry_of(&wide_then_scalar());
        assert_eq!(registry.index_extent(), 2);
        assert_eq!(registry.n_registered(), 2);
        assert_eq!(registry.packed_width(), PackedWidth::new(4));
        assert_eq!(registry.slot(ParamIdx::new(2)), None);

        assert_eq!(
            registry.slot(ParamIdx::new(1)).unwrap().offset,
            InputOffset::new(3)
        );
        assert_eq!(
            registry.slot(ParamIdx::new(0)).unwrap().width,
            Width::new(3)
        );
    }

    #[test]
    fn register_packs_each_name_after_the_last() {
        let mut registry = ParamRegistry::new();
        let (wide, wide_slot) = registry.register("wide", 3).unwrap();
        let (k, k_slot) = registry.register("k", 1).unwrap();

        assert_eq!((wide, k), (ParamIdx::new(0), ParamIdx::new(1)));
        assert_eq!(wide_slot.offset, InputOffset::new(0));
        assert_eq!(k_slot.offset, InputOffset::new(3));
        assert_eq!(registry.packed_width(), PackedWidth::new(4));
        assert_eq!(registry.names(), vec!["wide", "k"]);
        assert_eq!(registry.index_of("k"), Some(k));
        assert_eq!(registry.index_of("absent"), None);
    }

    #[test]
    fn re_registering_a_name_repeats_its_slot_and_refuses_another_width() {
        let mut registry = ParamRegistry::new();
        let first = registry.register("k", 2).unwrap();
        assert_eq!(registry.register("k", 2).unwrap(), first);
        assert_eq!(registry.packed_width(), PackedWidth::new(2));

        assert!(matches!(
            registry.register("k", 3),
            Err(CoreError::ParamWidthConflict {
                width: 3,
                registered: 2,
                ..
            })
        ));
    }

    #[test]
    fn a_slot_ending_past_u32_max_is_an_error_not_a_panic() {
        let max = u32::MAX as usize;
        let mut registry = ParamRegistry::new();
        assert!(matches!(
            registry.register("huge", max + 1),
            Err(CoreError::ParamSlotRange { offset: 0, .. })
        ));
        assert_eq!(registry, ParamRegistry::new());

        registry
            .register("fills", max)
            .expect("ends exactly at u32::MAX");
        assert!(matches!(
            registry.register("one_more", 1),
            Err(CoreError::ParamSlotRange { offset, width: 1, .. }) if offset == max
        ));

        let mut arena = Arena::new();
        let _ = arena.alloc(Node::InputParameter {
            name: "far".to_string(),
            index: 0,
            offset: max,
            width: 2,
        });
        assert!(matches!(
            ParamRegistry::from_arena(&arena),
            Err(CoreError::ParamSlotRange { .. })
        ));
    }

    #[test]
    fn a_scan_recovers_what_register_built() {
        let mut built = ParamRegistry::new();
        let mut arena = Arena::new();
        for (name, width) in [("wide", 3), ("k", 1), ("wide", 3)] {
            let node = input_node(&mut built, name, width);
            let _ = arena.alloc(node);
        }

        assert_eq!(registry_of(&arena), built);
    }

    #[test]
    fn an_index_with_no_packed_slot_is_not_registered() {
        let mut arena = Arena::new();
        let _a = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        for _ in 0..2 {
            let _ = arena.alloc(Node::Time);
        }
        let _tangent_only = arena.alloc(Node::TangentParameter { index: 3 });

        let registry = registry_of(&arena);
        assert_eq!(registry.index_extent(), 4);
        assert_eq!(registry.n_registered(), 1);
        assert_eq!(registry.packed_width(), PackedWidth::new(1));

        assert!(registry.slot(ParamIdx::new(0)).is_some());
        assert_eq!(registry.names(), ["a"]);
        for unslotted in [1, 2, 3] {
            let unslotted = ParamIdx::new(unslotted);
            assert_eq!(registry.slot(unslotted), None, "{unslotted:?} has no slot");
        }
    }

    #[test]
    fn an_index_past_the_arena_is_an_error_not_an_allocation() {
        for index in [5, usize::MAX] {
            let tangent = Node::TangentParameter { index };
            let input = Node::InputParameter {
                name: "k".to_string(),
                index,
                offset: 0,
                width: 1,
            };
            for node in [tangent, input] {
                let mut arena = Arena::new();
                let _ = arena.alloc(Node::InputParameter {
                    name: "p".to_string(),
                    index: 0,
                    offset: 1,
                    width: 1,
                });
                let _ = arena.alloc(node);
                assert!(matches!(
                    ParamRegistry::from_arena(&arena),
                    Err(CoreError::ParamIndexRange { index: got, n_nodes: 2 }) if got == index
                ));
            }
        }
    }

    #[test]
    fn two_slots_for_one_index_are_refused() {
        let mut arena = Arena::new();
        for offset in [0, 7] {
            let _ = arena.alloc(Node::InputParameter {
                name: "k".to_string(),
                index: 0,
                offset,
                width: 1,
            });
        }

        assert!(matches!(
            ParamRegistry::from_arena(&arena),
            Err(CoreError::ParamSlotConflict {
                index: 0,
                offset: 7,
                other_offset: 0,
                ..
            })
        ));
    }

    #[test]
    fn one_name_at_two_indices_is_refused() {
        let mut arena = Arena::new();
        for (index, offset) in [(0, 0), (1, 1)] {
            let _ = arena.alloc(Node::InputParameter {
                name: "k".to_string(),
                index,
                offset,
                width: 1,
            });
        }

        assert!(matches!(
            ParamRegistry::from_arena(&arena),
            Err(CoreError::ParamNameConflict {
                index: 1,
                other_index: 0,
                ..
            })
        ));
    }

    #[test]
    fn overlapping_slots_are_refused() {
        let mut arena = Arena::new();
        let _ = arena.alloc(Node::InputParameter {
            name: "wide".to_string(),
            index: 0,
            offset: 0,
            width: 3,
        });
        let _ = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 1,
            offset: 2,
            width: 1,
        });

        assert!(matches!(
            ParamRegistry::from_arena(&arena),
            Err(CoreError::ParamSlotOverlap {
                offset: 2,
                other_offset: 0,
                other_width: 3,
                ..
            })
        ));
    }

    #[test]
    fn a_hole_in_the_packed_space_is_spanned() {
        let mut arena = Arena::new();
        let _ = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let _ = arena.alloc(Node::InputParameter {
            name: "b".to_string(),
            index: 1,
            offset: 4,
            width: 1,
        });

        let registry = registry_of(&arena);
        assert_eq!(registry.n_registered(), 2);
        assert_eq!(registry.packed_width(), PackedWidth::new(5));
    }

    #[test]
    fn an_arena_without_inputs_reports_both_spaces_empty() {
        let registry = registry_of(&Arena::new());
        assert_eq!(registry, ParamRegistry::default());
        assert_eq!(registry.index_extent(), 0);
        assert_eq!(registry.n_registered(), 0);
        assert_eq!(registry.packed_width(), PackedWidth::new(0));
        assert!(registry.names().is_empty());
    }

    #[test]
    fn a_declared_input_width_is_refused_below_the_extent_and_kept_above_it() {
        let extent = PackedWidth::new(4);
        assert!(matches!(
            resolve_input_width(Some(PackedWidth::new(1)), extent),
            Err(CoreError::InputWidthTooSmall {
                declared: 1,
                extent: 4
            })
        ));
        assert_eq!(
            resolve_input_width(Some(PackedWidth::new(4)), extent).unwrap(),
            PackedWidth::new(4)
        );
        assert_eq!(
            resolve_input_width(Some(PackedWidth::new(6)), extent).unwrap(),
            PackedWidth::new(6)
        );
        assert_eq!(resolve_input_width(None, extent).unwrap(), extent);
    }
}

//! [`DagSnapshot`], the bincode form fixtures and benchmarks load a model's graph from.

use crate::arena::{Dag, DagRef};
use crate::node::CsrData;
use crate::params::PackedWidth;
use serde::{Deserialize, Serialize};

/// A compiled model's inputs, captured for fixtures and benchmarks.
///
/// The bincode encoding is positional, so field order is part of the format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagSnapshot {
    pub dag: Dag,
    pub n_states: usize,
    /// Length of the packed `inputs` slice the model's evaluators take.
    pub packed_input_width: PackedWidth,
    pub mass_matrix: Option<CsrData>,
    pub model_name: String,
}

impl DagSnapshot {
    pub const fn dag(&self) -> DagRef<'_> {
        self.dag.as_ref()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("DagSnapshot serialization failed")
    }

    /// Decode a snapshot, refusing truncated bytes or a graph that fails validation.
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_re_encode_to_their_own_bytes() {
        for bytes in [
            include_bytes!("../benches/fixtures/spm.bin").as_slice(),
            include_bytes!("../benches/fixtures/dfn.bin").as_slice(),
        ] {
            let snapshot = DagSnapshot::from_bytes(bytes).expect("fixture decodes");
            assert_eq!(snapshot.to_bytes(), bytes);
        }
    }

    #[test]
    fn truncated_bytes_are_an_error() {
        let bytes = include_bytes!("../benches/fixtures/spm.bin");
        assert!(DagSnapshot::from_bytes(&bytes[..bytes.len() / 2]).is_err());
    }
}

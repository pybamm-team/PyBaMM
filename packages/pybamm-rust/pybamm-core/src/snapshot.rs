#![cfg(feature = "serialize")]

use crate::arena::{Arena, NodeId};
use crate::node::CsrData;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagSnapshot {
    pub arena: Arena,
    pub root: NodeId,
    pub n_states: usize,
    pub n_params: usize,
    pub mass_matrix: Option<CsrData>,
    pub model_name: String,
}

impl DagSnapshot {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("DagSnapshot serialization failed")
    }

    pub fn from_bytes(data: &[u8]) -> Self {
        bincode::deserialize(data).expect("DagSnapshot deserialization failed")
    }
}

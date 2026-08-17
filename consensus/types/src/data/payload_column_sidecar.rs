//! Payload column sidecars, introduced in EIP-8142 (Block-in-Blobs).
//!
//! The execution payload is erasure-coded into `NUMBER_OF_COLUMNS` payload columns which replace the
//! EIP-7732 `execution_payload` gossip topic. Unlike data columns, payload columns are not
//! KZG-committed: the builder commits to them with a Merkle root (`payload_columns_root`) carried in
//! the signed `ExecutionPayloadBid`, and each sidecar carries a Merkle branch back to that root.
use std::sync::Arc;

use context_deserialize::context_deserialize;
use educe::Educe;
use merkle_proof::verify_merkle_proof;
use serde::{Deserialize, Serialize};
use ssz_derive::{Decode, Encode};
use ssz_types::{FixedVector, VariableList};
use tree_hash::TreeHash;
use tree_hash_derive::TreeHash;
use typenum::{U7, Unsigned};

use crate::{
    core::{Epoch, EthSpec, Hash256, Slot},
    data::{Cell, ColumnIndex},
    fork::ForkName,
};

/// The depth of the Merkle tree over the payload column roots, i.e. `log2(NUMBER_OF_COLUMNS)`.
///
/// `NUMBER_OF_COLUMNS` is 128 across every preset, which `test_proof_depth_matches_number_of_columns`
/// pins down.
pub type PayloadColumnsInclusionProofDepth = U7;

/// A single payload column: one cell per payload blob (row).
pub type PayloadColumn<E> = VariableList<Cell<E>, <E as EthSpec>::MaxBlobCommitmentsPerBlock>;

pub type PayloadColumnSidecarList<E> = Vec<Arc<PayloadColumnSidecar<E>>>;

#[cfg_attr(
    feature = "arbitrary",
    derive(arbitrary::Arbitrary),
    arbitrary(bound = "E: EthSpec")
)]
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode, TreeHash, Educe)]
#[educe(PartialEq, Hash(bound(E: EthSpec)))]
#[serde(bound = "E: EthSpec", deny_unknown_fields)]
#[context_deserialize(ForkName)]
pub struct PayloadColumnSidecar<E: EthSpec> {
    #[serde(with = "serde_utils::quoted_u64")]
    pub index: ColumnIndex,
    // TODO(EIP-8142): this iteration only ever carries a complete column. Supporting partial columns
    // over the partial-message protocol needs a `cells_present_bitmap` plus a per-cell Merkle
    // multiproof against `column_root`, so that a subset of cells is independently verifiable.
    // Tracking issue: https://github.com/sigp/lighthouse/issues/TODO
    #[serde(with = "ssz_types::serde_utils::list_of_hex_fixed_vec")]
    pub column: PayloadColumn<E>,
    /// Proves `column_root` at position `index` under the bid's `payload_columns_root`.
    pub column_inclusion_proof: FixedVector<Hash256, PayloadColumnsInclusionProofDepth>,
    pub slot: Slot,
    pub beacon_block_root: Hash256,
}

impl<E: EthSpec> PayloadColumnSidecar<E> {
    pub fn epoch(&self) -> Epoch {
        self.slot.epoch(E::slots_per_epoch())
    }

    /// The Merkle root of this column's cells, which is the leaf committed to by
    /// `payload_columns_root`.
    pub fn column_root(&self) -> Hash256 {
        self.column.tree_hash_root()
    }

    /// Verifies this column's inclusion under the `payload_columns_root` committed to by the
    /// builder in the signed `ExecutionPayloadBid`.
    pub fn verify_inclusion_proof(&self, payload_columns_root: Hash256) -> bool {
        verify_merkle_proof(
            self.column_root(),
            &self.column_inclusion_proof,
            PayloadColumnsInclusionProofDepth::to_usize(),
            self.index as usize,
            payload_columns_root,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MainnetEthSpec, MinimalEthSpec};
    use merkle_proof::MerkleTree;

    ssz_and_tree_hash_tests!(PayloadColumnSidecar<MainnetEthSpec>);

    /// `PayloadColumnsInclusionProofDepth` is hard-coded, so it must stay in step with
    /// `NUMBER_OF_COLUMNS` for every preset.
    #[test]
    fn test_proof_depth_matches_number_of_columns() {
        fn assert_depth<E: EthSpec>() {
            assert_eq!(
                1usize << PayloadColumnsInclusionProofDepth::to_usize(),
                E::number_of_columns(),
            );
        }
        assert_depth::<MainnetEthSpec>();
        assert_depth::<MinimalEthSpec>();
    }

    type E = MinimalEthSpec;

    fn make_column(marker: u8, cells: usize) -> PayloadColumn<E> {
        let cells = (0..cells)
            .map(|i| {
                let mut cell = Cell::<E>::default();
                cell[0] = marker;
                cell[1] = i as u8;
                cell
            })
            .collect::<Vec<_>>();
        VariableList::new(cells).expect("cell count is within bounds")
    }

    /// Builds the two-level commitment over `columns` and returns the sidecars plus the root.
    fn build(columns: Vec<PayloadColumn<E>>) -> (Vec<PayloadColumnSidecar<E>>, Hash256) {
        let depth = PayloadColumnsInclusionProofDepth::to_usize();
        let leaves = columns
            .iter()
            .map(|column| column.tree_hash_root())
            .collect::<Vec<_>>();
        let tree = MerkleTree::create(&leaves, depth);

        let sidecars = columns
            .into_iter()
            .enumerate()
            .map(|(index, column)| {
                let (_, proof) = tree
                    .generate_proof(index, depth)
                    .expect("index is within the tree");
                PayloadColumnSidecar {
                    index: index as u64,
                    column,
                    column_inclusion_proof: FixedVector::new(proof)
                        .expect("proof length equals depth"),
                    slot: Slot::new(1),
                    beacon_block_root: Hash256::repeat_byte(9),
                }
            })
            .collect();

        (sidecars, tree.hash())
    }

    #[test]
    fn inclusion_proof_verifies_for_every_column() {
        let columns = (0..E::number_of_columns())
            .map(|i| make_column(i as u8, 3))
            .collect();
        let (sidecars, root) = build(columns);

        assert_eq!(sidecars.len(), E::number_of_columns());
        for sidecar in &sidecars {
            assert!(
                sidecar.verify_inclusion_proof(root),
                "column {} should verify",
                sidecar.index
            );
        }
    }

    #[test]
    fn inclusion_proof_fails_for_mutated_cell() {
        let columns = (0..E::number_of_columns())
            .map(|i| make_column(i as u8, 3))
            .collect();
        let (mut sidecars, root) = build(columns);

        let sidecar = sidecars.get_mut(5).expect("column 5 exists");
        sidecar.column[0][0] ^= 0xff;
        assert!(!sidecar.verify_inclusion_proof(root));
    }

    #[test]
    fn inclusion_proof_fails_for_wrong_index() {
        let columns = (0..E::number_of_columns())
            .map(|i| make_column(i as u8, 3))
            .collect();
        let (mut sidecars, root) = build(columns);

        let sidecar = sidecars.get_mut(5).expect("column 5 exists");
        sidecar.index = 6;
        assert!(!sidecar.verify_inclusion_proof(root));
    }

    #[test]
    fn inclusion_proof_fails_against_wrong_root() {
        let columns = (0..E::number_of_columns())
            .map(|i| make_column(i as u8, 3))
            .collect();
        let (sidecars, _) = build(columns);

        let sidecar = sidecars.first().expect("column 0 exists");
        assert!(!sidecar.verify_inclusion_proof(Hash256::repeat_byte(0xaa)));
    }
}

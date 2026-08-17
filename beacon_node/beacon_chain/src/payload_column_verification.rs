//! Gossip verification for EIP-8142 payload column sidecars.
//!
//! Payload columns carry no signature of their own. They are authenticated through the chain
//! block signature -> `ExecutionPayloadBid` -> `payload_columns_root` -> column inclusion proof,
//! so verification needs the sidecar's beacon block to be known before the column can be judged.
use std::sync::Arc;

use bls::Signature;
use educe::Educe;
use slot_clock::SlotClock;
use store::DatabaseBlock;
use tracing::debug;
use types::{
    ColumnIndex, EthSpec, ExecutionPayloadEnvelope, Hash256, PayloadColumnSidecar,
    PayloadColumnSubnetId, SignedExecutionPayloadEnvelope, Slot,
};

use crate::payload_column_assembler::InsertOutcome;
use crate::payload_column_utils::payload_columns_to_payload_data;
use crate::{BeaconChain, BeaconChainError, BeaconChainTypes};

/// An error occurred while validating a gossip payload column.
#[derive(Debug)]
pub enum GossipPayloadColumnError {
    /// There was an error whilst processing the column. It is not known if it is valid or invalid.
    ///
    /// ## Peer scoring
    ///
    /// We were unable to process this column due to an internal error. It's unclear if the column
    /// is valid.
    BeaconChainError(Box<BeaconChainError>),
    /// The column was gossiped over an incorrect subnet.
    ///
    /// ## Peer scoring
    ///
    /// The column is invalid or the peer is faulty.
    InvalidSubnetId { received: u64, expected: u64 },
    /// The column index is out of range.
    ///
    /// ## Peer scoring
    ///
    /// The column is invalid and the peer is faulty.
    InvalidColumnIndex(ColumnIndex),
    /// The column is from a slot later than the current slot (with respect to gossip clock
    /// disparity).
    ///
    /// ## Peer scoring
    ///
    /// Assuming the local clock is correct, the peer has sent an invalid message.
    FutureSlot {
        message_slot: Slot,
        latest_permissible_slot: Slot,
    },
    /// The column is for a slot at or before the finalized slot.
    ///
    /// ## Peer scoring
    ///
    /// It's unclear if this column is valid, but it is useless to us.
    PastFinalizedSlot {
        column_slot: Slot,
        finalized_slot: Slot,
    },
    /// The beacon block this column refers to is not known.
    ///
    /// ## Peer scoring
    ///
    /// The block may simply not have arrived yet, so this is not a peer fault. The column should
    /// be queued for reprocessing.
    UnknownBeaconBlock(Hash256),
    /// The column's slot does not match the slot of its beacon block.
    ///
    /// ## Peer scoring
    ///
    /// The column is invalid and the peer is faulty.
    SlotMismatch { column_slot: Slot, block_slot: Slot },
    /// The beacon block is from before Gloas, so it carries no execution bid to commit to payload
    /// columns.
    ///
    /// ## Peer scoring
    ///
    /// The column is invalid and the peer is faulty.
    NotGloasBlock(Hash256),
    /// The column's inclusion proof does not verify against the bid's `payload_columns_root`.
    ///
    /// ## Peer scoring
    ///
    /// The column is invalid and the peer is faulty.
    InvalidInclusionProof,
    /// We have already seen a column with this index for this block.
    ///
    /// ## Peer scoring
    ///
    /// The column is valid but not useful. Not a peer fault, as gossip duplicates are expected.
    PriorKnown {
        block_root: Hash256,
        index: ColumnIndex,
    },
    /// The payload for this block has already been recovered.
    ///
    /// ## Peer scoring
    ///
    /// The column is valid but redundant. Not a peer fault.
    AlreadyReconstructed { block_root: Hash256 },
}

impl std::fmt::Display for GossipPayloadColumnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GossipPayloadColumnError {
    /// Whether the sending peer should be penalised for this error.
    pub fn penalize_peer(&self) -> bool {
        match self {
            GossipPayloadColumnError::InvalidSubnetId { .. }
            | GossipPayloadColumnError::InvalidColumnIndex(_)
            | GossipPayloadColumnError::FutureSlot { .. }
            | GossipPayloadColumnError::SlotMismatch { .. }
            | GossipPayloadColumnError::NotGloasBlock(_)
            | GossipPayloadColumnError::InvalidInclusionProof => true,
            GossipPayloadColumnError::BeaconChainError(_)
            | GossipPayloadColumnError::PastFinalizedSlot { .. }
            | GossipPayloadColumnError::UnknownBeaconBlock(_)
            | GossipPayloadColumnError::PriorKnown { .. }
            | GossipPayloadColumnError::AlreadyReconstructed { .. } => false,
        }
    }
}

impl From<BeaconChainError> for GossipPayloadColumnError {
    fn from(e: BeaconChainError) -> Self {
        GossipPayloadColumnError::BeaconChainError(Box::new(e))
    }
}

impl From<store::Error> for GossipPayloadColumnError {
    fn from(e: store::Error) -> Self {
        GossipPayloadColumnError::BeaconChainError(Box::new(BeaconChainError::DBError(e)))
    }
}

/// A payload column sidecar that has passed gossip validation and been stored in the assembler.
#[derive(Educe)]
#[educe(Debug(bound = "T: BeaconChainTypes"))]
pub struct GossipVerifiedPayloadColumn<T: BeaconChainTypes> {
    pub sidecar: Arc<PayloadColumnSidecar<T::EthSpec>>,
    /// How many distinct columns are now held for this block.
    pub columns_held: usize,
}

impl<T: BeaconChainTypes> GossipVerifiedPayloadColumn<T> {
    pub fn index(&self) -> ColumnIndex {
        self.sidecar.index
    }

    pub fn block_root(&self) -> Hash256 {
        self.sidecar.beacon_block_root
    }

    pub fn slot(&self) -> Slot {
        self.sidecar.slot
    }
}

impl<T: BeaconChainTypes> BeaconChain<T> {
    /// Validates a payload column sidecar received on `payload_column_sidecar_{subnet_id}` and, if
    /// it is new, stores it in the payload column assembler.
    pub fn verify_payload_column_for_gossip(
        &self,
        sidecar: Arc<PayloadColumnSidecar<T::EthSpec>>,
        subnet_id: PayloadColumnSubnetId,
    ) -> Result<GossipVerifiedPayloadColumn<T>, GossipPayloadColumnError> {
        let index = sidecar.index;
        let block_root = sidecar.beacon_block_root;
        let column_slot = sidecar.slot;

        // There is one subnet per column, so the index and the subnet must agree.
        if index >= T::EthSpec::number_of_columns() as u64 {
            return Err(GossipPayloadColumnError::InvalidColumnIndex(index));
        }
        if index != *subnet_id {
            return Err(GossipPayloadColumnError::InvalidSubnetId {
                received: *subnet_id,
                expected: index,
            });
        }

        // Cheap clock checks before anything that touches the store.
        let latest_permissible_slot = self
            .slot_clock
            .now_with_future_tolerance(self.spec.maximum_gossip_clock_disparity())
            .ok_or(BeaconChainError::UnableToReadSlot)?;
        if column_slot > latest_permissible_slot {
            return Err(GossipPayloadColumnError::FutureSlot {
                message_slot: column_slot,
                latest_permissible_slot,
            });
        }

        let finalized_slot = self
            .canonical_head
            .cached_head()
            .finalized_checkpoint()
            .epoch
            .start_slot(T::EthSpec::slots_per_epoch());
        if column_slot <= finalized_slot {
            return Err(GossipPayloadColumnError::PastFinalizedSlot {
                column_slot,
                finalized_slot,
            });
        }

        // The commitment lives in the block's execution bid, so the block must be known. If it is
        // not, the column may simply have outrun its block — the caller queues it for reprocessing.
        let Some(proto_block) = self
            .canonical_head
            .fork_choice_read_lock()
            .get_block(&block_root)
        else {
            return Err(GossipPayloadColumnError::UnknownBeaconBlock(block_root));
        };

        if proto_block.slot != column_slot {
            return Err(GossipPayloadColumnError::SlotMismatch {
                column_slot,
                block_slot: proto_block.slot,
            });
        }

        // Loading the block is the expensive step, so the assembler caches the root it yields for
        // the remaining columns of this block.
        let payload_columns_root = self.payload_column_assembler.payload_columns_root_or_init(
            block_root,
            column_slot,
            || load_payload_columns_root(self, block_root),
        )?;

        if !sidecar.verify_inclusion_proof(payload_columns_root) {
            return Err(GossipPayloadColumnError::InvalidInclusionProof);
        }

        match self
            .payload_column_assembler
            .insert(block_root, sidecar.clone())
        {
            InsertOutcome::Stored { columns_held } => {
                debug!(
                    %block_root,
                    column_index = index,
                    columns_held,
                    "Stored gossip payload column"
                );
                Ok(GossipVerifiedPayloadColumn {
                    sidecar,
                    columns_held,
                })
            }
            InsertOutcome::AlreadyKnown => {
                Err(GossipPayloadColumnError::PriorKnown { block_root, index })
            }
            InsertOutcome::AlreadyReconstructed => {
                Err(GossipPayloadColumnError::AlreadyReconstructed { block_root })
            }
        }
    }
}

impl<T: BeaconChainTypes> BeaconChain<T> {
    /// Rebuilds the execution payload envelope from a complete set of recovered payload columns.
    ///
    /// The envelope's `beacon_block_root`, `builder_index` and `parent_beacon_block_root` are read
    /// from the block rather than the wire, and the signature is set to infinity: EIP-8142 drops the
    /// envelope signature in favour of the bid-rooted Merkle commitment already verified per column.
    pub fn reconstruct_envelope_from_payload_columns(
        &self,
        block_root: Hash256,
        columns: &[PayloadColumnSidecar<T::EthSpec>],
    ) -> Result<SignedExecutionPayloadEnvelope<T::EthSpec>, BeaconChainError> {
        let data = payload_columns_to_payload_data(columns).map_err(|e| {
            BeaconChainError::UnableToBuildPayloadColumnSidecar(format!(
                "failed to decode payload columns: {e:?}"
            ))
        })?;

        let block = match self.store.try_get_full_block(&block_root)? {
            Some(DatabaseBlock::Full(block)) => block,
            Some(DatabaseBlock::Blinded(_)) | None => {
                return Err(BeaconChainError::MissingBeaconBlock(block_root));
            }
        };

        let bid = &block
            .message()
            .body()
            .signed_execution_payload_bid()?
            .message;

        Ok(SignedExecutionPayloadEnvelope {
            message: ExecutionPayloadEnvelope {
                payload: data.payload,
                execution_requests: data.execution_requests,
                builder_index: bid.builder_index,
                beacon_block_root: block_root,
                parent_beacon_block_root: bid.parent_block_root,
            },
            signature: Signature::infinity().map_err(|e| {
                BeaconChainError::UnableToBuildPayloadColumnSidecar(format!(
                    "failed to build infinity signature: {e:?}"
                ))
            })?,
        })
    }
}

/// Reads `payload_columns_root` out of the block's execution bid.
fn load_payload_columns_root<T: BeaconChainTypes>(
    chain: &BeaconChain<T>,
    block_root: Hash256,
) -> Result<Hash256, GossipPayloadColumnError> {
    let block = match chain.store.try_get_full_block(&block_root)? {
        Some(DatabaseBlock::Full(block)) => block,
        Some(DatabaseBlock::Blinded(_)) | None => {
            return Err(GossipPayloadColumnError::UnknownBeaconBlock(block_root));
        }
    };

    let bid = block
        .message()
        .body()
        .signed_execution_payload_bid()
        .map_err(|_| GossipPayloadColumnError::NotGloasBlock(block_root))?;

    Ok(bid.message.payload_columns_root)
}

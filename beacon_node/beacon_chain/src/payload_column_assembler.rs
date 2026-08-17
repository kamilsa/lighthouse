//! Collects EIP-8142 payload column sidecars until enough have arrived to recover the execution
//! payload.
//!
//! Each entry is keyed by beacon block root and caches the `payload_columns_root` taken from that
//! block's execution bid, so that verifying the 128th column for a block costs no more than
//! verifying the first.
use hashlink::lru_cache::LruCache;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use types::{ColumnIndex, Epoch, EthSpec, Hash256, PayloadColumnSidecar, Slot};

/// What happened when a column was offered to the assembler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    /// The column is new and was stored.
    Stored {
        /// How many distinct columns are now held for this block.
        columns_held: usize,
    },
    /// A column with this index was already held for this block.
    AlreadyKnown,
    /// The payload for this block has already been recovered, so the column is redundant.
    AlreadyReconstructed,
}

/// The columns held for a single beacon block.
struct Assembly<E: EthSpec> {
    /// Taken from the block's `ExecutionPayloadBid`; commits to every column of this payload.
    payload_columns_root: Hash256,
    slot: Slot,
    columns: HashMap<ColumnIndex, Arc<PayloadColumnSidecar<E>>>,
    /// Set once the payload has been recovered, so later columns are dropped rather than
    /// triggering a second reconstruction.
    reconstructed: bool,
}

pub struct PayloadColumnAssembler<E: EthSpec> {
    assemblies: RwLock<LruCache<Hash256, Assembly<E>>>,
}

impl<E: EthSpec> PayloadColumnAssembler<E> {
    pub fn new(capacity: usize) -> Self {
        Self {
            assemblies: RwLock::new(LruCache::new(capacity)),
        }
    }

    /// The number of columns needed to recover the payload: half of them, given the 2x erasure
    /// coding.
    pub fn columns_required_for_recovery() -> usize {
        E::number_of_columns().div_ceil(2)
    }

    /// Returns the cached `payload_columns_root` for `block_root`, initialising the entry from
    /// `load` on the first column seen for that block.
    ///
    /// `load` is only invoked on a cache miss, so verifying a block's columns loads its bid once
    /// rather than once per column.
    pub fn payload_columns_root_or_init<F, Err>(
        &self,
        block_root: Hash256,
        slot: Slot,
        load: F,
    ) -> Result<Hash256, Err>
    where
        F: FnOnce() -> Result<Hash256, Err>,
    {
        if let Some(assembly) = self.assemblies.write().get(&block_root) {
            return Ok(assembly.payload_columns_root);
        }

        // The lock is released while `load` hits the store, so several columns for the same block
        // can race here. Re-check under the write lock and keep the entry that got there first —
        // overwriting it would discard columns already stored and reset the `reconstructed` flag.
        let payload_columns_root = load()?;

        let mut assemblies = self.assemblies.write();
        if let Some(assembly) = assemblies.get(&block_root) {
            return Ok(assembly.payload_columns_root);
        }

        assemblies.insert(
            block_root,
            Assembly {
                payload_columns_root,
                slot,
                columns: HashMap::new(),
                reconstructed: false,
            },
        );

        Ok(payload_columns_root)
    }

    /// Stores a verified column, reporting whether it was new.
    ///
    /// The caller must have already verified the column's inclusion proof against the
    /// `payload_columns_root` for `block_root`.
    pub fn insert(
        &self,
        block_root: Hash256,
        sidecar: Arc<PayloadColumnSidecar<E>>,
    ) -> InsertOutcome {
        let mut assemblies = self.assemblies.write();
        let Some(assembly) = assemblies.get_mut(&block_root) else {
            // `payload_columns_root_or_init` creates the entry, so a miss here means the entry was
            // evicted between verification and insertion. Dropping the column is safe: it will be
            // re-offered by the next arrival for this block.
            return InsertOutcome::AlreadyKnown;
        };

        if assembly.reconstructed {
            return InsertOutcome::AlreadyReconstructed;
        }

        match assembly.columns.entry(sidecar.index) {
            std::collections::hash_map::Entry::Occupied(_) => InsertOutcome::AlreadyKnown,
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(sidecar);
                InsertOutcome::Stored {
                    columns_held: assembly.columns.len(),
                }
            }
        }
    }

    /// If enough columns are held to recover the payload, marks the block as reconstructed and
    /// returns the columns.
    ///
    /// Marking happens under the same lock as the check, so concurrent callers cannot both start a
    /// reconstruction for the same block.
    pub fn take_columns_for_recovery(
        &self,
        block_root: Hash256,
    ) -> Option<Vec<PayloadColumnSidecar<E>>> {
        let mut assemblies = self.assemblies.write();
        let assembly = assemblies.get_mut(&block_root)?;

        if assembly.reconstructed || assembly.columns.len() < Self::columns_required_for_recovery()
        {
            return None;
        }

        assembly.reconstructed = true;

        // Recovery wants a deterministic column ordering.
        let mut columns = assembly
            .columns
            .values()
            .map(|sidecar| sidecar.as_ref().clone())
            .collect::<Vec<_>>();
        columns.sort_by_key(|sidecar| sidecar.index);

        Some(columns)
    }

    /// Undoes the "reconstructed" mark, so that a failed recovery attempt can be retried when more
    /// columns arrive.
    pub fn clear_reconstructed(&self, block_root: Hash256) {
        if let Some(assembly) = self.assemblies.write().get_mut(&block_root) {
            assembly.reconstructed = false;
        }
    }

    /// The indices currently held for a block, used to work out which reconstructed columns still
    /// need publishing.
    pub fn held_indices(&self, block_root: Hash256) -> Vec<ColumnIndex> {
        self.assemblies
            .write()
            .get(&block_root)
            .map(|assembly| assembly.columns.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Drops assemblies for blocks older than `cutoff_epoch`.
    pub fn do_maintenance(&self, cutoff_epoch: Epoch) {
        let mut assemblies = self.assemblies.write();
        let to_remove = assemblies
            .iter()
            .filter(|(_, assembly)| assembly.slot.epoch(E::slots_per_epoch()) < cutoff_epoch)
            .map(|(root, _)| *root)
            .collect::<Vec<_>>();

        for root in to_remove {
            assemblies.remove(&root);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ssz_types::{FixedVector, VariableList};
    use types::MinimalEthSpec;

    type E = MinimalEthSpec;

    fn assembler() -> PayloadColumnAssembler<E> {
        PayloadColumnAssembler::new(4)
    }

    fn sidecar(index: ColumnIndex, slot: Slot) -> Arc<PayloadColumnSidecar<E>> {
        Arc::new(PayloadColumnSidecar {
            index,
            column: VariableList::empty(),
            column_inclusion_proof: FixedVector::default(),
            slot,
            beacon_block_root: Hash256::repeat_byte(1),
        })
    }

    fn init(assembler: &PayloadColumnAssembler<E>, root: Hash256, slot: Slot) -> Hash256 {
        assembler
            .payload_columns_root_or_init(root, slot, || Ok::<_, ()>(Hash256::repeat_byte(0xab)))
            .expect("load succeeds")
    }

    #[test]
    fn root_is_loaded_once_per_block() {
        let assembler = assembler();
        let root = Hash256::repeat_byte(1);

        let first = init(&assembler, root, Slot::new(1));
        assert_eq!(first, Hash256::repeat_byte(0xab));

        // A second call must not invoke `load` again.
        let second = assembler
            .payload_columns_root_or_init(root, Slot::new(1), || -> Result<Hash256, ()> {
                panic!("load should not be called for a cached block")
            })
            .expect("cached");
        assert_eq!(second, first);
    }

    #[test]
    fn racing_init_does_not_discard_stored_columns() {
        // Columns for the same block arrive concurrently, so two callers can both miss the cache
        // and both run `load`. The one that finishes second must not reset the entry.
        let assembler = assembler();
        let root = Hash256::repeat_byte(1);
        let slot = Slot::new(1);

        // `load` stands in for the racing caller: it wins the race while we are still "loading".
        let root_from_race = assembler
            .payload_columns_root_or_init(root, slot, || {
                init(&assembler, root, slot);
                assembler.insert(root, sidecar(0, slot));
                Ok::<_, ()>(Hash256::repeat_byte(0xab))
            })
            .expect("load succeeds");

        assert_eq!(root_from_race, Hash256::repeat_byte(0xab));
        assert_eq!(
            assembler.held_indices(root),
            vec![0],
            "the column stored by the winning caller must survive"
        );
    }

    #[test]
    fn duplicate_columns_are_rejected() {
        let assembler = assembler();
        let root = Hash256::repeat_byte(1);
        init(&assembler, root, Slot::new(1));

        assert_eq!(
            assembler.insert(root, sidecar(0, Slot::new(1))),
            InsertOutcome::Stored { columns_held: 1 }
        );
        assert_eq!(
            assembler.insert(root, sidecar(0, Slot::new(1))),
            InsertOutcome::AlreadyKnown
        );
        assert_eq!(
            assembler.insert(root, sidecar(1, Slot::new(1))),
            InsertOutcome::Stored { columns_held: 2 }
        );
    }

    #[test]
    fn recovery_unlocks_at_half_the_columns() {
        let assembler = assembler();
        let root = Hash256::repeat_byte(1);
        let slot = Slot::new(1);
        init(&assembler, root, slot);

        let required = PayloadColumnAssembler::<E>::columns_required_for_recovery();
        for index in 0..required.saturating_sub(1) {
            assembler.insert(root, sidecar(index as u64, slot));
            assert!(
                assembler.take_columns_for_recovery(root).is_none(),
                "should not recover with {} columns",
                index + 1
            );
        }

        assembler.insert(root, sidecar(required as u64 - 1, slot));
        let columns = assembler
            .take_columns_for_recovery(root)
            .expect("should recover once the threshold is reached");
        assert_eq!(columns.len(), required);
        // Ordering must be deterministic for recovery.
        assert!(columns.windows(2).all(|w| w[0].index < w[1].index));
    }

    #[test]
    fn recovery_happens_only_once() {
        let assembler = assembler();
        let root = Hash256::repeat_byte(1);
        let slot = Slot::new(1);
        init(&assembler, root, slot);

        let required = PayloadColumnAssembler::<E>::columns_required_for_recovery();
        for index in 0..required {
            assembler.insert(root, sidecar(index as u64, slot));
        }

        assert!(assembler.take_columns_for_recovery(root).is_some());
        assert!(
            assembler.take_columns_for_recovery(root).is_none(),
            "a second reconstruction must not be triggered"
        );
        assert_eq!(
            assembler.insert(root, sidecar(required as u64, slot)),
            InsertOutcome::AlreadyReconstructed,
            "late columns are redundant once the payload is recovered"
        );
    }

    #[test]
    fn failed_recovery_can_be_retried() {
        let assembler = assembler();
        let root = Hash256::repeat_byte(1);
        let slot = Slot::new(1);
        init(&assembler, root, slot);

        let required = PayloadColumnAssembler::<E>::columns_required_for_recovery();
        for index in 0..required {
            assembler.insert(root, sidecar(index as u64, slot));
        }
        assert!(assembler.take_columns_for_recovery(root).is_some());

        assembler.clear_reconstructed(root);
        assert!(
            assembler.take_columns_for_recovery(root).is_some(),
            "clearing the mark should allow another attempt"
        );
    }

    #[test]
    fn maintenance_drops_old_assemblies() {
        let assembler = assembler();
        let old = Hash256::repeat_byte(1);
        let recent = Hash256::repeat_byte(2);

        init(&assembler, old, Slot::new(0));
        init(&assembler, recent, Slot::new(100));
        assembler.insert(old, sidecar(0, Slot::new(0)));
        assembler.insert(recent, sidecar(0, Slot::new(100)));
        assert_eq!(assembler.held_indices(old), vec![0]);

        assembler.do_maintenance(Epoch::new(1));

        assert!(
            assembler.held_indices(old).is_empty(),
            "the epoch-0 assembly should have been dropped"
        );
        assert_eq!(
            assembler.held_indices(recent),
            vec![0],
            "the recent assembly should survive"
        );
    }
}

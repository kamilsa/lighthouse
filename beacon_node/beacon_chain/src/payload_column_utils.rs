//! Encoding, commitment and recovery for EIP-8142 payload columns.
//!
//! The execution payload is packed into blob-shaped field elements, erasure-coded with the same 2x
//! Reed-Solomon extension used by PeerDAS, and sliced into `NUMBER_OF_COLUMNS` payload columns. Any
//! `NUMBER_OF_COLUMNS / 2` columns are enough to recover the payload.
//!
//! Unlike data columns, payload columns are **not** KZG-committed. The builder commits to them with
//! a two-level Merkle tree — the cells of a column hash to a `column_root`, and the column roots hash
//! to `payload_columns_root`, which travels in the signed `ExecutionPayloadBid`. No KZG proof is ever
//! computed for a payload column, so both the encode and recover paths must stay clear of the
//! multi-scalar multiplication that proof generation performs.
use kzg::{Cell as KzgCell, CellRef as KzgCellRef, Error as KzgError, Kzg, KzgBlobRef};
use merkle_proof::{MerkleTree, MerkleTreeError};
use rayon::prelude::*;
use safe_arith::{ArithError, SafeArith};
use ssz::{Decode, Encode};
use ssz_derive::{Decode, Encode};
use ssz_types::FixedVector;
use tree_hash::TreeHash;
use typenum::Unsigned;
use types::{
    Blob, Cell, EthSpec, ExecutionPayloadGloas, ExecutionRequests, Hash256, PayloadColumn,
    PayloadColumnSidecar, PayloadColumnsInclusionProofDepth, Slot,
};

/// Bytes of a 32-byte field element that carry payload data. The most significant byte is always
/// zero so that every field element is a canonical BLS scalar (`value < BLS_MODULUS`).
const USABLE_BYTES_PER_FIELD_ELEMENT: usize = 31;

/// Length prefix written at the front of the encoded payload, in bytes.
const LENGTH_PREFIX_BYTES: usize = 4;

/// The subset of the execution payload envelope committed to by `payload_columns_root`.
///
/// Deliberately excludes `beacon_block_root` — the bid is built *before* the block that carries it,
/// so the block root is not yet known — and the builder signature, which EIP-8142 Phase 1 drops in
/// favour of the bid-rooted Merkle commitment.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct PayloadColumnData<E: EthSpec> {
    pub payload: ExecutionPayloadGloas<E>,
    pub execution_requests: ExecutionRequests<E>,
}

#[derive(Debug)]
pub enum PayloadColumnError {
    Kzg(KzgError),
    MerkleTree(MerkleTreeError),
    Arith(ArithError),
    Ssz(ssz::DecodeError),
    /// The encoded payload does not fit within `MaxBlobCommitmentsPerBlock` blobs.
    PayloadTooLarge {
        blobs: usize,
        max: usize,
    },
    /// A field element's most significant byte was non-zero.
    NonCanonicalFieldElement {
        offset: usize,
    },
    /// Bytes past the declared payload length were not zero padding.
    TrailingData,
    /// The declared length prefix exceeds the decoded byte count.
    InvalidLengthPrefix {
        declared: usize,
        available: usize,
    },
    /// Fewer than `NUMBER_OF_COLUMNS / 2` distinct columns were supplied for recovery.
    InsufficientColumns {
        supplied: usize,
        required: usize,
    },
    /// The supplied columns disagree on how many cells (rows) the payload occupies.
    InconsistentColumnLengths,
    /// Internal indexing failure that should be unreachable.
    Internal(String),
}

impl From<KzgError> for PayloadColumnError {
    fn from(e: KzgError) -> Self {
        Self::Kzg(e)
    }
}

impl From<MerkleTreeError> for PayloadColumnError {
    fn from(e: MerkleTreeError) -> Self {
        Self::MerkleTree(e)
    }
}

impl From<ArithError> for PayloadColumnError {
    fn from(e: ArithError) -> Self {
        Self::Arith(e)
    }
}

impl From<ssz::DecodeError> for PayloadColumnError {
    fn from(e: ssz::DecodeError) -> Self {
        Self::Ssz(e)
    }
}

/// Usable payload bytes per blob.
fn usable_bytes_per_blob<E: EthSpec>() -> Result<usize, ArithError> {
    E::field_elements_per_blob().safe_mul(USABLE_BYTES_PER_FIELD_ELEMENT)
}

/// Packs the SSZ-encoded `data` into one or more blobs.
///
/// The encoding is a 4-byte big-endian length prefix followed by the SSZ bytes, spread 31 bytes at a
/// time across the low bytes of each 32-byte field element, and zero-padded to a whole number of
/// blobs.
pub fn payload_data_to_blobs<E: EthSpec>(
    data: &PayloadColumnData<E>,
) -> Result<Vec<Blob<E>>, PayloadColumnError> {
    let ssz_bytes = data.as_ssz_bytes();

    let mut payload_bytes = Vec::with_capacity(ssz_bytes.len().saturating_add(LENGTH_PREFIX_BYTES));
    let length: u32 = ssz_bytes.len().try_into().map_err(|_| {
        PayloadColumnError::Internal("payload length does not fit in u32".to_string())
    })?;
    payload_bytes.extend_from_slice(&length.to_be_bytes());
    payload_bytes.extend_from_slice(&ssz_bytes);

    let usable_per_blob = usable_bytes_per_blob::<E>()?;
    // Ceiling division; at least one blob so that an empty payload still round-trips.
    let num_blobs = payload_bytes
        .len()
        .saturating_add(usable_per_blob.saturating_sub(1))
        .checked_div(usable_per_blob)
        .unwrap_or(0)
        .max(1);

    let max_blobs = E::max_blob_commitments_per_block();
    if num_blobs > max_blobs {
        return Err(PayloadColumnError::PayloadTooLarge {
            blobs: num_blobs,
            max: max_blobs,
        });
    }

    let mut blobs = Vec::with_capacity(num_blobs);
    for blob_index in 0..num_blobs {
        let mut blob_bytes = vec![0u8; E::bytes_per_blob()];
        let blob_start = blob_index.safe_mul(usable_per_blob)?;

        for fe_index in 0..E::field_elements_per_blob() {
            let chunk_start =
                blob_start.safe_add(fe_index.safe_mul(USABLE_BYTES_PER_FIELD_ELEMENT)?)?;
            if chunk_start >= payload_bytes.len() {
                break;
            }
            let chunk_end = chunk_start
                .safe_add(USABLE_BYTES_PER_FIELD_ELEMENT)?
                .min(payload_bytes.len());
            let chunk = payload_bytes
                .get(chunk_start..chunk_end)
                .ok_or_else(|| PayloadColumnError::Internal("chunk out of range".to_string()))?;

            // Leave byte 0 of the field element zero and write the data into bytes [1..32).
            let dest_start = fe_index
                .safe_mul(kzg::BYTES_PER_FIELD_ELEMENT)?
                .safe_add(1)?;
            let dest_end = dest_start.safe_add(chunk.len())?;
            let dest = blob_bytes
                .get_mut(dest_start..dest_end)
                .ok_or_else(|| PayloadColumnError::Internal("blob out of range".to_string()))?;
            dest.copy_from_slice(chunk);
        }

        blobs.push(
            Blob::<E>::new(blob_bytes)
                .map_err(|e| PayloadColumnError::Internal(format!("blob size mismatch: {e:?}")))?,
        );
    }

    Ok(blobs)
}

/// Inverse of [`payload_data_to_blobs`].
pub fn blobs_to_payload_data<E: EthSpec>(
    blobs: &[Blob<E>],
) -> Result<PayloadColumnData<E>, PayloadColumnError> {
    let usable_per_blob = usable_bytes_per_blob::<E>()?;
    let mut raw = Vec::with_capacity(blobs.len().saturating_mul(usable_per_blob));

    for (blob_index, blob) in blobs.iter().enumerate() {
        for fe_index in 0..E::field_elements_per_blob() {
            let fe_start = fe_index.safe_mul(kzg::BYTES_PER_FIELD_ELEMENT)?;
            let fe = blob
                .get(fe_start..fe_start.safe_add(kzg::BYTES_PER_FIELD_ELEMENT)?)
                .ok_or_else(|| {
                    PayloadColumnError::Internal("field element out of range".to_string())
                })?;

            match fe.first() {
                Some(0) => {}
                _ => {
                    return Err(PayloadColumnError::NonCanonicalFieldElement {
                        offset: blob_index
                            .safe_mul(E::bytes_per_blob())?
                            .safe_add(fe_start)?,
                    });
                }
            }

            raw.extend_from_slice(fe.get(1..).ok_or_else(|| {
                PayloadColumnError::Internal("field element too short".to_string())
            })?);
        }
    }

    let length_bytes: [u8; LENGTH_PREFIX_BYTES] = raw
        .get(0..LENGTH_PREFIX_BYTES)
        .and_then(|s| s.try_into().ok())
        .ok_or(PayloadColumnError::InvalidLengthPrefix {
            declared: 0,
            available: raw.len(),
        })?;
    let declared = u32::from_be_bytes(length_bytes) as usize;

    let body_end = declared.safe_add(LENGTH_PREFIX_BYTES)?;
    if body_end > raw.len() {
        return Err(PayloadColumnError::InvalidLengthPrefix {
            declared,
            available: raw.len(),
        });
    }

    let body = raw
        .get(LENGTH_PREFIX_BYTES..body_end)
        .ok_or_else(|| PayloadColumnError::Internal("body out of range".to_string()))?;

    // Everything after the payload must be zero padding — no smuggling extra bytes into the
    // committed data.
    let trailing = raw
        .get(body_end..)
        .ok_or_else(|| PayloadColumnError::Internal("trailing out of range".to_string()))?;
    if trailing.iter().any(|b| *b != 0) {
        return Err(PayloadColumnError::TrailingData);
    }

    PayloadColumnData::from_ssz_bytes(body).map_err(PayloadColumnError::from)
}

/// Erasure-codes `data` and returns one sidecar per column plus the `payload_columns_root` the
/// builder must place in its `ExecutionPayloadBid`.
pub fn build_payload_column_sidecars<E: EthSpec>(
    data: &PayloadColumnData<E>,
    beacon_block_root: Hash256,
    slot: Slot,
    kzg: &Kzg,
) -> Result<(Vec<PayloadColumnSidecar<E>>, Hash256), PayloadColumnError> {
    let blobs = payload_data_to_blobs(data)?;

    // `compute_cells` performs only the FFT extension — never use `compute_cells_and_proofs` here,
    // the KZG proofs would be discarded and their MSM dominates the cost.
    let rows = blobs
        .par_iter()
        .map(|blob| {
            let blob: KzgBlobRef<'_> = blob.as_ref().try_into().map_err(|e| {
                KzgError::InconsistentArrayLength(format!(
                    "blob should have a guaranteed size due to FixedVector: {e:?}"
                ))
            })?;
            kzg.compute_cells(blob)
        })
        .collect::<Result<Vec<_>, KzgError>>()?;

    let columns = transpose_rows_into_columns::<E>(&rows)?;
    build_sidecars_from_columns(columns, beacon_block_root, slot)
}

/// Returns just the `payload_columns_root` for `data`, for the builder at bid time and to bind an
/// envelope fetched over RPC to its signed bid.
pub fn compute_payload_columns_root<E: EthSpec>(
    data: &PayloadColumnData<E>,
    kzg: &Kzg,
) -> Result<Hash256, PayloadColumnError> {
    let (_, root) = build_payload_column_sidecars(data, Hash256::ZERO, Slot::new(0), kzg)?;
    Ok(root)
}

/// Recovers the full set of columns from any `NUMBER_OF_COLUMNS / 2` of them.
///
/// The returned sidecars carry freshly derived inclusion proofs, so they are ready to be gossiped
/// as re-seeded columns.
pub fn recover_payload_columns<E: EthSpec>(
    sidecars: &[PayloadColumnSidecar<E>],
    kzg: &Kzg,
) -> Result<Vec<PayloadColumnSidecar<E>>, PayloadColumnError> {
    let required = E::number_of_columns().safe_div(2)?;
    if sidecars.len() < required {
        return Err(PayloadColumnError::InsufficientColumns {
            supplied: sidecars.len(),
            required,
        });
    }

    let Some(first) = sidecars.first() else {
        return Err(PayloadColumnError::InsufficientColumns {
            supplied: 0,
            required,
        });
    };
    let num_rows = first.column.len();
    if sidecars.iter().any(|s| s.column.len() != num_rows) {
        return Err(PayloadColumnError::InconsistentColumnLengths);
    }

    let cell_ids = sidecars.iter().map(|s| s.index).collect::<Vec<_>>();

    // Recover row by row: every row is an independently erasure-coded blob.
    let rows = (0..num_rows)
        .into_par_iter()
        .map(|row| {
            let cells = sidecars
                .iter()
                .map(|sidecar| {
                    let cell = sidecar.column.get(row).ok_or_else(|| {
                        PayloadColumnError::Internal(format!("missing cell at row {row}"))
                    })?;
                    ssz_cell_to_kzg_cell_ref::<E>(cell)
                })
                .collect::<Result<Vec<KzgCellRef<'_>>, PayloadColumnError>>()?;

            kzg.recover_cells(&cell_ids, &cells)
                .map_err(PayloadColumnError::from)
        })
        .collect::<Result<Vec<_>, PayloadColumnError>>()?;

    let columns = transpose_rows_into_columns::<E>(&rows)?;
    let (recovered, _) = build_sidecars_from_columns(columns, first.beacon_block_root, first.slot)?;

    Ok(recovered)
}

/// Decodes a complete set of payload columns back into the execution payload.
///
/// Expects the full `NUMBER_OF_COLUMNS` columns, as returned by [`recover_payload_columns`]. The
/// original blob data is the first half of each row; the second half is the erasure-coded extension.
pub fn payload_columns_to_payload_data<E: EthSpec>(
    columns: &[PayloadColumnSidecar<E>],
) -> Result<PayloadColumnData<E>, PayloadColumnError> {
    let number_of_columns = E::number_of_columns();
    if columns.len() != number_of_columns {
        return Err(PayloadColumnError::InsufficientColumns {
            supplied: columns.len(),
            required: number_of_columns,
        });
    }

    let Some(first) = columns.first() else {
        return Err(PayloadColumnError::InsufficientColumns {
            supplied: 0,
            required: number_of_columns,
        });
    };
    let num_rows = first.column.len();
    if columns.iter().any(|column| column.column.len() != num_rows) {
        return Err(PayloadColumnError::InconsistentColumnLengths);
    }

    // The systematic (unextended) half of the codeword holds the original blob bytes.
    let data_columns = number_of_columns.safe_div(2)?;
    let mut blobs = Vec::with_capacity(num_rows);

    for row in 0..num_rows {
        let mut blob_bytes = Vec::with_capacity(E::bytes_per_blob());
        for index in 0..data_columns {
            let sidecar = columns
                .get(index)
                .ok_or_else(|| PayloadColumnError::Internal(format!("missing column {index}")))?;
            // Columns are ordered by index by `build_sidecars_from_columns`, but be explicit
            // rather than trusting position.
            if sidecar.index != index as u64 {
                return Err(PayloadColumnError::Internal(format!(
                    "column at position {index} has index {}",
                    sidecar.index
                )));
            }
            let cell = sidecar.column.get(row).ok_or_else(|| {
                PayloadColumnError::Internal(format!("missing cell at row {row}"))
            })?;
            blob_bytes.extend_from_slice(cell.as_ref());
        }

        blobs.push(
            Blob::<E>::new(blob_bytes)
                .map_err(|e| PayloadColumnError::Internal(format!("blob size mismatch: {e:?}")))?,
        );
    }

    blobs_to_payload_data(&blobs)
}

/// Turns per-blob cell arrays into per-column cell lists.
fn transpose_rows_into_columns<E: EthSpec>(
    rows: &[[KzgCell; kzg::CELLS_PER_EXT_BLOB]],
) -> Result<Vec<PayloadColumn<E>>, PayloadColumnError> {
    let number_of_columns = E::number_of_columns();
    let mut columns = vec![Vec::with_capacity(rows.len()); number_of_columns];

    for row_cells in rows {
        for col in 0..number_of_columns {
            let cell = row_cells
                .get(col)
                .ok_or_else(|| PayloadColumnError::Internal(format!("missing cell {col}")))?;
            let cell = Cell::<E>::try_from(cell.to_vec())
                .map_err(|e| PayloadColumnError::Internal(format!("BytesPerCell: {e:?}")))?;
            columns
                .get_mut(col)
                .ok_or_else(|| PayloadColumnError::Internal(format!("missing column {col}")))?
                .push(cell);
        }
    }

    columns
        .into_iter()
        .map(|cells| {
            PayloadColumn::<E>::try_from(cells).map_err(|e| {
                PayloadColumnError::Internal(format!("MaxBlobCommitmentsPerBlock: {e:?}"))
            })
        })
        .collect()
}

/// Builds the two-level Merkle commitment over `columns` and attaches an inclusion proof to each.
fn build_sidecars_from_columns<E: EthSpec>(
    columns: Vec<PayloadColumn<E>>,
    beacon_block_root: Hash256,
    slot: Slot,
) -> Result<(Vec<PayloadColumnSidecar<E>>, Hash256), PayloadColumnError> {
    let depth = PayloadColumnsInclusionProofDepth::to_usize();
    let leaves = columns
        .iter()
        .map(|column| column.tree_hash_root())
        .collect::<Vec<_>>();
    let tree = MerkleTree::create(&leaves, depth);
    let payload_columns_root = tree.hash();

    let sidecars = columns
        .into_iter()
        .enumerate()
        .map(|(index, column)| {
            let (_, proof) = tree.generate_proof(index, depth)?;
            Ok(PayloadColumnSidecar {
                index: index as u64,
                column,
                column_inclusion_proof: FixedVector::new(proof).map_err(|e| {
                    PayloadColumnError::Internal(format!("proof length mismatch: {e:?}"))
                })?,
                slot,
                beacon_block_root,
            })
        })
        .collect::<Result<Vec<_>, PayloadColumnError>>()?;

    Ok((sidecars, payload_columns_root))
}

/// Converts an SSZ cell to the fixed-size array reference `rust_eth_kzg` expects.
fn ssz_cell_to_kzg_cell_ref<E: EthSpec>(
    cell: &Cell<E>,
) -> Result<KzgCellRef<'_>, PayloadColumnError> {
    let cell_bytes: &[u8] = cell.as_ref();
    cell_bytes.try_into().map_err(|e| {
        PayloadColumnError::Kzg(KzgError::InconsistentArrayLength(format!(
            "expected cell to have size BYTES_PER_CELL, guaranteed by FixedVector: {e:?}"
        )))
    })
}

/// Convenience for the common case of building the committed data straight from an envelope's
/// contents.
pub fn payload_column_data<E: EthSpec>(
    payload: &ExecutionPayloadGloas<E>,
    execution_requests: &ExecutionRequests<E>,
) -> PayloadColumnData<E> {
    PayloadColumnData {
        payload: payload.clone(),
        execution_requests: execution_requests.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kzg::trusted_setup::get_trusted_setup;
    use ssz_types::VariableList;
    use std::sync::LazyLock;
    use types::{MainnetEthSpec, Transaction, Transactions};

    type E = MainnetEthSpec;

    static KZG: LazyLock<Kzg> = LazyLock::new(|| {
        Kzg::new_from_trusted_setup(&get_trusted_setup()).expect("should create kzg")
    });

    /// Builds payload data whose SSZ encoding spans roughly `blobs` blobs.
    fn payload_data(transaction_bytes: usize) -> PayloadColumnData<E> {
        let mut payload = ExecutionPayloadGloas::<E>::default();
        if transaction_bytes > 0 {
            let tx: Transaction<<E as EthSpec>::MaxBytesPerTransaction> =
                VariableList::new(vec![0xab; transaction_bytes]).expect("tx within bounds");
            payload.transactions =
                Transactions::<E>::new(vec![tx]).expect("transaction list within bounds");
        }
        PayloadColumnData {
            payload,
            execution_requests: ExecutionRequests::default(),
        }
    }

    #[test]
    fn blob_round_trip_empty_payload() {
        let data = payload_data(0);
        let blobs = payload_data_to_blobs(&data).expect("should encode");
        assert_eq!(blobs.len(), 1, "an empty payload still occupies one blob");
        assert_eq!(blobs_to_payload_data(&blobs).expect("should decode"), data);
    }

    #[test]
    fn blob_round_trip_single_blob() {
        let data = payload_data(1024);
        let blobs = payload_data_to_blobs(&data).expect("should encode");
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs_to_payload_data(&blobs).expect("should decode"), data);
    }

    #[test]
    fn blob_round_trip_multiple_blobs() {
        // Comfortably larger than one blob's worth of usable bytes.
        let usable = usable_bytes_per_blob::<E>().expect("no overflow");
        let data = payload_data(usable * 2 + 17);
        let blobs = payload_data_to_blobs(&data).expect("should encode");
        assert!(blobs.len() >= 3, "expected >= 3 blobs, got {}", blobs.len());
        assert_eq!(blobs_to_payload_data(&blobs).expect("should decode"), data);
    }

    #[test]
    fn decode_rejects_non_canonical_field_element() {
        let data = payload_data(64);
        let mut blobs = payload_data_to_blobs(&data).expect("should encode");
        // Set the MSB of the second field element.
        blobs[0][kzg::BYTES_PER_FIELD_ELEMENT] = 1;
        assert!(matches!(
            blobs_to_payload_data::<E>(&blobs),
            Err(PayloadColumnError::NonCanonicalFieldElement { .. })
        ));
    }

    #[test]
    fn decode_rejects_trailing_data() {
        let data = payload_data(64);
        let mut blobs = payload_data_to_blobs(&data).expect("should encode");
        // Write a non-zero byte into the padding at the very end of the last blob.
        let last = blobs.last_mut().expect("at least one blob");
        let len = last.len();
        last[len - 1] = 0xff;
        assert!(matches!(
            blobs_to_payload_data::<E>(&blobs),
            Err(PayloadColumnError::TrailingData)
        ));
    }

    #[test]
    fn sidecars_verify_against_returned_root() {
        let data = payload_data(4096);
        let block_root = Hash256::repeat_byte(3);
        let (sidecars, root) = build_payload_column_sidecars(&data, block_root, Slot::new(7), &KZG)
            .expect("should build sidecars");

        assert_eq!(sidecars.len(), E::number_of_columns());
        for sidecar in &sidecars {
            assert!(
                sidecar.verify_inclusion_proof(root),
                "column {} should verify",
                sidecar.index
            );
            assert_eq!(sidecar.beacon_block_root, block_root);
            assert_eq!(sidecar.slot, Slot::new(7));
        }
    }

    #[test]
    fn recovery_from_half_the_columns_reproduces_everything() {
        let data = payload_data(4096);
        let block_root = Hash256::repeat_byte(3);
        let slot = Slot::new(7);
        let (sidecars, root) = build_payload_column_sidecars(&data, block_root, slot, &KZG)
            .expect("should build sidecars");

        // Keep every other column, so exactly half survive.
        let half = sidecars
            .iter()
            .filter(|s| s.index % 2 == 0)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(half.len(), E::number_of_columns() / 2);

        let recovered = recover_payload_columns(&half, &KZG).expect("should recover");

        // Cell-for-cell equality is what pins the recovered ordering to `compute_cells`.
        assert_eq!(recovered, sidecars);
        for sidecar in &recovered {
            assert!(sidecar.verify_inclusion_proof(root));
        }
    }

    #[test]
    fn recovered_columns_decode_to_the_original_payload() {
        let usable = usable_bytes_per_blob::<E>().expect("no overflow");
        let data = payload_data(usable + 128);
        let (sidecars, _) = build_payload_column_sidecars(&data, Hash256::ZERO, Slot::new(1), &KZG)
            .expect("should build sidecars");

        let half = sidecars
            .iter()
            .filter(|s| s.index >= (E::number_of_columns() as u64) / 2)
            .cloned()
            .collect::<Vec<_>>();
        let recovered = recover_payload_columns(&half, &KZG).expect("should recover");

        assert_eq!(
            payload_columns_to_payload_data(&recovered).expect("should decode"),
            data
        );
    }

    #[test]
    fn decoding_requires_the_full_column_set() {
        let data = payload_data(1024);
        let (sidecars, _) = build_payload_column_sidecars(&data, Hash256::ZERO, Slot::new(1), &KZG)
            .expect("should build sidecars");

        // A full set decodes without any recovery step.
        assert_eq!(
            payload_columns_to_payload_data(&sidecars).expect("should decode"),
            data
        );

        // A partial set does not — recovery must run first.
        let partial = sidecars
            .into_iter()
            .take(E::number_of_columns() - 1)
            .collect::<Vec<_>>();
        assert!(matches!(
            payload_columns_to_payload_data(&partial),
            Err(PayloadColumnError::InsufficientColumns { .. })
        ));
    }

    #[test]
    fn recovery_rejects_too_few_columns() {
        let data = payload_data(1024);
        let (sidecars, _) = build_payload_column_sidecars(&data, Hash256::ZERO, Slot::new(1), &KZG)
            .expect("should build sidecars");

        let too_few = sidecars
            .into_iter()
            .take(E::number_of_columns() / 2 - 1)
            .collect::<Vec<_>>();
        assert!(matches!(
            recover_payload_columns(&too_few, &KZG),
            Err(PayloadColumnError::InsufficientColumns { .. })
        ));
    }

    #[test]
    fn compute_root_matches_build() {
        let data = payload_data(2048);
        let (_, root) = build_payload_column_sidecars(&data, Hash256::ZERO, Slot::new(1), &KZG)
            .expect("should build sidecars");
        assert_eq!(
            compute_payload_columns_root(&data, &KZG).expect("should compute root"),
            root,
            "the root must not depend on block root or slot"
        );
    }
}

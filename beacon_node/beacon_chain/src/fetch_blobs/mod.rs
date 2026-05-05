//! This module implements an optimisation to fetch blobs via JSON-RPC from the EL.
//! If a blob has already been seen in the public mempool, then it is often unnecessary to wait for
//! it to arrive on P2P gossip. This PR uses a new JSON-RPC method (`engine_getBlobsV1`) which
//! allows the CL to load the blobs quickly from the EL's blob pool.
//!
//! Once the node fetches the blobs from EL, it then publishes the remaining blobs that it hasn't seen
//! on P2P gossip to the network. From PeerDAS onwards, together with the increase in blob count,
//! broadcasting blobs requires a much higher bandwidth, and is only done by high capacity
//! supernodes.

mod fetch_blobs_beacon_adapter;
#[cfg(test)]
mod tests;

use crate::blob_verification::{GossipBlobError, KzgVerifiedBlob};
use crate::data_column_verification::{
    KzgVerifiedCustodyDataColumn, KzgVerifiedCustodyPartialDataColumn, KzgVerifiedPartialDataColumn,
};
#[cfg_attr(test, double)]
use crate::fetch_blobs::fetch_blobs_beacon_adapter::FetchBlobsBeaconAdapter;
use crate::kzg_utils::blobs_to_partial_data_columns;
use crate::observed_data_sidecars::ObservationKey;
use crate::{
    AvailabilityProcessingStatus, BeaconChain, BeaconChainError, BeaconChainTypes, BlockError,
    metrics,
};
use execution_layer::Error as ExecutionLayerError;
use execution_layer::json_structures::{BlobAndProofV1, BlobAndProofV2, BlobAndProofV3, JsonBlobCellsAndProofsV1, custody_columns_to_bitarray};
use metrics::{TryExt, inc_counter};
#[cfg(test)]
use mockall_double::double;
use slot_clock::timestamp_now;
use state_processing::per_block_processing::deneb::kzg_commitment_to_versioned_hash;
use std::sync::Arc;
use tracing::{debug, instrument, warn};
use types::data::{BlobSidecarError, ColumnIndex, DataColumnSidecarError, PartialDataColumnHeader};
use types::{BeaconStateError, BlobSidecar, EthSpec, ExecutionBlockHash, Hash256, KzgProof, VersionedHash};

/// Result from engine get blobs to be passed onto `DataAvailabilityChecker` and published to the
/// gossip network. The blobs / data columns have not been marked as observed yet, as they may not
/// be published immediately.
#[derive(Debug)]
pub enum EngineGetBlobsOutput<T: BeaconChainTypes> {
    Blobs(Vec<KzgVerifiedBlob<T::EthSpec>>),
    /// A filtered list of custody data columns to be imported into the `DataAvailabilityChecker`.
    CustodyColumns(Vec<KzgVerifiedCustodyDataColumn<T::EthSpec>>),
}

#[derive(Debug)]
pub enum FetchEngineBlobError {
    BeaconStateError(BeaconStateError),
    BeaconChainError(Box<BeaconChainError>),
    BlobProcessingError(BlockError),
    BlobSidecarError(BlobSidecarError),
    DataColumnSidecarError(DataColumnSidecarError),
    ExecutionLayerMissing,
    InternalError(String),
    GossipBlob(GossipBlobError),
    KzgError(kzg::Error),
    RequestFailed(ExecutionLayerError),
    RuntimeShutdown,
    TokioJoin(tokio::task::JoinError),
}

/// Fetches blobs from the EL mempool and processes them. It also broadcasts unseen blobs or
/// data columns (PeerDAS onwards) to the network, using the supplied `publish_fn`.
#[instrument(skip_all)]
pub async fn fetch_and_process_engine_blobs<T: BeaconChainTypes>(
    chain: Arc<BeaconChain<T>>,
    block_root: Hash256,
    header: Arc<PartialDataColumnHeader<T::EthSpec>>,
    custody_columns: &[ColumnIndex],
    publish_fn: impl Fn(EngineGetBlobsOutput<T>) + Send + 'static,
) -> Result<Option<AvailabilityProcessingStatus>, FetchEngineBlobError> {
    fetch_and_process_engine_blobs_inner(
        FetchBlobsBeaconAdapter::new(chain),
        block_root,
        header,
        custody_columns,
        publish_fn,
    )
    .await
}

/// Internal implementation of fetch blobs, which uses `FetchBlobsBeaconAdapter` instead of
/// `BeaconChain` for better testability.
async fn fetch_and_process_engine_blobs_inner<T: BeaconChainTypes>(
    chain_adapter: FetchBlobsBeaconAdapter<T>,
    block_root: Hash256,
    header: Arc<PartialDataColumnHeader<T::EthSpec>>,
    custody_columns: &[ColumnIndex],
    publish_fn: impl Fn(EngineGetBlobsOutput<T>) + Send + 'static,
) -> Result<Option<AvailabilityProcessingStatus>, FetchEngineBlobError> {
    let versioned_hashes = header
        .kzg_commitments
        .iter()
        .map(kzg_commitment_to_versioned_hash)
        .collect::<Vec<_>>();
    if versioned_hashes.is_empty() {
        debug!("Fetch blobs not triggered - none required");
        return Ok(None);
    };

    debug!(
        num_expected_blobs = versioned_hashes.len(),
        "Fetching blobs from the EL"
    );

    if chain_adapter
        .spec()
        .is_peer_das_enabled_for_epoch(header.slot().epoch(T::EthSpec::slots_per_epoch()))
    {
        // Try V4 first if the EL supports it — fetches only custody cells instead of full blobs.
        let supports_v4 = chain_adapter.supports_get_blobs_v4().await?;
        if supports_v4 {
            if let Some(exec_block_hash) = chain_adapter.get_execution_block_hash(&block_root)? {
                return fetch_and_process_blobs_v4(
                    chain_adapter,
                    block_root,
                    header,
                    exec_block_hash,
                    custody_columns,
                    publish_fn,
                )
                .await;
            }
        }
        fetch_and_process_blobs_v2_or_v3(
            chain_adapter,
            block_root,
            header,
            versioned_hashes,
            custody_columns,
            publish_fn,
        )
        .await
    } else {
        fetch_and_process_blobs_v1(
            chain_adapter,
            block_root,
            &header,
            versioned_hashes,
            publish_fn,
        )
        .await
    }
}

#[instrument(skip_all, level = "debug")]
async fn fetch_and_process_blobs_v1<T: BeaconChainTypes>(
    chain_adapter: FetchBlobsBeaconAdapter<T>,
    block_root: Hash256,
    header: &PartialDataColumnHeader<T::EthSpec>,
    versioned_hashes: Vec<VersionedHash>,
    publish_fn: impl Fn(EngineGetBlobsOutput<T>) + Send + Sized,
) -> Result<Option<AvailabilityProcessingStatus>, FetchEngineBlobError> {
    let num_expected_blobs = versioned_hashes.len();
    metrics::observe(&metrics::BLOBS_FROM_EL_EXPECTED, num_expected_blobs as f64);
    debug!(num_expected_blobs, "Fetching blobs from the EL");
    let response = chain_adapter
        .get_blobs_v1(versioned_hashes)
        .await
        .inspect_err(|_| {
            inc_counter(&metrics::BLOBS_FROM_EL_ERROR_TOTAL);
        })?;

    let num_fetched_blobs = response.iter().filter(|opt| opt.is_some()).count();
    metrics::observe(&metrics::BLOBS_FROM_EL_RECEIVED, num_fetched_blobs as f64);

    if num_fetched_blobs == 0 {
        debug!(num_expected_blobs, "No blobs fetched from the EL");
        inc_counter(&metrics::BLOBS_FROM_EL_MISS_TOTAL);
        return Ok(None);
    } else {
        debug!(
            num_expected_blobs,
            num_fetched_blobs, "Received blobs from the EL"
        );
        inc_counter(&metrics::BLOBS_FROM_EL_HIT_TOTAL);
    }

    if chain_adapter.fork_choice_contains_block(&block_root) {
        // Avoid computing sidecars if the block has already been imported.
        debug!(
            info = "block has already been imported",
            "Ignoring EL blobs response"
        );
        return Ok(None);
    }

    let mut blob_sidecar_list = build_blob_sidecars(header, response)?;

    let observation_key = ObservationKey::new_proposer_key(
        header.signed_block_header.message.proposer_index,
        header.slot(),
    );

    if let Some(observed_blobs) = chain_adapter.blobs_known_for_observation_key(observation_key) {
        blob_sidecar_list.retain(|blob| !observed_blobs.contains(&blob.blob_index()));
        if blob_sidecar_list.is_empty() {
            debug!(
                info = "blobs have already been seen on gossip",
                "Ignoring EL blobs response"
            );
            return Ok(None);
        }
    }

    if let Some(known_blobs) = chain_adapter.cached_blob_indexes(&block_root) {
        blob_sidecar_list.retain(|blob| !known_blobs.contains(&blob.blob_index()));
        if blob_sidecar_list.is_empty() {
            debug!(
                info = "blobs have already been imported into data availability checker",
                "Ignoring EL blobs response"
            );
            return Ok(None);
        }
    }

    // Up until this point we have not observed the blobs in the gossip cache, which allows them to
    // arrive independently while this function is running. In `publish_fn` we will observe them
    // and then publish any blobs that had not already been observed.
    publish_fn(EngineGetBlobsOutput::Blobs(blob_sidecar_list.clone()));

    let availability_processing_status = chain_adapter
        .process_engine_blobs(
            header.slot(),
            block_root,
            EngineGetBlobsOutput::Blobs(blob_sidecar_list),
        )
        .await?;

    Ok(Some(availability_processing_status))
}

#[instrument(skip_all, level = "debug")]
async fn fetch_and_process_blobs_v2_or_v3<T: BeaconChainTypes>(
    chain_adapter: FetchBlobsBeaconAdapter<T>,
    block_root: Hash256,
    header: Arc<PartialDataColumnHeader<T::EthSpec>>,
    versioned_hashes: Vec<VersionedHash>,
    custody_columns_indices: &[ColumnIndex],
    publish_fn: impl Fn(EngineGetBlobsOutput<T>) + Send + 'static,
) -> Result<Option<AvailabilityProcessingStatus>, FetchEngineBlobError> {
    let num_expected_blobs = versioned_hashes.len();
    let slot = header.slot();

    metrics::observe(&metrics::BLOBS_FROM_EL_EXPECTED, num_expected_blobs as f64);

    let get_blobs_v3 = chain_adapter.supports_get_blobs_v3().await?;
    let response = if get_blobs_v3 {
        debug!(num_expected_blobs, "Fetching available blobs from the EL");
        // Track request count and duration for standardized metrics
        inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V3_REQUESTS_TOTAL);
        let _timer =
            metrics::start_timer(&metrics::BEACON_ENGINE_GET_BLOBS_V3_REQUEST_DURATION_SECONDS);

        chain_adapter
            .get_blobs_v3(versioned_hashes)
            .await
            .inspect_err(|_| {
                inc_counter(&metrics::BLOBS_FROM_EL_ERROR_TOTAL);
            })?
    } else {
        debug!(num_expected_blobs, "Fetching all blobs from the EL");

        // Track request count and duration for standardized metrics
        inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V2_REQUESTS_TOTAL);
        let _timer =
            metrics::start_timer(&metrics::BEACON_ENGINE_GET_BLOBS_V2_REQUEST_DURATION_SECONDS);

        let response = chain_adapter
            .get_blobs_v2(versioned_hashes)
            .await
            .inspect_err(|_| {
                inc_counter(&metrics::BLOBS_FROM_EL_ERROR_TOTAL);
            })?;

        // Track successful response
        inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V2_RESPONSES_TOTAL);

        response.map(|vec| vec.into_iter().map(Some).collect())
    };

    let Some(blobs_and_proofs) = response else {
        debug!(num_expected_blobs, "No blobs fetched from the EL");
        inc_counter(&metrics::BLOBS_FROM_EL_MISS_TOTAL);
        return Ok(None);
    };

    let num_fetched_blobs = blobs_and_proofs.iter().filter(|opt| opt.is_some()).count();
    metrics::observe(&metrics::BLOBS_FROM_EL_RECEIVED, num_fetched_blobs as f64);

    if num_fetched_blobs != num_expected_blobs {
        if !get_blobs_v3 {
            // This scenario is not supposed to happen if the EL is spec compliant.
            // It should either return all requested blobs or none, but NOT partial responses.
            // If we attempt to compute columns with partial blobs, we'd end up with invalid columns.
            warn!(
                num_fetched_blobs,
                num_expected_blobs, "The EL did not return all requested blobs"
            );
            inc_counter(&metrics::BLOBS_FROM_EL_MISS_TOTAL);
            return Ok(None);
        } else {
            inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V3_PARTIAL_RESPONSES_TOTAL);
            debug!(
                num_fetched_blobs,
                num_expected_blobs, "Blobs partially received from the EL"
            );
        }
    } else {
        debug!(num_fetched_blobs, "All blobs received from the EL");
        inc_counter(&metrics::BLOBS_FROM_EL_HIT_TOTAL);
        if get_blobs_v3 {
            inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V3_COMPLETE_RESPONSES_TOTAL);
        }
    }

    if chain_adapter.fork_choice_contains_block(&block_root) {
        // Avoid computing columns if the block has already been imported.
        debug!(
            info = "block has already been imported",
            "Ignoring EL blobs response"
        );
        return Ok(None);
    }

    let chain_adapter = Arc::new(chain_adapter);
    let custody_columns_to_import = compute_custody_columns_to_import(
        &chain_adapter,
        block_root,
        &header,
        blobs_and_proofs,
        custody_columns_indices,
    )
    .await?;

    if custody_columns_to_import.is_empty() {
        debug!(
            info = "No new data columns to import",
            "Ignoring EL blobs response"
        );
        return Ok(None);
    }

    let full_columns = match chain_adapter.partial_assembler() {
        Some(assembler) => {
            // Initialize the partial assembler with the columns from the engine and return any full
            // columns for publishing
            assembler
                .merge_partials(block_root, custody_columns_to_import, header)
                .ok_or_else(|| {
                    FetchEngineBlobError::InternalError(
                        "Failed to merge partials into assembler".to_string(),
                    )
                })?
                .full_columns
        }
        None => {
            // Partial columns are disabled, so let's try to directly convert the columns we got
            // from the EL into full columns.
            custody_columns_to_import
                .into_iter()
                .filter_map(|col| col.try_into_full(&header))
                .collect()
        }
    };

    // Publish complete columns
    if !full_columns.is_empty() {
        publish_fn(EngineGetBlobsOutput::CustodyColumns(full_columns.clone()));
    }
    // We publish all partials at the calling site, regardless of result, as previous publishs
    // have been blocked, waiting for the results of this call

    // Process complete columns through DA checker
    let availability_processing_status = if !full_columns.is_empty() {
        chain_adapter
            .process_engine_blobs(
                slot,
                block_root,
                EngineGetBlobsOutput::CustodyColumns(full_columns),
            )
            .await?
    } else {
        // No complete columns yet, still missing components
        AvailabilityProcessingStatus::MissingComponents(slot, block_root)
    };

    Ok(Some(availability_processing_status))
}

/// Offload the data column computation to a blocking task to avoid holding up the async runtime.
async fn compute_custody_columns_to_import<T: BeaconChainTypes>(
    chain_adapter: &Arc<FetchBlobsBeaconAdapter<T>>,
    block_root: Hash256,
    header: &PartialDataColumnHeader<T::EthSpec>,
    blobs_and_proofs: Vec<BlobAndProofV3<T::EthSpec>>,
    custody_columns_indices: &[ColumnIndex],
) -> Result<Vec<KzgVerifiedCustodyPartialDataColumn<T::EthSpec>>, FetchEngineBlobError> {
    let kzg = chain_adapter.kzg().clone();
    let spec = chain_adapter.spec().clone();
    let chain_adapter_cloned = chain_adapter.clone();
    let custody_columns_indices = custody_columns_indices.to_vec();
    let header = header.clone();
    chain_adapter
        .executor()
        .spawn_blocking_handle(
            move || {
                let mut timer = metrics::start_timer_vec(
                    &metrics::DATA_COLUMN_SIDECAR_COMPUTATION,
                    &[&blobs_and_proofs.len().to_string()],
                );

                let blob_and_proof_refs = blobs_and_proofs
                    .iter()
                    .map(|option| {
                        option
                            .as_ref()
                            .map(|BlobAndProofV2 { blob, proofs }| (blob, proofs.as_ref()))
                    })
                    .collect::<Vec<_>>();
                let data_columns_result =
                    blobs_to_partial_data_columns(blob_and_proof_refs, &header, &kzg, &spec)
                        .discard_timer_on_break(&mut timer);
                drop(timer);

                // This filtering ensures we only import and publish the custody columns.
                // `DataAvailabilityChecker` requires a strict match on custody columns count to
                // consider a block available.
                let mut custody_columns = data_columns_result
                    .map(|data_columns| {
                        data_columns
                            .into_iter()
                            .filter(|col| custody_columns_indices.contains(&col.index))
                            .map(|col| {
                                KzgVerifiedCustodyPartialDataColumn::from_asserted_custody(
                                    KzgVerifiedPartialDataColumn::from_execution_verified(
                                        Arc::new(col),
                                    ),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .map_err(FetchEngineBlobError::DataColumnSidecarError)?;

                // Only consider columns that are not already observed on gossip.
                let observation_key =
                    ObservationKey::from_partial_column_header(&header, block_root, &spec);

                if let Some(observed_columns) =
                    chain_adapter_cloned.data_column_known_for_observation_key(observation_key)
                {
                    custody_columns.retain(|col| !observed_columns.contains(&col.index()));
                    if custody_columns.is_empty() {
                        return Ok(vec![]);
                    }
                }

                // Only consider columns that are not already known to data availability.
                if let Some(known_columns) =
                    chain_adapter_cloned.cached_data_column_indexes(&block_root)
                {
                    custody_columns.retain(|col| !known_columns.contains(&col.index()));
                    if custody_columns.is_empty() {
                        return Ok(vec![]);
                    }
                }

                Ok(custody_columns)
            },
            "compute_custody_columns_to_import",
        )
        .ok_or(FetchEngineBlobError::RuntimeShutdown)?
        .await
        .map_err(FetchEngineBlobError::TokioJoin)?
}

fn build_blob_sidecars<E: EthSpec>(
    header: &PartialDataColumnHeader<E>,
    response: Vec<Option<BlobAndProofV1<E>>>,
) -> Result<Vec<KzgVerifiedBlob<E>>, FetchEngineBlobError> {
    let mut sidecars = vec![];
    for (index, blob_and_proof) in response
        .into_iter()
        .enumerate()
        .filter_map(|(index, opt_blob)| Some((index, opt_blob?)))
    {
        let blob_sidecar = BlobSidecar::new_with_existing_proof(
            index,
            blob_and_proof.blob,
            header.clone(),
            blob_and_proof.proof,
        )
        .map_err(FetchEngineBlobError::BlobSidecarError)?;

        sidecars.push(KzgVerifiedBlob::from_execution_verified(
            Arc::new(blob_sidecar),
            timestamp_now(),
        ));
    }

    Ok(sidecars)
}

/// Fetches cells for custody columns from the EL via `engine_getBlobsV4` and processes them.
///
/// Unlike V2/V3 which fetch full blobs and compute all 128 columns locally, V4 requests only the
/// cells the node custodies, saving bandwidth and computation.
#[instrument(skip_all, level = "debug")]
async fn fetch_and_process_blobs_v4<T: BeaconChainTypes>(
    chain_adapter: FetchBlobsBeaconAdapter<T>,
    block_root: Hash256,
    header: Arc<PartialDataColumnHeader<T::EthSpec>>,
    exec_block_hash: ExecutionBlockHash,
    custody_columns_indices: &[ColumnIndex],
    publish_fn: impl Fn(EngineGetBlobsOutput<T>) + Send + 'static,
) -> Result<Option<AvailabilityProcessingStatus>, FetchEngineBlobError> {
    let slot = header.slot();
    let num_expected_blobs = header.kzg_commitments.len();

    if num_expected_blobs == 0 {
        debug!("Fetch blobs V4 not triggered - none required");
        return Ok(None);
    }

    metrics::observe(&metrics::BLOBS_FROM_EL_EXPECTED, num_expected_blobs as f64);
    inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V4_REQUESTS_TOTAL);
    let _timer = metrics::start_timer(&metrics::BEACON_ENGINE_GET_BLOBS_V4_REQUEST_DURATION_SECONDS);

    let cell_index_bitarray = custody_columns_to_bitarray(custody_columns_indices);

    debug!(
        num_expected_blobs,
        num_custody_columns = custody_columns_indices.len(),
        "Fetching custody cells from the EL via getBlobsV4"
    );

    let response = chain_adapter
        .get_blobs_v4(exec_block_hash, cell_index_bitarray)
        .await
        .inspect_err(|_| {
            inc_counter(&metrics::BLOBS_FROM_EL_ERROR_TOTAL);
        })?;

    let Some(blobs_cells_and_proofs) = response else {
        debug!(num_expected_blobs, "No blobs fetched from the EL (V4)");
        inc_counter(&metrics::BLOBS_FROM_EL_MISS_TOTAL);
        return Ok(None);
    };

    let num_fetched = blobs_cells_and_proofs.len();
    metrics::observe(&metrics::BLOBS_FROM_EL_RECEIVED, num_fetched as f64);

    if num_fetched == 0 {
        debug!(num_expected_blobs, "No cells fetched from the EL (V4)");
        inc_counter(&metrics::BLOBS_FROM_EL_MISS_TOTAL);
        return Ok(None);
    }

    if num_fetched != num_expected_blobs {
        debug!(
            num_fetched,
            num_expected_blobs, "Partial cells received from the EL (V4)"
        );
        inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V4_PARTIAL_RESPONSES_TOTAL);
    } else {
        debug!(num_fetched, "All cell blobs received from the EL (V4)");
        inc_counter(&metrics::BLOBS_FROM_EL_HIT_TOTAL);
        inc_counter(&metrics::BEACON_ENGINE_GET_BLOBS_V4_COMPLETE_RESPONSES_TOTAL);
    }

    if chain_adapter.fork_choice_contains_block(&block_root) {
        debug!(
            info = "block has already been imported",
            "Ignoring EL blobs response (V4)"
        );
        return Ok(None);
    }

    // Convert cells+proofs from V4 response into DataColumnSidecars
    let chain_adapter = Arc::new(chain_adapter);
    let custody_columns_to_import = compute_custody_columns_from_cells(
        &chain_adapter,
        block_root,
        &header,
        blobs_cells_and_proofs,
        custody_columns_indices,
    )
    .await?;

    if custody_columns_to_import.is_empty() {
        debug!(
            info = "No new data columns to import",
            "Ignoring EL blobs response (V4)"
        );
        return Ok(None);
    }

    let full_columns = match chain_adapter.partial_assembler() {
        Some(assembler) => assembler
            .merge_partials(block_root, custody_columns_to_import, header.clone())
            .ok_or_else(|| {
                FetchEngineBlobError::InternalError(
                    "Failed to merge partials into assembler".to_string(),
                )
            })?
            .full_columns,
        None => custody_columns_to_import
            .into_iter()
            .filter_map(|col| col.try_into_full(&header))
            .collect(),
    };

    if !full_columns.is_empty() {
        publish_fn(EngineGetBlobsOutput::CustodyColumns(full_columns.clone()));
    }

    let availability_processing_status = if !full_columns.is_empty() {
        chain_adapter
            .process_engine_blobs(
                slot,
                block_root,
                EngineGetBlobsOutput::CustodyColumns(full_columns),
            )
            .await?
    } else {
        AvailabilityProcessingStatus::MissingComponents(slot, block_root)
    };

    Ok(Some(availability_processing_status))
}

/// Convert V4 response (cells+proofs per blob) into partial data column sidecars for custody columns.
async fn compute_custody_columns_from_cells<T: BeaconChainTypes>(
    chain_adapter: &Arc<FetchBlobsBeaconAdapter<T>>,
    block_root: Hash256,
    header: &PartialDataColumnHeader<T::EthSpec>,
    blobs_cells_and_proofs: Vec<JsonBlobCellsAndProofsV1<T::EthSpec>>,
    custody_columns_indices: &[ColumnIndex],
) -> Result<Vec<KzgVerifiedCustodyPartialDataColumn<T::EthSpec>>, FetchEngineBlobError> {
    let spec = chain_adapter.spec().clone();
    let chain_adapter_cloned = chain_adapter.clone();
    let custody_columns_indices = custody_columns_indices.to_vec();
    let header = header.clone();
    let bitarray = custody_columns_to_bitarray(&custody_columns_indices);

    chain_adapter
        .executor()
        .spawn_blocking_handle(
            move || {
                use types::data::{Cell, DataColumn};
                use ssz_types::VariableList;

                let mut columns: Vec<KzgVerifiedCustodyPartialDataColumn<T::EthSpec>> = Vec::new();

                // Check which columns are already known
                let observation_key =
                    ObservationKey::from_partial_column_header(&header, block_root, &spec);

                let observed_columns = chain_adapter_cloned
                    .data_column_known_for_observation_key(observation_key);
                let known_columns = chain_adapter_cloned.cached_data_column_indexes(&block_root);

                // Pre-compute dense index mapping: for each column index, what is its position
                // in the dense array (position among set bits).
                let dense_index_map: Vec<(ColumnIndex, usize)> = custody_columns_indices
                    .iter()
                    .enumerate()
                    .map(|(dense_idx, &col_idx)| (col_idx, dense_idx))
                    .collect();

                for (col_idx, dense_idx) in dense_index_map {
                    // Skip if already observed on gossip
                    if let Some(ref observed) = observed_columns {
                        if observed.contains(&col_idx) {
                            continue;
                        }
                    }
                    // Skip if already in DA checker cache
                    if let Some(ref known) = known_columns {
                        if known.contains(&col_idx) {
                            continue;
                        }
                    }

                    // Collect cells and proofs for this column index across all blobs.
                    // The V4 response has one BlobCellsAndProofsV1 per blob; each has dense arrays
                    // indexed by position among set bits in cellIndexBitarray.
                    let mut column_cells: Vec<Cell<T::EthSpec>> = Vec::new();
                    let mut column_proofs: Vec<KzgProof> = Vec::new();
                    let mut num_present = 0usize;

                    for blob_cells in &blobs_cells_and_proofs {
                        let cell = blob_cells.blob_cells.get(dense_idx).and_then(|c| c.as_ref());
                        let proof = blob_cells.proofs.get(dense_idx).and_then(|p| p.as_ref());

                        match (cell, proof) {
                            (Some(cell), Some(proof)) => {
                                column_cells.push(cell.clone());
                                column_proofs.push(*proof);
                                num_present += 1;
                            }
                            _ => {
                                // Cell or proof not available for this blob — push placeholder.
                                // A full column requires cells for ALL blobs, so this column will
                                // remain partial.
                            }
                        }
                    }

                    if column_cells.is_empty() {
                        continue;
                    }

                    let column: DataColumn<T::EthSpec> = VariableList::try_from(column_cells)
                        .map_err(|e| {
                            FetchEngineBlobError::InternalError(format!(
                                "Failed to create column: {:?}",
                                e
                            ))
                        })?;
                    let proofs: VariableList<KzgProof, <T::EthSpec as EthSpec>::MaxBlobCommitmentsPerBlock> = VariableList::try_from(column_proofs).map_err(|e| {
                        FetchEngineBlobError::InternalError(format!(
                            "Failed to create proofs: {:?}",
                            e
                        ))
                    })?;

                    // TODO: Construct PartialDataColumnSidecar from cells and proofs.
                    // This requires implementing construction logic that builds a sidecar
                    // directly from pre-computed cells rather than from full blobs.
                    // The sidecar needs to track which cells are present via CellBitmap.
                    let _ = (column, proofs, num_present, bitarray);
                }

                Ok(columns)
            },
            "compute_custody_columns_from_cells_v4",
        )
        .ok_or(FetchEngineBlobError::RuntimeShutdown)?
        .await
        .map_err(FetchEngineBlobError::TokioJoin)?
}

/// Get bit at position `i` from a 16-byte (128-bit) bitarray.
fn bitarray_get(bitarray: &[u8; 16], i: usize) -> bool {
    if i >= 128 {
        return false;
    }
    (bitarray[i / 8] >> (i % 8)) & 1 == 1
}

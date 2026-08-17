//! Identifies each payload column subnet by an integer identifier.
//!
//! Introduced in EIP-8142 (Block-in-Blobs). Unlike data column subnets, every node subscribes to
//! every payload column subnet, so there is one subnet per column and no custody-based selection.
use std::{
    fmt::{self, Display},
    ops::{Deref, DerefMut},
};

use serde::{Deserialize, Serialize};

use crate::{core::EthSpec, data::ColumnIndex};

#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PayloadColumnSubnetId(#[serde(with = "serde_utils::quoted_u64")] u64);

impl fmt::Debug for PayloadColumnSubnetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl PayloadColumnSubnetId {
    pub fn new(id: u64) -> Self {
        id.into()
    }

    /// There is exactly one payload column subnet per column, so the subnet id and the column index
    /// are the same value.
    pub fn from_column_index(column_index: ColumnIndex) -> Self {
        column_index.into()
    }
}

impl Display for PayloadColumnSubnetId {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.0)
    }
}

impl Deref for PayloadColumnSubnetId {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PayloadColumnSubnetId {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<u64> for PayloadColumnSubnetId {
    fn from(x: u64) -> Self {
        Self(x)
    }
}

impl From<PayloadColumnSubnetId> for u64 {
    fn from(val: PayloadColumnSubnetId) -> Self {
        val.0
    }
}

impl From<&PayloadColumnSubnetId> for u64 {
    fn from(val: &PayloadColumnSubnetId) -> Self {
        val.0
    }
}

/// All payload column subnets. Every node subscribes to all of them.
pub fn all_payload_column_sidecar_subnets<E: EthSpec>()
-> impl Iterator<Item = PayloadColumnSubnetId> {
    (0..E::number_of_columns() as u64).map(PayloadColumnSubnetId::new)
}

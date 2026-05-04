//! Manifest types for native VoxCPM model bundles.

#[derive(Debug, Clone)]
pub struct BundleManifest;

#[derive(Debug, Clone)]
pub struct ComponentFiles;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    VoxCpm05,
    VoxCpm15,
    VoxCpm2,
}


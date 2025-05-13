//! Memory-efficient operations and views for large arrays.
//!
//! This module provides utilities for working with large arrays efficiently by:
//! - Using chunk-wise processing to reduce memory requirements
//! - Creating memory-efficient views that avoid full data copies
//! - Implementing lazy evaluation and operation fusion for improved performance
//! - Supporting out-of-core processing for data that doesn't fit in RAM
//! - Smart prefetching for improved performance with predictable access patterns

mod adaptive_chunking;
mod chunked;
mod compressed_memmap;
mod fusion;
mod lazy_array;
mod memmap;
mod memmap_chunks;
mod memmap_slice;
mod out_of_core;
mod prefetch;
mod adaptive_prefetch;
mod pattern_recognition;
mod resource_aware;
mod cross_file_prefetch;
mod validation;
mod views;
mod zerocopy;

pub use adaptive_chunking::{
    AdaptiveChunking, AdaptiveChunkingBuilder, AdaptiveChunkingParams, AdaptiveChunkingResult,
};
pub use chunked::{
    chunk_wise_binary_op, chunk_wise_op, chunk_wise_reduce, ChunkedArray, ChunkingStrategy,
    OPTIMAL_CHUNK_SIZE,
};
pub use compressed_memmap::{
    CompressedFileMetadata, CompressedMemMapBuilder, CompressedMemMappedArray, CompressionAlgorithm,
};
pub use fusion::{register_fusion, FusedOp, OpFusion};
pub use lazy_array::{evaluate, LazyArray, LazyOp, LazyOpKind};
pub use memmap::{create_mmap, create_temp_mmap, open_mmap, AccessMode, MemoryMappedArray};
#[cfg(feature = "parallel")]
pub use memmap_chunks::MemoryMappedChunksParallel;
pub use memmap_chunks::{ChunkIter, MemoryMappedChunkIter, MemoryMappedChunks};
pub use memmap_slice::{MemoryMappedSlice, MemoryMappedSlicing};
pub use out_of_core::{create_disk_array, load_chunks, DiskBackedArray, OutOfCoreArray};
pub use prefetch::{
    AccessPattern, PrefetchConfig, PrefetchConfigBuilder, PrefetchStats,
    PrefetchingCompressedArray, Prefetching,
};
pub use adaptive_prefetch::{
    PrefetchStrategy, AdaptivePatternTracker, AdaptivePrefetchConfig,
    AdaptivePrefetchConfigBuilder, PatternTrackerFactory,
};
pub use pattern_recognition::{
    ComplexPattern, PatternRecognizer, PatternRecognitionConfig, Confidence,
    RecognizedPattern,
};
pub use resource_aware::{
    ResourceMonitor, ResourceSnapshot, ResourceSummary, ResourceType,
    ResourceAwareConfig, ResourceAwareConfigBuilder, ResourceAwarePrefetcher,
};
pub use cross_file_prefetch::{
    CrossFilePrefetchManager, DatasetPrefetcher, DatasetId, DataAccess, AccessType,
    CrossFilePrefetchConfig, CrossFilePrefetchConfigBuilder, CrossFilePrefetchRegistry,
};
pub use views::{diagonal_view, transpose_view, view_as, view_mut_as, ArrayView, ViewMut};
pub use zerocopy::{ArithmeticOps, BroadcastOps, ZeroCopyOps};

// Re-export commonly used items in a prelude module for convenience
pub mod prelude {
    pub use super::{
        chunk_wise_binary_op, chunk_wise_op, chunk_wise_reduce, create_mmap, create_temp_mmap,
        evaluate, open_mmap, view_as, view_mut_as, AccessMode, AdaptiveChunking,
        AdaptiveChunkingBuilder, ArithmeticOps, ArrayView, BroadcastOps, ChunkIter, ChunkedArray,
        CompressedMemMapBuilder, CompressionAlgorithm, LazyArray, MemoryMappedArray,
        MemoryMappedChunkIter, MemoryMappedChunks, MemoryMappedSlice, MemoryMappedSlicing,
        OutOfCoreArray, PrefetchConfig, PrefetchConfigBuilder, PrefetchingCompressedArray,
        Prefetching, ViewMut, ZeroCopyOps,
        // Advanced prefetching
        AdaptivePatternTracker, AdaptivePrefetchConfig, PrefetchStrategy,
        ComplexPattern, PatternRecognizer, ResourceAwarePrefetcher, ResourceAwareConfig,
        CrossFilePrefetchManager, DatasetPrefetcher, DatasetId,
    };

    #[cfg(feature = "parallel")]
    pub use super::MemoryMappedChunksParallel;
}

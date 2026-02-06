# Changelog

All notable changes to Eric's Semantic Search for ComfyUI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2025-01-23

### Added

- **Core Architecture**
  - Two-stage multimodal retrieval pipeline (embedding + reranking)
  - FAISS-based vector index with persistent storage
  - SQLite metadata database for image tracking
  - Automatic thumbnail generation for fast previews

- **Model Support**
  - Qwen3-VL-Embedding-2B and 8B models
  - Qwen3-VL-Reranker-2B and 8B models
  - Automatic model path resolution (local or HuggingFace download)
  - Configurable attention implementations (sdpa, eager, sage)

- **Model Nodes**
  - `Load Embedding Model` - Load embedding models with resolution and attention options
  - `Load Reranker Model` - Load reranker models for Stage 2 refinement

- **Index Nodes**
  - `Load/Create Index` - Create or load persistent indexes
  - `Add Folder to Index` - Batch index images from folders
  - `Remove Folder from Index` - Remove indexed content
  - `Validate Index` - Check integrity and remove missing files
  - `Get Index Info` - Display index statistics

- **Search Nodes**
  - `Search by Text` - Natural language image search
  - `Search by Image` - Visual similarity search
  - `Rerank Results` - Two-stage reranking for improved precision
  - `Filter by Score` - Threshold-based result filtering
  - `Combine Results` - Merge result sets (union/intersection/concat)

- **Output Nodes**
  - `Preview Results` - Visual thumbnail grid with scores
  - `Load Result Images` - Load full-resolution images
  - `Get Result Paths` - Extract file paths for downstream use

- **Configuration Options**
  - Configurable max_resolution (256x256 to 1536x1536)
  - Score threshold filtering (min_score parameter)
  - Batch size control for indexing
  - Multiple attention implementations

### Technical Notes

- Requires `qwen-vl-utils>=0.0.14` for batch processing support
- FAISS index uses Inner Product (IP) similarity for cosine distance
- Embeddings stored as float32 (converted from bfloat16)
- Compatible with Blackwell (sm120) GPUs using sdpa/eager attention

---

## [1.1.0] - 2025-06-21

### Added

- **Video Support**
  - Full video file indexing with configurable frame extraction
  - `Search by Video` node - use a video as the search query
  - Video thumbnails with play button overlay for visual distinction
  - Support for .mp4, .avi, .mov, .mkv, .webm, .wmv, .flv, .m4v formats
  - Configurable extraction FPS (default: 1.0) and max frames (default: 32)
  - Video metadata tracking (duration, frame count, media type) in database

- **FAISS Index Types**
  - `Flat` index - exact search, 100% recall (default)
  - `IVF-Flat` index - faster approximate search (~95-99% recall)
  - `HNSW` index - very fast graph-based search
  - Index type selection in `Load/Create Index` node
  - `Rebuild Index` node - convert between index types without re-encoding
  - Auto-training buffer for IVF indexes (trains at 1000+ vectors)

- **New Nodes**
  - `Rebuild Index` - convert existing index to different FAISS type
  - `Search by Video` - use video files as search queries

- **Additional Resolution Option**
  - Added 478×478 resolution option for embedding models

### Fixed

- **Critical Reranker Performance Fix** (5x speedup)
  - Removed `do_resize=False` bug that bypassed max_pixels constraint
  - Images now properly resized before processing
  - GPU utilization improved from 17% to near 100%
  - 256×256: ~3s per image (was ~14s)
  - Fixed `aten::slow_conv_dilated3d` fallback on Blackwell GPUs

### Changed

- `Add Folder to Index` now accepts `include_videos` parameter
- Database schema now includes media_type, frame_count, duration_seconds columns
- Automatic schema migration for existing databases

### Technical Notes

- New dependency: OpenCV (cv2) for video processing
- Video frames are averaged into single embedding vector
- IVF index automatically trains when buffer reaches 1000 vectors
- HNSW uses efConstruction=200, M=32 for quality/speed balance

---

## [1.2.0] - 2026-01-31

### Added

- **Index Compaction**
  - `Compact Index` node - remove deleted vectors and reclaim space
  - Tracks deleted vector IDs in `deleted_vectors` table
  - Rebuilds FAISS index keeping only active vectors
  - Reassigns sequential vector IDs with safe two-pass database update
  - Optionally change index type during compaction
  - Shows wasted space percentage in node output
  - `needs_compaction()` method returns (needs_compact, deleted_count, wasted_pct)
  - `get_all_vector_ids()`, `update_vector_ids()` database methods
  - Automatic database vacuum after compaction

### Fixed

- **Deadlock Bug in SemanticIndex**
  - Changed `threading.Lock()` to `threading.RLock()` (reentrant lock)
  - Fixed hang when `compact()` called `save()` while holding lock
  - Affects any method that calls other locking methods internally

### Technical Notes

- `update_vector_ids()` uses two-pass approach to avoid ID collisions:
  1. First pass: offset all IDs by 1,000,000
  2. Second pass: assign final sequential IDs
- Compaction extracts vectors from old FAISS index via `reconstruct()`
- Total nodes: 18 (was 17)

---

## [1.3.0] - 2026-02-01

### Added

- **PDF/Document Support**
  - Full PDF file indexing with page-by-page processing
  - Each PDF page gets its own embedding for precise search
  - `Search by Document` node - use a PDF page as the search query
  - PDF thumbnails with page number badge for multi-page documents
  - Virtual path format: `file.pdf#page=3` for page references
  - Support for .pdf format
  - Configurable settings: PDF_DPI (150), PDF_MAX_PAGES (100)

- **New Nodes**
  - `Search by Document` - search using a PDF page as query
  - Parameters: pdf_path, page_number, top_k, min_score, instruction

- **Updated Nodes**
  - `Add Folder to Index` - new `include_documents` parameter (default: True)
  - Status output now shows PDF count and total pages indexed

- **Database Schema Updates**
  - Added `page_number` column for PDF page tracking
  - Added `parent_document` column for linking pages to parent PDFs
  - Auto-migration for existing databases

### Technical Notes

- Uses PyMuPDF (fitz) for PDF processing (already installed via ComfyUI)
- PDF pages rendered at 150 DPI for embedding, 72 DPI for thumbnails
- Maximum 100 pages per PDF (configurable via PDF_MAX_PAGES)
- Total nodes: 19 (was 18)

---

## [1.4.0] - 2026-01-31

### Added

- **Advanced Search Features**
  - `Search with Exclusion` node - exclude results similar to specified terms
    - Separate `query` and `exclude` inputs (comma-separated exclusion terms)
    - Configurable `exclusion_threshold` (0-1, default 0.3)
    - Two-pass search: gets 3x candidates, then filters by negative similarity
  - `Search Multi-Index` node - search across 1-4 indexes simultaneously
    - Merges results with optional score normalization
    - Deduplicates by file path, keeping highest score
    - `merge_search_results()` static method on SemanticIndex

- **Matryoshka Dimension Reduction**
  - New `embedding_dim` parameter on `Load Embedding Model`
  - Options: Full (4096/2048), 2048, 1024, 512, 256
  - Smaller dimensions = smaller index, faster search, slightly less accuracy
  - Embeddings truncated and re-normalized after generation

- **Dimension Validation**
  - Model-index dimension mismatch detection
  - Clear error messages with guidance on how to fix
  - Validation in AddFolderToIndex and all search nodes

- **New Methods**
  - `SemanticIndex.search_by_text_with_exclusion()` - two-pass exclusion search
  - `SemanticIndex.merge_search_results()` - static method for combining results

### Technical Notes

- Matryoshka representation: first N dimensions of Qwen3-VL embeddings are still meaningful
- FAISS `reconstruct()` used to get candidate vectors for exclusion filtering
- Score normalization for multi-index uses per-index min-max scaling
- Total nodes: 21 (was 19)

---

## [1.5.0] - 2026-02-03

### Added

- **Result Type Filtering**
  - New `result_type` parameter on all search nodes
  - Options: `all`, `images`, `videos`, `documents`, `media` (images+videos)
  - Filter applied post-search, so top_k applies before filtering
  - Added `media_type` field to SearchResult dataclass
  - New `filter_results_by_type()` helper function

- **Nodes Updated**
  - `Search by Text` - added result_type dropdown
  - `Search by Image` - added result_type dropdown
  - `Search by Video` - added result_type dropdown
  - `Search by Document` - added result_type dropdown
  - `Search with Exclusion` - added result_type dropdown
  - `Search Multi-Index` - added result_type dropdown

### Fixed

- **Video OOM During Indexing**
  - Reduced video frame processing resolution to 512×512 (was using full resolution)
  - Added batch processing: 8 frames at a time with CUDA cache clearing
  - Fixes out-of-memory errors on long videos with many frames

- **Video Thumbnail Creation**
  - Fixed `create_video_thumbnail()` receiving wrong parameter type
  - Now correctly receives file path instead of integer

- **Document Skip Check**
  - Documents are now correctly skipped if already indexed
  - Added `is_indexed()` check with media_type parameter

- **Checkpoint Saving**
  - Reduced CHECKPOINT_INTERVAL from 100 to 10 batches
  - More frequent saves during long indexing operations

### Changed

- **Video Frame Extraction**
  - Changed from uniform 8-frame sampling to FPS-based sampling
  - Default: 2 FPS sampling with 120 max frames
  - Better temporal coverage for long videos
  - Configurable via VIDEO_DEFAULT_FPS and VIDEO_MAX_FRAMES in config

### Technical Notes

- VIDEO_FRAME_MAX_PIXELS = 512 * 512 (262,144 pixels)
- CUDA cache cleared between frame batches to prevent OOM
- Result type filtering uses path extension matching

---

## [Unreleased]

### Planned
- Hybrid search with metadata filtering
- Index description and tags metadata

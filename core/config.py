"""
Eric's Semantic Search - Configuration

Description: Configuration constants, paths, and settings for the semantic search system.
             Defines storage locations, model registry, supported file formats, and
             default parameters for indexing and search operations.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

See LICENSE.md for complete license information.
"""

from pathlib import Path
from typing import List

# Base storage location (outside ComfyUI for persistence)
BASE_PATH = Path("H:/semantic_search")

# Subdirectories
INDEXES_PATH = BASE_PATH / "indexes"
MODELS_PATH = BASE_PATH / "models"

# Ensure directories exist
INDEXES_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Model registry: short name -> (local folder, HuggingFace repo ID)
MODEL_REGISTRY = {
    "Qwen3-VL-Embedding-2B": ("Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Embedding-2B"),
    "Qwen3-VL-Embedding-8B": ("Qwen3-VL-Embedding-8B", "Qwen/Qwen3-VL-Embedding-8B"),
    "Qwen3-VL-Reranker-2B": ("Qwen3-VL-Reranker-2B", "Qwen/Qwen3-VL-Reranker-2B"),
    "Qwen3-VL-Reranker-8B": ("Qwen3-VL-Reranker-8B", "Qwen/Qwen3-VL-Reranker-8B"),
}

# Default models (short names)
DEFAULT_EMBEDDING_MODEL = "Qwen3-VL-Embedding-8B"
DEFAULT_RERANKER_MODEL = "Qwen3-VL-Reranker-8B"

# Available model options for node dropdowns
EMBEDDING_MODELS = [
    "Qwen3-VL-Embedding-2B",
    "Qwen3-VL-Embedding-8B",
]

RERANKER_MODELS = [
    "Qwen3-VL-Reranker-2B",
    "Qwen3-VL-Reranker-8B",
]

# Thumbnail settings
DEFAULT_THUMBNAIL_SIZE = 768  # Max dimension (preserves aspect ratio)
THUMBNAIL_QUALITY = 85        # JPEG quality
THUMBNAIL_FORMAT = "JPEG"

# Indexing settings
DEFAULT_BATCH_SIZE = 16       # Images per batch during indexing
CHECKPOINT_INTERVAL = 10      # Save index every N batches (more frequent saves)

# Search settings
DEFAULT_TOP_K = 50            # Default number of results
MAX_TOP_K = 500               # Maximum allowed results

# Supported image formats
SUPPORTED_IMAGE_EXTENSIONS: List[str] = [
    # Common formats
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif",
    # RAW formats (via rawpy)
    ".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".raf",
]

# Supported video formats
SUPPORTED_VIDEO_EXTENSIONS: List[str] = [
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v",
]

# Combined for backwards compatibility
SUPPORTED_EXTENSIONS: List[str] = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS

# Video processing settings
VIDEO_DEFAULT_FPS = 2.0       # Frames per second to sample (2 fps = good temporal coverage)
VIDEO_MAX_FRAMES = 120        # Maximum frames (caps at 60 seconds of 2fps sampling)
VIDEO_MIN_FRAMES = 4          # Minimum frames for meaningful video embedding
VIDEO_FRAME_MAX_PIXELS = 512 * 512  # Max pixels per video frame (512x512, much smaller than images)

# Supported document formats
SUPPORTED_DOCUMENT_EXTENSIONS: List[str] = [
    ".pdf",
]

# PDF processing settings
PDF_DPI = 150                 # DPI for rendering PDF pages (72=native, 150=good quality, 300=print)
PDF_MAX_PAGES = 100           # Maximum pages to index from a single PDF
PDF_THUMBNAIL_DPI = 72        # Lower DPI for thumbnails (faster)

# File hash settings (for change detection)
HASH_CHUNK_SIZE = 65536       # 64KB chunks for hashing large files
USE_FAST_HASH = True          # Use file size + mtime instead of content hash for speed


def resolve_model_path(model_name: str) -> str:
    """
    Resolve model name to path. Checks local MODELS_PATH first,
    raises FileNotFoundError with download instructions if not found.
    
    Args:
        model_name: Short name (e.g. "Qwen3-VL-Embedding-8B") or full path
        
    Returns:
        Local path to the model
        
    Raises:
        FileNotFoundError: If model not found locally, with download instructions
    """
    # Already a valid local path
    if Path(model_name).exists():
        return model_name
    
    # Registered short name
    if model_name in MODEL_REGISTRY:
        local_folder, hf_repo = MODEL_REGISTRY[model_name]
        local_path = MODELS_PATH / local_folder
        
        if (local_path / "config.json").exists():
            print(f"[SemanticSearch] Using local model: {local_path}")
            return str(local_path)
        else:
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {local_path}\n"
                f"Download it with:\n"
                f"  huggingface-cli download {hf_repo} --local-dir {local_path}"
            )
    
    # Unknown model name
    raise FileNotFoundError(
        f"Unknown model: {model_name}\n"
        f"Available models: {list(MODEL_REGISTRY.keys())}"
    )


def get_index_path(index_name: str) -> Path:
    """Get the directory path for a specific index"""
    return INDEXES_PATH / index_name


def get_index_db_path(index_name: str) -> Path:
    """Get the SQLite database path for an index"""
    return get_index_path(index_name) / "metadata.db"


def get_index_faiss_path(index_name: str) -> Path:
    """Get the FAISS index file path"""
    return get_index_path(index_name) / "vectors.faiss"


def get_index_config_path(index_name: str) -> Path:
    """Get the config JSON path for an index"""
    return get_index_path(index_name) / "config.json"


def get_thumbnails_path(index_name: str) -> Path:
    """Get the thumbnails directory for an index"""
    thumb_path = get_index_path(index_name) / "thumbnails"
    thumb_path.mkdir(parents=True, exist_ok=True)
    return thumb_path

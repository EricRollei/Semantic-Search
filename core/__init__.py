"""
Eric's Semantic Search - Core Package

Description: Core modules providing the foundation for semantic search functionality.
             Includes configuration, database management, index management, model wrappers,
             and utility functions for thumbnail generation and media processing.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

See LICENSE.md for complete license information.
"""

from .config import (
    BASE_PATH,
    INDEXES_PATH,
    MODELS_PATH,
    MODEL_REGISTRY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    DEFAULT_THUMBNAIL_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    SUPPORTED_EXTENSIONS,
    resolve_model_path,
    get_index_path,
    get_index_db_path,
    get_index_faiss_path,
    get_thumbnails_path,
)

from .database import DatabaseManager, ImageRecord, IndexedFolder

from .thumbnail_manager import (
    ThumbnailManager,
    compute_file_hash,
    get_file_info,
    is_supported_image,
    is_supported_media,
    load_image,
    resize_for_thumbnail,
    collect_images_from_folder,
    collect_media_from_folder,
)

from .video_utils import (
    VideoInfo,
    is_video_file,
    get_video_info,
    extract_frames,
    extract_keyframe,
    create_video_thumbnail,
    frames_to_video_input,
)

from .model_wrapper import (
    EmbeddingModelWrapper,
    RerankerModelWrapper,
    get_cached_model,
    clear_model_cache,
    get_device,
)

from .index_manager import (
    SemanticIndex,
    SearchResult,
    SearchResults,
    get_or_create_index,
    list_indexes,
    delete_index,
)

from .index_factory import (
    IndexFactory,
    IndexConfig,
    IndexType,
    INDEX_TYPE_DESCRIPTIONS,
)

__all__ = [
    # Config
    "BASE_PATH",
    "INDEXES_PATH", 
    "MODELS_PATH",
    "MODEL_REGISTRY",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "EMBEDDING_MODELS",
    "RERANKER_MODELS",
    "DEFAULT_THUMBNAIL_SIZE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "SUPPORTED_EXTENSIONS",
    "resolve_model_path",
    "get_index_path",
    "get_index_db_path",
    "get_index_faiss_path",
    "get_thumbnails_path",
    # Database
    "DatabaseManager",
    "ImageRecord",
    "IndexedFolder",
    # Thumbnails
    "ThumbnailManager",
    "compute_file_hash",
    "get_file_info",
    "is_supported_image",
    "is_supported_media",
    "load_image",
    "resize_for_thumbnail",
    "collect_images_from_folder",
    "collect_media_from_folder",
    # Video Utils
    "VideoInfo",
    "is_video_file",
    "get_video_info",
    "extract_frames",
    "extract_keyframe",
    "create_video_thumbnail",
    "frames_to_video_input",
    # Models
    "EmbeddingModelWrapper",
    "RerankerModelWrapper",
    "get_cached_model",
    "clear_model_cache",
    "get_device",
    # Index
    "SemanticIndex",
    "SearchResult",
    "SearchResults",
    "get_or_create_index",
    "list_indexes",
    "delete_index",
    # Index Factory
    "IndexFactory",
    "IndexConfig",
    "IndexType",
    "INDEX_TYPE_DESCRIPTIONS",
]

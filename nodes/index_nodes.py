"""
Eric's Semantic Search - Index Nodes

Description: ComfyUI nodes for index management - creating, loading, populating,
             validating, compacting, and rebuilding semantic search indexes.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

See LICENSE.md for complete license information.
"""

import os
from pathlib import Path
from typing import Tuple

import comfy.utils

from ..core import (
    SemanticIndex,
    get_or_create_index,
    list_indexes,
    DEFAULT_THUMBNAIL_SIZE,
    DEFAULT_BATCH_SIZE,
    EmbeddingModelWrapper,
    INDEX_TYPE_DESCRIPTIONS,
    IndexType,
)


# Index type options for dropdown
INDEX_TYPE_OPTIONS = [
    "Flat (Exact)",      # flat
    "IVF-Flat (Fast)",   # ivf_flat
    "HNSW (Very Fast)",  # hnsw
]

INDEX_TYPE_MAP = {
    "Flat (Exact)": "flat",
    "IVF-Flat (Fast)": "ivf_flat",
    "HNSW (Very Fast)": "hnsw",
}


def get_existing_indexes():
    """Get list of existing indexes for dropdown"""
    indexes = list_indexes()
    if not indexes:
        indexes = ["default"]
    return indexes


class LoadOrCreateIndex:
    """Load an existing index or create a new one"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "load_index"
    RETURN_TYPES = ("SEMANTIC_INDEX",)
    RETURN_NAMES = ("index",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index_name": ("STRING", {"default": "default"}),
            },
            "optional": {
                "embedding_dim": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 8192,
                    "step": 512,
                    "tooltip": "Embedding dimension (4096 for 8B model, 2048 for 2B model)",
                }),
                "index_type": (INDEX_TYPE_OPTIONS, {
                    "default": "Flat (Exact)",
                    "tooltip": "FAISS index type. Flat=exact, IVF-Flat=fast approximate, HNSW=very fast approximate. Only applies to new indexes.",
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, index_name, embedding_dim=4096, index_type="Flat (Exact)"):
        # Always check for changes
        return float("nan")
    
    def load_index(
        self,
        index_name: str,
        embedding_dim: int = 4096,
        index_type: str = "Flat (Exact)",
    ):
        """Load or create the index"""
        index_type_str = INDEX_TYPE_MAP.get(index_type, "flat")
        index = get_or_create_index(index_name, embedding_dim, index_type_str)
        return (index,)


class AddFolderToIndex:
    """Index all images, videos, and documents in a folder"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "add_folder"
    RETURN_TYPES = ("SEMANTIC_INDEX", "STRING")
    RETURN_NAMES = ("index", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
                "model": ("EMBEDDING_MODEL",),
                "folder_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {
                    "default": DEFAULT_BATCH_SIZE,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                }),
                "thumbnail_size": ("INT", {
                    "default": DEFAULT_THUMBNAIL_SIZE,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                }),
                "include_videos": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also index video files (mp4, mkv, avi, etc.)",
                }),
                "include_documents": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also index PDF documents (each page indexed separately)",
                }),
            }
        }
    
    def add_folder(
        self,
        index: SemanticIndex,
        model: EmbeddingModelWrapper,
        folder_path: str,
        recursive: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE,
        include_videos: bool = True,
        include_documents: bool = True,
    ):
        """Index all images, videos, and documents in the folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            return (index, f"Error: Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            return (index, f"Error: Not a directory: {folder_path}")
        
        # Validate embedding dimensions match
        model._ensure_initialized()
        model_dim = model.embedding_dim
        index_dim = index.embedding_dim
        
        if model_dim != index_dim:
            return (index, f"Error: Dimension mismatch! Model outputs {model_dim}-dim embeddings but index expects {index_dim}-dim. Create a new index with matching dimension.")
        
        # Create progress bar
        pbar = comfy.utils.ProgressBar(100)
        
        def progress_callback(current, total, status):
            if total > 0:
                pbar.update_absolute(int(current * 100 / total), 100)
        
        # Index the folder
        stats = index.index_folder(
            folder_path=str(folder_path),
            model=model,
            recursive=recursive,
            batch_size=batch_size,
            thumbnail_size=thumbnail_size,
            include_videos=include_videos,
            include_documents=include_documents,
            progress_callback=progress_callback,
        )
        
        # Build status string
        status_parts = [f"Added: {stats['added']}", f"Skipped: {stats['skipped']}", f"Failed: {stats['failed']}"]
        if stats.get('videos', 0) > 0:
            status_parts.append(f"Videos: {stats['videos']}")
        if stats.get('documents', 0) > 0:
            status_parts.append(f"PDFs: {stats['documents']} ({stats.get('pages', 0)} pages)")
        status = ", ".join(status_parts)
        
        return (index, status)


class RemoveFolderFromIndex:
    """Remove a folder and its images from the index"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "remove_folder"
    RETURN_TYPES = ("SEMANTIC_INDEX", "STRING")
    RETURN_NAMES = ("index", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
                "folder_path": ("STRING", {"default": ""}),
            },
        }
    
    def remove_folder(self, index: SemanticIndex, folder_path: str):
        """Remove folder from index"""
        count = index.remove_folder(folder_path)
        status = f"Removed {count} images from index"
        return (index, status)


class ValidateIndex:
    """Check for stale entries (deleted files) and optionally clean them"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "validate"
    RETURN_TYPES = ("SEMANTIC_INDEX", "INT", "STRING")
    RETURN_NAMES = ("index", "stale_count", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
            },
            "optional": {
                "auto_clean": ("BOOLEAN", {"default": False}),
            }
        }
    
    def validate(self, index: SemanticIndex, auto_clean: bool = False):
        """Validate the index"""
        result = index.validate(auto_clean=auto_clean)
        
        stale_count = result["stale_count"]
        
        if auto_clean:
            status = f"Cleaned {stale_count} stale entries, removed {result['orphaned_thumbnails_removed']} orphaned thumbnails"
        else:
            status = f"Found {stale_count} stale entries"
            if stale_count > 0:
                status += " (set auto_clean=True to remove)"
        
        return (index, stale_count, status)


class GetIndexInfo:
    """Get statistics about an index"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "get_info"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
            },
        }
    
    def get_info(self, index: SemanticIndex):
        """Get index statistics"""
        stats = index.get_stats()
        
        lines = [
            f"Index: {stats['index_name']}",
            f"Type: {stats.get('index_type', 'flat').upper()}",
            f"Images: {stats['image_count']}",
            f"Vectors: {stats['vector_count']}",
            f"Embedding dim: {stats['embedding_dim']}",
            f"Trained: {stats.get('is_trained', True)}",
            f"Folders tracked: {stats['folder_count']}",
            "",
            "Storage:",
            f"  FAISS: {stats['faiss_size_mb']} MB",
            f"  Database: {stats['db_size_mb']} MB",
            f"  Thumbnails: {stats['thumbnail_cache_mb']} MB ({stats['thumbnail_count']} files)",
        ]
        
        # Add compaction info if there are deleted vectors
        deleted = stats.get('deleted_vectors', 0)
        if deleted > 0:
            lines.append("")
            lines.append("Compaction:")
            lines.append(f"  Deleted vectors: {deleted}")
            lines.append(f"  Wasted space: {stats.get('wasted_percentage', 0)}%")
            if stats.get('needs_compaction', False):
                lines.append("  ⚠️ Compaction recommended!")
        
        lines.append("")
        lines.append(f"Model: {stats['model_used'] or 'N/A'}")
        lines.append(f"Created: {stats['created_at'] or 'N/A'}")
        lines.append(f"Last indexed: {stats['last_indexed'] or 'N/A'}")
        
        if stats['folders']:
            lines.append("")
            lines.append("Indexed folders:")
            for folder in stats['folders']:
                lines.append(f"  * {folder}")
        
        return ("\n".join(lines),)


class RebuildIndex:
    """
    Rebuild the FAISS index, optionally changing the index type.
    
    This compacts the index (removes gaps from deleted entries) and can
    convert between index types (e.g., Flat to HNSW for faster search).
    """
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "rebuild"
    RETURN_TYPES = ("SEMANTIC_INDEX", "STRING")
    RETURN_NAMES = ("index", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
            },
            "optional": {
                "new_index_type": (["Keep Current"] + INDEX_TYPE_OPTIONS, {
                    "default": "Keep Current",
                    "tooltip": "New index type. 'Keep Current' will compact without changing type.",
                }),
            }
        }
    
    def rebuild(
        self,
        index: SemanticIndex,
        new_index_type: str = "Keep Current",
    ):
        """Rebuild the index"""
        # Determine new type
        if new_index_type == "Keep Current":
            new_type_str = None
        else:
            new_type_str = INDEX_TYPE_MAP.get(new_index_type, "flat")
        
        # Create a simple progress tracker
        def progress_callback(current, total, status):
            print(f"[SemanticSearch] Rebuild: {status} ({current}/{total})")
        
        result = index.rebuild(
            new_index_type=new_type_str,
            progress_callback=progress_callback,
        )
        
        if result["success"]:
            if result["old_type"] != result["new_type"]:
                status = f"Rebuilt index: {result['old_type']} → {result['new_type']} ({result['new_count']} vectors)"
            else:
                status = f"Compacted index: {result['old_count']} → {result['new_count']} vectors"
        else:
            status = f"Rebuild failed: {result.get('error', 'Unknown error')}"
        
        return (index, status)


class CompactIndex:
    """
    Compact the index by removing deleted vectors.
    
    When you remove images from the index, the vectors remain in FAISS
    as orphaned entries. Compaction rebuilds the index with only the
    valid vectors, reclaiming memory and disk space.
    
    Use Get Index Info to check if compaction is recommended.
    """
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "compact"
    RETURN_TYPES = ("SEMANTIC_INDEX", "STRING")
    RETURN_NAMES = ("index", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
            },
            "optional": {
                "new_index_type": (["Keep Current"] + INDEX_TYPE_OPTIONS, {
                    "default": "Keep Current",
                    "tooltip": "Optionally change index type during compaction.",
                }),
            }
        }
    
    def compact(
        self,
        index: SemanticIndex,
        new_index_type: str = "Keep Current",
    ):
        """Compact the index"""
        # Check if compaction is needed
        needs_compact, deleted_count, wasted_pct = index.needs_compaction()
        
        if not needs_compact and deleted_count == 0:
            return (index, "No compaction needed - no deleted vectors")
        
        # Determine new type
        if new_index_type == "Keep Current":
            new_type_str = None
        else:
            new_type_str = INDEX_TYPE_MAP.get(new_index_type, "flat")
        
        # Create a simple progress tracker
        def progress_callback(current, total, status):
            print(f"[SemanticSearch] Compact: {status} ({current}/{total})")
        
        result = index.compact(
            new_index_type=new_type_str,
            progress_callback=progress_callback,
        )
        
        if result["success"]:
            if result.get("compacted", False):
                status = (
                    f"Compacted index: {result['old_count']} → {result['new_count']} vectors "
                    f"(removed {result['removed_count']} orphaned)"
                )
            else:
                status = result.get("message", "No compaction performed")
        else:
            status = f"Compaction failed: {result.get('error', 'Unknown error')}"
        
        return (index, status)

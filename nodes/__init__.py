"""
Eric's Semantic Search - ComfyUI Nodes Package

Description: ComfyUI node definitions for semantic search functionality.
             Exports all node classes and mappings for ComfyUI registration.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

See LICENSE.md for complete license information.
"""

from .model_nodes import LoadEmbeddingModel, LoadRerankerModel
from .index_nodes import (
    LoadOrCreateIndex,
    AddFolderToIndex,
    RemoveFolderFromIndex,
    ValidateIndex,
    GetIndexInfo,
    RebuildIndex,
    CompactIndex,
)
from .search_nodes import (
    SearchByText,
    SearchByImage,
    SearchByVideo,
    SearchByDocument,
    SearchWithExclusion,
    SearchMultiIndex,
    RerankResults,
    CombineResults,
    FilterByScore,
)
from .output_nodes import (
    PreviewResults,
    LoadResultImages,
    GetResultPaths,
)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Model nodes
    "Eric_LoadEmbeddingModel": LoadEmbeddingModel,
    "Eric_LoadRerankerModel": LoadRerankerModel,
    # Index nodes
    "Eric_LoadOrCreateIndex": LoadOrCreateIndex,
    "Eric_AddFolderToIndex": AddFolderToIndex,
    "Eric_RemoveFolderFromIndex": RemoveFolderFromIndex,
    "Eric_ValidateIndex": ValidateIndex,
    "Eric_GetIndexInfo": GetIndexInfo,
    "Eric_RebuildIndex": RebuildIndex,
    "Eric_CompactIndex": CompactIndex,
    # Search nodes
    "Eric_SearchByText": SearchByText,
    "Eric_SearchByImage": SearchByImage,
    "Eric_SearchByVideo": SearchByVideo,
    "Eric_SearchByDocument": SearchByDocument,
    "Eric_SearchWithExclusion": SearchWithExclusion,
    "Eric_SearchMultiIndex": SearchMultiIndex,
    "Eric_RerankResults": RerankResults,
    "Eric_CombineResults": CombineResults,
    "Eric_FilterByScore": FilterByScore,
    # Output nodes
    "Eric_PreviewResults": PreviewResults,
    "Eric_LoadResultImages": LoadResultImages,
    "Eric_GetResultPaths": GetResultPaths,
}

# Display names for ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    # Model nodes
    "Eric_LoadEmbeddingModel": "Load Embedding Model",
    "Eric_LoadRerankerModel": "Load Reranker Model",
    # Index nodes
    "Eric_LoadOrCreateIndex": "Load/Create Index",
    "Eric_AddFolderToIndex": "Add Folder to Index",
    "Eric_RemoveFolderFromIndex": "Remove Folder from Index",
    "Eric_ValidateIndex": "Validate Index",
    "Eric_GetIndexInfo": "Get Index Info",
    "Eric_RebuildIndex": "Rebuild Index",
    "Eric_CompactIndex": "Compact Index",
    # Search nodes
    "Eric_SearchByText": "Search by Text",
    "Eric_SearchByImage": "Search by Image",
    "Eric_SearchByVideo": "Search by Video",
    "Eric_SearchByDocument": "Search by Document",
    "Eric_SearchWithExclusion": "Search with Exclusion",
    "Eric_SearchMultiIndex": "Search Multi-Index",
    "Eric_RerankResults": "Rerank Results",
    "Eric_CombineResults": "Combine Results",
    "Eric_FilterByScore": "Filter by Score",
    # Output nodes
    "Eric_PreviewResults": "Preview Results",
    "Eric_LoadResultImages": "Load Result Images",
    "Eric_GetResultPaths": "Get Result Paths",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

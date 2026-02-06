"""
Eric's Semantic Search - Search Nodes

Description: ComfyUI nodes for searching - text search, image similarity, video search,
             document search, multi-index search, exclusion search, reranking,
             result filtering, and result combination.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

See LICENSE.md for complete license information.
"""

from typing import List, Optional
import torch
import numpy as np

from ..core import (
    SemanticIndex,
    SearchResults,
    SearchResult,
    EmbeddingModelWrapper,
    RerankerModelWrapper,
    DEFAULT_TOP_K,
    MAX_TOP_K,
)


# Result type filter options
RESULT_TYPE_OPTIONS = ["all", "images", "videos", "documents", "media"]


def validate_dimensions(index: SemanticIndex, model: EmbeddingModelWrapper) -> Optional[str]:
    """
    Validate that model and index embedding dimensions match.
    Returns error message if mismatch, None if OK.
    """
    model._ensure_initialized()
    model_dim = model.embedding_dim
    index_dim = index.embedding_dim
    
    if model_dim != index_dim:
        return f"Dimension mismatch! Model outputs {model_dim}-dim but index has {index_dim}-dim embeddings. Use matching model/index dimensions."
    return None


def filter_results_by_type(results: SearchResults, result_type: str) -> SearchResults:
    """
    Filter search results by media type.
    
    Args:
        results: SearchResults to filter
        result_type: "all", "images", "videos", "documents", or "media" (images+videos)
        
    Returns:
        Filtered SearchResults
    """
    if result_type == "all":
        return results
    
    type_map = {
        "images": ["image"],
        "videos": ["video"],
        "documents": ["document"],
        "media": ["image", "video"],
    }
    
    allowed_types = type_map.get(result_type, ["image", "video", "document"])
    
    filtered = [r for r in results.results if r.media_type in allowed_types]
    
    return SearchResults(
        results=filtered,
        query_type=results.query_type,
        query_info=results.query_info,
    )


class SearchByText:
    """Search for images using a text query"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "search"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
                "model": ("EMBEDDING_MODEL",),
                "query": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "top_k": ("INT", {
                    "default": DEFAULT_TOP_K,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                }),
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum score threshold (0-1). Results below this score are filtered out.",
                }),
                "result_type": (RESULT_TYPE_OPTIONS, {
                    "default": "all",
                    "tooltip": "Filter results by media type: all, images, videos, documents, or media (images+videos)",
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom instruction for the embedding model",
                }),
            }
        }
    
    def search(
        self,
        index: SemanticIndex,
        model: EmbeddingModelWrapper,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0,
        result_type: str = "all",
        instruction: str = "",
    ):
        """Search by text query"""
        if not query.strip():
            return (SearchResults(results=[], query_type="text", query_info=""),)
        
        # Validate dimensions
        dim_error = validate_dimensions(index, model)
        if dim_error:
            return (SearchResults(results=[], query_type="text", query_info=f"ERROR: {dim_error}"),)
        
        results = index.search_by_text(
            text=query,
            model=model,
            top_k=top_k,
            instruction=instruction if instruction.strip() else None,
        )
        
        # Filter by result type
        results = filter_results_by_type(results, result_type)
        
        # Filter by minimum score
        if min_score > 0.0:
            filtered = [r for r in results.results if r.score >= min_score]
            results = SearchResults(
                results=filtered,
                query_type=results.query_type,
                query_info=results.query_info,
            )
        
        return (results,)


class SearchByImage:
    """Search for similar images using an image query"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "search"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
                "model": ("EMBEDDING_MODEL",),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_path": ("STRING", {"default": ""}),
                "top_k": ("INT", {
                    "default": DEFAULT_TOP_K,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                }),
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum score threshold (0-1). Results below this score are filtered out.",
                }),
                "result_type": (RESULT_TYPE_OPTIONS, {
                    "default": "all",
                    "tooltip": "Filter results by media type: all, images, videos, documents, or media (images+videos)",
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    def search(
        self,
        index: SemanticIndex,
        model: EmbeddingModelWrapper,
        image=None,
        image_path: str = "",
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0,
        result_type: str = "all",
        instruction: str = "",
    ):
        """Search by image"""
        from PIL import Image
        import tempfile
        import os
        
        results = None
        
        # Determine image source
        if image is not None:
            # Convert ComfyUI tensor to PIL and save temp file
            # ComfyUI images are (B, H, W, C) tensors in 0-1 range
            if torch.is_tensor(image):
                img_np = image[0].cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    pil_image.save(f.name)
                    temp_path = f.name
                
                try:
                    results = index.search_by_image(
                        image_path=temp_path,
                        model=model,
                        top_k=top_k,
                        instruction=instruction if instruction.strip() else None,
                    )
                finally:
                    os.unlink(temp_path)
        
        elif image_path.strip():
            results = index.search_by_image(
                image_path=image_path,
                model=model,
                top_k=top_k,
                instruction=instruction if instruction.strip() else None,
            )
        
        if results is None:
            return (SearchResults(results=[], query_type="image", query_info=""),)
        
        # Filter by result type
        results = filter_results_by_type(results, result_type)
        
        # Filter by minimum score
        if min_score > 0.0:
            filtered = [r for r in results.results if r.score >= min_score]
            results = SearchResults(
                results=filtered,
                query_type=results.query_type,
                query_info=results.query_info,
            )
        
        return (results,)


class SearchByVideo:
    """Search for similar content using a video query"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "search"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
                "model": ("EMBEDDING_MODEL",),
                "video_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "top_k": ("INT", {
                    "default": DEFAULT_TOP_K,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                }),
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum score threshold (0-1). Results below this score are filtered out.",
                }),
                "result_type": (RESULT_TYPE_OPTIONS, {
                    "default": "all",
                    "tooltip": "Filter results by media type: all, images, videos, documents, or media (images+videos)",
                }),
                "max_frames": ("INT", {
                    "default": 32,
                    "min": 4,
                    "max": 64,
                    "step": 4,
                    "tooltip": "Maximum number of frames to extract from video for query",
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    def search(
        self,
        index: SemanticIndex,
        model: EmbeddingModelWrapper,
        video_path: str = "",
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0,
        result_type: str = "all",
        max_frames: int = 32,
        instruction: str = "",
    ):
        """Search by video"""
        if not video_path.strip():
            return (SearchResults(results=[], query_type="video", query_info=""),)
        
        results = index.search_by_video(
            video_path=video_path,
            model=model,
            top_k=top_k,
            instruction=instruction if instruction.strip() else None,
            max_frames=max_frames,
        )
        
        # Filter by result type
        results = filter_results_by_type(results, result_type)
        
        # Filter by minimum score
        if min_score > 0.0:
            filtered = [r for r in results.results if r.score >= min_score]
            results = SearchResults(
                results=filtered,
                query_type=results.query_type,
                query_info=results.query_info,
            )
        
        return (results,)


class SearchByDocument:
    """Search for similar content using a PDF page as query"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "search"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
                "model": ("EMBEDDING_MODEL",),
                "pdf_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "page_number": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Which page to use as the query (1-indexed)",
                }),
                "top_k": ("INT", {
                    "default": DEFAULT_TOP_K,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                }),
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum score threshold (0-1). Results below this score are filtered out.",
                }),
                "result_type": (RESULT_TYPE_OPTIONS, {
                    "default": "all",
                    "tooltip": "Filter results by media type: all, images, videos, documents, or media (images+videos)",
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom instruction for the search (e.g., 'Find similar invoices')",
                }),
            }
        }
    
    def search(
        self,
        index: SemanticIndex,
        model: EmbeddingModelWrapper,
        pdf_path: str = "",
        page_number: int = 1,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0,
        result_type: str = "all",
        instruction: str = "",
    ):
        """Search by PDF page"""
        from pathlib import Path
        
        if not pdf_path.strip():
            return (SearchResults(results=[], query_type="document", query_info=""),)
        
        if not Path(pdf_path).exists():
            return (SearchResults(results=[], query_type="document", query_info=f"File not found: {pdf_path}"),)
        
        results = index.search_by_document(
            pdf_path=pdf_path,
            model=model,
            top_k=top_k,
            instruction=instruction if instruction.strip() else None,
            page_number=page_number,
        )
        
        # Filter by result type
        results = filter_results_by_type(results, result_type)
        
        # Filter by minimum score
        if min_score > 0.0:
            filtered = [r for r in results.results if r.score >= min_score]
            results = SearchResults(
                results=filtered,
                query_type=results.query_type,
                query_info=results.query_info,
            )
        
        return (results,)


class RerankResults:
    """Rerank search results using the reranker model for better precision"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "rerank"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("SEARCH_RESULTS",),
                "reranker": ("RERANKER_MODEL",),
            },
            "optional": {
                "query_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Override query text for reranking",
                }),
                "top_k": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum reranker score threshold (0-1).",
                }),
                "instruction": ("STRING", {
                    "default": "Retrieve images relevant to the query.",
                    "multiline": False,
                }),
            }
        }
    
    def rerank(
        self,
        results: SearchResults,
        reranker: RerankerModelWrapper,
        query_text: str = "",
        top_k: int = 10,
        min_score: float = 0.0,
        instruction: str = "Retrieve images relevant to the query.",
    ):
        """Rerank results"""
        if len(results) == 0:
            return (results,)
        
        # Determine query
        if query_text.strip():
            query = {"text": query_text}
        elif results.query_type == "text" and results.query_info:
            query = {"text": results.query_info}
        elif results.query_type == "image" and results.query_info:
            query = {"image": results.query_info}
        else:
            # Can't rerank without a query
            return (results,)
        
        # Build documents from results (using thumbnails for efficiency)
        documents = [{"image": r.thumbnail_path} for r in results.results]
        
        # Rerank
        scores = reranker.rerank(
            query=query,
            documents=documents,
            instruction=instruction if instruction.strip() else None,
        )
        
        # Sort by new scores
        scored_results = list(zip(results.results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by min_score and take top_k
        reranked = []
        for result, score in scored_results:
            if score >= min_score:
                reranked.append(SearchResult(
                    file_path=result.file_path,
                    thumbnail_path=result.thumbnail_path,
                    score=score,
                    vector_id=result.vector_id,
                ))
            if len(reranked) >= top_k:
                break
        
        return (SearchResults(
            results=reranked,
            query_type=results.query_type,
            query_info=results.query_info,
        ),)


class FilterByScore:
    """Filter search results by minimum score threshold"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "filter"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("SEARCH_RESULTS",),
                "min_score": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum score threshold. Results with scores below this are removed.",
                }),
            },
            "optional": {
                "max_results": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                    "tooltip": "Maximum number of results to return after filtering.",
                }),
            }
        }
    
    def filter(
        self,
        results: SearchResults,
        min_score: float = 0.1,
        max_results: int = 100,
    ):
        """Filter results by score threshold"""
        filtered = [r for r in results.results if r.score >= min_score]
        filtered = filtered[:max_results]
        
        return (SearchResults(
            results=filtered,
            query_type=results.query_type,
            query_info=results.query_info,
        ),)


class CombineResults:
    """Combine two search result sets"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "combine"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results_a": ("SEARCH_RESULTS",),
                "results_b": ("SEARCH_RESULTS",),
                "mode": (["union", "intersection", "concat"], {"default": "union"}),
            },
            "optional": {
                "max_results": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                }),
            }
        }
    
    def combine(
        self,
        results_a: SearchResults,
        results_b: SearchResults,
        mode: str,
        max_results: int = 100,
    ):
        """Combine results"""
        if mode == "union":
            # Union: all unique results, deduplicated by file_path
            seen = {}
            for r in list(results_a) + list(results_b):
                if r.file_path not in seen:
                    seen[r.file_path] = r
                else:
                    # Keep higher score
                    if r.score > seen[r.file_path].score:
                        seen[r.file_path] = r
            
            combined = sorted(seen.values(), key=lambda x: x.score, reverse=True)
            
        elif mode == "intersection":
            # Intersection: only results in both
            paths_a = {r.file_path for r in results_a}
            paths_b = {r.file_path for r in results_b}
            common = paths_a & paths_b
            
            # Combine scores
            scores_a = {r.file_path: r for r in results_a}
            scores_b = {r.file_path: r for r in results_b}
            
            combined = []
            for path in common:
                r_a = scores_a[path]
                r_b = scores_b[path]
                combined.append(SearchResult(
                    file_path=path,
                    thumbnail_path=r_a.thumbnail_path,
                    score=(r_a.score + r_b.score) / 2,  # Average score
                    vector_id=r_a.vector_id,
                ))
            
            combined.sort(key=lambda x: x.score, reverse=True)
            
        else:  # concat
            # Simple concatenation
            combined = list(results_a) + list(results_b)
        
        # Limit results
        combined = combined[:max_results]
        
        return (SearchResults(
            results=combined,
            query_type="combined",
            query_info=f"{results_a.query_info} + {results_b.query_info}",
        ),)


class SearchWithExclusion:
    """Search by text with negative terms to exclude certain results"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "search"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("SEMANTIC_INDEX",),
                "model": ("EMBEDDING_MODEL",),
                "query": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Main search query (what you want to find)",
                }),
                "exclude": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Terms to exclude, comma-separated (e.g., 'beach, sand, water')",
                }),
            },
            "optional": {
                "top_k": ("INT", {
                    "default": DEFAULT_TOP_K,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                }),
                "exclusion_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Similarity threshold for exclusion. Higher = stricter exclusion (fewer results).",
                }),
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "result_type": (RESULT_TYPE_OPTIONS, {
                    "default": "all",
                    "tooltip": "Filter results by media type: all, images, videos, documents, or media (images+videos)",
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    def search(
        self,
        index: SemanticIndex,
        model: EmbeddingModelWrapper,
        query: str,
        exclude: str,
        top_k: int = DEFAULT_TOP_K,
        exclusion_threshold: float = 0.3,
        min_score: float = 0.0,
        result_type: str = "all",
        instruction: str = "",
    ):
        """Search with exclusions"""
        if not query.strip():
            return (SearchResults(results=[], query_type="text_with_exclusion", query_info=""),)
        
        # Parse negative terms (comma-separated)
        negative_terms = [t.strip() for t in exclude.split(",") if t.strip()]
        
        if negative_terms:
            results = index.search_by_text_with_exclusion(
                positive_query=query,
                negative_terms=negative_terms,
                model=model,
                top_k=top_k,
                exclusion_threshold=exclusion_threshold,
                instruction=instruction if instruction.strip() else None,
            )
        else:
            # No exclusions, just do normal search
            results = index.search_by_text(
                text=query,
                model=model,
                top_k=top_k,
                instruction=instruction if instruction.strip() else None,
            )
        
        # Filter by result type
        results = filter_results_by_type(results, result_type)
        
        # Filter by minimum score
        if min_score > 0.0:
            filtered = [r for r in results.results if r.score >= min_score]
            results = SearchResults(
                results=filtered,
                query_type=results.query_type,
                query_info=results.query_info,
            )
        
        return (results,)


class SearchMultiIndex:
    """Search across multiple indexes simultaneously"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "search"
    RETURN_TYPES = ("SEARCH_RESULTS",)
    RETURN_NAMES = ("results",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("EMBEDDING_MODEL",),
                "query": ("STRING", {"default": "", "multiline": True}),
                "index_1": ("SEMANTIC_INDEX",),
            },
            "optional": {
                "index_2": ("SEMANTIC_INDEX",),
                "index_3": ("SEMANTIC_INDEX",),
                "index_4": ("SEMANTIC_INDEX",),
                "top_k": ("INT", {
                    "default": DEFAULT_TOP_K,
                    "min": 1,
                    "max": MAX_TOP_K,
                    "step": 1,
                    "tooltip": "Total results across all indexes",
                }),
                "normalize_scores": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize scores to 0-1 range per index before merging",
                }),
                "min_score": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "result_type": (RESULT_TYPE_OPTIONS, {
                    "default": "all",
                    "tooltip": "Filter results by media type: all, images, videos, documents, or media (images+videos)",
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    def search(
        self,
        model: EmbeddingModelWrapper,
        query: str,
        index_1: SemanticIndex,
        index_2: SemanticIndex = None,
        index_3: SemanticIndex = None,
        index_4: SemanticIndex = None,
        top_k: int = DEFAULT_TOP_K,
        normalize_scores: bool = True,
        min_score: float = 0.0,
        result_type: str = "all",
        instruction: str = "",
    ):
        """Search across multiple indexes"""
        if not query.strip():
            return (SearchResults(results=[], query_type="multi_index", query_info=""),)
        
        # Collect all indexes
        indexes = [index_1]
        if index_2 is not None:
            indexes.append(index_2)
        if index_3 is not None:
            indexes.append(index_3)
        if index_4 is not None:
            indexes.append(index_4)
        
        # Search each index
        results_list = []
        per_index_k = max(top_k, 50)  # Get enough from each for good merging
        
        for idx in indexes:
            result = idx.search_by_text(
                text=query,
                model=model,
                top_k=per_index_k,
                instruction=instruction if instruction.strip() else None,
            )
            results_list.append(result)
        
        # Merge results
        merged = SemanticIndex.merge_search_results(
            results_list=results_list,
            top_k=top_k,
            normalize_scores=normalize_scores,
        )
        
        # Filter by result type
        merged = filter_results_by_type(merged, result_type)
        
        # Filter by minimum score
        if min_score > 0.0:
            filtered = [r for r in merged.results if r.score >= min_score]
            merged = SearchResults(
                results=filtered,
                query_type=merged.query_type,
                query_info=merged.query_info,
            )
        
        return (merged,)

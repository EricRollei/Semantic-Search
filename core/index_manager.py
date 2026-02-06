"""
Eric's Semantic Search - Index Manager

Description: Core index management coordinating FAISS vector search, SQLite metadata,
             and thumbnail storage. Handles indexing of images, videos, and documents,
             as well as search operations and index maintenance.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Dependencies:
- FAISS (MIT License): Facebook AI Similarity Search - https://github.com/facebookresearch/faiss
- NumPy (BSD License)

See LICENSE.md for complete license information.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import threading

import faiss

from .config import (
    get_index_path,
    get_index_faiss_path,
    get_index_config_path,
    get_thumbnails_path,
    DEFAULT_THUMBNAIL_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TOP_K,
    CHECKPOINT_INTERVAL,
    VIDEO_FRAME_MAX_PIXELS,
)
from .database import DatabaseManager, ImageRecord
from .thumbnail_manager import (
    ThumbnailManager,
    collect_images_from_folder,
    collect_media_from_folder,
    collect_documents_from_folder,
    get_file_info,
    compute_file_hash,
    is_supported_media,
)
from .video_utils import (
    is_video_file,
    extract_frames,
    get_video_info,
)
from .pdf_utils import (
    is_pdf_file,
    get_pdf_info,
    extract_all_pages,
    get_pdf_page_path,
    check_pymupdf,
)
from .model_wrapper import EmbeddingModelWrapper
from .index_factory import (
    IndexFactory,
    IndexConfig,
    IndexType,
    INDEX_TYPE_DESCRIPTIONS,
)


@dataclass
class SearchResult:
    """A single search result"""
    file_path: str
    thumbnail_path: str
    score: float
    vector_id: int
    media_type: str = "image"  # "image", "video", or "document"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "thumbnail_path": self.thumbnail_path,
            "score": float(self.score),
            "vector_id": self.vector_id,
            "media_type": self.media_type,
        }


@dataclass
class SearchResults:
    """Collection of search results"""
    results: List[SearchResult]
    query_type: str  # "text" or "image"
    query_info: str  # The query text or image path
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def __getitem__(self, idx) -> SearchResult:
        return self.results[idx]
    
    def get_paths(self) -> List[str]:
        """Get list of file paths"""
        return [r.file_path for r in self.results]
    
    def get_thumbnails(self) -> List[str]:
        """Get list of thumbnail paths"""
        return [r.thumbnail_path for r in self.results]
    
    def get_scores(self) -> List[float]:
        """Get list of similarity scores"""
        return [r.score for r in self.results]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "query_info": self.query_info,
            "count": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }


class SemanticIndex:
    """
    Manages a semantic search index combining:
    - FAISS for vector similarity search
    - SQLite for metadata storage
    - Thumbnail cache for fast previews
    """
    
    def __init__(
        self,
        index_name: str,
        embedding_dim: int = 4096,
        index_type: str = "flat",
    ):
        """
        Initialize or load an existing index.
        
        Args:
            index_name: Name of the index (used for directory structure)
            embedding_dim: Dimension of embedding vectors (4096 for 8B, 2048 for 2B)
            index_type: Type of FAISS index ("flat", "ivf_flat", "hnsw", "ivf_pq")
        """
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.index_type_str = index_type
        
        # Paths
        self.index_path = get_index_path(index_name)
        self.faiss_path = get_index_faiss_path(index_name)
        self.config_path = get_index_config_path(index_name)
        
        # Ensure directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db = DatabaseManager(index_name)
        self.thumbnails = ThumbnailManager(index_name)
        
        # Load or create config first (may contain index settings)
        self._config = self._load_or_create_config()
        
        # Load or create FAISS index
        self._faiss_index = None
        self._index_config = None
        self._load_or_create_faiss()
        
        # Thread safety - use RLock for re-entrancy (methods that hold the lock can call other locking methods)
        self._lock = threading.RLock()
        
        # Training buffer for IVF indexes
        self._training_buffer = []
        self._training_buffer_size = 0
    
    def _load_or_create_faiss(self):
        """Load existing FAISS index or create new one"""
        if self.faiss_path.exists():
            print(f"[SemanticSearch] Loading existing FAISS index from {self.faiss_path}")
            self._faiss_index = faiss.read_index(str(self.faiss_path))
            # Update embedding dim from loaded index
            self.embedding_dim = self._faiss_index.d
            
            # Detect and store index config
            detected_type = IndexFactory.detect_index_type(self._faiss_index)
            self._index_config = IndexConfig(
                index_type=detected_type,
                dimension=self.embedding_dim,
                nlist=self._config.get("index_params", {}).get("nlist", 100),
                nprobe=self._config.get("index_params", {}).get("nprobe", 10),
                hnsw_m=self._config.get("index_params", {}).get("hnsw_m", 32),
                hnsw_ef_search=self._config.get("index_params", {}).get("hnsw_ef_search", 64),
                is_trained=getattr(self._faiss_index, 'is_trained', True),
            )
            print(f"[SemanticSearch] Index type: {detected_type.value}, vectors: {self._faiss_index.ntotal}")
        else:
            # Create new index with specified type
            index_type = IndexType.from_string(self.index_type_str)
            print(f"[SemanticSearch] Creating new FAISS index: type={index_type.value}, dim={self.embedding_dim}")
            
            self._index_config = IndexConfig(
                index_type=index_type,
                dimension=self.embedding_dim,
            )
            self._faiss_index = IndexFactory.create_index(self._index_config)
            
            # Store index type in config
            self._config["index_type"] = index_type.value
            self._config["index_params"] = self._index_config.to_dict()
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load or create index configuration"""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        else:
            config = {
                "index_name": self.index_name,
                "embedding_dim": self.embedding_dim,
                "created_at": datetime.now().isoformat(),
                "model_used": None,
            }
            self._save_config(config)
            return config
    
    def _save_config(self, config: Dict[str, Any] = None):
        """Save configuration to disk"""
        if config is None:
            config = self._config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def save(self):
        """Save FAISS index and config to disk"""
        with self._lock:
            faiss.write_index(self._faiss_index, str(self.faiss_path))
            self._save_config()
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        file_paths: List[str],
        thumbnail_paths: List[str],
        file_infos: List[Tuple[float, int]],  # (mtime, size) for each file
        media_types: Optional[List[str]] = None,
        frame_counts: Optional[List[Optional[int]]] = None,
        durations: Optional[List[Optional[float]]] = None,
    ) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            vectors: Embedding vectors (n, embedding_dim)
            file_paths: Paths to original image/video files
            thumbnail_paths: Paths to cached thumbnails
            file_infos: (mtime, size) tuples for change detection
            media_types: "image" or "video" for each file (default: all "image")
            frame_counts: Frame count for videos (None for images)
            durations: Duration in seconds for videos (None for images)
            
        Returns:
            List of assigned vector IDs
        """
        if len(vectors) == 0:
            return []
        
        # Default media types
        if media_types is None:
            media_types = ["image"] * len(vectors)
        if frame_counts is None:
            frame_counts = [None] * len(vectors)
        if durations is None:
            durations = [None] * len(vectors)
        
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        with self._lock:
            # Handle IVF training if needed
            if IndexFactory.needs_training(self._faiss_index):
                min_training = IndexFactory.get_min_training_size(self._index_config)
                
                # Buffer vectors for training
                self._training_buffer.append(vectors)
                self._training_buffer_size += len(vectors)
                
                if self._training_buffer_size >= min_training:
                    # Combine all buffered vectors and train
                    all_vectors = np.vstack(self._training_buffer)
                    print(f"[SemanticSearch] Training index with {len(all_vectors)} vectors...")
                    
                    if IndexFactory.train_index(self._faiss_index, all_vectors):
                        print("[SemanticSearch] Training complete, adding buffered vectors...")
                        # Now add all buffered vectors
                        self._faiss_index.add(all_vectors)
                        self._index_config.is_trained = True
                        self._index_config.training_size = len(all_vectors)
                        self._config["index_params"] = self._index_config.to_dict()
                    else:
                        print("[SemanticSearch] Training failed, falling back to Flat index")
                        # Fallback to flat index
                        self._index_config.index_type = IndexType.FLAT
                        self._faiss_index = IndexFactory.create_index(self._index_config)
                        self._faiss_index.add(all_vectors)
                    
                    # Clear buffer
                    self._training_buffer = []
                    self._training_buffer_size = 0
                else:
                    print(f"[SemanticSearch] Buffering vectors for training: {self._training_buffer_size}/{min_training}")
                    # Still need to record in database for later
                    # We'll add to FAISS once trained
            else:
                # Index is ready, add directly
                self._faiss_index.add(vectors)
            
            # Get starting vector ID
            start_id = self._faiss_index.ntotal - len(vectors)
            if start_id < 0:
                # Vectors were buffered, not yet added
                start_id = self.db.get_max_vector_id() + 1 if self.db.get_max_vector_id() >= 0 else 0
            
            # Add to database
            vector_ids = []
            for i, (path, thumb, (mtime, size)) in enumerate(zip(file_paths, thumbnail_paths, file_infos)):
                vector_id = start_id + i
                file_hash = f"{size}_{mtime}"  # Fast hash
                
                self.db.add_image(
                    file_path=str(path),
                    file_hash=file_hash,
                    file_mtime=mtime,
                    file_size=size,
                    thumbnail_path=str(thumb),
                    vector_id=vector_id,
                    media_type=media_types[i],
                    frame_count=frame_counts[i],
                    duration_seconds=durations[i],
                )
                vector_ids.append(vector_id)
            
            return vector_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
    ) -> SearchResults:
        """
        Search for similar images by vector.
        
        Args:
            query_vector: Query embedding (1D or 2D with batch size 1)
            top_k: Number of results to return
            
        Returns:
            SearchResults object
        """
        # Ensure correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = np.ascontiguousarray(query_vector.astype(np.float32))
        
        with self._lock:
            # Search FAISS
            scores, indices = self._faiss_index.search(query_vector, top_k)
            scores = scores[0]  # Remove batch dimension
            indices = indices[0]
        
        # Filter out invalid results (index -1 means not enough results)
        valid_mask = indices >= 0
        scores = scores[valid_mask]
        indices = indices[valid_mask]
        
        # Fetch metadata from database
        records = self.db.get_images_by_vector_ids(indices.tolist())
        
        # Build results
        results = []
        for score, vector_id in zip(scores, indices):
            if vector_id in records:
                record = records[vector_id]
                # Check if file still exists
                if Path(record.file_path).exists():
                    results.append(SearchResult(
                        file_path=record.file_path,
                        thumbnail_path=record.thumbnail_path,
                        score=float(score),
                        vector_id=int(vector_id),
                        media_type=getattr(record, 'media_type', 'image'),
                    ))
        
        return SearchResults(results=results, query_type="vector", query_info="")
    
    def search_by_text(
        self,
        text: str,
        model: EmbeddingModelWrapper,
        top_k: int = 50,
        instruction: Optional[str] = None,
    ) -> SearchResults:
        """
        Search for images by text query.
        
        Args:
            text: Query text
            model: Embedding model
            top_k: Number of results
            instruction: Optional instruction for the model
            
        Returns:
            SearchResults object
        """
        # Encode query
        query_vector = model.encode_text(text, instruction=instruction)
        
        # Search
        results = self.search(query_vector, top_k)
        results.query_type = "text"
        results.query_info = text
        
        return results
    
    def search_by_image(
        self,
        image_path: str,
        model: EmbeddingModelWrapper,
        top_k: int = 50,
        instruction: Optional[str] = None,
    ) -> SearchResults:
        """
        Search for similar images by image query.
        
        Args:
            image_path: Path to query image
            model: Embedding model
            top_k: Number of results
            instruction: Optional instruction for the model
            
        Returns:
            SearchResults object
        """
        # Encode query
        query_vector = model.encode_image(image_path, instruction=instruction)
        
        # Search
        results = self.search(query_vector, top_k)
        results.query_type = "image"
        results.query_info = str(image_path)
        
        return results
    
    def search_by_video(
        self,
        video_path: str,
        model: EmbeddingModelWrapper,
        top_k: int = 50,
        instruction: Optional[str] = None,
        max_frames: int = 32,
    ) -> SearchResults:
        """
        Search for similar content using a video query.
        
        Args:
            video_path: Path to query video
            model: Embedding model
            top_k: Number of results
            instruction: Optional instruction for the model
            max_frames: Maximum frames to extract from video
            
        Returns:
            SearchResults object
        """
        from .config import VIDEO_DEFAULT_FPS
        
        # Extract frames from video
        frames = extract_frames(
            video_path,
            max_frames=max_frames,
            target_fps=VIDEO_DEFAULT_FPS,
            uniform_sampling=True,
        )
        
        if not frames:
            return SearchResults(
                results=[],
                query_type="video",
                query_info=str(video_path),
            )
        
        # Encode query video frames
        query_vector = model.encode_video_frames(frames, instruction=instruction)
        
        # Search
        results = self.search(query_vector, top_k)
        results.query_type = "video"
        results.query_info = str(video_path)
        
        return results
    
    def search_by_document(
        self,
        pdf_path: str,
        model: EmbeddingModelWrapper,
        top_k: int = DEFAULT_TOP_K,
        instruction: str = "",
        page_number: int = 1,
    ) -> SearchResults:
        """
        Search for similar content using a PDF page as query.
        
        Args:
            pdf_path: Path to PDF file
            model: Embedding model
            top_k: Number of results
            instruction: Optional instruction for the model
            page_number: Which page to use as query (1-indexed)
            
        Returns:
            SearchResults object
        """
        from .pdf_utils import extract_page_image
        
        # Extract the specified page as image
        page_image = extract_page_image(pdf_path, page_number)
        
        if page_image is None:
            return SearchResults(
                results=[],
                query_type="document",
                query_info=f"{pdf_path}#page={page_number}",
            )
        
        # Encode query page
        default_instruction = instruction or "Represent this document page for retrieval."
        query_vector = model.encode(
            [{"image": page_image}],
            instruction=default_instruction,
        )
        
        # Search
        results = self.search(query_vector.reshape(-1), top_k)
        results.query_type = "document"
        results.query_info = f"{pdf_path}#page={page_number}"
        
        return results
    
    def search_by_text_with_exclusion(
        self,
        positive_query: str,
        negative_terms: List[str],
        model: EmbeddingModelWrapper,
        top_k: int = DEFAULT_TOP_K,
        exclusion_threshold: float = 0.3,
        instruction: Optional[str] = None,
    ) -> SearchResults:
        """
        Search for images by text query while excluding results similar to negative terms.
        
        This uses a two-pass approach:
        1. First pass: Get top_k*2 results for positive query
        2. Second pass: Filter out results similar to any negative term
        
        Args:
            positive_query: Main search query
            negative_terms: List of terms to exclude (e.g., ["beach", "sand"])
            model: Embedding model
            top_k: Number of final results
            exclusion_threshold: Similarity threshold for exclusion (0-1). 
                                 Results with similarity >= threshold to negative terms are filtered.
            instruction: Optional instruction for the model
            
        Returns:
            SearchResults object
        """
        # First pass: get more candidates than needed
        candidate_count = min(top_k * 3, 500)  # Get 3x candidates for filtering
        candidates = self.search_by_text(
            text=positive_query,
            model=model,
            top_k=candidate_count,
            instruction=instruction,
        )
        
        if not candidates.results or not negative_terms:
            # No negatives, just return top_k
            return SearchResults(
                results=candidates.results[:top_k],
                query_type="text_with_exclusion",
                query_info=f"{positive_query} NOT ({', '.join(negative_terms)})",
            )
        
        # Encode all negative terms
        negative_embeddings = []
        for term in negative_terms:
            neg_vec = model.encode_text(term, instruction=instruction)
            negative_embeddings.append(neg_vec)
        
        # Get embeddings for candidates from FAISS index
        filtered_results = []
        for result in candidates.results:
            with self._lock:
                try:
                    candidate_vec = self._faiss_index.reconstruct(result.vector_id)
                except RuntimeError:
                    # Can't reconstruct, skip this result
                    continue
            
            # Check similarity to all negative terms
            should_exclude = False
            for neg_vec in negative_embeddings:
                # Compute cosine similarity (vectors are already normalized)
                similarity = float(np.dot(candidate_vec, neg_vec.flatten()))
                if similarity >= exclusion_threshold:
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        return SearchResults(
            results=filtered_results[:top_k],
            query_type="text_with_exclusion",
            query_info=f"{positive_query} NOT ({', '.join(negative_terms)})",
        )
    
    @staticmethod
    def merge_search_results(
        results_list: List[SearchResults],
        top_k: int = DEFAULT_TOP_K,
        normalize_scores: bool = True,
    ) -> SearchResults:
        """
        Merge results from multiple indexes, deduplicating by file path.
        
        Args:
            results_list: List of SearchResults from different indexes
            top_k: Maximum results to return
            normalize_scores: Whether to normalize scores to 0-1 range per index
            
        Returns:
            Merged SearchResults
        """
        if not results_list:
            return SearchResults(results=[], query_type="multi_index", query_info="")
        
        # Collect all results with normalization
        all_results: Dict[str, SearchResult] = {}
        query_infos = []
        
        for sr in results_list:
            if sr.query_info:
                query_infos.append(sr.query_info)
            
            if not sr.results:
                continue
            
            # Normalize scores if requested
            if normalize_scores and sr.results:
                max_score = max(r.score for r in sr.results)
                min_score = min(r.score for r in sr.results)
                score_range = max_score - min_score if max_score > min_score else 1.0
            
            for result in sr.results:
                # Normalize score to 0-1
                if normalize_scores and score_range > 0:
                    normalized_score = (result.score - min_score) / score_range
                else:
                    normalized_score = result.score
                
                # Deduplicate by file path, keeping highest score
                if result.file_path in all_results:
                    if normalized_score > all_results[result.file_path].score:
                        all_results[result.file_path] = SearchResult(
                            file_path=result.file_path,
                            thumbnail_path=result.thumbnail_path,
                            score=normalized_score,
                            vector_id=result.vector_id,
                        )
                else:
                    all_results[result.file_path] = SearchResult(
                        file_path=result.file_path,
                        thumbnail_path=result.thumbnail_path,
                        score=normalized_score,
                        vector_id=result.vector_id,
                    )
        
        # Sort by score descending and take top_k
        merged = sorted(all_results.values(), key=lambda x: x.score, reverse=True)[:top_k]
        
        return SearchResults(
            results=merged,
            query_type="multi_index",
            query_info=" + ".join(query_infos) if query_infos else "multi-index search",
        )

    def index_folder(
        self,
        folder_path: str,
        model: EmbeddingModelWrapper,
        recursive: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE,
        include_videos: bool = True,
        include_documents: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, int]:
        """
        Index all images, videos, and documents in a folder.
        
        Args:
            folder_path: Path to folder
            model: Embedding model
            recursive: Search subdirectories
            batch_size: Images per batch
            thumbnail_size: Thumbnail size
            include_videos: If True, also index video files
            include_documents: If True, also index PDF documents
            progress_callback: Callback(current, total, status)
            
        Returns:
            Dict with stats: {"added": n, "skipped": n, "failed": n, "videos": n, "documents": n, "pages": n}
        """
        folder_path = Path(folder_path)
        
        # Collect media files (images and optionally videos)
        all_media = collect_media_from_folder(folder_path, recursive, include_videos)
        
        # Collect documents if requested
        all_documents = []
        if include_documents and check_pymupdf():
            all_documents_raw = collect_documents_from_folder(folder_path, recursive)
            # Filter out already-indexed documents
            for doc_path in all_documents_raw:
                mtime, size = get_file_info(doc_path)
                if self.db.image_needs_update(str(doc_path), mtime, size):
                    all_documents.append(doc_path)
            docs_skipped = len(all_documents_raw) - len(all_documents)
            if docs_skipped > 0:
                skipped += docs_skipped
        
        total = len(all_media) + len(all_documents)
        
        if progress_callback:
            progress_callback(0, total, f"Found {len(all_media)} media, {len(all_documents)} documents")
        
        # Separate images and videos
        images_to_process = []
        videos_to_process = []
        skipped = 0
        
        for media_path in all_media:
            mtime, size = get_file_info(media_path)
            if self.db.image_needs_update(str(media_path), mtime, size):
                if is_video_file(media_path):
                    videos_to_process.append(media_path)
                else:
                    images_to_process.append(media_path)
            else:
                skipped += 1
        
        if progress_callback:
            progress_callback(skipped, total, f"Processing {len(images_to_process)} images, {len(videos_to_process)} videos")
        
        # Track folder in database
        self.db.add_folder(str(folder_path), recursive)
        
        # Process images in batches
        added = 0
        failed = 0
        videos_added = 0
        
        import time as time_module
        total_thumb_time = 0
        total_encode_time = 0
        total_db_time = 0
        batch_count = 0
        indexing_start = time_module.time()
        
        print(f"[SemanticSearch] Starting indexing: {len(images_to_process)} images, {len(videos_to_process)} videos, {len(all_documents)} documents")
        print(f"[SemanticSearch] Batch size: {batch_size}, Thumbnail size: {thumbnail_size}")
        
        for batch_start in range(0, len(images_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(images_to_process))
            batch_paths = images_to_process[batch_start:batch_end]
            
            # Generate thumbnails
            t0 = time_module.time()
            thumbnails = []
            valid_paths = []
            valid_infos = []
            
            for img_path in batch_paths:
                thumb = self.thumbnails.get_or_create_thumbnail(img_path)
                if thumb:
                    thumbnails.append(thumb)
                    valid_paths.append(img_path)
                    valid_infos.append(get_file_info(img_path))
                else:
                    failed += 1
            
            if not thumbnails:
                continue
            
            thumb_time = time_module.time() - t0
            total_thumb_time += thumb_time
            
            # Compute embeddings from thumbnails
            try:
                t1 = time_module.time()
                inputs = [{"image": str(t)} for t in thumbnails]
                embeddings = model.encode(
                    inputs,
                    instruction="Represent this image for retrieval.",
                )
                encode_time = time_module.time() - t1
                total_encode_time += encode_time
                
                # Add to index
                t2 = time_module.time()
                self.add_vectors(
                    embeddings,
                    [str(p) for p in valid_paths],
                    [str(t) for t in thumbnails],
                    valid_infos,
                )
                db_time = time_module.time() - t2
                total_db_time += db_time
                
                added += len(valid_paths)
                batch_count += 1
                
                # Log timing every 10 batches
                if batch_count % 10 == 0:
                    avg_encode = total_encode_time / batch_count
                    avg_thumb = total_thumb_time / batch_count
                    avg_db = total_db_time / batch_count
                    imgs_per_sec = (batch_count * batch_size) / (total_encode_time + total_thumb_time + total_db_time)
                    print(f"[SemanticSearch] Batch {batch_count}: Thumbnail={avg_thumb:.2f}s, Encode={avg_encode:.2f}s, DB={avg_db:.2f}s, Throughput={imgs_per_sec:.2f} img/s")
                
            except Exception as e:
                print(f"[SemanticSearch] Batch encoding failed: {e}")
                failed += len(valid_paths)
            
            # Progress update
            processed = skipped + added + failed
            if progress_callback:
                progress_callback(processed, total, f"Indexed {added} images")
            
            # Checkpoint save
            if (batch_start // batch_size) % CHECKPOINT_INTERVAL == 0:
                self.save()
        
        # Image phase summary
        if batch_count > 0:
            images_elapsed = time_module.time() - indexing_start
            print(f"[SemanticSearch] === IMAGES COMPLETE ===")
            print(f"[SemanticSearch]   Indexed: {added} images in {images_elapsed:.1f}s ({added/images_elapsed:.2f} img/s)")
            print(f"[SemanticSearch]   Avg per batch: Thumb={total_thumb_time/batch_count:.2f}s, Encode={total_encode_time/batch_count:.2f}s, DB={total_db_time/batch_count:.2f}s")
        
        # Process videos one at a time (they're more complex)
        video_start = time_module.time()
        for i, video_path in enumerate(videos_to_process):
            try:
                video_added = self._index_video(video_path, model, thumbnail_size)
                if video_added:
                    videos_added += 1
                    added += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[SemanticSearch] Video encoding failed for {video_path}: {e}")
                failed += 1
            
            # Progress update
            processed = skipped + added + failed
            if progress_callback:
                progress_callback(processed, total, f"Indexed {added} media ({videos_added} videos)")
            
            # Checkpoint save periodically
            if i % 10 == 0:
                self.save()
        
        # Video phase summary
        if videos_added > 0:
            video_elapsed = time_module.time() - video_start
            print(f"[SemanticSearch] === VIDEOS COMPLETE ===")
            print(f"[SemanticSearch]   Indexed: {videos_added} videos in {video_elapsed:.1f}s ({videos_added/video_elapsed:.2f} vid/s)")
        
        # Process documents (PDFs) - each page becomes a separate entry
        documents_added = 0
        pages_added = 0
        doc_start = time_module.time()
        
        for i, doc_path in enumerate(all_documents):
            try:
                doc_pages = self._index_pdf(doc_path, model, thumbnail_size)
                if doc_pages > 0:
                    documents_added += 1
                    pages_added += doc_pages
                    added += doc_pages
                    # Log progress every 10 documents
                    if (i + 1) % 10 == 0:
                        doc_elapsed = time_module.time() - doc_start
                        print(f"[SemanticSearch] Documents: {documents_added} docs, {pages_added} pages in {doc_elapsed:.1f}s")
                else:
                    failed += 1
            except Exception as e:
                print(f"[SemanticSearch] PDF encoding failed for {doc_path}: {e}")
                failed += 1
            
            # Progress update
            processed_files = len(all_media) + i + 1
            if progress_callback:
                progress_callback(processed_files, total, f"Indexed {added} items ({documents_added} docs, {pages_added} pages)")
            
            # Checkpoint save periodically
            if i % 5 == 0:
                self.save()
        
        # Document phase summary
        if documents_added > 0:
            doc_elapsed = time_module.time() - doc_start
            print(f"[SemanticSearch] === DOCUMENTS COMPLETE ===")
            print(f"[SemanticSearch]   Indexed: {documents_added} docs ({pages_added} pages) in {doc_elapsed:.1f}s ({pages_added/doc_elapsed:.2f} pages/s)")
        
        # Final save
        self.save()
        
        # Total summary
        total_elapsed = time_module.time() - indexing_start
        print(f"[SemanticSearch] === INDEXING COMPLETE ===")
        print(f"[SemanticSearch]   Total: {added} items indexed, {skipped} skipped, {failed} failed")
        print(f"[SemanticSearch]   Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        if added > 0:
            print(f"[SemanticSearch]   Average: {total_elapsed/added:.2f}s per item, {added/total_elapsed:.2f} items/s")
        
        # Update config with model name
        self._config["model_used"] = model.model_name
        self._config["last_indexed"] = datetime.now().isoformat()
        self._save_config()
        
        return {
            "added": added, 
            "skipped": skipped, 
            "failed": failed, 
            "videos": videos_added,
            "documents": documents_added,
            "pages": pages_added,
        }
    
    def _index_video(
        self,
        video_path: Path,
        model: EmbeddingModelWrapper,
        thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE,
    ) -> bool:
        """
        Index a single video file.
        
        Args:
            video_path: Path to video file
            model: Embedding model
            thumbnail_size: Size for thumbnail
            
        Returns:
            True if successfully indexed, False otherwise
        """
        from .config import VIDEO_DEFAULT_FPS, VIDEO_MAX_FRAMES, VIDEO_MIN_FRAMES
        
        video_path = Path(video_path)
        
        # Get video info
        video_info = get_video_info(video_path)
        if video_info is None:
            print(f"[SemanticSearch] Failed to get video info: {video_path}")
            return False
        
        # Extract frames using FPS-based sampling (2 fps default)
        frames = extract_frames(
            video_path,
            max_frames=VIDEO_MAX_FRAMES,
            target_fps=VIDEO_DEFAULT_FPS,
            uniform_sampling=False,  # Use FPS-based sampling for better temporal coverage
        )
        
        if not frames or len(frames) < VIDEO_MIN_FRAMES:
            print(f"[SemanticSearch] Failed to extract frames from: {video_path}")
            return False
        
        # Create thumbnail
        thumb_path = self.thumbnails.get_or_create_thumbnail(video_path)
        if thumb_path is None:
            print(f"[SemanticSearch] Failed to create thumbnail for: {video_path}")
            return False
        
        # Generate embedding from frames (single embedding for the whole video)
        # Uses batched processing and lower resolution to avoid OOM
        try:
            embedding = model.encode_video_frames(
                frames,
                instruction="Represent this video for retrieval.",
                max_frames_per_batch=8,  # Process 8 frames at a time
                frame_max_pixels=VIDEO_FRAME_MAX_PIXELS,  # 512x512 default
            )
            embedding = embedding.reshape(1, -1)  # Ensure 2D array
            
            # Add to index with video metadata
            mtime, size = get_file_info(video_path)
            self.add_video_vector(
                embedding,
                str(video_path),
                str(thumb_path),
                (mtime, size),
                frame_count=len(frames),
                duration_seconds=video_info.duration_seconds,
            )
            
            # Clear CUDA cache after video processing to free memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"[SemanticSearch] Failed to encode video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            # Try to clear cache even on failure
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
    
    def add_video_vector(
        self,
        embedding: np.ndarray,
        file_path: str,
        thumbnail_path: str,
        file_info: Tuple[float, int],
        frame_count: int,
        duration_seconds: float,
    ):
        """
        Add a single video embedding to the index.
        
        Args:
            embedding: Video embedding (1, dim)
            file_path: Path to video file
            thumbnail_path: Path to thumbnail
            file_info: (mtime, size) tuple
            frame_count: Number of frames extracted
            duration_seconds: Video duration
        """
        # Use add_vectors with video metadata
        self.add_vectors(
            vectors=embedding,
            file_paths=[file_path],
            thumbnail_paths=[thumbnail_path],
            file_infos=[file_info],
            media_types=["video"],
            frame_counts=[frame_count],
            durations=[duration_seconds],
        )
    
    def _index_pdf(
        self,
        pdf_path: Path,
        model: EmbeddingModelWrapper,
        thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE,
    ) -> int:
        """
        Index a single PDF file (all pages).
        
        Args:
            pdf_path: Path to PDF file
            model: Embedding model
            thumbnail_size: Size for thumbnail
            
        Returns:
            Number of pages successfully indexed (0 if failed)
        """
        from .config import PDF_DPI, PDF_MAX_PAGES
        
        pdf_path = Path(pdf_path)
        
        # Get PDF info
        pdf_info = get_pdf_info(str(pdf_path))
        if pdf_info is None:
            print(f"[SemanticSearch] Failed to get PDF info: {pdf_path}")
            return 0
        
        # Extract all pages as images
        pages = extract_all_pages(
            str(pdf_path),
            max_pages=PDF_MAX_PAGES,
            dpi=PDF_DPI,
        )
        
        if not pages:
            print(f"[SemanticSearch] Failed to extract pages from: {pdf_path}")
            return 0
        
        mtime, size = get_file_info(pdf_path)
        pages_indexed = 0
        
        # Index each page separately
        for page in pages:
            try:
                # Create a virtual path for this page
                page_path = get_pdf_page_path(str(pdf_path), page.page_number)
                
                # Check if already indexed
                if not self.db.image_needs_update(page_path, mtime, size):
                    pages_indexed += 1  # Already indexed, count as success
                    continue
                
                # Create thumbnail for this page
                thumb_path = self.thumbnails.create_thumbnail(page_path, force=True, page_number=page.page_number)
                if thumb_path is None:
                    # Try using the page image directly
                    page.image.thumbnail((thumbnail_size, thumbnail_size))
                    thumb_path = self.thumbnails.thumbnails_path / f"{pdf_path.stem}_p{page.page_number}.jpg"
                    page.image.save(thumb_path, "JPEG", quality=85)
                
                # Generate embedding from page image
                embedding = model.encode(
                    [{"image": page.image}],
                    instruction="Represent this document page for retrieval.",
                )
                
                # Add to index
                self.add_document_vector(
                    embedding=embedding,
                    page_path=page_path,
                    thumbnail_path=str(thumb_path),
                    file_info=(mtime, size),
                    page_number=page.page_number,
                    parent_document=str(pdf_path),
                    total_pages=pdf_info.page_count,
                )
                
                pages_indexed += 1
                
            except Exception as e:
                print(f"[SemanticSearch] Failed to index page {page.page_number} of {pdf_path}: {e}")
                continue
        
        return pages_indexed
    
    def add_document_vector(
        self,
        embedding: np.ndarray,
        page_path: str,
        thumbnail_path: str,
        file_info: Tuple[float, int],
        page_number: int,
        parent_document: str,
        total_pages: int = None,
    ):
        """
        Add a single document page embedding to the index.
        
        Args:
            embedding: Page embedding (1, dim)
            page_path: Virtual path (e.g., "file.pdf#page=3")
            thumbnail_path: Path to thumbnail
            file_info: (mtime, size) tuple
            page_number: Page number (1-indexed)
            parent_document: Path to parent PDF
            total_pages: Total pages in document (optional, for metadata)
        """
        # Ensure 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        with self._lock:
            # Add to FAISS
            self._faiss_index.add(embedding.astype(np.float32))
            vector_id = self._faiss_index.ntotal - 1
            
            # Add to database with document metadata
            mtime, size = file_info
            file_hash = f"{size}_{mtime}"
            
            self.db.add_image(
                file_path=page_path,
                file_hash=file_hash,
                file_mtime=mtime,
                file_size=size,
                thumbnail_path=thumbnail_path,
                vector_id=vector_id,
                media_type="document",
                page_number=page_number,
                parent_document=parent_document,
            )
    
    def remove_folder(self, folder_path: str) -> int:
        """
        Remove a folder and all its images from the index.
        
        Note: This marks records for removal but doesn't compact the FAISS index.
        Call rebuild() to reclaim space.
        
        Args:
            folder_path: Path to folder
            
        Returns:
            Number of images removed
        """
        # Remove from database (and get count)
        count = self.db.remove_images_in_folder(folder_path)
        
        # Remove folder tracking
        self.db.remove_folder(folder_path)
        
        return count
    
    def validate(self, auto_clean: bool = False) -> Dict[str, Any]:
        """
        Validate the index and find stale entries.
        
        Args:
            auto_clean: If True, remove stale entries
            
        Returns:
            Dict with validation results
        """
        stale = self.db.get_stale_images()
        stale_count = len(stale)
        
        if auto_clean and stale:
            paths = [s.file_path for s in stale]
            self.db.remove_images_batch(paths)
            
            # Also clean up orphaned thumbnails
            valid_paths = {r.file_path for r in self.db.get_all_images()}
            orphaned = self.thumbnails.cleanup_orphaned_thumbnails(valid_paths)
        else:
            orphaned = 0
        
        return {
            "stale_count": stale_count,
            "stale_paths": [s.file_path for s in stale] if not auto_clean else [],
            "cleaned": auto_clean,
            "orphaned_thumbnails_removed": orphaned,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        db_stats = self.db.get_stats()
        
        # Get FAISS index stats
        faiss_stats = IndexFactory.get_index_stats(self._faiss_index)
        
        # Compaction info
        needs_compact, deleted_count, wasted_pct = self.needs_compaction()
        
        return {
            **db_stats,
            "vector_count": self._faiss_index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": faiss_stats["index_type"],
            "is_trained": faiss_stats["is_trained"],
            "faiss_path": str(self.faiss_path),
            "faiss_size_mb": round(self.faiss_path.stat().st_size / (1024 * 1024), 2) if self.faiss_path.exists() else 0,
            "thumbnail_count": self.thumbnails.get_thumbnail_count(),
            "thumbnail_cache_mb": self.thumbnails.get_cache_size_mb(),
            "model_used": self._config.get("model_used"),
            "created_at": self._config.get("created_at"),
            "last_indexed": self._config.get("last_indexed"),
            "index_params": faiss_stats,
            # Compaction stats
            "deleted_vectors": deleted_count,
            "wasted_percentage": round(wasted_pct, 1),
            "needs_compaction": needs_compact,
        }
    
    def rebuild(
        self,
        new_index_type: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Rebuild the FAISS index, optionally changing the index type.
        
        This also compacts the index by removing gaps from deleted entries.
        
        Args:
            new_index_type: New index type ("flat", "ivf_flat", "hnsw", "ivf_pq")
                          If None, keeps the current type
            progress_callback: Optional callback(current, total, status)
            
        Returns:
            Dict with rebuild stats
        """
        with self._lock:
            old_type = IndexFactory.detect_index_type(self._faiss_index)
            old_count = self._faiss_index.ntotal
            
            # Determine new config
            if new_index_type:
                new_type = IndexType.from_string(new_index_type)
            else:
                new_type = old_type
            
            new_config = IndexConfig(
                index_type=new_type,
                dimension=self.embedding_dim,
                nlist=self._config.get("index_params", {}).get("nlist", 100),
                nprobe=self._config.get("index_params", {}).get("nprobe", 10),
                hnsw_m=self._config.get("index_params", {}).get("hnsw_m", 32),
                hnsw_ef_search=self._config.get("index_params", {}).get("hnsw_ef_search", 64),
            )
            
            # Rebuild
            new_index, success = IndexFactory.rebuild_index(
                self._faiss_index,
                new_config,
                progress_callback,
            )
            
            if success:
                self._faiss_index = new_index
                self._index_config = new_config
                self._config["index_type"] = new_type.value
                self._config["index_params"] = new_config.to_dict()
                self.save()
                
                return {
                    "success": True,
                    "old_type": old_type.value,
                    "new_type": new_type.value,
                    "old_count": old_count,
                    "new_count": new_index.ntotal,
                }
            else:
                return {
                    "success": False,
                    "error": "Rebuild failed - could not extract vectors from old index",
                }
    
    def compact(
        self,
        new_index_type: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Compact the index by removing deleted vectors.
        
        This rebuilds the FAISS index keeping only vectors that are still
        referenced in the database, reassigns sequential vector IDs, and
        updates the database mappings.
        
        Args:
            new_index_type: Optionally change index type during compaction
            progress_callback: Optional callback(current, total, status)
            
        Returns:
            Dict with compaction stats
        """
        with self._lock:
            old_count = self._faiss_index.ntotal
            deleted_count = self.db.get_deleted_vector_count()
            
            if deleted_count == 0:
                return {
                    "success": True,
                    "compacted": False,
                    "message": "No deleted vectors to compact",
                    "vector_count": old_count,
                }
            
            # Get valid vector IDs from database
            valid_vector_ids = self.db.get_all_vector_ids()
            new_count = len(valid_vector_ids)
            
            if progress_callback:
                progress_callback(0, new_count, f"Compacting: {old_count} -> {new_count} vectors...")
            
            # Determine new config
            if new_index_type:
                new_type = IndexType.from_string(new_index_type)
            else:
                new_type = IndexFactory.detect_index_type(self._faiss_index)
            
            new_config = IndexConfig(
                index_type=new_type,
                dimension=self.embedding_dim,
                nlist=self._config.get("index_params", {}).get("nlist", 100),
                nprobe=self._config.get("index_params", {}).get("nprobe", 10),
                hnsw_m=self._config.get("index_params", {}).get("hnsw_m", 32),
                hnsw_ef_search=self._config.get("index_params", {}).get("hnsw_ef_search", 64),
            )
            
            # Compact the FAISS index
            new_index, id_mapping, success = IndexFactory.compact_index(
                self._faiss_index,
                valid_vector_ids,
                new_config,
                progress_callback,
            )
            
            if not success:
                return {
                    "success": False,
                    "error": "Compaction failed - could not extract vectors from index",
                }
            
            # Update database with new vector IDs
            if progress_callback:
                progress_callback(new_count, new_count, "Updating database mappings...")
            
            self.db.update_vector_ids(id_mapping)
            
            # Clear deleted vectors table
            self.db.clear_deleted_vectors()
            
            # Update index
            self._faiss_index = new_index
            self._index_config = new_config
            self._config["index_type"] = new_type.value
            self._config["index_params"] = new_config.to_dict()
            self.save()
            
            # Vacuum database
            self.db.vacuum()
            
            if progress_callback:
                progress_callback(new_count, new_count, "Compaction complete")
            
            return {
                "success": True,
                "compacted": True,
                "old_count": old_count,
                "new_count": new_count,
                "removed_count": old_count - new_count,
                "index_type": new_type.value,
            }
    
    def needs_compaction(self) -> Tuple[bool, int, float]:
        """
        Check if compaction is recommended.
        
        Returns:
            (needs_compaction, deleted_count, wasted_percentage)
        """
        deleted_count = self.db.get_deleted_vector_count()
        total_vectors = self._faiss_index.ntotal
        
        if total_vectors == 0:
            return False, 0, 0.0
        
        wasted_pct = (deleted_count / total_vectors) * 100
        
        # Recommend compaction if >10% wasted or >1000 deleted vectors
        needs_compact = wasted_pct > 10.0 or deleted_count > 1000
        
        return needs_compact, deleted_count, wasted_pct
    
    def get_index_type(self) -> str:
        """Get the current index type as a string"""
        return IndexFactory.detect_index_type(self._faiss_index).value
    
    def close(self):
        """Close the index and free resources"""
        self.save()
        self.db.close()


# Global index cache
_index_cache: Dict[str, SemanticIndex] = {}
_index_lock = threading.Lock()


def get_or_create_index(
    index_name: str,
    embedding_dim: int = 4096,
    index_type: str = "flat",
) -> SemanticIndex:
    """
    Get or create a cached index instance.
    
    Args:
        index_name: Name of the index
        embedding_dim: Embedding dimension (only used for new indexes)
        index_type: Type of index ("flat", "ivf_flat", "hnsw", "ivf_pq")
        
    Returns:
        SemanticIndex instance
    """
    with _index_lock:
        if index_name not in _index_cache:
            _index_cache[index_name] = SemanticIndex(index_name, embedding_dim, index_type)
        return _index_cache[index_name]


def list_indexes() -> List[str]:
    """List all available indexes"""
    from .config import INDEXES_PATH
    
    indexes = []
    if INDEXES_PATH.exists():
        for item in INDEXES_PATH.iterdir():
            if item.is_dir() and (item / "metadata.db").exists():
                indexes.append(item.name)
    return indexes


def delete_index(index_name: str) -> bool:
    """
    Delete an index and all its data.
    
    Args:
        index_name: Name of the index to delete
        
    Returns:
        True if deleted, False if not found
    """
    import shutil
    
    with _index_lock:
        # Remove from cache
        if index_name in _index_cache:
            _index_cache[index_name].close()
            del _index_cache[index_name]
    
    # Delete directory
    index_path = get_index_path(index_name)
    if index_path.exists():
        shutil.rmtree(index_path)
        return True
    
    return False

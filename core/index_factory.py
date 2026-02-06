"""
FAISS Index Factory - creates and manages different index types

Supported index types:
- Flat: Exact search, slowest but 100% accurate
- IVFFlat: Inverted file index, fast approximate search
- HNSW: Hierarchical Navigable Small World, very fast approximate search
- IVFPQ: Product Quantization, fastest but lower accuracy (for very large indexes)
"""

import faiss
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IndexType(Enum):
    """Supported FAISS index types"""
    FLAT = "flat"           # Exact search - IndexFlatIP
    IVF_FLAT = "ivf_flat"   # Fast approximate - IndexIVFFlat
    HNSW = "hnsw"           # Very fast approximate - IndexHNSWFlat
    IVF_PQ = "ivf_pq"       # Compressed approximate - IndexIVFPQ
    
    @classmethod
    def from_string(cls, s: str) -> "IndexType":
        """Convert string to IndexType"""
        s = s.lower().replace("-", "_")
        for member in cls:
            if member.value == s:
                return member
        # Fallback
        return cls.FLAT


@dataclass
class IndexConfig:
    """Configuration for a FAISS index"""
    index_type: IndexType
    dimension: int
    
    # IVF parameters
    nlist: int = 100            # Number of clusters for IVF
    nprobe: int = 10            # Number of clusters to search
    
    # HNSW parameters
    hnsw_m: int = 32            # Number of connections per layer
    hnsw_ef_construction: int = 200  # Construction-time search depth
    hnsw_ef_search: int = 64    # Search-time search depth
    
    # PQ parameters
    pq_m: int = 8               # Number of subquantizers
    pq_bits: int = 8            # Bits per subquantizer
    
    # Training status
    is_trained: bool = False
    training_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "index_type": self.index_type.value,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "pq_m": self.pq_m,
            "pq_bits": self.pq_bits,
            "is_trained": self.is_trained,
            "training_size": self.training_size,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IndexConfig":
        """Create from dictionary"""
        return cls(
            index_type=IndexType.from_string(d.get("index_type", "flat")),
            dimension=d.get("dimension", 4096),
            nlist=d.get("nlist", 100),
            nprobe=d.get("nprobe", 10),
            hnsw_m=d.get("hnsw_m", 32),
            hnsw_ef_construction=d.get("hnsw_ef_construction", 200),
            hnsw_ef_search=d.get("hnsw_ef_search", 64),
            pq_m=d.get("pq_m", 8),
            pq_bits=d.get("pq_bits", 8),
            is_trained=d.get("is_trained", False),
            training_size=d.get("training_size", 0),
        )


# Default parameters for each index type
INDEX_TYPE_DESCRIPTIONS = {
    IndexType.FLAT: {
        "name": "Flat (Exact)",
        "description": "Exact search, 100% accurate but slowest. Best for < 50K images.",
        "needs_training": False,
        "supports_removal": False,
    },
    IndexType.IVF_FLAT: {
        "name": "IVF-Flat (Fast)",
        "description": "Approximate search using clustering. ~95-99% recall, much faster. Best for 50K-500K images.",
        "needs_training": True,
        "supports_removal": False,
    },
    IndexType.HNSW: {
        "name": "HNSW (Very Fast)",
        "description": "Graph-based search, excellent speed/accuracy tradeoff. Best for 10K-1M images.",
        "needs_training": False,
        "supports_removal": False,  # HNSW doesn't support removal in FAISS
    },
    IndexType.IVF_PQ: {
        "name": "IVF-PQ (Compressed)",
        "description": "Compressed vectors, lowest memory but ~90-95% recall. Best for 1M+ images.",
        "needs_training": True,
        "supports_removal": False,
    },
}


class IndexFactory:
    """Factory for creating and managing FAISS indexes"""
    
    @staticmethod
    def create_index(config: IndexConfig) -> faiss.Index:
        """
        Create a new FAISS index based on configuration.
        
        Args:
            config: Index configuration
            
        Returns:
            FAISS index (may need training before use for IVF types)
        """
        dim = config.dimension
        
        if config.index_type == IndexType.FLAT:
            # Simple flat index - exact search
            index = faiss.IndexFlatIP(dim)
            
        elif config.index_type == IndexType.IVF_FLAT:
            # IVF with flat storage - needs training
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, config.nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = config.nprobe
            
        elif config.index_type == IndexType.HNSW:
            # HNSW graph index - no training needed
            index = faiss.IndexHNSWFlat(dim, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = config.hnsw_ef_construction
            index.hnsw.efSearch = config.hnsw_ef_search
            
        elif config.index_type == IndexType.IVF_PQ:
            # IVF with product quantization - needs training
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, config.nlist, config.pq_m, config.pq_bits)
            index.nprobe = config.nprobe
            
        else:
            raise ValueError(f"Unknown index type: {config.index_type}")
        
        return index
    
    @staticmethod
    def needs_training(index: faiss.Index) -> bool:
        """Check if an index needs training before vectors can be added"""
        # IVF and PQ indexes need training
        if hasattr(index, 'is_trained'):
            return not index.is_trained
        return False
    
    @staticmethod
    def train_index(index: faiss.Index, vectors: np.ndarray) -> bool:
        """
        Train an index on sample vectors.
        
        Args:
            index: FAISS index to train
            vectors: Training vectors (should be representative sample)
            
        Returns:
            True if training succeeded
        """
        if not IndexFactory.needs_training(index):
            return True
        
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        try:
            index.train(vectors)
            return index.is_trained
        except Exception as e:
            print(f"[SemanticSearch] Index training failed: {e}")
            return False
    
    @staticmethod
    def get_min_training_size(config: IndexConfig) -> int:
        """Get minimum number of vectors needed for training"""
        if config.index_type in (IndexType.IVF_FLAT, IndexType.IVF_PQ):
            # Need at least nlist vectors, but more is better
            return max(config.nlist * 10, 1000)
        return 0
    
    @staticmethod
    def detect_index_type(index: faiss.Index) -> IndexType:
        """Detect the type of an existing FAISS index"""
        index_str = str(type(index).__name__)
        
        if "IVFPQ" in index_str or isinstance(index, faiss.IndexIVFPQ):
            return IndexType.IVF_PQ
        elif "IVFFlat" in index_str or isinstance(index, faiss.IndexIVFFlat):
            return IndexType.IVF_FLAT
        elif "HNSW" in index_str or isinstance(index, faiss.IndexHNSWFlat):
            return IndexType.HNSW
        else:
            return IndexType.FLAT
    
    @staticmethod
    def get_index_stats(index: faiss.Index) -> Dict[str, Any]:
        """Get statistics about an index"""
        stats = {
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "index_type": IndexFactory.detect_index_type(index).value,
            "is_trained": getattr(index, 'is_trained', True),
        }
        
        # Type-specific stats
        if hasattr(index, 'nlist'):
            stats["nlist"] = index.nlist
        if hasattr(index, 'nprobe'):
            stats["nprobe"] = index.nprobe
        if hasattr(index, 'hnsw'):
            stats["hnsw_m"] = index.hnsw.M if hasattr(index.hnsw, 'M') else None
            stats["hnsw_ef_search"] = index.hnsw.efSearch
        
        return stats
    
    @staticmethod
    def rebuild_index(
        old_index: faiss.Index,
        new_config: IndexConfig,
        progress_callback=None,
    ) -> Tuple[faiss.Index, bool]:
        """
        Rebuild an index with a new configuration.
        
        This extracts all vectors from the old index and adds them to a new one.
        
        Args:
            old_index: Existing FAISS index
            new_config: Configuration for new index
            progress_callback: Optional callback(current, total, status)
            
        Returns:
            (new_index, success)
        """
        n_vectors = old_index.ntotal
        dim = old_index.d
        
        if n_vectors == 0:
            # Empty index, just create new one
            return IndexFactory.create_index(new_config), True
        
        if progress_callback:
            progress_callback(0, n_vectors, "Extracting vectors from old index...")
        
        # Extract all vectors from old index
        # For most index types, we can reconstruct vectors
        try:
            vectors = np.zeros((n_vectors, dim), dtype=np.float32)
            for i in range(n_vectors):
                vectors[i] = old_index.reconstruct(i)
        except RuntimeError:
            # Some index types don't support reconstruction (e.g., some PQ variants)
            print("[SemanticSearch] Warning: Cannot reconstruct vectors from this index type")
            return old_index, False
        
        if progress_callback:
            progress_callback(n_vectors // 2, n_vectors, "Creating new index...")
        
        # Create new index
        new_index = IndexFactory.create_index(new_config)
        
        # Train if needed
        if IndexFactory.needs_training(new_index):
            if progress_callback:
                progress_callback(n_vectors // 2, n_vectors, "Training new index...")
            
            # Use all vectors for training (or sample if too many)
            train_vectors = vectors
            if n_vectors > 100000:
                # Sample for training
                indices = np.random.choice(n_vectors, 100000, replace=False)
                train_vectors = vectors[indices]
            
            if not IndexFactory.train_index(new_index, train_vectors):
                print("[SemanticSearch] Warning: Training failed, falling back to Flat index")
                new_config.index_type = IndexType.FLAT
                new_index = IndexFactory.create_index(new_config)
        
        if progress_callback:
            progress_callback(n_vectors * 3 // 4, n_vectors, "Adding vectors to new index...")
        
        # Add vectors to new index
        new_index.add(vectors)
        
        if progress_callback:
            progress_callback(n_vectors, n_vectors, "Rebuild complete")
        
        return new_index, True
    
    @staticmethod
    def set_search_params(index: faiss.Index, config: IndexConfig):
        """Update search parameters on an existing index"""
        if hasattr(index, 'nprobe'):
            index.nprobe = config.nprobe
        if hasattr(index, 'hnsw') and hasattr(index.hnsw, 'efSearch'):
            index.hnsw.efSearch = config.hnsw_ef_search
    
    @staticmethod
    def compact_index(
        old_index: faiss.Index,
        valid_vector_ids: list,
        new_config: Optional[IndexConfig] = None,
        progress_callback=None,
    ) -> Tuple[faiss.Index, Dict[int, int], bool]:
        """
        Compact an index by keeping only specified vector IDs.
        
        This extracts only the vectors at the specified IDs, creates a new
        index with sequential IDs (0, 1, 2, ...), and returns a mapping
        from old IDs to new IDs.
        
        Args:
            old_index: Existing FAISS index
            valid_vector_ids: List of vector IDs to keep (in desired order)
            new_config: Configuration for new index (default: detect from old)
            progress_callback: Optional callback(current, total, status)
            
        Returns:
            (new_index, id_mapping, success)
            id_mapping maps old_vector_id -> new_vector_id
        """
        n_valid = len(valid_vector_ids)
        n_total = old_index.ntotal
        dim = old_index.d
        
        if n_valid == 0:
            # No vectors to keep, return empty index
            if new_config is None:
                new_config = IndexConfig(
                    index_type=IndexFactory.detect_index_type(old_index),
                    dimension=dim,
                )
            return IndexFactory.create_index(new_config), {}, True
        
        if progress_callback:
            progress_callback(0, n_valid, f"Extracting {n_valid} vectors (removing {n_total - n_valid} deleted)...")
        
        # Sort valid IDs to extract in order
        sorted_valid_ids = sorted(valid_vector_ids)
        
        # Create ID mapping: old_id -> new_sequential_id
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_valid_ids)}
        
        # Extract only the valid vectors
        try:
            vectors = np.zeros((n_valid, dim), dtype=np.float32)
            for new_id, old_id in enumerate(sorted_valid_ids):
                vectors[new_id] = old_index.reconstruct(old_id)
                
                if progress_callback and (new_id + 1) % 1000 == 0:
                    progress_callback(new_id + 1, n_valid, f"Extracted {new_id + 1}/{n_valid} vectors...")
        except RuntimeError as e:
            print(f"[SemanticSearch] Warning: Cannot reconstruct vectors: {e}")
            return old_index, {}, False
        
        if progress_callback:
            progress_callback(n_valid, n_valid, "Creating compacted index...")
        
        # Determine new config
        if new_config is None:
            detected_type = IndexFactory.detect_index_type(old_index)
            new_config = IndexConfig(
                index_type=detected_type,
                dimension=dim,
            )
        
        # Create new index
        new_index = IndexFactory.create_index(new_config)
        
        # Train if needed
        if IndexFactory.needs_training(new_index):
            if progress_callback:
                progress_callback(n_valid, n_valid, "Training compacted index...")
            
            # Use all vectors for training (or sample if too many)
            train_vectors = vectors
            if n_valid > 100000:
                indices = np.random.choice(n_valid, 100000, replace=False)
                train_vectors = vectors[indices]
            
            if not IndexFactory.train_index(new_index, train_vectors):
                print("[SemanticSearch] Warning: Training failed, falling back to Flat index")
                new_config.index_type = IndexType.FLAT
                new_index = IndexFactory.create_index(new_config)
        
        # Add vectors to new index
        new_index.add(vectors)
        
        if progress_callback:
            progress_callback(n_valid, n_valid, f"Compaction complete: {n_total} -> {n_valid} vectors")
        
        return new_index, id_mapping, True

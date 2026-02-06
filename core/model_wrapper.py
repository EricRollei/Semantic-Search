"""
Eric's Semantic Search - Model Wrapper

Description: Unified interface for Qwen3-VL-Embedding and Qwen3-VL-Reranker models.
             Handles model loading, caching, and provides methods for computing
             embeddings and reranking scores. Supports Matryoshka dimension reduction.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Dependencies:
- Qwen3-VL Models (Qwen License): https://huggingface.co/Qwen
- Transformers (Apache 2.0): Hugging Face - https://github.com/huggingface/transformers
- PyTorch (BSD License): Meta Platforms - https://pytorch.org

See LICENSE.md for complete license information.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import threading

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    MODELS_PATH,
    resolve_model_path,
)


# Global model cache to avoid reloading
_model_cache = {}
_cache_lock = threading.Lock()


def get_device(preferred: str = "auto") -> str:
    """Determine the best available device"""
    if preferred != "auto":
        return preferred
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingModelWrapper:
    """
    Wrapper for Qwen3-VL-Embedding model.
    
    Provides a consistent interface for computing embeddings from images and text.
    Supports Matryoshka dimension reduction for smaller, faster indexes.
    """
    
    # Valid Matryoshka dimensions (must be power of 2 or model max)
    MATRYOSHKA_DIMS = {
        "8B": [4096, 2048, 1024, 512, 256],
        "2B": [2048, 1024, 512, 256],
    }
    
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_pixels: int = 1024 * 1024,  # 1MP default - good accuracy/speed balance
        min_pixels: int = 4096,
        attention_type: str = "sdpa",  # sdpa, eager, or sage
        use_flash_attention: bool = False,  # Deprecated, use attention_type instead
        embedding_dim: Optional[int] = None,  # None = full dimension, or specify reduced dim
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name_or_path: Model short name (e.g. "Qwen3-VL-Embedding-8B") or local path
            device: Device to run on ("auto", "cuda", "cuda:0", "cpu", etc.)
            torch_dtype: Data type for model weights
            max_pixels: Maximum pixels per image (affects accuracy and speed)
            min_pixels: Minimum pixels per image
            attention_type: Attention implementation ("sdpa", "eager", "sage")
            use_flash_attention: Deprecated - use attention_type instead
            embedding_dim: Output embedding dimension. None for full dimension,
                          or specify a reduced dimension (1024, 512, 256) for Matryoshka.
        """
        self.model_name = model_name_or_path
        self.resolved_path = resolve_model_path(model_name_or_path)
        self.device = get_device(device)
        self.torch_dtype = torch_dtype
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._requested_dim = embedding_dim  # User-requested dimension
        
        # Handle deprecated use_flash_attention
        if use_flash_attention:
            self.attention_type = "flash_attention_2"
        else:
            self.attention_type = attention_type
        
        self._model = None
        self._initialized = False
        self.embedding_dim = None  # Actual output dimension (set after init)
        self._full_dim = None  # Full model dimension
    
    def _ensure_initialized(self):
        """Lazy initialization of the model"""
        if self._initialized:
            return
        
        print(f"[SemanticSearch] Loading embedding model: {self.resolved_path}")
        print(f"[SemanticSearch] Device: {self.device}, dtype: {self.torch_dtype}")
        
        # Import local Qwen3VLEmbedder
        from .qwen3_vl_embedding import Qwen3VLEmbedder
        
        print(f"[SemanticSearch] Using attention implementation: {self.attention_type}")
        print(f"[SemanticSearch] Max pixels: {self.max_pixels} ({int(self.max_pixels**0.5)}x{int(self.max_pixels**0.5)} equivalent)")
        
        self._model = Qwen3VLEmbedder(
            model_name_or_path=self.resolved_path,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attention_type,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        
        # Determine embedding dimension from model config
        self._determine_embedding_dim()
        self._initialized = True
        print(f"[SemanticSearch] Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _determine_embedding_dim(self):
        """Determine the embedding dimension from model config and user request"""
        # Determine full model dimension
        # For 2B model: 2048, for 8B model: 4096
        if "2B" in self.model_name or "2b" in self.model_name:
            self._full_dim = 2048
            valid_dims = self.MATRYOSHKA_DIMS["2B"]
        else:
            self._full_dim = 4096
            valid_dims = self.MATRYOSHKA_DIMS["8B"]
        
        # Apply user-requested dimension
        if self._requested_dim is None:
            # Use full dimension
            self.embedding_dim = self._full_dim
        elif self._requested_dim in valid_dims:
            self.embedding_dim = self._requested_dim
        else:
            # Invalid dimension requested, use closest valid one
            closest = min(valid_dims, key=lambda x: abs(x - self._requested_dim))
            print(f"[SemanticSearch] Warning: Requested dim {self._requested_dim} not valid for this model. Using {closest}.")
            self.embedding_dim = closest
    
    def encode(
        self,
        inputs: List[Dict[str, Any]],
        instruction: Optional[str] = None,
        batch_size: int = 8,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of inputs into embeddings.
        
        Args:
            inputs: List of input dicts. Each dict can contain:
                - "image": PIL.Image, file path, or URL
                - "text": Text string
                - "instruction": Per-input instruction (overrides global)
            instruction: Global instruction for all inputs
            batch_size: Batch size for processing (used for chunking large batches)
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            NumPy array of shape (n_inputs, embedding_dim)
        """
        self._ensure_initialized()
        
        # Prepare inputs with instruction
        prepared = []
        for inp in inputs:
            item = dict(inp)
            if instruction and "instruction" not in item:
                item["instruction"] = instruction
            prepared.append(item)
        
        # Process in batches if needed
        all_embeddings = []
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i:i + batch_size]
            embeddings = self._model.process(batch, normalize=normalize)
            
            if isinstance(embeddings, torch.Tensor):
                # Convert bfloat16 -> float32 before numpy (numpy doesn't support bfloat16)
                embeddings = embeddings.float().cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        if len(all_embeddings) == 1:
            result = all_embeddings[0].astype(np.float32)
        else:
            result = np.vstack(all_embeddings).astype(np.float32)
        
        # Apply Matryoshka dimension reduction if requested
        if self.embedding_dim < self._full_dim:
            result = result[:, :self.embedding_dim]
            # Re-normalize after truncation (important for Matryoshka!)
            if normalize:
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)  # Avoid div by zero
                result = result / norms
        
        return result
    
    def encode_image(
        self,
        image: Union[str, Path, Image.Image],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a single image.
        
        Args:
            image: PIL Image, file path, or URL
            instruction: Optional instruction
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        if isinstance(image, (str, Path)):
            image = str(image)
        
        inputs = [{"image": image}]
        if instruction:
            inputs[0]["instruction"] = instruction
        
        embeddings = self.encode(inputs)
        return embeddings[0]
    
    def encode_text(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a single text.
        
        Args:
            text: Text string
            instruction: Optional instruction
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        inputs = [{"text": text}]
        if instruction:
            inputs[0]["instruction"] = instruction
        
        embeddings = self.encode(inputs)
        return embeddings[0]
    
    def encode_images_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        instruction: Optional[str] = None,
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode multiple images.
        
        Args:
            images: List of PIL Images, file paths, or URLs
            instruction: Optional instruction for all images
            batch_size: Processing batch size
            show_progress: Show progress
            
        Returns:
            2D numpy array of shape (n_images, embedding_dim)
        """
        inputs = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = str(img)
            inputs.append({"image": img})
        
        return self.encode(inputs, instruction=instruction, batch_size=batch_size, show_progress=show_progress)
    
    def encode_video(
        self,
        video: Union[str, Path, List[Image.Image]],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a video (file path or list of frames).
        
        Args:
            video: Video file path or list of PIL Image frames
            instruction: Optional instruction
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        if isinstance(video, (str, Path)):
            video = str(video)
        
        inputs = [{"video": video}]
        if instruction:
            inputs[0]["instruction"] = instruction
        
        embeddings = self.encode(inputs)
        return embeddings[0]
    
    def encode_video_frames(
        self,
        frames: List[Image.Image],
        instruction: Optional[str] = None,
        max_frames_per_batch: int = 8,
        frame_max_pixels: int = 512 * 512,
    ) -> np.ndarray:
        """
        Encode a video represented as a list of frames.
        
        Processes frames in smaller batches to avoid OOM, then averages embeddings.
        Also reduces frame resolution to limit memory usage.
        
        Args:
            frames: List of PIL Image frames
            instruction: Optional instruction
            max_frames_per_batch: Max frames to process at once (default 8)
            frame_max_pixels: Max pixels per frame for video (default 512x512)
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        import torch
        
        # Resize frames to reduce memory - videos don't need 1024x1024 per frame
        def resize_frame(img, max_pixels):
            w, h = img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                new_w, new_h = int(w * scale), int(h * scale)
                # Ensure dimensions are multiples of 28 (Qwen3-VL requirement)
                new_w = max(28, (new_w // 28) * 28)
                new_h = max(28, (new_h // 28) * 28)
                return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return img
        
        # Resize all frames
        resized_frames = [resize_frame(f, frame_max_pixels) for f in frames]
        
        # If few frames, process all at once
        if len(resized_frames) <= max_frames_per_batch:
            inputs = [{"video": resized_frames}]
            if instruction:
                inputs[0]["instruction"] = instruction
            embeddings = self.encode(inputs)
            # Clear CUDA cache after video processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return embeddings[0]
        
        # Process in batches and average embeddings
        all_embeddings = []
        for i in range(0, len(resized_frames), max_frames_per_batch):
            batch_frames = resized_frames[i:i + max_frames_per_batch]
            inputs = [{"video": batch_frames}]
            if instruction:
                inputs[0]["instruction"] = instruction
            
            batch_embeddings = self.encode(inputs)
            all_embeddings.append(batch_embeddings[0])
            
            # Clear cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average all batch embeddings
        avg_embedding = np.mean(all_embeddings, axis=0)
        # Re-normalize the averaged embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding
    
    @property
    def is_loaded(self) -> bool:
        return self._initialized


class RerankerModelWrapper:
    """
    Wrapper for Qwen3-VL-Reranker model.
    
    Provides fine-grained relevance scoring for query-document pairs.
    """
    
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANKER_MODEL,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_pixels: int = 512 * 512,  # Lower for reranker since it uses thumbnails
        min_pixels: int = 4096,
        attention_type: str = "sdpa",  # sdpa, eager, or sage
        use_flash_attention: bool = False,  # Deprecated
    ):
        """
        Initialize the reranker model.
        
        Args:
            model_name_or_path: Model short name or local path
            device: Device to run on
            torch_dtype: Data type for model weights
            max_pixels: Maximum pixels per image
            min_pixels: Minimum pixels per image
            attention_type: Attention implementation ("sdpa", "eager", "sage")
            use_flash_attention: Deprecated - use attention_type instead
        """
        self.model_name = model_name_or_path
        self.resolved_path = resolve_model_path(model_name_or_path)
        self.device = get_device(device)
        self.torch_dtype = torch_dtype
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        # Handle deprecated use_flash_attention
        if use_flash_attention:
            self.attention_type = "flash_attention_2"
        else:
            self.attention_type = attention_type
        
        self._model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of the model"""
        if self._initialized:
            return
        
        print(f"[SemanticSearch] Loading reranker model: {self.resolved_path}")
        
        # Import local Qwen3VLReranker
        from .qwen3_vl_reranker import Qwen3VLReranker
        
        print(f"[SemanticSearch] Using attention implementation: {self.attention_type}")
        
        self._model = Qwen3VLReranker(
            model_name_or_path=self.resolved_path,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attention_type,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        
        self._initialized = True
        print("[SemanticSearch] Reranker model loaded")
    
    def rerank(
        self,
        query: Dict[str, Any],
        documents: List[Dict[str, Any]],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Compute relevance scores for query-document pairs.
        
        Args:
            query: Query dict with "text" and/or "image"
            documents: List of document dicts
            instruction: Optional instruction
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        self._ensure_initialized()
        
        inputs = {
            "query": query,
            "documents": documents,
        }
        if instruction:
            inputs["instruction"] = instruction
        
        scores = self._model.process(inputs)
        
        if isinstance(scores, torch.Tensor):
            # Convert to float32 before converting to list (in case of bfloat16)
            scores = scores.float().cpu().tolist()
        
        return scores
    
    @property
    def is_loaded(self) -> bool:
        return self._initialized


def get_cached_model(
    model_type: str,
    model_name: str,
    **kwargs
) -> Union[EmbeddingModelWrapper, RerankerModelWrapper]:
    """
    Get or create a cached model instance.
    
    Args:
        model_type: "embedding" or "reranker"
        model_name: Model name/path
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model wrapper instance
    """
    cache_key = f"{model_type}:{model_name}"
    
    with _cache_lock:
        if cache_key not in _model_cache:
            if model_type == "embedding":
                _model_cache[cache_key] = EmbeddingModelWrapper(model_name, **kwargs)
            elif model_type == "reranker":
                _model_cache[cache_key] = RerankerModelWrapper(model_name, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        return _model_cache[cache_key]


def clear_model_cache():
    """Clear all cached models to free memory"""
    global _model_cache
    with _cache_lock:
        _model_cache.clear()
    
    # Also clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""
Eric's Semantic Search - Model Nodes

Description: ComfyUI nodes for loading Qwen3-VL embedding and reranker models
             with configurable resolution, attention type, and dimension settings.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

See LICENSE.md for complete license information.
"""

import torch
from ..core import (
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    EmbeddingModelWrapper,
    RerankerModelWrapper,
    get_device,
)

# Resolution presets for max_pixels (name -> pixel count)
RESOLUTION_PRESETS = {
    "128x128 (ultra-fast)": 128 * 128, # 16,384 - testing only
    "256x256 (fast)": 256 * 256,      # 65,536 - fastest practical
    "384x384": 384 * 384,              # 147,456
    "478x478": 478 * 478,              # 228,484
    "512x512": 512 * 512,              # 262,144
    "768x768": 768 * 768,              # 589,824
    "1024x1024 (1MP)": 1024 * 1024,   # 1,048,576 - good balance
    "1280x1280 (1.6MP)": 1280 * 1280, # 1,638,400
    "1536x1536 (2.4MP)": 1536 * 1536, # 2,359,296 - high quality
}

# Attention implementation options
ATTENTION_TYPES = ["sdpa", "eager", "sage"]

# Matryoshka embedding dimensions
EMBEDDING_DIM_OPTIONS = {
    "Full (4096/2048)": None,      # Use model's native dimension
    "2048": 2048,
    "1024": 1024,
    "512": 512,
    "256": 256,
}


class LoadEmbeddingModel:
    """Load a Qwen3-VL-Embedding model for semantic search"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "load_model"
    RETURN_TYPES = ("EMBEDDING_MODEL",)
    RETURN_NAMES = ("model",)
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cuda", "cuda:0", "cuda:1", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if f"cuda:{i}" not in devices:
                    devices.append(f"cuda:{i}")
        
        return {
            "required": {
                "model_name": (EMBEDDING_MODELS, {"default": EMBEDDING_MODELS[-1]}),
                "device": (devices, {"default": "auto"}),
            },
            "optional": {
                "max_resolution": (list(RESOLUTION_PRESETS.keys()), {
                    "default": "1024x1024 (1MP)",
                    "tooltip": "Maximum image resolution for encoding. Higher = more accurate but slower.",
                }),
                "attention_type": (ATTENTION_TYPES, {
                    "default": "sdpa",
                    "tooltip": "Attention implementation. sdpa=default, eager=fallback, sage=SageAttention (if installed)",
                }),
                "embedding_dim": (list(EMBEDDING_DIM_OPTIONS.keys()), {
                    "default": "Full (4096/2048)",
                    "tooltip": "Embedding dimension. Full=native (4096 for 8B, 2048 for 2B). Reduced dims use Matryoshka representation for smaller/faster indexes.",
                }),
            }
        }
    
    def load_model(
        self,
        model_name: str,
        device: str,
        max_resolution: str = "1024x1024 (1MP)",
        attention_type: str = "sdpa",
        embedding_dim: str = "Full (4096/2048)",
    ):
        """Load the embedding model"""
        max_pixels = RESOLUTION_PRESETS.get(max_resolution, 1024 * 1024)
        dim = EMBEDDING_DIM_OPTIONS.get(embedding_dim, None)
        
        model = EmbeddingModelWrapper(
            model_name_or_path=model_name,
            device=device,
            max_pixels=max_pixels,
            attention_type=attention_type,
            embedding_dim=dim,
        )
        
        # Trigger initialization
        model._ensure_initialized()
        
        return (model,)


class LoadRerankerModel:
    """Load a Qwen3-VL-Reranker model for result refinement"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "load_model"
    RETURN_TYPES = ("RERANKER_MODEL",)
    RETURN_NAMES = ("model",)
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cuda", "cuda:0", "cuda:1", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if f"cuda:{i}" not in devices:
                    devices.append(f"cuda:{i}")
        
        return {
            "required": {
                "model_name": (RERANKER_MODELS, {"default": RERANKER_MODELS[-1]}),
                "device": (devices, {"default": "auto"}),
            },
            "optional": {
                "max_resolution": (list(RESOLUTION_PRESETS.keys()), {
                    "default": "512x512",
                    "tooltip": "Maximum image resolution for reranking. Lower is faster since reranking uses thumbnails.",
                }),
                "attention_type": (ATTENTION_TYPES, {
                    "default": "sdpa",
                    "tooltip": "Attention implementation. sdpa=default, eager=fallback, sage=SageAttention (if installed)",
                }),
            }
        }
    
    def load_model(
        self,
        model_name: str,
        device: str,
        max_resolution: str = "512x512",
        attention_type: str = "sdpa",
    ):
        """Load the reranker model"""
        max_pixels = RESOLUTION_PRESETS.get(max_resolution, 512 * 512)
        
        model = RerankerModelWrapper(
            model_name_or_path=model_name,
            device=device,
            max_pixels=max_pixels,
            attention_type=attention_type,
        )
        
        # Trigger initialization
        model._ensure_initialized()
        
        return (model,)

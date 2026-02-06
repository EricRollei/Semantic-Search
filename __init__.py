"""
Eric's Semantic Search - Main Package

Description: Multimodal semantic search for ComfyUI using Qwen3-VL models.
             Enables natural language image search, visual similarity search,
             and two-stage retrieval with embedding + reranking.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/

2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at eric@historic.camera or eric@rollei.us for licensing options.

Dependencies:
- Qwen3-VL Models (Qwen License): https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- FAISS (MIT License): Facebook AI Similarity Search
- Transformers (Apache 2.0): Hugging Face
- PyTorch (BSD License): Meta Platforms

See LICENSE.md for complete license information.
"""

__version__ = "1.5.0"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

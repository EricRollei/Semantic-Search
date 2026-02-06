#!/usr/bin/env python
"""
Eric's Semantic Search - Standalone Indexer

Description: Command-line tool for batch indexing images, videos, and documents.
             Designed to run in a separate Python environment with PyTorch 2.10+
             to avoid the PyTorch 2.9.x 3D conv performance regression.

Usage:
    python standalone_indexer.py --index <index_name> --folder <path> [options]

Example:
    python standalone_indexer.py --index my_images --folder "D:/Photos" --recursive --batch-size 32
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Dependencies:
- Qwen3-VL Models (Qwen License): Alibaba Qwen Team
- PyTorch 2.10+ recommended (BSD License): Meta Platforms
- FAISS (MIT License): Facebook AI Research

See LICENSE.md for complete license information.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))


def check_pytorch_version():
    """Check PyTorch version and warn if 2.9.x"""
    import torch
    version = torch.__version__
    major_minor = ".".join(version.split(".")[:2])
    
    print(f"[Indexer] PyTorch version: {version}")
    print(f"[Indexer] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Indexer] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Indexer] CUDA version: {torch.version.cuda}")
    
    if major_minor == "2.9":
        print("\n" + "=" * 70)
        print("WARNING: PyTorch 2.9.x has a 4x performance regression for 3D convs!")
        print("This will make Qwen3-VL vision encoding extremely slow.")
        print("Recommended: Use PyTorch 2.10+ or 2.8.x for fast indexing.")
        print("=" * 70 + "\n")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted. Please upgrade PyTorch:")
            print("  pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
            sys.exit(1)
    elif major_minor in ("2.10", "2.8"):
        print(f"[Indexer] PyTorch {major_minor} - Good! 3D conv performance is optimal.")
    
    return version


def main():
    parser = argparse.ArgumentParser(
        description="Standalone indexer for Eric Semantic Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a folder recursively
  python standalone_indexer.py --index photos --folder "D:/Photos" --recursive

  # Index with specific settings
  python standalone_indexer.py --index artwork --folder "E:/Art" --batch-size 32 --thumbnail-size 512

  # Use the 2B model for faster indexing
  python standalone_indexer.py --index quick --folder "D:/Test" --model Qwen3-VL-Embedding-2B
        """
    )
    
    parser.add_argument("--index", "-i", required=True, 
                        help="Name of the index to create/update")
    parser.add_argument("--folder", "-f", required=True,
                        help="Path to folder containing images/videos/PDFs")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Search subdirectories recursively")
    parser.add_argument("--batch-size", "-b", type=int, default=16,
                        help="Images per batch (default: 16)")
    parser.add_argument("--thumbnail-size", "-t", type=int, default=768,
                        help="Thumbnail size in pixels (default: 768)")
    parser.add_argument("--model", "-m", default="Qwen3-VL-Embedding-8B",
                        choices=["Qwen3-VL-Embedding-2B", "Qwen3-VL-Embedding-8B"],
                        help="Embedding model to use (default: 8B)")
    parser.add_argument("--max-resolution", default="1024x1024 (1MP)",
                        help="Max image resolution for embedding")
    parser.add_argument("--include-videos", action="store_true", default=True,
                        help="Index video files (default: True)")
    parser.add_argument("--include-documents", action="store_true", default=True,
                        help="Index PDF documents (default: True)")
    parser.add_argument("--index-type", default="flat",
                        choices=["flat", "ivf", "hnsw"],
                        help="FAISS index type (default: flat)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model precision (default: bfloat16)")
    parser.add_argument("--no-version-check", action="store_true",
                        help="Skip PyTorch version check")
    
    args = parser.parse_args()
    
    # Check PyTorch version
    if not args.no_version_check:
        check_pytorch_version()
    
    # Import after version check (to show warning before slow imports)
    print("[Indexer] Loading modules...")
    
    import torch
    from core.config import MODELS_PATH, INDEXES_PATH, MODEL_REGISTRY
    from core.model_wrapper import EmbeddingModelWrapper
    from core.index_manager import SemanticIndex
    
    # Determine model path
    model_info = MODEL_REGISTRY.get(args.model)
    if model_info:
        local_folder, hf_repo = model_info
        model_path = MODELS_PATH / local_folder
        if model_path.exists():
            model_path_str = str(model_path)
        else:
            model_path_str = hf_repo
            print(f"[Indexer] Model not found locally, will download from: {hf_repo}")
    else:
        model_path_str = args.model
    
    # Resolution presets
    resolution_presets = {
        "128x128 (ultra-fast)": 128 * 128,
        "256x256 (fast)": 256 * 256,
        "384x384": 384 * 384,
        "478x478": 478 * 478,
        "512x512": 512 * 512,
        "768x768": 768 * 768,
        "1024x1024 (1MP)": 1024 * 1024,
        "1280x1280 (1.6MP)": 1280 * 1280,
        "1536x1536 (2.4MP)": 1536 * 1536,
    }
    max_pixels = resolution_presets.get(args.max_resolution, 1024 * 1024)
    
    # Dtype mapping
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    print(f"\n[Indexer] Configuration:")
    print(f"  Index name: {args.index}")
    print(f"  Folder: {args.folder}")
    print(f"  Recursive: {args.recursive}")
    print(f"  Model: {args.model}")
    print(f"  Max resolution: {args.max_resolution} ({max_pixels:,} pixels)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Thumbnail size: {args.thumbnail_size}")
    print(f"  Index type: {args.index_type}")
    print(f"  Include videos: {args.include_videos}")
    print(f"  Include documents: {args.include_documents}")
    print()
    
    # Load model
    print("[Indexer] Loading embedding model...")
    model = EmbeddingModelWrapper(
        model_name_or_path=model_path_str,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch_dtype,
        attention_type="sdpa",
        max_pixels=max_pixels,
    )
    # Trigger lazy initialization to load the model and set embedding_dim
    model._ensure_initialized()
    print(f"[Indexer] Model loaded. Embedding dimension: {model.embedding_dim}")
    
    # Create/load index
    index_path = INDEXES_PATH / args.index
    print(f"[Indexer] Index location: {index_path}")
    
    index_manager = SemanticIndex(
        index_name=args.index,
        embedding_dim=model.embedding_dim,
        index_type=args.index_type,
    )
    
    # Progress callback
    def progress_callback(current, total, status):
        pct = (current / total * 100) if total > 0 else 0
        bar_len = 40
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {pct:5.1f}% ({current}/{total}) {status}", end="", flush=True)
    
    # Run indexing
    print(f"\n[Indexer] Starting indexing of: {args.folder}")
    print("-" * 60)
    
    results = index_manager.index_folder(
        folder_path=args.folder,
        model=model,
        recursive=args.recursive,
        batch_size=args.batch_size,
        thumbnail_size=args.thumbnail_size,
        include_videos=args.include_videos,
        include_documents=args.include_documents,
        progress_callback=progress_callback,
    )
    
    print("\n" + "-" * 60)
    print(f"\n[Indexer] Indexing complete!")
    print(f"  Added: {results['added']}")
    print(f"  Skipped (already indexed): {results['skipped']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Videos indexed: {results['videos']}")
    print(f"  Documents indexed: {results['documents']} ({results['pages']} pages)")
    print(f"\n  Index saved to: {index_path}")
    print(f"\n  You can now use this index in ComfyUI with the SearchBy* nodes!")


if __name__ == "__main__":
    main()

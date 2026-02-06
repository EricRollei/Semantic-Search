#!/usr/bin/env python3
"""
Eric's Semantic Search - Interactive Indexer

Description: User-friendly menu-driven interface for indexing images, videos, and
             documents. Provides an interactive console for managing indexes,
             viewing statistics, and batch indexing operations.

Usage:
    python interactive_indexer.py
    
For optimal performance, run from a venv with PyTorch 2.10+:
    H:\semantic_search\indexer_venv\Scripts\python.exe interactive_indexer.py
             
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

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# Check PyTorch version before proceeding
def check_environment():
    """Check that we're running in the correct environment"""
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        
        print(f"\n{'='*60}")
        print("  Eric Semantic Search - Interactive Indexer")
        print(f"{'='*60}")
        print(f"\n  PyTorch version: {version}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  WARNING: CUDA not available, will use CPU (slow)")
        
        if major < 2 or (major == 2 and minor < 10):
            print(f"\n  ⚠️  WARNING: PyTorch {version} has slow 3D convolutions!")
            print("  ⚠️  For best performance, use PyTorch 2.10+")
            print("  ⚠️  Consider running from the indexer venv instead.")
            response = input("\n  Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(0)
        else:
            print(f"  ✓ PyTorch 2.10+ detected - optimal performance!")
        
        print()
        return True
    except ImportError:
        print("ERROR: PyTorch not installed!")
        return False

# Configuration paths
SEMANTIC_SEARCH_BASE = Path("H:/semantic_search")
MODELS_PATH = SEMANTIC_SEARCH_BASE / "models"
INDEXES_PATH = SEMANTIC_SEARCH_BASE / "indexes"
CONFIG_FILE = SEMANTIC_SEARCH_BASE / "indexer_config.json"

# Ensure directories exist
INDEXES_PATH.mkdir(parents=True, exist_ok=True)

# Model options
MODELS = {
    "1": ("Qwen3-VL-Embedding-8B", "8B model - Higher accuracy, more VRAM (~18GB)"),
    "2": ("Qwen3-VL-Embedding-2B", "2B model - Faster, less VRAM (~6GB)"),
}

# Resolution presets
RESOLUTIONS = {
    "1": ("128x128 (ultra-fast)", 128 * 128, "Very fast, lower accuracy"),
    "2": ("256x256 (fast)", 256 * 256, "Fast indexing, decent accuracy"),
    "3": ("384x384", 384 * 384, "Balanced speed/accuracy"),
    "4": ("512x512", 512 * 512, "Good accuracy"),
    "5": ("768x768", 768 * 768, "High accuracy"),
    "6": ("1024x1024 (1MP)", 1024 * 1024, "Best accuracy (default)"),
    "7": ("1280x1280 (1.6MP)", 1280 * 1280, "Maximum quality"),
}

# Index types
INDEX_TYPES = {
    "1": ("flat", "Flat (exact search)", "Best accuracy, slower for large indexes"),
    "2": ("ivf", "IVF (approximate)", "Fast for 10K+ items, good accuracy"),
    "3": ("hnsw", "HNSW (graph-based)", "Very fast search, higher memory"),
}

# Data type options
DTYPES = {
    "1": ("bfloat16", "BFloat16 (recommended)", "Best for modern GPUs"),
    "2": ("float16", "Float16", "Good compatibility"),
    "3": ("float32", "Float32", "Maximum precision, more VRAM"),
}


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def load_config() -> Dict[str, Any]:
    """Load saved configuration"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "model": "Qwen3-VL-Embedding-8B",
        "resolution": "1024x1024 (1MP)",
        "max_pixels": 1024 * 1024,
        "index_type": "flat",
        "dtype": "bfloat16",
        "batch_size": 16,
        "thumbnail_size": 768,
        "include_videos": True,
        "include_documents": True,
        "recursive": True,
        "last_folders": [],
    }


def save_config(config: Dict[str, Any]):
    """Save configuration for next session"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except:
        pass


def print_menu(title: str, options: Dict[str, tuple], current: str = None):
    """Print a formatted menu"""
    print(f"\n  {title}")
    print("  " + "-" * 50)
    for key, value in options.items():
        name = value[0] if isinstance(value, tuple) else value
        desc = value[1] if isinstance(value, tuple) and len(value) > 1 else ""
        marker = " ← current" if current and name == current else ""
        print(f"  [{key}] {name}{marker}")
        if desc:
            print(f"      {desc}")
    print()


def get_choice(prompt: str, valid: List[str], default: str = None) -> str:
    """Get user choice with validation"""
    while True:
        if default:
            choice = input(f"  {prompt} [{default}]: ").strip() or default
        else:
            choice = input(f"  {prompt}: ").strip()
        
        if choice.lower() in [v.lower() for v in valid]:
            return choice
        print(f"  Invalid choice. Please enter one of: {', '.join(valid)}")


def get_folder_path(prompt: str = "Enter folder path") -> Optional[Path]:
    """Get and validate a folder path"""
    while True:
        path_str = input(f"  {prompt}: ").strip()
        if not path_str:
            return None
        
        # Remove quotes if present
        path_str = path_str.strip('"').strip("'")
        path = Path(path_str)
        
        if path.exists() and path.is_dir():
            return path
        else:
            print(f"  ✗ Folder not found: {path}")
            retry = input("  Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None


def get_index_name(existing_indexes: List[str]) -> tuple[str, bool]:
    """Get index name and whether to create new or add to existing"""
    print("\n  Available indexes:")
    print("  " + "-" * 50)
    
    if existing_indexes:
        for i, name in enumerate(existing_indexes, 1):
            index_path = INDEXES_PATH / name
            # Count items if possible
            try:
                db_path = index_path / "metadata.db"
                if db_path.exists():
                    import sqlite3
                    conn = sqlite3.connect(str(db_path))
                    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                    conn.close()
                    print(f"  [{i}] {name} ({count:,} items)")
                else:
                    print(f"  [{i}] {name}")
            except:
                print(f"  [{i}] {name}")
    else:
        print("  (No existing indexes)")
    
    print(f"  [N] Create NEW index")
    print()
    
    while True:
        choice = input("  Select index or [N] for new: ").strip()
        
        if choice.upper() == 'N' or not existing_indexes:
            # Create new index
            while True:
                name = input("  Enter name for new index: ").strip()
                if not name:
                    print("  Name cannot be empty")
                    continue
                # Sanitize name
                name = "".join(c for c in name if c.isalnum() or c in "-_")
                if name in existing_indexes:
                    print(f"  Index '{name}' already exists. Choose another name or select it from the list.")
                    continue
                return name, True  # True = new index
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(existing_indexes):
                return existing_indexes[idx - 1], False  # False = existing index
        except ValueError:
            pass
        
        print("  Invalid choice")


def list_existing_indexes() -> List[str]:
    """Get list of existing index names"""
    indexes = []
    if INDEXES_PATH.exists():
        for item in INDEXES_PATH.iterdir():
            if item.is_dir():
                # Check if it looks like a valid index
                if (item / "index.faiss").exists() or (item / "metadata.db").exists():
                    indexes.append(item.name)
    return sorted(indexes)


def configure_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Interactive settings configuration"""
    while True:
        clear_screen()
        print(f"\n{'='*60}")
        print("  Settings Configuration")
        print(f"{'='*60}")
        
        print(f"\n  Current Settings:")
        print(f"  " + "-" * 50)
        print(f"  [1] Model:        {config['model']}")
        print(f"  [2] Resolution:   {config['resolution']}")
        print(f"  [3] Index Type:   {config['index_type']}")
        print(f"  [4] Precision:    {config['dtype']}")
        print(f"  [5] Batch Size:   {config['batch_size']}")
        print(f"  [6] Thumb Size:   {config['thumbnail_size']}px")
        print(f"  [7] Videos:       {'Yes' if config['include_videos'] else 'No'}")
        print(f"  [8] Documents:    {'Yes' if config['include_documents'] else 'No'}")
        print(f"  [9] Recursive:    {'Yes' if config['recursive'] else 'No'}")
        print()
        print(f"  [S] Save and return to main menu")
        print(f"  [R] Reset to defaults")
        print()
        
        choice = input("  Select setting to change: ").strip().upper()
        
        if choice == 'S':
            save_config(config)
            return config
        elif choice == 'R':
            config = load_config()  # Reset
            config = {  # Defaults
                "model": "Qwen3-VL-Embedding-8B",
                "resolution": "1024x1024 (1MP)",
                "max_pixels": 1024 * 1024,
                "index_type": "flat",
                "dtype": "bfloat16",
                "batch_size": 16,
                "thumbnail_size": 768,
                "include_videos": True,
                "include_documents": True,
                "recursive": True,
                "last_folders": config.get("last_folders", []),
            }
        elif choice == '1':
            print_menu("Select Embedding Model", MODELS, config['model'])
            c = get_choice("Choice", list(MODELS.keys()), "1")
            config['model'] = MODELS[c][0]
        elif choice == '2':
            print_menu("Select Resolution", {k: (v[0], v[2]) for k, v in RESOLUTIONS.items()}, config['resolution'])
            c = get_choice("Choice", list(RESOLUTIONS.keys()), "6")
            config['resolution'] = RESOLUTIONS[c][0]
            config['max_pixels'] = RESOLUTIONS[c][1]
        elif choice == '3':
            print_menu("Select Index Type", {k: (v[1], v[2]) for k, v in INDEX_TYPES.items()}, config['index_type'])
            c = get_choice("Choice", list(INDEX_TYPES.keys()), "1")
            config['index_type'] = INDEX_TYPES[c][0]
        elif choice == '4':
            print_menu("Select Precision", {k: (v[1], v[2]) for k, v in DTYPES.items()}, config['dtype'])
            c = get_choice("Choice", list(DTYPES.keys()), "1")
            config['dtype'] = DTYPES[c][0]
        elif choice == '5':
            try:
                bs = int(input("  Batch size (1-64) [16]: ").strip() or "16")
                config['batch_size'] = max(1, min(64, bs))
            except ValueError:
                pass
        elif choice == '6':
            try:
                ts = int(input("  Thumbnail size (256-1024) [768]: ").strip() or "768")
                config['thumbnail_size'] = max(256, min(1024, ts))
            except ValueError:
                pass
        elif choice == '7':
            config['include_videos'] = not config['include_videos']
        elif choice == '8':
            config['include_documents'] = not config['include_documents']
        elif choice == '9':
            config['recursive'] = not config['recursive']
    
    return config


def run_indexing(config: Dict[str, Any], index_name: str, folders: List[Path], is_new: bool):
    """Run the actual indexing process"""
    import torch
    
    # Add the package to path
    package_path = Path(__file__).parent
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    
    from core.model_wrapper import EmbeddingModelWrapper
    from core.index_manager import SemanticIndex
    
    # Resolve model path
    model_path = MODELS_PATH / config['model']
    if not model_path.exists():
        print(f"\n  ✗ Model not found: {model_path}")
        print(f"  Please download the model first.")
        input("\n  Press Enter to continue...")
        return
    
    # Dtype mapping
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[config['dtype']]
    
    print(f"\n{'='*60}")
    print("  Starting Indexing")
    print(f"{'='*60}")
    print(f"\n  Index: {index_name} ({'new' if is_new else 'adding to existing'})")
    print(f"  Model: {config['model']}")
    print(f"  Resolution: {config['resolution']}")
    print(f"  Index type: {config['index_type']}")
    print(f"  Folders to index:")
    for folder in folders:
        print(f"    - {folder}")
    print()
    
    confirm = input("  Start indexing? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    print("\n  Loading model...")
    model = EmbeddingModelWrapper(
        model_name_or_path=str(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch_dtype,
        attention_type="sdpa",
        max_pixels=config['max_pixels'],
    )
    model._ensure_initialized()
    print(f"  ✓ Model loaded. Embedding dimension: {model.embedding_dim}")
    
    # Create or load index
    index_path = INDEXES_PATH / index_name
    print(f"  Index location: {index_path}")
    
    if is_new:
        index_manager = SemanticIndex(
            index_name=index_name,
            embedding_dim=model.embedding_dim,
            index_type=config['index_type'],
        )
    else:
        index_manager = SemanticIndex(index_name=index_name)
    
    # Process each folder
    total_added = 0
    total_skipped = 0
    total_failed = 0
    
    for folder in folders:
        print(f"\n  Indexing: {folder}")
        print("  " + "-" * 50)
        
        result = index_manager.index_folder(
            folder_path=str(folder),
            embedding_model=model,
            recursive=config['recursive'],
            batch_size=config['batch_size'],
            thumbnail_size=config['thumbnail_size'],
            include_videos=config['include_videos'],
            include_documents=config['include_documents'],
        )
        
        total_added += result.get('added', 0)
        total_skipped += result.get('skipped', 0)
        total_failed += result.get('failed', 0)
    
    print(f"\n{'='*60}")
    print("  Indexing Complete!")
    print(f"{'='*60}")
    print(f"\n  Results:")
    print(f"    Added:   {total_added:,}")
    print(f"    Skipped: {total_skipped:,}")
    print(f"    Failed:  {total_failed:,}")
    print(f"\n  Index saved to: {index_path}")
    print(f"\n  You can now use this index in ComfyUI with SearchBy* nodes!")
    
    # Update last folders in config
    folder_strs = [str(f) for f in folders]
    config['last_folders'] = folder_strs[:10]  # Keep last 10
    save_config(config)
    
    input("\n  Press Enter to continue...")


def main_menu():
    """Main interactive menu"""
    config = load_config()
    
    while True:
        clear_screen()
        print(f"\n{'='*60}")
        print("  Eric Semantic Search - Interactive Indexer")
        print(f"{'='*60}")
        
        print(f"\n  Current Settings: {config['model']} | {config['resolution']} | {config['index_type']}")
        
        existing_indexes = list_existing_indexes()
        print(f"  Existing Indexes: {len(existing_indexes)}")
        
        print(f"\n  Main Menu")
        print("  " + "-" * 50)
        print("  [1] Index New Folder")
        print("  [2] Add Folder to Existing Index")
        print("  [3] View/Manage Indexes")
        print("  [4] Settings")
        print("  [5] Quick Re-index (last folder)")
        print()
        print("  [Q] Quit")
        print()
        
        choice = input("  Select option: ").strip().upper()
        
        if choice == 'Q':
            print("\n  Goodbye!")
            sys.exit(0)
        
        elif choice == '1':
            # Index new folder
            clear_screen()
            print(f"\n{'='*60}")
            print("  Index New Folder")
            print(f"{'='*60}")
            
            # Get index name
            index_name, is_new = get_index_name(existing_indexes)
            if not is_new:
                print(f"\n  Note: Adding to existing index '{index_name}'")
            
            # Get folder(s)
            folders = []
            print("\n  Enter folder(s) to index (empty line when done):")
            while True:
                folder = get_folder_path(f"Folder {len(folders)+1}")
                if folder is None:
                    if not folders:
                        print("  At least one folder is required")
                        continue
                    break
                folders.append(folder)
                print(f"  ✓ Added: {folder}")
                more = input("  Add another folder? (y/n): ").strip().lower()
                if more != 'y':
                    break
            
            # Run indexing
            run_indexing(config, index_name, folders, is_new)
        
        elif choice == '2':
            # Add to existing index
            clear_screen()
            print(f"\n{'='*60}")
            print("  Add Folder to Existing Index")
            print(f"{'='*60}")
            
            if not existing_indexes:
                print("\n  No existing indexes found. Create one first!")
                input("\n  Press Enter to continue...")
                continue
            
            # Select existing index
            index_name, _ = get_index_name(existing_indexes)
            
            # Get folder(s)
            folders = []
            print("\n  Enter folder(s) to add (empty line when done):")
            while True:
                folder = get_folder_path(f"Folder {len(folders)+1}")
                if folder is None:
                    if not folders:
                        print("  At least one folder is required")
                        continue
                    break
                folders.append(folder)
                print(f"  ✓ Added: {folder}")
                more = input("  Add another folder? (y/n): ").strip().lower()
                if more != 'y':
                    break
            
            # Run indexing (adding to existing)
            run_indexing(config, index_name, folders, is_new=False)
        
        elif choice == '3':
            # View/manage indexes
            clear_screen()
            print(f"\n{'='*60}")
            print("  Manage Indexes")
            print(f"{'='*60}")
            
            if not existing_indexes:
                print("\n  No indexes found.")
            else:
                print(f"\n  Found {len(existing_indexes)} indexes:\n")
                for name in existing_indexes:
                    index_path = INDEXES_PATH / name
                    try:
                        import sqlite3
                        db_path = index_path / "metadata.db"
                        if db_path.exists():
                            conn = sqlite3.connect(str(db_path))
                            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                            conn.close()
                            
                            # Get folder size
                            size = sum(f.stat().st_size for f in index_path.rglob('*') if f.is_file())
                            size_mb = size / (1024 * 1024)
                            
                            print(f"  • {name}")
                            print(f"      Items: {count:,} | Size: {size_mb:.1f} MB")
                            print(f"      Path: {index_path}")
                        else:
                            print(f"  • {name} (no database)")
                    except Exception as e:
                        print(f"  • {name} (error: {e})")
                    print()
            
            input("\n  Press Enter to continue...")
        
        elif choice == '4':
            # Settings
            config = configure_settings(config)
        
        elif choice == '5':
            # Quick re-index
            if not config.get('last_folders'):
                print("\n  No previous folders to re-index")
                input("\n  Press Enter to continue...")
                continue
            
            clear_screen()
            print(f"\n{'='*60}")
            print("  Quick Re-index")
            print(f"{'='*60}")
            
            print("\n  Last indexed folders:")
            for i, folder in enumerate(config['last_folders'], 1):
                exists = "✓" if Path(folder).exists() else "✗"
                print(f"  [{i}] {exists} {folder}")
            
            # Get index name
            print()
            index_name, is_new = get_index_name(existing_indexes)
            
            # Filter to existing folders
            folders = [Path(f) for f in config['last_folders'] if Path(f).exists()]
            if not folders:
                print("\n  No valid folders from last session")
                input("\n  Press Enter to continue...")
                continue
            
            run_indexing(config, index_name, folders, is_new)


if __name__ == "__main__":
    if check_environment():
        print("  Loading modules...")
        main_menu()

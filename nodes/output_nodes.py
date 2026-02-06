"""
Eric's Semantic Search - Output Nodes

Description: ComfyUI nodes for displaying and using search results - preview grids,
             loading full-resolution images, and extracting file paths.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

See LICENSE.md for complete license information.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from ..core import SearchResults, load_image


class PreviewResults:
    """Create a thumbnail grid preview of search results"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "preview"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_grid", "paths_list")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("SEARCH_RESULTS",),
            },
            "optional": {
                "columns": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "thumbnail_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 32,
                }),
                "max_images": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "show_scores": ("BOOLEAN", {"default": True}),
            }
        }
    
    def preview(
        self,
        results: SearchResults,
        columns: int = 5,
        thumbnail_size: int = 256,
        max_images: int = 25,
        show_scores: bool = True,
    ):
        """Create preview grid"""
        from PIL import ImageDraw, ImageFont
        
        # Limit results
        items = list(results)[:max_images]
        
        if not items:
            # Return empty image
            empty = np.zeros((256, 256, 3), dtype=np.float32)
            paths_text = "No results"
            return (torch.from_numpy(empty).unsqueeze(0), paths_text)
        
        # Calculate grid dimensions
        n_images = len(items)
        rows = (n_images + columns - 1) // columns
        
        # Create grid image
        grid_width = columns * thumbnail_size
        grid_height = rows * thumbnail_size
        grid = Image.new("RGB", (grid_width, grid_height), (32, 32, 32))
        
        # Try to get a font for labels
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(grid)
        
        # Place thumbnails
        for idx, result in enumerate(items):
            row = idx // columns
            col = idx % columns
            x = col * thumbnail_size
            y = row * thumbnail_size
            
            # Load thumbnail
            try:
                thumb = Image.open(result.thumbnail_path).convert("RGB")
                # Resize to fit cell while maintaining aspect ratio
                thumb.thumbnail((thumbnail_size - 4, thumbnail_size - 20 if show_scores else thumbnail_size - 4))
                
                # Center in cell
                paste_x = x + (thumbnail_size - thumb.width) // 2
                paste_y = y + 2
                grid.paste(thumb, (paste_x, paste_y))
                
                # Draw score
                if show_scores:
                    score_text = f"{result.score:.3f}"
                    text_y = y + thumbnail_size - 18
                    draw.text((x + 4, text_y), score_text, fill=(255, 255, 255), font=font)
                    
            except Exception as e:
                # Draw error placeholder
                draw.rectangle([x + 2, y + 2, x + thumbnail_size - 2, y + thumbnail_size - 2], 
                              outline=(128, 0, 0), width=2)
                draw.text((x + 4, y + 4), "Error", fill=(255, 0, 0), font=font)
        
        # Convert to tensor
        grid_np = np.array(grid).astype(np.float32) / 255.0
        grid_tensor = torch.from_numpy(grid_np).unsqueeze(0)
        
        # Build paths list
        paths_list = "\n".join([f"{i+1}. {r.file_path}" for i, r in enumerate(items)])
        
        return (grid_tensor, paths_list)


class LoadResultImages:
    """Load full-resolution images from search results"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "load_images"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("SEARCH_RESULTS",),
            },
            "optional": {
                "max_images": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                }),
                "max_dimension": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 128,
                    "tooltip": "Resize images if larger than this dimension",
                }),
            }
        }
    
    def load_images(
        self,
        results: SearchResults,
        max_images: int = 4,
        max_dimension: int = 1024,
    ):
        """Load images from results"""
        items = list(results)[:max_images]
        
        if not items:
            # Return empty batch
            empty = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
            return (empty,)
        
        images = []
        
        for result in items:
            try:
                img = load_image(Path(result.file_path))
                if img is None:
                    continue
                
                # Resize if needed
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Convert to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Convert to numpy
                img_np = np.array(img).astype(np.float32) / 255.0
                images.append(img_np)
                
            except Exception as e:
                print(f"[SemanticSearch] Failed to load {result.file_path}: {e}")
                continue
        
        if not images:
            empty = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
            return (empty,)
        
        # Pad to same size for batching
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        
        padded = []
        for img in images:
            h, w = img.shape[:2]
            pad_h = max_h - h
            pad_w = max_w - w
            
            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            
            padded.append(img)
        
        # Stack into batch
        batch = np.stack(padded, axis=0)
        batch_tensor = torch.from_numpy(batch)
        
        return (batch_tensor,)


class GetResultPaths:
    """Extract file paths from search results"""
    
    CATEGORY = "Eric/SemanticSearch"
    FUNCTION = "get_paths"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("paths_list", "paths_newline")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("SEARCH_RESULTS",),
            },
            "optional": {
                "max_results": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                }),
                "include_scores": ("BOOLEAN", {"default": False}),
            }
        }
    
    def get_paths(
        self,
        results: SearchResults,
        max_results: int = 50,
        include_scores: bool = False,
    ):
        """Get paths from results"""
        items = list(results)[:max_results]
        
        if include_scores:
            paths = [f"{r.file_path}\t{r.score:.4f}" for r in items]
        else:
            paths = [r.file_path for r in items]
        
        # Comma-separated for lists
        paths_list = ",".join(paths)
        
        # Newline-separated for display
        paths_newline = "\n".join(paths)
        
        return (paths_list, paths_newline)

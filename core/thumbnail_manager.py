"""
Eric's Semantic Search - Thumbnail Manager

Description: Thumbnail generation and caching for indexed images, videos, and documents.
             Handles multiple image formats including HEIC/HEIF and RAW files,
             with support for video keyframe extraction and PDF page rendering.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Dependencies:
- Pillow (HPND License): Python Imaging Library - https://pillow.readthedocs.io
- pillow-heif (BSD License): HEIF/HEIC support for Pillow
- rawpy (MIT License, optional): RAW image file support
- OpenCV (Apache 2.0, optional): Video frame extraction
- PyMuPDF (AGPL/Commercial, optional): PDF rendering

See LICENSE.md for complete license information.
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple
import os

from PIL import Image
import pillow_heif  # HEIF/HEIC support

from .config import (
    DEFAULT_THUMBNAIL_SIZE,
    THUMBNAIL_QUALITY,
    THUMBNAIL_FORMAT,
    HASH_CHUNK_SIZE,
    USE_FAST_HASH,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    get_thumbnails_path,
)
from .video_utils import is_video_file, extract_keyframe, _add_play_indicator
from .pdf_utils import is_pdf_file, create_pdf_thumbnail, parse_pdf_page_path

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

# Try to import rawpy for RAW support
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("[SemanticSearch] rawpy not installed - RAW file support disabled")


# RAW extensions that need rawpy
RAW_EXTENSIONS = {".raw", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".raf"}


def compute_file_hash(file_path: Path, fast: bool = USE_FAST_HASH) -> str:
    """
    Compute a hash for change detection.
    
    If fast=True, uses size + mtime (much faster for large files).
    If fast=False, uses actual content hash (slower but detects in-place edits).
    """
    file_path = Path(file_path)
    
    if fast:
        # Fast mode: combine size and mtime into a hash-like string
        stat = file_path.stat()
        return f"{stat.st_size}_{stat.st_mtime_ns}"
    else:
        # Full content hash (MD5 is fine for change detection, not security)
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(HASH_CHUNK_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()


def get_file_info(file_path: Path) -> Tuple[float, int]:
    """Get mtime and size for a file"""
    stat = file_path.stat()
    return stat.st_mtime, stat.st_size


def is_supported_image(file_path: Path) -> bool:
    """Check if a file is a supported image format"""
    suffix = file_path.suffix.lower()
    if suffix in RAW_EXTENSIONS and not HAS_RAWPY:
        return False
    return suffix in SUPPORTED_EXTENSIONS


def is_supported_media(file_path: Path) -> bool:
    """Check if a file is a supported image OR video format"""
    return is_supported_image(file_path) or is_video_file(file_path)


def load_image(file_path: Path) -> Optional[Image.Image]:
    """
    Load an image from any supported format.
    Returns PIL Image in RGB mode, or None if loading fails.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    try:
        if suffix in RAW_EXTENSIONS:
            if not HAS_RAWPY:
                return None
            # Load RAW with rawpy
            with rawpy.imread(str(file_path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,  # Faster processing, still plenty of resolution for thumbnails
                    no_auto_bright=False,
                    output_bps=8,
                )
            return Image.fromarray(rgb)
        else:
            # Standard formats via Pillow (including HEIF via pillow_heif)
            img = Image.open(file_path)
            # Handle EXIF orientation
            img = _apply_exif_orientation(img)
            # Convert to RGB if needed
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            elif img.mode == "RGBA":
                # Composite onto white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            return img
    except Exception as e:
        print(f"[SemanticSearch] Failed to load image {file_path}: {e}")
        return None


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag if present"""
    try:
        exif = img.getexif()
        if exif:
            orientation = exif.get(274)  # 274 is the orientation tag
            if orientation:
                rotations = {
                    3: Image.Transpose.ROTATE_180,
                    6: Image.Transpose.ROTATE_270,
                    8: Image.Transpose.ROTATE_90,
                }
                if orientation in rotations:
                    img = img.transpose(rotations[orientation])
                # Handle mirrored orientations
                if orientation in (2, 4):
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if orientation in (5, 7):
                    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    except Exception:
        pass  # Ignore EXIF errors
    return img


def resize_for_thumbnail(img: Image.Image, max_size: int = DEFAULT_THUMBNAIL_SIZE) -> Image.Image:
    """
    Resize image so longest side is max_size, preserving aspect ratio.
    Uses high-quality Lanczos resampling.
    """
    width, height = img.size
    
    if width <= max_size and height <= max_size:
        return img
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


class ThumbnailManager:
    """Manages thumbnail generation and caching for an index"""
    
    def __init__(self, index_name: str, thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE):
        self.index_name = index_name
        self.thumbnail_size = thumbnail_size
        self.thumbnails_path = get_thumbnails_path(index_name)
    
    def get_thumbnail_filename(self, original_path: Path) -> str:
        """Generate a unique thumbnail filename based on the original path"""
        # Use hash of full path to handle files with same name in different folders
        path_hash = hashlib.md5(str(original_path).encode()).hexdigest()[:16]
        return f"{path_hash}.jpg"
    
    def get_thumbnail_path(self, original_path: Path) -> Path:
        """Get the full path where a thumbnail would be stored"""
        return self.thumbnails_path / self.get_thumbnail_filename(original_path)
    
    def thumbnail_exists(self, original_path: Path) -> bool:
        """Check if a thumbnail already exists for an image"""
        return self.get_thumbnail_path(original_path).exists()
    
    def get_or_create_thumbnail(self, original_path: Path) -> Optional[Path]:
        """
        Get existing thumbnail or create a new one.
        Returns the thumbnail path, or None if creation failed.
        """
        original_path = Path(original_path)
        thumb_path = self.get_thumbnail_path(original_path)
        
        if thumb_path.exists():
            return thumb_path
        
        return self.create_thumbnail(original_path)
    
    def create_thumbnail(self, original_path: Path, force: bool = False, page_number: int = None) -> Optional[Path]:
        """
        Create a thumbnail for an image, video, or PDF page.
        
        Args:
            original_path: Path to the original image/video/PDF (or PDF#page=N path)
            force: If True, recreate even if thumbnail exists
            page_number: For PDFs, which page to render (1-indexed). If None, uses page 1.
            
        Returns:
            Path to the thumbnail, or None if creation failed
        """
        original_path = Path(original_path) if not isinstance(original_path, str) else original_path
        
        # Check if this is a PDF page path (e.g., "file.pdf#page=3")
        pdf_path, parsed_page = parse_pdf_page_path(str(original_path))
        if parsed_page is not None:
            page_number = parsed_page
            original_path = Path(pdf_path)
        
        thumb_path = self.get_thumbnail_path(original_path if page_number is None else Path(f"{original_path}#page={page_number}"))
        
        if thumb_path.exists() and not force:
            return thumb_path
        
        # Check if this is a PDF file
        if is_pdf_file(str(original_path)):
            # Use PDF thumbnail generator
            img = create_pdf_thumbnail(
                str(original_path), 
                page_number=page_number or 1,
                size=(self.thumbnail_size, self.thumbnail_size),
            )
            if img is None:
                return None
        # Check if this is a video file
        elif is_video_file(original_path):
            # Use video keyframe extractor + play indicator
            img = extract_keyframe(str(original_path), position=0.1)
            if img is None:
                return None
            # Resize and add play indicator
            img = resize_for_thumbnail(img, self.thumbnail_size)
            img = _add_play_indicator(img)
        else:
            # Load original image
            img = load_image(original_path)
            if img is None:
                return None
            # Resize
            img = resize_for_thumbnail(img, self.thumbnail_size)
        
        # Save thumbnail
        try:
            img.save(
                thumb_path,
                THUMBNAIL_FORMAT,
                quality=THUMBNAIL_QUALITY,
                optimize=True,
            )
            return thumb_path
        except Exception as e:
            print(f"[SemanticSearch] Failed to save thumbnail for {original_path}: {e}")
            return None
    
    def delete_thumbnail(self, original_path: Path) -> bool:
        """Delete a thumbnail if it exists. Returns True if deleted."""
        thumb_path = self.get_thumbnail_path(original_path)
        if thumb_path.exists():
            thumb_path.unlink()
            return True
        return False
    
    def get_thumbnail_as_pil(self, original_path: Path) -> Optional[Image.Image]:
        """Get thumbnail as a PIL Image (loads from cache or creates)"""
        thumb_path = self.get_or_create_thumbnail(original_path)
        if thumb_path is None:
            return None
        
        try:
            return Image.open(thumb_path).convert("RGB")
        except Exception as e:
            print(f"[SemanticSearch] Failed to load thumbnail {thumb_path}: {e}")
            return None
    
    def cleanup_orphaned_thumbnails(self, valid_paths: set) -> int:
        """
        Remove thumbnails that don't correspond to any indexed image.
        
        Args:
            valid_paths: Set of original image paths that should have thumbnails
            
        Returns:
            Number of orphaned thumbnails deleted
        """
        # Build set of expected thumbnail filenames
        expected_filenames = {self.get_thumbnail_filename(p) for p in valid_paths}
        
        deleted = 0
        for thumb_file in self.thumbnails_path.glob("*.jpg"):
            if thumb_file.name not in expected_filenames:
                thumb_file.unlink()
                deleted += 1
        
        return deleted
    
    def get_cache_size_mb(self) -> float:
        """Get total size of thumbnail cache in MB"""
        total = sum(f.stat().st_size for f in self.thumbnails_path.glob("*.jpg"))
        return round(total / (1024 * 1024), 2)
    
    def get_thumbnail_count(self) -> int:
        """Get number of cached thumbnails"""
        return len(list(self.thumbnails_path.glob("*.jpg")))


def collect_images_from_folder(folder_path: Path, recursive: bool = True) -> list:
    """
    Collect all supported image files from a folder.
    
    Args:
        folder_path: Path to search
        recursive: If True, search subdirectories
        
    Returns:
        List of Path objects for found images
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return []
    
    images = []
    
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            # Case-insensitive glob
            images.extend(folder_path.rglob(f"*{ext}"))
            images.extend(folder_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            images.extend(folder_path.glob(f"*{ext}"))
            images.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # Filter out any non-files and deduplicate
    images = list(set(p for p in images if p.is_file() and is_supported_image(p)))
    
    return sorted(images)


def collect_media_from_folder(folder_path: Path, recursive: bool = True, include_videos: bool = True) -> list:
    """
    Collect all supported image and video files from a folder.
    
    Args:
        folder_path: Path to search
        recursive: If True, search subdirectories
        include_videos: If True, include video files
        
    Returns:
        List of Path objects for found media files
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return []
    
    media = []
    all_extensions = list(SUPPORTED_EXTENSIONS)
    if include_videos:
        all_extensions.extend(SUPPORTED_VIDEO_EXTENSIONS)
    
    if recursive:
        for ext in all_extensions:
            # Case-insensitive glob
            media.extend(folder_path.rglob(f"*{ext}"))
            media.extend(folder_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in all_extensions:
            media.extend(folder_path.glob(f"*{ext}"))
            media.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # Filter out any non-files and deduplicate
    media = list(set(p for p in media if p.is_file() and is_supported_media(p)))
    
    return sorted(media)


def collect_documents_from_folder(folder_path: Path, recursive: bool = True) -> list:
    """
    Collect all supported document files (PDFs) from a folder.
    
    Args:
        folder_path: Path to search
        recursive: If True, search subdirectories
        
    Returns:
        List of Path objects for found document files
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return []
    
    documents = []
    
    if recursive:
        for ext in SUPPORTED_DOCUMENT_EXTENSIONS:
            documents.extend(folder_path.rglob(f"*{ext}"))
            documents.extend(folder_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in SUPPORTED_DOCUMENT_EXTENSIONS:
            documents.extend(folder_path.glob(f"*{ext}"))
            documents.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # Filter out any non-files and deduplicate
    documents = list(set(p for p in documents if p.is_file()))
    
    return sorted(documents)

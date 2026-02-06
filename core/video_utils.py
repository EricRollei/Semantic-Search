"""
Video processing utilities for Eric Semantic Search

Handles video frame extraction, thumbnail generation, and metadata.
Uses OpenCV for video processing.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import (
    SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_DEFAULT_FPS,
    VIDEO_MAX_FRAMES,
    VIDEO_MIN_FRAMES,
)

logger = logging.getLogger(__name__)

# Try to import cv2 (opencv-python)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Video support disabled. Install with: pip install opencv-python")


@dataclass
class VideoInfo:
    """Metadata about a video file"""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "codec": self.codec,
        }


def is_video_file(file_path: str) -> bool:
    """Check if file is a supported video format"""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def get_video_info(video_path: str) -> Optional[VideoInfo]:
    """
    Get metadata about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        VideoInfo object or None if file cannot be read
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV not available for video processing")
        return None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return VideoInfo(
            path=str(video_path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            codec=codec,
        )
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {e}")
        return None


def extract_frames(
    video_path: str,
    target_fps: float = VIDEO_DEFAULT_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    min_frames: int = VIDEO_MIN_FRAMES,
    uniform_sampling: bool = True,
) -> List[Image.Image]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        target_fps: Target frames per second to extract (for time-based sampling)
        max_frames: Maximum number of frames to extract
        min_frames: Minimum number of frames (pad with last frame if needed)
        uniform_sampling: If True, sample frames uniformly; if False, use target_fps
        
    Returns:
        List of PIL Images
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV not available for video processing")
        return []
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0 or video_fps <= 0:
            logger.error(f"Invalid video metadata: {video_path}")
            cap.release()
            return []
        
        # Determine which frame indices to extract
        if uniform_sampling:
            # Uniformly sample across video duration
            num_frames = min(max_frames, total_frames)
            if num_frames < min_frames:
                num_frames = min(min_frames, total_frames)
            
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Sample at target_fps
            frame_interval = max(1, int(video_fps / target_fps))
            frame_indices = list(range(0, total_frames, frame_interval))[:max_frames]
        
        frames = []
        last_frame = None
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                last_frame = pil_image
            elif last_frame is not None:
                # Use last successful frame as fallback
                frames.append(last_frame)
        
        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < min_frames and last_frame is not None:
            frames.append(last_frame)
        
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        return []


def extract_keyframe(
    video_path: str,
    position: float = 0.1,
) -> Optional[Image.Image]:
    """
    Extract a single keyframe from a video for thumbnail.
    
    Args:
        video_path: Path to video file
        position: Position in video (0.0 = start, 1.0 = end), default 10%
        
    Returns:
        PIL Image or None
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV not available for video processing")
        return None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(total_frames * position)
        target_frame = max(0, min(target_frame, total_frames - 1))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting keyframe from {video_path}: {e}")
        return None


def create_video_thumbnail(
    video_path: str,
    output_path: str,
    max_size: int = 768,
    position: float = 0.1,
    quality: int = 85,
    add_play_indicator: bool = True,
) -> Optional[str]:
    """
    Create a thumbnail image from a video file.
    
    Args:
        video_path: Path to video file
        output_path: Where to save the thumbnail
        max_size: Maximum dimension (preserves aspect ratio)
        position: Position in video to extract (0.0-1.0)
        quality: JPEG quality
        add_play_indicator: If True, add a play button overlay
        
    Returns:
        Path to saved thumbnail or None
    """
    frame = extract_keyframe(video_path, position)
    if frame is None:
        return None
    
    # Resize to max_size
    width, height = frame.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Optionally add play button indicator
    if add_play_indicator:
        frame = _add_play_indicator(frame)
    
    # Save
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.save(output_path, "JPEG", quality=quality)
        return output_path
    except Exception as e:
        logger.error(f"Error saving video thumbnail: {e}")
        return None


def _add_play_indicator(image: Image.Image, size_ratio: float = 0.15) -> Image.Image:
    """
    Add a semi-transparent play button indicator to an image.
    
    Args:
        image: PIL Image to modify
        size_ratio: Size of play button relative to image
        
    Returns:
        Modified image with play button
    """
    try:
        from PIL import ImageDraw
        
        # Create a copy to avoid modifying original
        img = image.copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        width, height = img.size
        
        # Create overlay for the play button
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Calculate play button size and position
        button_size = int(min(width, height) * size_ratio)
        center_x = width // 2
        center_y = height // 2
        
        # Draw semi-transparent circle background
        circle_radius = button_size
        draw.ellipse(
            [center_x - circle_radius, center_y - circle_radius,
             center_x + circle_radius, center_y + circle_radius],
            fill=(0, 0, 0, 128)
        )
        
        # Draw play triangle
        triangle_size = button_size * 0.6
        # Triangle points (pointing right)
        points = [
            (center_x - triangle_size * 0.4, center_y - triangle_size * 0.6),
            (center_x - triangle_size * 0.4, center_y + triangle_size * 0.6),
            (center_x + triangle_size * 0.6, center_y),
        ]
        draw.polygon(points, fill=(255, 255, 255, 200))
        
        # Composite
        result = Image.alpha_composite(img, overlay)
        return result.convert('RGB')
        
    except Exception as e:
        logger.warning(f"Could not add play indicator: {e}")
        return image


def frames_to_video_input(frames: List[Image.Image]) -> List[Image.Image]:
    """
    Prepare frames for Qwen3-VL video input format.
    
    The Qwen3-VL model expects video as a list of PIL Images.
    This function ensures they're in the correct format.
    
    Args:
        frames: List of PIL Images
        
    Returns:
        List of PIL Images ready for model input
    """
    processed = []
    for frame in frames:
        # Ensure RGB mode
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')
        processed.append(frame)
    return processed

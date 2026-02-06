"""
PDF processing utilities for Eric Semantic Search.

Uses PyMuPDF (fitz) for fast PDF page extraction without external dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import io

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

from PIL import Image

from . import config


@dataclass
class PDFInfo:
    """Information about a PDF file"""
    path: str
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    file_size: int = 0


@dataclass
class PDFPage:
    """A single page from a PDF"""
    page_number: int  # 1-indexed
    image: Image.Image
    width: int
    height: int
    parent_document: str  # Path to source PDF


def is_pdf_file(path: str) -> bool:
    """Check if a file is a PDF based on extension"""
    return Path(path).suffix.lower() == ".pdf"


def check_pymupdf() -> bool:
    """Check if PyMuPDF is available"""
    return HAS_PYMUPDF


def get_pdf_info(pdf_path: str) -> Optional[PDFInfo]:
    """
    Get information about a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        PDFInfo object or None if file cannot be read
    """
    if not HAS_PYMUPDF:
        print("[SemanticSearch] Warning: PyMuPDF not installed. Install with: pip install pymupdf")
        return None
    
    path = Path(pdf_path)
    if not path.exists():
        return None
    
    try:
        doc = fitz.open(pdf_path)
        info = PDFInfo(
            path=str(path),
            page_count=len(doc),
            title=doc.metadata.get("title") if doc.metadata else None,
            author=doc.metadata.get("author") if doc.metadata else None,
            subject=doc.metadata.get("subject") if doc.metadata else None,
            file_size=path.stat().st_size,
        )
        doc.close()
        return info
    except Exception as e:
        print(f"[SemanticSearch] Error reading PDF {pdf_path}: {e}")
        return None


def extract_page_image(
    pdf_path: str,
    page_number: int,
    dpi: int = None,
) -> Optional[Image.Image]:
    """
    Extract a single page from a PDF as an image.
    
    Args:
        pdf_path: Path to PDF file
        page_number: Page number (1-indexed)
        dpi: Resolution for rendering (default: config.PDF_DPI)
        
    Returns:
        PIL Image or None if extraction fails
    """
    if not HAS_PYMUPDF:
        return None
    
    if dpi is None:
        dpi = getattr(config, 'PDF_DPI', 150)
    
    try:
        doc = fitz.open(pdf_path)
        
        # Convert to 0-indexed
        page_idx = page_number - 1
        if page_idx < 0 or page_idx >= len(doc):
            doc.close()
            return None
        
        page = doc[page_idx]
        
        # Calculate zoom factor from DPI (72 is PDF's default DPI)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        return img
        
    except Exception as e:
        print(f"[SemanticSearch] Error extracting page {page_number} from {pdf_path}: {e}")
        return None


def extract_all_pages(
    pdf_path: str,
    max_pages: int = None,
    dpi: int = None,
) -> List[PDFPage]:
    """
    Extract all pages from a PDF as images.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to extract (default: config.PDF_MAX_PAGES)
        dpi: Resolution for rendering (default: config.PDF_DPI)
        
    Returns:
        List of PDFPage objects
    """
    if not HAS_PYMUPDF:
        print("[SemanticSearch] Warning: PyMuPDF not installed")
        return []
    
    if max_pages is None:
        max_pages = getattr(config, 'PDF_MAX_PAGES', 100)
    if dpi is None:
        dpi = getattr(config, 'PDF_DPI', 150)
    
    pages = []
    
    try:
        doc = fitz.open(pdf_path)
        num_pages = min(len(doc), max_pages)
        
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        for page_idx in range(num_pages):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            pages.append(PDFPage(
                page_number=page_idx + 1,  # 1-indexed
                image=img,
                width=pix.width,
                height=pix.height,
                parent_document=str(pdf_path),
            ))
        
        doc.close()
        
    except Exception as e:
        print(f"[SemanticSearch] Error extracting pages from {pdf_path}: {e}")
    
    return pages


def create_pdf_thumbnail(
    pdf_path: str,
    page_number: int = 1,
    size: Tuple[int, int] = None,
    dpi: int = None,
) -> Optional[Image.Image]:
    """
    Create a thumbnail from a PDF page.
    
    For first page, just returns the page thumbnail.
    For other pages, adds a small page number indicator.
    
    Args:
        pdf_path: Path to PDF file
        page_number: Which page to use (1-indexed, default: 1)
        size: Target thumbnail size (default: from config)
        dpi: Initial render DPI (default: config.PDF_DPI)
        
    Returns:
        PIL Image thumbnail or None if creation fails
    """
    if size is None:
        size = (config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE)
    
    # Extract the page as image
    img = extract_page_image(pdf_path, page_number, dpi)
    if img is None:
        return None
    
    # Create thumbnail
    img.thumbnail(size, Image.Resampling.LANCZOS)
    
    # Create square canvas with white background (typical for documents)
    thumb = Image.new("RGB", size, (255, 255, 255))
    
    # Center the page on canvas
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    thumb.paste(img, (x, y))
    
    # Add page number indicator if not first page
    if page_number > 1:
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(thumb)
            
            # Draw page number badge in bottom-right
            text = f"p.{page_number}"
            
            # Try to use a font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Position in bottom-right with padding
            padding = 4
            x = size[0] - text_w - padding - 6
            y = size[1] - text_h - padding - 6
            
            # Draw background rectangle
            draw.rectangle(
                [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
                fill=(50, 50, 50),
            )
            
            # Draw text
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
        except Exception as e:
            # Page number is optional, don't fail if it doesn't work
            pass
    
    return thumb


def get_pdf_page_path(pdf_path: str, page_number: int) -> str:
    """
    Generate a unique identifier for a PDF page.
    
    This creates a virtual path that can be stored in the database
    to uniquely identify a specific page of a PDF.
    
    Format: /path/to/document.pdf#page=N
    
    Args:
        pdf_path: Path to PDF file
        page_number: Page number (1-indexed)
        
    Returns:
        Virtual path string like "document.pdf#page=3"
    """
    return f"{pdf_path}#page={page_number}"


def parse_pdf_page_path(page_path: str) -> Tuple[str, Optional[int]]:
    """
    Parse a PDF page path back into PDF path and page number.
    
    Args:
        page_path: Path that may contain #page=N suffix
        
    Returns:
        (pdf_path, page_number) where page_number is None if not a page path
    """
    if "#page=" in page_path:
        pdf_path, page_part = page_path.rsplit("#page=", 1)
        try:
            page_number = int(page_part)
            return pdf_path, page_number
        except ValueError:
            pass
    
    return page_path, None

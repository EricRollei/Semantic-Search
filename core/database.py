"""
Eric's Semantic Search - Database Manager

Description: SQLite database wrapper for tracking indexed images, videos, and documents.
             Manages metadata storage, folder tracking, and provides query interfaces
             for the semantic search system.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Dependencies:
- SQLite3 (Python standard library)

See LICENSE.md for complete license information.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading

from .config import get_index_db_path


@dataclass
class ImageRecord:
    """Represents an indexed image, video, or document page"""
    id: int
    file_path: str
    file_hash: str
    file_mtime: float
    file_size: int
    thumbnail_path: str
    vector_id: int
    indexed_at: datetime
    media_type: str = "image"  # "image", "video", or "document"
    frame_count: Optional[int] = None  # Number of frames extracted (for videos)
    duration_seconds: Optional[float] = None  # Video duration in seconds
    page_number: Optional[int] = None  # Page number (for PDF pages, 1-indexed)
    parent_document: Optional[str] = None  # Path to parent document (for PDF pages)
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ImageRecord":
        return cls(
            id=row["id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            file_mtime=row["file_mtime"],
            file_size=row["file_size"],
            thumbnail_path=row["thumbnail_path"],
            vector_id=row["vector_id"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None,
            media_type=row["media_type"] if "media_type" in row.keys() else "image",
            frame_count=row["frame_count"] if "frame_count" in row.keys() else None,
            duration_seconds=row["duration_seconds"] if "duration_seconds" in row.keys() else None,
            page_number=row["page_number"] if "page_number" in row.keys() else None,
            parent_document=row["parent_document"] if "parent_document" in row.keys() else None,
        )


@dataclass
class IndexedFolder:
    """Represents a tracked source folder"""
    id: int
    folder_path: str
    recursive: bool
    added_at: datetime
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "IndexedFolder":
        return cls(
            id=row["id"],
            folder_path=row["folder_path"],
            recursive=bool(row["recursive"]),
            added_at=datetime.fromisoformat(row["added_at"]) if row["added_at"] else None,
        )


class DatabaseManager:
    """Thread-safe SQLite database manager for index metadata"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.db_path = get_index_db_path(index_name)
        self._local = threading.local()
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._init_schema()
    
    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local connection"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for better concurrency
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn
    
    def _init_schema(self):
        """Create database tables if they don't exist"""
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path       TEXT UNIQUE NOT NULL,
                    file_hash       TEXT NOT NULL,
                    file_mtime      REAL NOT NULL,
                    file_size       INTEGER NOT NULL,
                    thumbnail_path  TEXT,
                    vector_id       INTEGER NOT NULL,
                    indexed_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    media_type      TEXT DEFAULT 'image',
                    frame_count     INTEGER,
                    duration_seconds REAL,
                    page_number     INTEGER,
                    parent_document TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path);
                CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);
                CREATE INDEX IF NOT EXISTS idx_images_vector_id ON images(vector_id);
                
                CREATE TABLE IF NOT EXISTS indexed_folders (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_path     TEXT UNIQUE NOT NULL,
                    recursive       INTEGER NOT NULL DEFAULT 1,
                    added_at        TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_folders_path ON indexed_folders(folder_path);
                
                -- Track deleted vector IDs for compaction
                CREATE TABLE IF NOT EXISTS deleted_vectors (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_id       INTEGER NOT NULL,
                    deleted_at      TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_deleted_vectors ON deleted_vectors(vector_id);
            """)
        
        # Migration: Add new columns to existing databases
        self._migrate_schema()
    
    def _migrate_schema(self):
        """Add new columns to existing databases (for backwards compatibility)"""
        cursor = self._conn.execute("PRAGMA table_info(images)")
        existing_columns = {row["name"] for row in cursor.fetchall()}
        
        migrations = [
            ("media_type", "TEXT DEFAULT 'image'"),
            ("frame_count", "INTEGER"),
            ("duration_seconds", "REAL"),
            ("page_number", "INTEGER"),
            ("parent_document", "TEXT"),
        ]
        
        for column_name, column_def in migrations:
            if column_name not in existing_columns:
                try:
                    self._conn.execute(f"ALTER TABLE images ADD COLUMN {column_name} {column_def}")
                    print(f"[SemanticSearch] Migrated database: added column '{column_name}'")
                except sqlite3.OperationalError:
                    pass  # Column already exists
    
    # ==================== Image Operations ====================
    
    def add_image(
        self,
        file_path: str,
        file_hash: str,
        file_mtime: float,
        file_size: int,
        thumbnail_path: str,
        vector_id: int,
        media_type: str = "image",
        frame_count: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        page_number: Optional[int] = None,
        parent_document: Optional[str] = None,
    ) -> int:
        """Add or update an image/video/document record. Returns the record ID."""
        with self._conn:
            cursor = self._conn.execute("""
                INSERT INTO images (file_path, file_hash, file_mtime, file_size, thumbnail_path, 
                                    vector_id, media_type, frame_count, duration_seconds,
                                    page_number, parent_document)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    file_mtime = excluded.file_mtime,
                    file_size = excluded.file_size,
                    thumbnail_path = excluded.thumbnail_path,
                    vector_id = excluded.vector_id,
                    media_type = excluded.media_type,
                    frame_count = excluded.frame_count,
                    duration_seconds = excluded.duration_seconds,
                    page_number = excluded.page_number,
                    parent_document = excluded.parent_document,
                    indexed_at = CURRENT_TIMESTAMP
            """, (file_path, file_hash, file_mtime, file_size, thumbnail_path, vector_id,
                  media_type, frame_count, duration_seconds, page_number, parent_document))
            return cursor.lastrowid
    
    def get_image_by_path(self, file_path: str) -> Optional[ImageRecord]:
        """Get an image record by file path"""
        cursor = self._conn.execute(
            "SELECT * FROM images WHERE file_path = ?", (file_path,)
        )
        row = cursor.fetchone()
        return ImageRecord.from_row(row) if row else None
    
    def get_image_by_vector_id(self, vector_id: int) -> Optional[ImageRecord]:
        """Get an image record by its FAISS vector ID"""
        cursor = self._conn.execute(
            "SELECT * FROM images WHERE vector_id = ?", (vector_id,)
        )
        row = cursor.fetchone()
        return ImageRecord.from_row(row) if row else None
    
    def get_images_by_vector_ids(self, vector_ids: List[int]) -> Dict[int, ImageRecord]:
        """Get multiple image records by vector IDs. Returns dict mapping vector_id -> record."""
        if not vector_ids:
            return {}
        
        placeholders = ",".join("?" * len(vector_ids))
        cursor = self._conn.execute(
            f"SELECT * FROM images WHERE vector_id IN ({placeholders})", vector_ids
        )
        return {row["vector_id"]: ImageRecord.from_row(row) for row in cursor.fetchall()}
    
    def get_all_images(self) -> List[ImageRecord]:
        """Get all image records"""
        cursor = self._conn.execute("SELECT * FROM images ORDER BY id")
        return [ImageRecord.from_row(row) for row in cursor.fetchall()]
    
    def get_images_in_folder(self, folder_path: str, recursive: bool = True) -> List[ImageRecord]:
        """Get all images within a folder"""
        folder_path = str(folder_path).rstrip("/\\")
        if recursive:
            pattern = folder_path + "%"
            cursor = self._conn.execute(
                "SELECT * FROM images WHERE file_path LIKE ?", (pattern,)
            )
        else:
            # Non-recursive: match direct children only
            pattern = folder_path + "[/\\\\][^/\\\\]*"
            cursor = self._conn.execute(
                "SELECT * FROM images WHERE file_path GLOB ?", (pattern,)
            )
        return [ImageRecord.from_row(row) for row in cursor.fetchall()]
    
    def remove_image(self, file_path: str) -> bool:
        """Remove an image record. Returns True if a record was deleted."""
        with self._conn:
            # First get the vector_id so we can track it
            cursor = self._conn.execute(
                "SELECT vector_id FROM images WHERE file_path = ?", (file_path,)
            )
            row = cursor.fetchone()
            if row:
                vector_id = row[0]
                # Track the deleted vector_id
                self._conn.execute(
                    "INSERT INTO deleted_vectors (vector_id) VALUES (?)", (vector_id,)
                )
                # Delete the record
                self._conn.execute(
                    "DELETE FROM images WHERE file_path = ?", (file_path,)
                )
                return True
            return False
    
    def remove_images_batch(self, file_paths: List[str]) -> int:
        """Remove multiple image records. Returns count of deleted records."""
        if not file_paths:
            return 0
        with self._conn:
            placeholders = ",".join("?" * len(file_paths))
            # First get vector_ids to track
            cursor = self._conn.execute(
                f"SELECT vector_id FROM images WHERE file_path IN ({placeholders})", file_paths
            )
            vector_ids = [row[0] for row in cursor.fetchall()]
            
            # Track deleted vector_ids
            if vector_ids:
                self._conn.executemany(
                    "INSERT INTO deleted_vectors (vector_id) VALUES (?)",
                    [(vid,) for vid in vector_ids]
                )
            
            # Delete records
            cursor = self._conn.execute(
                f"DELETE FROM images WHERE file_path IN ({placeholders})", file_paths
            )
            return cursor.rowcount
    
    def remove_images_in_folder(self, folder_path: str) -> int:
        """Remove all images within a folder. Returns count of deleted records."""
        folder_path = str(folder_path).rstrip("/\\")
        pattern = folder_path + "%"
        with self._conn:
            # First get vector_ids to track
            cursor = self._conn.execute(
                "SELECT vector_id FROM images WHERE file_path LIKE ?", (pattern,)
            )
            vector_ids = [row[0] for row in cursor.fetchall()]
            
            # Track deleted vector_ids
            if vector_ids:
                self._conn.executemany(
                    "INSERT INTO deleted_vectors (vector_id) VALUES (?)",
                    [(vid,) for vid in vector_ids]
                )
            
            # Delete records
            cursor = self._conn.execute(
                "DELETE FROM images WHERE file_path LIKE ?", (pattern,)
            )
            return cursor.rowcount
    
    def image_needs_update(self, file_path: str, file_mtime: float, file_size: int) -> bool:
        """Check if an image needs re-indexing based on mtime/size change"""
        record = self.get_image_by_path(file_path)
        if record is None:
            return True
        return record.file_mtime != file_mtime or record.file_size != file_size
    
    def get_image_count(self) -> int:
        """Get total number of indexed images"""
        cursor = self._conn.execute("SELECT COUNT(*) FROM images")
        return cursor.fetchone()[0]
    
    def get_max_vector_id(self) -> int:
        """Get the highest vector_id in use (for appending new vectors)"""
        cursor = self._conn.execute("SELECT MAX(vector_id) FROM images")
        result = cursor.fetchone()[0]
        return result if result is not None else -1
    
    # ==================== Folder Operations ====================
    
    def add_folder(self, folder_path: str, recursive: bool = True) -> int:
        """Add a folder to track. Returns the folder ID."""
        with self._conn:
            cursor = self._conn.execute("""
                INSERT INTO indexed_folders (folder_path, recursive)
                VALUES (?, ?)
                ON CONFLICT(folder_path) DO UPDATE SET
                    recursive = excluded.recursive,
                    added_at = CURRENT_TIMESTAMP
            """, (str(folder_path), int(recursive)))
            return cursor.lastrowid
    
    def get_folder(self, folder_path: str) -> Optional[IndexedFolder]:
        """Get a folder record"""
        cursor = self._conn.execute(
            "SELECT * FROM indexed_folders WHERE folder_path = ?", (str(folder_path),)
        )
        row = cursor.fetchone()
        return IndexedFolder.from_row(row) if row else None
    
    def get_all_folders(self) -> List[IndexedFolder]:
        """Get all tracked folders"""
        cursor = self._conn.execute("SELECT * FROM indexed_folders ORDER BY added_at")
        return [IndexedFolder.from_row(row) for row in cursor.fetchall()]
    
    def remove_folder(self, folder_path: str) -> bool:
        """Remove a folder from tracking (does not remove images). Returns True if deleted."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM indexed_folders WHERE folder_path = ?", (str(folder_path),)
            )
            return cursor.rowcount > 0
    
    def get_folder_count(self) -> int:
        """Get number of tracked folders"""
        cursor = self._conn.execute("SELECT COUNT(*) FROM indexed_folders")
        return cursor.fetchone()[0]
    
    # ==================== Maintenance ====================
    
    def get_stale_images(self) -> List[ImageRecord]:
        """Find images whose files no longer exist on disk"""
        all_images = self.get_all_images()
        stale = []
        for img in all_images:
            if not Path(img.file_path).exists():
                stale.append(img)
        return stale
    
    def vacuum(self):
        """Optimize database file size"""
        self._conn.execute("VACUUM")
    
    def close(self):
        """Close the database connection"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    # ==================== Compaction Operations ====================
    
    def get_deleted_vector_ids(self) -> List[int]:
        """Get all deleted vector IDs that are still in FAISS but removed from DB"""
        cursor = self._conn.execute("SELECT DISTINCT vector_id FROM deleted_vectors")
        return [row[0] for row in cursor.fetchall()]
    
    def get_deleted_vector_count(self) -> int:
        """Get count of deleted vectors pending compaction"""
        cursor = self._conn.execute("SELECT COUNT(DISTINCT vector_id) FROM deleted_vectors")
        return cursor.fetchone()[0]
    
    def get_all_vector_ids(self) -> List[int]:
        """Get all valid vector IDs currently in the database (sorted)"""
        cursor = self._conn.execute("SELECT vector_id FROM images ORDER BY vector_id")
        return [row[0] for row in cursor.fetchall()]
    
    def get_vector_id_mapping(self) -> Dict[int, int]:
        """
        Create mapping from old vector IDs to new sequential IDs.
        Returns dict mapping old_id -> new_id (0-based sequential)
        """
        old_ids = self.get_all_vector_ids()
        return {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    
    def update_vector_ids(self, id_mapping: Dict[int, int]):
        """
        Update vector IDs in the database according to mapping.
        
        Uses a two-pass approach to avoid ID collisions:
        1. First pass: offset all IDs by a large number to avoid overlap
        2. Second pass: assign final sequential IDs
        
        Args:
            id_mapping: Dict mapping old_vector_id -> new_vector_id
        """
        if not id_mapping:
            return
            
        # Use offset larger than any possible vector ID
        max_id = max(max(id_mapping.keys()), max(id_mapping.values()))
        offset = max_id + 1000000
        
        with self._conn:
            # Pass 1: Move all affected IDs to temporary offset range
            for old_id in id_mapping.keys():
                self._conn.execute(
                    "UPDATE images SET vector_id = ? WHERE vector_id = ?",
                    (old_id + offset, old_id)
                )
            
            # Pass 2: Move from offset to final IDs
            for old_id, new_id in id_mapping.items():
                self._conn.execute(
                    "UPDATE images SET vector_id = ? WHERE vector_id = ?",
                    (new_id, old_id + offset)
                )
    
    def clear_deleted_vectors(self):
        """Clear the deleted_vectors table after successful compaction"""
        with self._conn:
            self._conn.execute("DELETE FROM deleted_vectors")
    
    # ==================== Stats ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "index_name": self.index_name,
            "image_count": self.get_image_count(),
            "folder_count": self.get_folder_count(),
            "folders": [f.folder_path for f in self.get_all_folders()],
            "deleted_vector_count": self.get_deleted_vector_count(),
            "db_path": str(self.db_path),
            "db_size_mb": round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0,
        }

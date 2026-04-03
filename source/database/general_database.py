import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class ImageDatabaseManager:
    """Manages image metadata and tagging in SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager with optional custom database path."""
        if db_path is None:
            self.database_path = (
                Path(__file__).parent.joinpath("resource").joinpath("imagedatabase.db")
            )
            self.db_path = str(self.database_path)
        else:
            self.db_path = db_path
            self.database_path = Path(db_path)

        # Ensure directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_database_exists()

    @contextmanager
    def _get_connection(self):
        """Context manager for safe database access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _ensure_database_exists(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    original_filename TEXT NOT NULL,
                    file_size INTEGER,
                    mime_type TEXT,
                    hash_value TEXT NOT NULL UNIQUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_name TEXT NOT NULL UNIQUE,
                    color_hex TEXT,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create image_tags junction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(image_id, tag_id),
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
                    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_hash_value
                ON images(hash_value)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_tags_image_id
                ON image_tags(image_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_tags_tag_id
                ON image_tags(tag_id)
            """)

    def compute_hash(self, file_path: str, block_size: int = 4096) -> str:
        """Compute MD5 hash of an image file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]  # Use first 8 characters of MD5

    def _guess_mime_type(self, file_path: str) -> str:
        """Guess MIME type based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".tiff": "image/tiff",
            ".svg": "image/svg+xml",
        }
        return mime_types.get(ext, "image/unknown")

    def add_image(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
        tags: Optional[List[str]] = None,
        create_tags: bool = False,
    ) -> Tuple[int, List[str]]:
        """Add an image with tags to the database."""

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file information
        file_size = os.path.getsize(file_path)
        mime_type = self._guess_mime_type(file_path)
        hash_value = self.compute_hash(file_path)

        # Check if image already exists by hash
        result = self._get_image_by_hash(hash_value)
        if result:
            image_id, _ = result
            return (image_id, [])

        # Insert new image
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO images
                    (file_path, original_filename, file_size, mime_type, hash_value, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        os.path.abspath(file_path),
                        original_filename or os.path.basename(file_path),
                        file_size,
                        mime_type,
                        hash_value,
                        datetime.now().isoformat(),
                    ),
                )
                image_id: int = cursor.lastrowid or 0

                added_tags: List[str] = []

                # Add tags to image
                if tags and create_tags:
                    for tag_name in tags:
                        tag_id = self._create_or_get_tag(tag_name)
                        try:
                            cursor.execute(
                                """
                                INSERT INTO image_tags (image_id, tag_id, created_at)
                                VALUES (?, ?, ?)
                            """,
                                (image_id, tag_id, datetime.now().isoformat()),
                            )
                            added_tags.append(tag_name)
                        except sqlite3.IntegrityError:
                            pass
                elif tags:
                    for tag_name in tags:
                        tag_id = self._get_tag_by_name(tag_name)
                        if not tag_id:
                            raise ValueError(
                                f"Tag '{tag_name}' does not exist. Create it first."
                            )
                        try:
                            cursor.execute(
                                """
                                INSERT INTO image_tags (image_id, tag_id, created_at)
                                VALUES (?, ?, ?)
                            """,
                                (image_id, tag_id, datetime.now().isoformat()),
                            )
                            added_tags.append(tag_name)
                        except sqlite3.IntegrityError:
                            pass

            return (image_id, added_tags)

        except sqlite3.IntegrityError as e:
            # Handle duplicate insertion
            result = self._get_image_by_hash(hash_value)
            if result:
                image_id, _ = result
                return (image_id, [])
            raise e

    def _create_or_get_tag(self, tag_name: str, color_hex: Optional[str] = None) -> int:
        """Create tag if it doesn't exist, otherwise return existing tag ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Try to get existing tag
            cursor.execute("SELECT id FROM tags WHERE tag_name = ?", (tag_name,))
            result = cursor.fetchone()

            if result:
                return result["id"]

            # Create new tag with default color
            cursor.execute(
                """
                INSERT INTO tags (tag_name, color_hex)
                VALUES (?, ?)
            """,
                (tag_name, color_hex or self._generate_default_color(tag_name)),
            )

            cursor.execute("SELECT id FROM tags WHERE tag_name = ?", (tag_name,))
            return cursor.fetchone()["id"]

    def _get_tag_by_name(self, tag_name: str) -> Optional[int]:
        """Retrieve tag ID by name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM tags WHERE tag_name = ?", (tag_name,))
            result = cursor.fetchone()

            if result:
                return result["id"]

            return None

    def _get_image_by_hash(self, hash_value: str) -> Optional[Tuple[int, str]]:
        """Find image by hash value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, created_at FROM images WHERE hash_value = ?", (hash_value,)
            )
            result = cursor.fetchone()

            if result:
                return (result["id"], result["created_at"])

            return None

    def _get_images_by_tags(self, tag_names: List[str]) -> List[sqlite3.Row]:
        """Retrieve images filtered by tags."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            placeholders = ",".join(["?" for _ in tag_names])
            query = f"""
                SELECT i.*
                FROM images i
                JOIN image_tags it ON i.id = it.image_id
                JOIN tags t ON it.tag_id = t.id
                WHERE t.tag_name IN ({placeholders})
                GROUP BY i.id
            """

            cursor.execute(query, tag_names)
            return list(cursor.fetchall())

    def _get_image_with_tags(
        self, image_id: int
    ) -> Optional[Tuple[sqlite3.Row, List[str]]]:
        """Retrieve an image with its associated tags."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get image with tags
            cursor.execute(
                """
                SELECT i.*, GROUP_CONCAT(t.tag_name, ',') as tags
                FROM images i
                LEFT JOIN image_tags it ON i.id = it.image_id
                LEFT JOIN tags t ON it.tag_id = t.id
                WHERE i.id = ?
                GROUP BY i.id
            """,
                (image_id,),
            )

            result = cursor.fetchone()

            if result:
                tags = result["tags"].split(",") if result["tags"] else []
                return (result, tags)

        return None

    def search_by_partial_name(self, query: str) -> List[sqlite3.Row]:
        """Search for images by partial filename or tag name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            search_query = """
                SELECT i.*,
                       GROUP_CONCAT(t.tag_name || ' (' || t.color_hex || ')', ', ') as tag_str
                FROM images i
                LEFT JOIN image_tags it ON i.id = it.image_id
                LEFT JOIN tags t ON it.tag_id = t.id
                WHERE i.original_filename LIKE ? OR t.tag_name LIKE ?
                GROUP BY i.id
            """

            search_term = f"%{query}%"
            cursor.execute(search_query, (search_term, search_term))
            return list(cursor.fetchall())

    def _generate_default_color(self, tag_name: str) -> str:
        """Generate a deterministic color based on tag name."""
        # Simple hash-based color generation
        hash_val = int(hashlib.md5(tag_name.encode()).hexdigest(), 16)

        # Distribute colors across the spectrum
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF
        b = hash_val & 0xFF

        return f"#{r:02x}{g:02x}{b:02x}"

    def get_images(self, tags: list[str]) -> list[str]:
        image_rows = self._get_images_by_tags(tags)
        image_paths = [_["file_path"] for _ in image_rows]
        index = 0
        
        while index < len(image_paths):
            if not os.path.exists(image_paths[index]):
                self.remove_image(image_paths[index])
                image_paths.pop(index)
                continue
            
            index += 1

        return image_paths

    def remove_image(self, file_path: str) -> bool:
        """Remove an image from the database by file path."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM images WHERE file_path = ?", (file_path,))
            return cursor.rowcount > 0


if __name__ == "__main__":
    # Test the database manager
    db = ImageDatabaseManager()
    print(f"Database initialized at: {db.db_path}")

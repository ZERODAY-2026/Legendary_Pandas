"""
File format and security validator for incoming legal documents.
Checks file extension, real magic bytes, file size, and basic readability.
"""

from pathlib import Path
from backend.app.config import settings


class ValidationError(Exception):
    """Raised when document fails validation checks."""

    pass


def validate_file(file_path: Path, filename: str, content: bytes) -> str:
    """Validate document prior to ingestion.

    Returns the detected format ('pdf' or 'txt').
    """
    if settings.MAX_FILE_SIZE_BYTES is not None and len(content) > settings.MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ValidationError(f"Unsupported file extension '{ext}'. Only .pdf and .txt are supported.")

    # Validate magic bytes / real file format
    if ext == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ValidationError("File has .pdf extension but does not contain valid PDF header.")
        return "pdf"
    elif ext == ".txt":
        # Check text encoding readability
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content.decode("latin-1")
            except Exception as e:
                raise ValidationError(f"Unable to decode text file contents: {e}")
        return "txt"

    raise ValidationError(f"Unknown format for file: {filename}")

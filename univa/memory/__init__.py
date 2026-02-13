"""
Timeline-centric memory for video2video projects.

This package is intentionally framework-light:
- SQLite is the source of truth (per-project DB)
- Optional: sync summaries into an agent framework's "text memory"
"""

from .service import ProjectMemoryService

__all__ = ["ProjectMemoryService"]


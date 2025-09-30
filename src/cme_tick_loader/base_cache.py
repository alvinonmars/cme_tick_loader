"""Base cache implementation for shared functionality"""

import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod


class BaseCache(ABC):
    """Base class for cache implementations"""

    def __init__(self, base_path=None, cache_subdir='cache', cache_dir=None):
        """
        Initialize base cache

        Args:
            base_path: Base path for data (auto-detects if None)
            cache_subdir: Subdirectory name under .cache/
            cache_dir: Custom cache directory (overrides default)
        """
        if base_path is None:
            from .config import get_default_base_path
            base_path = get_default_base_path()
        self.base_path = Path(base_path)

        if cache_dir is None:
            self.cache_dir = self.base_path / '.cache' / cache_subdir
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def get_cache_key(self, *args, **kwargs):
        """Generate cache filename - must be implemented by subclasses"""
        pass

    def save_data(self, data, cache_key):
        """Save data to cache"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            data.to_pickle(cache_path)
        except Exception as e:
            print(f"Warning: Could not save cache {cache_key}: {e}")

    def load_data(self, cache_key):
        """Load data from cache"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                return pd.read_pickle(cache_path)
            except Exception:
                return None
        return None

    def exists(self, cache_key):
        """Check if cache exists"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        return cache_path.exists()

    def clear_cache(self, pattern=None):
        """Clear cache files"""
        if pattern:
            for cache_file in self.cache_dir.glob(f"{pattern}.pkl"):
                cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

    def get_cache_info(self):
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'cache_dir': str(self.cache_dir),
            'file_count': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in cache_files]
        }
"""Simple in-memory TTL cache implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """Store cached value with expiration metadata."""

    value: V
    expires_at: float


class TTLCache(Generic[K, V]):
    """Very small utility cache for prompt optimization responses."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._data: Dict[K, CacheEntry[V]] = {}

    def set(self, key: K, value: V) -> None:
        """Insert a value into the cache."""
        self._data[key] = CacheEntry(value=value, expires_at=time.time() + self.ttl_seconds)

    def get(self, key: K) -> Optional[V]:
        """Retrieve a cached value if it has not expired."""
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.time():
            self._data.pop(key, None)
            return None
        return entry.value

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()

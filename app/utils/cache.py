"""
Caching module for System Dynamics Backend
Implements LRU cache for parsed ASTs, dependency graphs, and evaluation order
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import logging
import threading

from app.models import Element, Link

logger = logging.getLogger(__name__)


def hash_model(elements: List[Element], links: List[Link]) -> str:
    """
    Generate a hash for a model based on elements and links

    The hash is computed from:
    - Element IDs, types, names, equations, initial values, and values
    - Link IDs, sources, and targets

    Args:
        elements: List of model elements
        links: List of connections between elements

    Returns:
        Hexadecimal hash string representing the model
    """
    # Create a deterministic representation of the model
    model_data = {
        "elements": sorted(
            [
                {
                    "id": elem.id,
                    "type": elem.type,
                    "name": elem.name,
                    "initial": elem.initial,
                    "equation": elem.equation or "",
                    "value": elem.value,
                }
                for elem in elements
            ],
            key=lambda x: x["id"],
        ),
        "links": sorted(
            [
                {"id": link.id, "source": link.source, "target": link.target}
                for link in links
            ],
            key=lambda x: (x["source"], x["target"], x["id"]),
        ),
    }

    # Convert to JSON string and hash
    model_json = json.dumps(model_data, sort_keys=True, ensure_ascii=False)
    hash_obj = hashlib.sha256(model_json.encode("utf-8"))
    return hash_obj.hexdigest()


class ModelCache:
    """
    LRU cache for model parsing results

    Caches:
    - Parsed ASTs (mapping element ID to AST)
    - Dependency graph
    - Sorted evaluation order

    Uses LRU eviction with max size of 50 entries
    """

    def __init__(self, max_size: int = 50):
        """
        Initialize cache with maximum size

        Args:
            max_size: Maximum number of cached models (default: 50)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()  # Thread-safe access
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, model_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data for a model

        Args:
            model_hash: Hash of the model

        Returns:
            Dictionary containing 'asts', 'dependency_graph', and 'evaluation_order',
            or None if not cached
        """
        with self._lock:
            if model_hash in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_hash)
                self._stats["hits"] += 1
                logger.debug(f"Cache hit for model hash: {model_hash[:16]}...")
                return self._cache[model_hash].copy()
            else:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss for model hash: {model_hash[:16]}...")
                return None

    def put(
        self,
        model_hash: str,
        asts: Dict[str, Any],
        dependency_graph: Dict[str, List[str]],
        evaluation_order: List[str],
    ) -> None:
        """
        Store parsed model data in cache

        Args:
            model_hash: Hash of the model
            asts: Dictionary mapping element IDs to parsed ASTs
            dependency_graph: Dependency graph dictionary
            evaluation_order: Topologically sorted evaluation order
        """
        with self._lock:
            # Remove oldest entry if cache is full
            if len(self._cache) >= self.max_size and model_hash not in self._cache:
                # Remove least recently used (first item)
                oldest_hash, _ = self._cache.popitem(last=False)
                self._stats["evictions"] += 1
                logger.debug(
                    f"Cache eviction: removed model hash {oldest_hash[:16]}..."
                )

            # Store new entry (or update existing)
            self._cache[model_hash] = {
                "asts": asts.copy(),
                "dependency_graph": dependency_graph.copy(),
                "evaluation_order": evaluation_order.copy(),
            }
            # Move to end (most recently used)
            self._cache.move_to_end(model_hash)
            logger.debug(
                f"Cached model hash: {model_hash[:16]}... (cache size: {len(self._cache)})"
            )

    def clear(self) -> None:
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._stats["hits"] = 0
            self._stats["misses"] = 0
            self._stats["evictions"] = 0
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics:
            - size: Current number of cached entries
            - max_size: Maximum cache size
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Hit rate as percentage
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                (self._stats["hits"] / total_requests * 100)
                if total_requests > 0
                else 0.0
            )

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": round(hit_rate, 2),
            }


# Global cache instance
# Initialize cache with default size (will be updated on first access)
_model_cache: Optional[ModelCache] = None
_cache_lock = threading.Lock()


def _get_model_cache() -> ModelCache:
    """Get or create the global model cache instance"""
    global _model_cache
    if _model_cache is None:
        with _cache_lock:
            if _model_cache is None:
                from app.config import get_settings
                config = get_settings()
                _model_cache = ModelCache(max_size=config.cache_max_size)
                logger.info(f"Initialized model cache with max_size={config.cache_max_size}")
    return _model_cache


def get_cache() -> ModelCache:
    """
    Get the global model cache instance

    Returns:
        Global ModelCache instance
    """
    return _get_model_cache()

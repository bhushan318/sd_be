"""
Result store for simulation results
Implements in-memory storage for simulation results with optional persistence
"""

import os
import uuid
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import threading
import logging

logger = logging.getLogger(__name__)


class SimulationResult:
    """
    Container for simulation result data

    Attributes:
        id: Unique identifier for the result
        time: List of time points
        results: Dictionary mapping element IDs to time series data
        created_at: Timestamp when result was created
        expires_at: Optional expiration timestamp
    """

    def __init__(
        self,
        time: List[float],
        results: Dict[str, List[float]],
        result_id: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ):
        """
        Initialize simulation result

        Args:
            time: List of time points
            results: Dictionary mapping element IDs to time series data
            result_id: Optional unique identifier (generated if not provided)
            ttl_hours: Optional time-to-live in hours (default: 24)
        """
        self.id = result_id or str(uuid.uuid4())
        self.time = time
        self.results = results
        self.created_at = datetime.now(timezone.utc)

        # Default TTL: 24 hours
        ttl = ttl_hours if ttl_hours is not None else 24
        self.expires_at = self.created_at + timedelta(hours=ttl)

    def is_expired(self) -> bool:
        """
        Check if result has expired based on expiration timestamp
        
        Returns:
            True if result has expired, False otherwise
        """
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format
        
        Returns:
            Dictionary with 'id', 'time', 'results', 'created_at', and 'expires_at' keys
        """
        return {
            "id": self.id,
            "time": self.time,
            "results": self.results,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    def filter_elements(
        self, element_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Filter results by element IDs

        Args:
            element_ids: Optional list of element IDs to include. If None, returns all.

        Returns:
            Dictionary with filtered time and results
        """
        if element_ids is None:
            return {"time": self.time, "results": self.results}

        # Filter results to only include requested elements
        filtered_results = {
            elem_id: self.results[elem_id]
            for elem_id in element_ids
            if elem_id in self.results
        }

        return {"time": self.time, "results": filtered_results}


class ResultStore:
    """
    Thread-safe in-memory store for simulation results

    Uses LRU-like eviction based on expiration time.
    Results are automatically cleaned up when expired.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize result store

        Args:
            max_size: Maximum number of results to store (default: 1000)
        """
        self._store: Dict[str, SimulationResult] = {}
        self._lock = threading.RLock()
        self.max_size = max_size

    def store(self, result: SimulationResult) -> str:
        """
        Store a simulation result

        Args:
            result: SimulationResult to store

        Returns:
            Result ID
        """
        with self._lock:
            # Clean up expired results before storing
            self._cleanup_expired()

            # If at capacity, remove oldest result
            if len(self._store) >= self.max_size:
                self._evict_oldest()

            self._store[result.id] = result
            logger.debug(f"Stored simulation result: {result.id}")
            return result.id

    def get(self, result_id: str) -> Optional[SimulationResult]:
        """
        Retrieve a simulation result by ID

        Args:
            result_id: Unique result identifier

        Returns:
            SimulationResult if found and not expired, None otherwise
        """
        with self._lock:
            result = self._store.get(result_id)

            if result is None:
                return None

            # Check if expired
            if result.is_expired():
                del self._store[result_id]
                logger.debug(f"Result {result_id} expired and removed")
                return None

            return result

    def delete(self, result_id: str) -> bool:
        """
        Delete a result by ID

        Args:
            result_id: Unique result identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if result_id in self._store:
                del self._store[result_id]
                logger.debug(f"Deleted simulation result: {result_id}")
                return True
            return False

    def _cleanup_expired(self) -> None:
        """
        Remove all expired results from the store
        
        This method should only be called while holding self._lock
        to ensure thread-safe operation.
        """
        expired_ids = [
            result_id
            for result_id, result in self._store.items()
            if result.is_expired()
        ]

        for result_id in expired_ids:
            del self._store[result_id]
            logger.debug(f"Cleaned up expired result: {result_id}")

    def _evict_oldest(self) -> None:
        """
        Evict the oldest result
        
        Note: This method should only be called while holding self._lock
        to ensure thread-safe operation.
        """
        if not self._store:
            return

        # Find oldest result by created_at (atomic operation within lock)
        # Use list() to create a snapshot of keys to avoid issues if dict changes
        oldest_id = min(
            list(self._store.keys()),
            key=lambda rid: self._store[rid].created_at
        )
        # Delete is atomic, but we've already locked the dict access above
        if oldest_id in self._store:  # Double-check it still exists
            del self._store[oldest_id]
            logger.debug(f"Evicted oldest result: {oldest_id}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get store statistics

        Returns:
            Dictionary with store statistics
        """
        with self._lock:
            self._cleanup_expired()
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "results": [
                    {
                        "id": result.id,
                        "created_at": result.created_at.isoformat(),
                        "expires_at": result.expires_at.isoformat(),
                        "element_count": len(result.results),
                        "time_points": len(result.time),
                    }
                    for result in self._store.values()
                ],
            }


# Global result store instance (singleton pattern)
_result_store: Optional[ResultStore] = None
_store_lock = threading.Lock()


def get_result_store() -> ResultStore:
    """
    Get the global result store instance

    Returns:
        ResultStore singleton instance
    """
    global _result_store

    if _result_store is None:
        with _store_lock:
            if _result_store is None:
                from app.config import get_settings
                config = get_settings()
                _result_store = ResultStore(max_size=config.result_store_max_size)
                logger.info(f"Initialized result store with max_size={config.result_store_max_size}")

    return _result_store
